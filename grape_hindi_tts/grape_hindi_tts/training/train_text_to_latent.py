"""
Stage 2: Text-to-Latent Module Training for SupertonicTTS Hindi.

Flow matching pipeline using components from two sources:
  ┌─────────────────────────────────────────────────────────────────┐
  │  FROM F5-TTS (SWivid/F5-TTS, Apache 2.0)                       │
  │    • OT-CFM interpolation: z_t = (1−(1−σ)t)z₀ + t·z₁         │
  │    • Velocity target:       u_t = z₁ − (1−σ)z₀               │
  │    • Sway-sampling schedule for efficient inference ODE         │
  │    • CFG double-batch trick at inference                        │
  │    • ConvNeXtV2Block with GRN (replaces our hand-written V1)   │
  ├─────────────────────────────────────────────────────────────────┤
  │  FROM SupertonicTTS paper (Supertone Inc., 2025)               │
  │    • Masked L1 loss (not MSE)                                  │
  │    • Context-sharing batch expansion (Algorithm 1, Ke=4)       │
  │    • SPFM — Self-Purifying Flow Matching                       │
  │    • LARoPE in text/reference cross-attention                  │
  │    • Utterance-level duration predictor (Stage 3)              │
  └─────────────────────────────────────────────────────────────────┘

Training stages:
  1. (pre-req) Speech Autoencoder must be trained first
  2. (pre-req) Latents must be precomputed with precompute_latents.py
  3. This script: trains TextToLatentModule + wires SupertonicCFM
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ── SupertonicTTS model components ──────────────────────────────────────────
from supertonic_hindi_tts.models.text_to_latent import TextToLatentModule
from supertonic_hindi_tts.models.speech_autoencoder import SpeechAutoencoder

# ── F5-TTS flow matching engine ──────────────────────────────────────────────
from supertonic_hindi_tts.third_party.f5_tts_cfm import (
    SupertonicCFM,          # wraps VFEstimator with OT-CFM + SPFM + CFG
    ot_cfm_interpolate,     # z_t, u_t computation
    cfm_loss,               # masked L1
    euler_solve,            # Euler ODE solver with sway-sampling
    SPFMFilter,             # self-purifying sample filter
)

# ── Training utilities ────────────────────────────────────────────────────────
from supertonic_hindi_tts.training.trainer_utils import (
    setup_training,
    create_optimizer,
    create_scheduler,
    save_checkpoint,
    load_checkpoint,
    AverageMeter,
    count_parameters,
    get_lr,
    gradient_clip,
    load_config,
)

logger = logging.getLogger(__name__)


# ============================================================================
# TTL Dataset  (precomputed latents + text token IDs)
# ============================================================================

class TTLDataset(Dataset):
    """Text-to-Latent training dataset.

    Expects precomputed latent .npy files alongside the manifest.
    Each manifest entry:
        {
          "audio_path":   "...",
          "latent_path":  "...",    ← path to .npy file of shape (C, T)
          "text":         "...",
          "text_tokens":  [...],    ← list of int token IDs
          "speaker_id":   "...",
          "duration":     float,    ← seconds
          "gender":       "M/F"
        }
    """

    def __init__(
        self,
        manifest_path: str,
        latent_mean: Optional[torch.Tensor] = None,
        latent_std: Optional[torch.Tensor] = None,
        max_duration: float = 30.0,
        min_duration: float = 0.5,
        ref_crop_min_sec: float = 0.2,
        ref_crop_max_sec: float = 9.0,
        compression_factor: int = 6,
        hop_length: int = 512,
        sample_rate: int = 44100,
    ):
        self.ref_crop_min = ref_crop_min_sec
        self.ref_crop_max = ref_crop_max_sec
        self.Kc = compression_factor
        self.hop = hop_length
        self.sr  = sample_rate
        self.latent_mean = latent_mean
        self.latent_std  = latent_std

        with open(manifest_path) as f:
            all_items = [json.loads(l) for l in f if l.strip()]

        # Filter by duration
        self.items = [
            it for it in all_items
            if min_duration <= it.get("duration", 999) <= max_duration
        ]
        logger.info(f"TTLDataset: {len(self.items)} samples after duration filter")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict:
        item = self.items[idx]

        # Load precomputed latent  (C, T)
        latent = torch.from_numpy(np.load(item["latent_path"])).float()
        C, T = latent.shape

        # Channel-wise normalisation using precomputed statistics
        if self.latent_mean is not None and self.latent_std is not None:
            latent = (latent - self.latent_mean[:, None]) / (self.latent_std[:, None] + 1e-8)

        # Compress latents along time: (C, T) → (Kc·C, T//Kc)
        # (SupertonicTTS §3.2.1 — temporal compression by factor Kc=6)
        T_pad = (self.Kc - T % self.Kc) % self.Kc
        if T_pad:
            latent = F.pad(latent, (0, T_pad))
        T_compressed = latent.shape[1] // self.Kc
        compressed = latent.reshape(C * self.Kc, T_compressed)  # (Kc·C, T//Kc)

        # Reference crop: random segment in [ref_min, ref_max] seconds
        # Must not exceed 50 % of the total duration (paper §3.2.4)
        max_ref_sec = min(self.ref_crop_max, item["duration"] * 0.5)
        ref_dur_sec = np.random.uniform(self.ref_crop_min, max(self.ref_crop_min, max_ref_sec))
        ref_frames  = int(ref_dur_sec * self.sr / self.hop / self.Kc)
        ref_frames  = max(1, min(ref_frames, T_compressed - 1))

        ref_start = np.random.randint(0, max(1, T_compressed - ref_frames))
        ref_latent = compressed[:, ref_start: ref_start + ref_frames]  # (Kc·C, T_ref)

        # Reference mask: 1 everywhere EXCEPT the reference region (for loss)
        ref_mask = torch.ones(1, T_compressed)
        ref_mask[:, ref_start: ref_start + ref_frames] = 0.0

        # Text tokens
        tokens = torch.tensor(item["text_tokens"], dtype=torch.long)

        return {
            "compressed_latent": compressed,    # (Kc·C, T_comp)
            "ref_latent":        ref_latent,    # (Kc·C, T_ref)
            "ref_mask":          ref_mask,      # (1, T_comp)
            "text_tokens":       tokens,        # (L,)
            "duration":          torch.tensor(item["duration"], dtype=torch.float32),
        }


def ttl_collate(batch: list) -> Dict[str, torch.Tensor]:
    """Pad variable-length sequences to batch."""
    max_comp = max(b["compressed_latent"].shape[1] for b in batch)
    max_ref  = max(b["ref_latent"].shape[1] for b in batch)
    max_text = max(b["text_tokens"].shape[0] for b in batch)
    C = batch[0]["compressed_latent"].shape[0]

    comp_pad, ref_pad, mask_pad, tok_pad = [], [], [], []
    for b in batch:
        c = b["compressed_latent"]
        r = b["ref_latent"]
        m = b["ref_mask"]
        t = b["text_tokens"]

        comp_pad.append(F.pad(c, (0, max_comp - c.shape[1])))
        ref_pad.append(F.pad(r, (0, max_ref - r.shape[1])))
        mask_pad.append(F.pad(m, (0, max_comp - m.shape[1])))
        tok_pad.append(F.pad(t, (0, max_text - t.shape[0])))

    return {
        "compressed_latent": torch.stack(comp_pad),   # (B, Kc·C, T_comp)
        "ref_latent":        torch.stack(ref_pad),    # (B, Kc·C, T_ref)
        "ref_mask":          torch.stack(mask_pad),   # (B, 1, T_comp)
        "text_tokens":       torch.stack(tok_pad),    # (B, L)
        "duration":          torch.stack([b["duration"] for b in batch]),
    }


# ============================================================================
# Latent Statistics helper
# ============================================================================

def load_latent_stats(stats_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load channel-wise latent mean and std from precompute_latents output."""
    stats = np.load(stats_path)
    mean = torch.from_numpy(stats["mean"]).float()
    std  = torch.from_numpy(stats["std"]).float()
    return mean, std


# ============================================================================
# Main Trainer
# ============================================================================

class TTLTrainer:
    """Text-to-Latent trainer using SupertonicCFM (F5-TTS CFM backbone)."""

    def __init__(self, config: Dict, device: torch.device, output_dir: Path):
        self.config     = config
        self.device     = device
        self.output_dir = output_dir
        self.step       = 0

        ttl_cfg = config["text_to_latent"]
        data_cfg = config["data"]

        # ── Model ─────────────────────────────────────────────────────────
        self.ttl_module = TextToLatentModule(
            compressed_channels = config["latent"]["compressed_channels"],  # 144
            text_vocab_size     = config["text"]["vocab_size"],
            text_embed_dim      = ttl_cfg["text_encoder"]["hidden_dim"],
            ref_hidden_dim      = ttl_cfg["reference_encoder"]["hidden_dim"],
            vf_hidden_dim       = ttl_cfg["vf_estimator"]["hidden_dim"],
            n_text_convnext     = ttl_cfg["text_encoder"]["n_convnext_blocks"],
            n_ref_convnext      = ttl_cfg["reference_encoder"]["n_convnext_blocks"],
            n_vf_main_blocks    = ttl_cfg["vf_estimator"]["n_main_blocks"],
            time_embed_dim      = ttl_cfg["vf_estimator"]["time_embed_dim"],
            n_learnable_queries = ttl_cfg["reference_encoder"]["n_learnable_queries"],
            p_uncond            = ttl_cfg["training"]["p_uncond"],
        ).to(device)

        # ── F5-TTS SupertonicCFM wrapper ───────────────────────────────────
        # Injects our VFEstimator into the F5-TTS OT-CFM + SPFM engine.
        self.cfm = SupertonicCFM(
            vf_estimator = self.ttl_module.vf_estimator,
            sigma_min    = ttl_cfg["training"].get("sigma_min", 1e-8),
            p_uncond     = ttl_cfg["training"]["p_uncond"],
            Ke           = ttl_cfg["training"]["expansion_factor"],      # 4
            spfm_warmup  = config.get("spfm", {}).get("warmup_iterations", 40_000),
        ).to(device)

        logger.info(f"TTL module parameters : {count_parameters(self.ttl_module):,}")
        logger.info(f"  TextEncoder         : {count_parameters(self.ttl_module.text_encoder):,}")
        logger.info(f"  ReferenceEncoder    : {count_parameters(self.ttl_module.reference_encoder):,}")
        logger.info(f"  VFEstimator         : {count_parameters(self.ttl_module.vf_estimator):,}")

        # ── DGX Spark optimisation: torch.compile ─────────────────────────
        if config.get("dgx_spark", {}).get("compile_model", False):
            logger.info("Compiling model with torch.compile (GB10 optimisation) …")
            self.ttl_module = torch.compile(self.ttl_module)

        # ── Gradient checkpointing ────────────────────────────────────────
        if config.get("dgx_spark", {}).get("gradient_checkpointing", False):
            self.ttl_module.enable_gradient_checkpointing()

        # ── Optimizer & scheduler ─────────────────────────────────────────
        self.optimizer = create_optimizer(
            self.ttl_module,
            lr=ttl_cfg["training"]["learning_rate"],
            weight_decay=ttl_cfg["training"].get("optimizer_weight_decay",
                                  config.get("text_to_latent", {}).get("training", {}).get("optimizer", {}) or 0.01),
        )
        self.scheduler = create_scheduler(
            self.optimizer,
            total_steps    = ttl_cfg["training"]["num_iterations"],
            decay_interval = ttl_cfg["training"]["lr_decay_every"],
            decay_factor   = ttl_cfg["training"]["lr_decay_factor"],
        )

        # ── Mixed precision (BF16 on GB10) ────────────────────────────────
        precision = config.get("project", {}).get("mixed_precision", "bf16")
        self.use_amp = precision in ("bf16", "fp16")
        amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        self.scaler = GradScaler(enabled=(precision == "fp16"))
        self.amp_dtype = amp_dtype

        # ── Latent normalisation stats ────────────────────────────────────
        stats_path = data_cfg.get("latent_stats_path")
        if stats_path and Path(stats_path).exists():
            self.latent_mean, self.latent_std = load_latent_stats(stats_path)
            self.latent_mean = self.latent_mean.to(device)
            self.latent_std  = self.latent_std.to(device)
            logger.info(f"Loaded latent normalisation stats from {stats_path}")
        else:
            self.latent_mean = self.latent_std = None
            logger.warning("No latent stats found — running without normalisation")

        # ── Metric trackers ────────────────────────────────────────────────
        self.loss_meter   = AverageMeter("loss")
        self.best_val     = float("inf")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def build_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        data_cfg = self.config["data"]
        ttl_cfg  = self.config["text_to_latent"]["training"]

        train_ds = TTLDataset(
            manifest_path   = data_cfg["train_manifest"],
            latent_mean     = self.latent_mean,
            latent_std      = self.latent_std,
            ref_crop_min_sec = ttl_cfg["ref_crop_min_sec"],
            ref_crop_max_sec = ttl_cfg["ref_crop_max_sec"],
        )
        val_ds = TTLDataset(
            manifest_path   = data_cfg["val_manifest"],
            latent_mean     = self.latent_mean,
            latent_std      = self.latent_std,
            ref_crop_min_sec = ttl_cfg["ref_crop_min_sec"],
            ref_crop_max_sec = ttl_cfg["ref_crop_max_sec"],
        )

        train_loader = DataLoader(
            train_ds,
            batch_size  = ttl_cfg["batch_size"],
            shuffle     = True,
            num_workers = self.config["data"].get("num_workers", 8),
            collate_fn  = ttl_collate,
            pin_memory  = self.config.get("dgx_spark", {}).get("pin_memory", True),
            drop_last   = True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size  = ttl_cfg["batch_size"],
            shuffle     = False,
            num_workers = 4,
            collate_fn  = ttl_collate,
            pin_memory  = False,
        )
        return train_loader, val_loader

    # ------------------------------------------------------------------
    # Train step  (uses F5-TTS SupertonicCFM)
    # ------------------------------------------------------------------

    def train_step(self, batch: Dict) -> float:
        """One training iteration.

        Flow (Algorithm 1 from SupertonicTTS paper + F5-TTS CFM):
          1. Encode text tokens → text embeddings  (TextEncoder)
          2. Encode reference latents → ref key/value  (ReferenceEncoder)
          3. [F5-TTS OT-CFM] Expand batch by Ke, sample (z_t, u_t) pairs
          4. [F5-TTS CFG]    Drop conditions with p_uncond
          5. [SPFM]          After warmup, additionally drop unreliable samples
          6. Predict velocity with VFEstimator
          7. Masked L1 loss  (SupertonicTTS Eq. 1)
        """
        z1       = batch["compressed_latent"].to(self.device)  # (B, Kc·C, T)
        ref      = batch["ref_latent"].to(self.device)         # (B, Kc·C, T_ref)
        ref_mask = batch["ref_mask"].to(self.device)           # (B, 1, T)
        text_tok = batch["text_tokens"].to(self.device)        # (B, L)

        self.optimizer.zero_grad(set_to_none=True)

        with autocast(dtype=self.amp_dtype, enabled=self.use_amp):
            # ── Encode text + reference (once per real batch) ─────────────
            text_emb = self.ttl_module.text_encoder(text_tok)   # (B, L, D_text)
            ref_emb  = self.ttl_module.reference_encoder(ref)   # (B, 50, D_ref) K,V

            # ── F5-TTS OT-CFM: context-sharing batch expansion ────────────
            B, C, T = z1.shape
            Ke = self.cfm.Ke

            z0    = torch.randn(B * Ke, C, T, device=self.device, dtype=z1.dtype)
            t_val = torch.rand(B * Ke, device=self.device, dtype=z1.dtype)
            z1_ke = z1.repeat_interleave(Ke, dim=0)              # (B·Ke, C, T)

            # OT-CFM interpolation (F5-TTS formulation)
            z_t, u_t = ot_cfm_interpolate(z0, z1_ke, t_val, self.cfm.sigma_min)

            # ── CFG dropout ───────────────────────────────────────────────
            use_cond = (torch.rand(B * Ke, device=self.device) >= self.cfm.p_uncond)

            # ── SPFM per-sample filtering (after warmup) ──────────────────
            if self.step >= self.cfm.spfm.warmup:
                def _vf_fn(zt, ts, use_cond_flag, **_kw):
                    te_ke  = text_emb.repeat_interleave(Ke, dim=0)
                    re_ke  = ref_emb.repeat_interleave(Ke, dim=0)
                    return self.ttl_module.vf_estimator(
                        zt, ts, te_ke, re_ke, use_cond=use_cond_flag
                    )
                spfm_mask = self.cfm.spfm.should_condition(
                    self.step, z1_ke, _vf_fn,
                    cond_kwargs={}, sigma_min=self.cfm.sigma_min,
                )
                use_cond = use_cond & spfm_mask

            # ── VF Estimator forward ──────────────────────────────────────
            text_ke = text_emb.repeat_interleave(Ke, dim=0)     # (B·Ke, L, D_t)
            ref_ke  = ref_emb.repeat_interleave(Ke, dim=0)      # (B·Ke, 50, D_r)

            v_pred = self.ttl_module.vf_estimator(
                z_t, t_val, text_ke, ref_ke, use_cond=use_cond
            )

            # ── Masked L1 loss (SupertonicTTS Eq. 1, NOT MSE like F5-TTS) ─
            mask_ke = ref_mask.repeat_interleave(Ke, dim=0) if ref_mask is not None else None
            loss = cfm_loss(v_pred, u_t, mask=mask_ke)

        # ── Backward ──────────────────────────────────────────────────────
        if self.scaler.is_enabled():
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            gradient_clip(self.ttl_module, max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            gradient_clip(self.ttl_module, max_norm=1.0)
            self.optimizer.step()

        self.scheduler.step()
        return loss.item()

    # ------------------------------------------------------------------
    # Validation step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> float:
        self.ttl_module.eval()
        val_loss = AverageMeter("val_loss")

        for batch in val_loader:
            z1   = batch["compressed_latent"].to(self.device)
            ref  = batch["ref_latent"].to(self.device)
            mask = batch["ref_mask"].to(self.device)
            text = batch["text_tokens"].to(self.device)

            with autocast(dtype=self.amp_dtype, enabled=self.use_amp):
                B, C, T = z1.shape
                z0 = torch.randn_like(z1)
                t  = torch.rand(B, device=self.device)
                z_t, u_t = ot_cfm_interpolate(z0, z1, t, self.cfm.sigma_min)

                text_emb = self.ttl_module.text_encoder(text)
                ref_emb  = self.ttl_module.reference_encoder(ref)
                v_pred   = self.ttl_module.vf_estimator(
                    z_t, t, text_emb, ref_emb, use_cond=True
                )
                loss = cfm_loss(v_pred, u_t, mask=mask)

            val_loss.update(loss.item(), B)

        self.ttl_module.train()
        return val_loss.avg

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def train(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        num_iterations: int,
        log_every:  int = 100,
        val_every:  int = 5_000,
        save_every: int = 10_000,
        resume_path: Optional[str] = None,
        tracker=None,
    ) -> None:
        if resume_path:
            self.step = load_checkpoint(
                self.ttl_module, self.optimizer, resume_path, self.device
            )
            logger.info(f"Resumed from step {self.step}")

        self.ttl_module.train()
        interrupted = False

        def _handle_signal(sig, frame):
            nonlocal interrupted
            logger.warning("Interrupt received — finishing current step …")
            interrupted = True

        signal.signal(signal.SIGINT,  _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)

        data_iter = iter(train_loader)
        pbar = tqdm(total=num_iterations, initial=self.step,
                    desc="TTL Training", dynamic_ncols=True)

        while self.step < num_iterations and not interrupted:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            loss_val = self.train_step(batch)
            self.loss_meter.update(loss_val)
            self.step += 1

            pbar.update(1)
            pbar.set_postfix(loss=f"{self.loss_meter.avg:.4f}",
                             lr=f"{get_lr(self.optimizer):.2e}")

            # ── Logging ──────────────────────────────────────────────────
            if self.step % log_every == 0:
                spfm_active = int(self.step >= self.cfm.spfm.warmup)
                metrics = {
                    "train/flow_loss":   self.loss_meter.avg,
                    "train/lr":          get_lr(self.optimizer),
                    "train/spfm_active": float(spfm_active),
                }
                if tracker:
                    tracker.log_scalars(metrics, self.step)
                logger.info(
                    f"[step {self.step:>7d}] loss={self.loss_meter.avg:.4f} "
                    f"lr={get_lr(self.optimizer):.2e} "
                    f"spfm={'on' if spfm_active else 'warmup'}"
                )
                self.loss_meter.reset()

            # ── Validation ───────────────────────────────────────────────
            if self.step % val_every == 0:
                val_loss = self.validate(val_loader)
                logger.info(f"[step {self.step}] val_loss={val_loss:.4f}")
                if tracker:
                    tracker.log_scalars({"val/flow_loss": val_loss}, self.step)

                if val_loss < self.best_val:
                    self.best_val = val_loss
                    best_path = self.output_dir / "best_ttl.pt"
                    save_checkpoint(
                        self.ttl_module, self.optimizer, self.step, best_path
                    )
                    logger.info(f"  ↳ New best val_loss={val_loss:.4f} — saved best_ttl.pt")
                    if tracker:
                        tracker.log_model_artifact(
                            str(best_path), name="best-ttl-module",
                            metadata={"step": self.step, "val_flow_loss": val_loss},
                        )

            # ── Checkpoint ───────────────────────────────────────────────
            if self.step % save_every == 0:
                save_checkpoint(
                    self.ttl_module, self.optimizer, self.step,
                    self.output_dir / f"ttl_step_{self.step:07d}.pt"
                )

        pbar.close()
        # Final checkpoint
        final_path = self.output_dir / "ttl_final.pt"
        save_checkpoint(self.ttl_module, self.optimizer, self.step, final_path)
        logger.info(f"Training complete at step {self.step}. Best val_loss={self.best_val:.4f}")
        if tracker:
            tracker.log_scalars({"summary/best_val_flow_loss": self.best_val}, self.step)
            tracker.close()


# ============================================================================
# Entry point
# ============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SupertonicTTS Text-to-Latent module")
    p.add_argument("--config",      required=True,  help="Path to base_config.yaml")
    p.add_argument("--output_dir",  default="./outputs/ttl", help="Output directory")
    p.add_argument("--resume",      default=None,   help="Resume from checkpoint path")
    p.add_argument("--log_every",   type=int, default=100)
    p.add_argument("--val_every",   type=int, default=5_000)
    p.add_argument("--save_every",  type=int, default=10_000)
    p.add_argument("--num_iterations", type=int, default=None,
                   help="Override number of iterations from config")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device, tracker = setup_training(
        config, output_dir, stage="ttl",
        wandb_tags=["stage2", "flow-matching", "hindi-tts"],
        wandb_notes="OT-CFM + LARoPE + SPFM + context-sharing batch expansion",
    )

    num_iters = (
        args.num_iterations
        or config["text_to_latent"]["training"]["num_iterations"]
    )

    trainer = TTLTrainer(config, device, output_dir)
    train_loader, val_loader = trainer.build_dataloaders()

    # Log model summary to W&B run overview
    tracker.log_model_summary({
        "ttl/total_params":       count_parameters(trainer.ttl_module),
        "ttl/text_encoder_params": count_parameters(trainer.ttl_module.text_encoder),
        "ttl/ref_encoder_params":  count_parameters(trainer.ttl_module.reference_encoder),
        "ttl/vf_estimator_params": count_parameters(trainer.ttl_module.vf_estimator),
        "ttl/Ke_expansion":        config["text_to_latent"]["training"]["expansion_factor"],
        "ttl/spfm_warmup":         config.get("spfm", {}).get("warmup_iterations", 40_000),
    })

    logger.info("=" * 60)
    logger.info("SupertonicTTS — Text-to-Latent Training")
    logger.info(f"  Flow matching: OT-CFM (F5-TTS) + SPFM + LARoPE")
    logger.info(f"  Context expansion Ke : {config['text_to_latent']['training']['expansion_factor']}")
    logger.info(f"  Iterations           : {num_iters:,}")
    logger.info(f"  Device               : {device}")
    logger.info(f"  Mixed precision      : {config.get('project', {}).get('mixed_precision', 'bf16')}")
    if tracker.wandb_url:
        logger.info(f"  W&B dashboard        : {tracker.wandb_url}")
    logger.info("=" * 60)

    trainer.train(
        train_loader,
        val_loader,
        num_iterations = num_iters,
        log_every      = args.log_every,
        val_every      = args.val_every,
        save_every     = args.save_every,
        resume_path    = args.resume,
        tracker        = tracker,
    )


if __name__ == "__main__":
    main()
