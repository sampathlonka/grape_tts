"""
Logging, metrics tracking, and experiment monitoring utilities.

Supports:
  • Python logging (file + console)
  • TensorBoard
  • Weights & Biases (wandb) — loaded from .env via python-dotenv

W&B logged objects per training stage:
  Autoencoder : recon_loss, adv_loss, fm_loss, g_loss, d_loss, grad_norm,
                lr, reconstructed_audio, mel_spectrogram
  TTL         : flow_loss, val_loss, lr, grad_norm, spfm_drop_rate,
                generated_audio, mel_spectrogram, text_alignment_table
  Duration    : dur_loss, val_dur_loss, lr
  Evaluation  : WER, CER, UTMOS, PESQ, STOI, SIM, RTF,
                per_speaker_table, synthesised_audio, mel_comparison
"""

from __future__ import annotations

import io
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# .env loader  (called once at module import)
# ─────────────────────────────────────────────────────────────────────────────

def _load_dotenv(env_path: Optional[str] = None) -> None:
    """Load .env file into os.environ using python-dotenv."""
    try:
        from dotenv import load_dotenv
        path = Path(env_path) if env_path else Path(".env")
        if not path.exists():
            # Walk up to find .env
            for parent in Path(__file__).parents:
                candidate = parent / ".env"
                if candidate.exists():
                    path = candidate
                    break
        if path.exists():
            load_dotenv(dotenv_path=str(path), override=False)
            logging.getLogger(__name__).debug(f"Loaded .env from {path}")
    except ImportError:
        logging.getLogger(__name__).warning(
            "python-dotenv not installed — .env not loaded. "
            "Install with: pip install python-dotenv"
        )


_load_dotenv()   # auto-load on import


# ─────────────────────────────────────────────────────────────────────────────
# Python logger setup
# ─────────────────────────────────────────────────────────────────────────────

def setup_logger(
    name: str,
    log_dir: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Configure a logger with console + optional file handler."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    for h in logger.handlers[:]:
        logger.removeHandler(h)

    fmt = logging.Formatter(
        "[%(asctime)s] %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        fh  = logging.FileHandler(log_dir / f"{name}_{ts}.log")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ─────────────────────────────────────────────────────────────────────────────
# Running average
# ─────────────────────────────────────────────────────────────────────────────

class MetricsTracker:
    """Track metrics across training steps with running statistics."""

    def __init__(self):
        self.metrics: Dict[str, List[float]] = defaultdict(list)
        self.step_count = 0

    def update(self, **kwargs: float) -> None:
        for k, v in kwargs.items():
            if v is not None:
                self.metrics[k].append(float(v))
        self.step_count += 1

    def get_mean(self, name: str) -> float:
        vals = self.metrics.get(name, [])
        return sum(vals) / len(vals) if vals else 0.0

    def get_recent_mean(self, name: str, n: int = 100) -> float:
        vals = self.metrics.get(name, [])
        return sum(vals[-n:]) / len(vals[-n:]) if vals else 0.0

    def get_latest(self, name: str) -> Optional[float]:
        vals = self.metrics.get(name, [])
        return vals[-1] if vals else None

    def reset(self) -> None:
        self.metrics.clear()
        self.step_count = 0

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        import statistics
        out = {}
        for name, vals in self.metrics.items():
            if vals:
                out[name] = {
                    "mean":   sum(vals) / len(vals),
                    "min":    min(vals),
                    "max":    max(vals),
                    "std":    statistics.stdev(vals) if len(vals) > 1 else 0.0,
                    "latest": vals[-1],
                }
        return out


# ─────────────────────────────────────────────────────────────────────────────
# TensorBoard logger
# ─────────────────────────────────────────────────────────────────────────────

class TensorboardLogger:
    """Thin wrapper around torch SummaryWriter."""

    def __init__(self, log_dir: str):
        from torch.utils.tensorboard import SummaryWriter
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(self.log_dir))

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        self.writer.add_scalar(tag, value, step)

    def log_scalars(self, metrics: Dict[str, float], step: int) -> None:
        for tag, value in metrics.items():
            self.writer.add_scalar(tag, value, step)

    def log_histogram(self, tag: str, values: Any, step: int) -> None:
        self.writer.add_histogram(tag, values, step)

    def log_audio(self, tag: str, audio: Any, step: int, sr: int = 44100) -> None:
        self.writer.add_audio(tag, audio, step, sample_rate=sr)

    def log_figure(self, tag: str, fig: Any, step: int) -> None:
        self.writer.add_figure(tag, fig, step)

    def log_text(self, tag: str, text: str, step: int) -> None:
        self.writer.add_text(tag, text, step)

    def flush(self) -> None:
        self.writer.flush()

    def close(self) -> None:
        self.writer.close()


# ─────────────────────────────────────────────────────────────────────────────
# W&B logger — full implementation
# ─────────────────────────────────────────────────────────────────────────────

class WandbLogger:
    """
    Comprehensive Weights & Biases logger for SupertonicTTS.

    Reads credentials from environment (loaded from .env):
        WANDB_API_KEY  — required
        WANDB_PROJECT  — optional (overridden by constructor arg)
        WANDB_ENTITY   — optional (overridden by constructor arg)

    Logs per training stage:
    ┌─────────────────┬────────────────────────────────────────────────────┐
    │ Autoencoder     │ recon_loss, adv_loss, fm_loss, g_loss, d_loss,     │
    │                 │ grad_norm, lr, audio samples, mel spectrograms      │
    ├─────────────────┼────────────────────────────────────────────────────┤
    │ Text-to-Latent  │ flow_loss, val_loss, lr, grad_norm, spfm_stats,   │
    │                 │ generated audio, mel spectrograms, alignment table  │
    ├─────────────────┼────────────────────────────────────────────────────┤
    │ Duration        │ dur_loss, val_dur_loss, lr, pred vs GT scatter      │
    ├─────────────────┼────────────────────────────────────────────────────┤
    │ Evaluation      │ WER, CER, UTMOS, PESQ, STOI, SIM, RTF,            │
    │                 │ per-speaker table, synthesised audio, mel compare   │
    └─────────────────┴────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        project:  Optional[str] = None,
        entity:   Optional[str] = None,
        name:     Optional[str] = None,
        config:   Optional[Dict[str, Any]] = None,
        tags:     Optional[List[str]] = None,
        notes:    Optional[str] = None,
        resume:   bool = False,
        run_id:   Optional[str] = None,
    ):
        try:
            import wandb
        except ImportError:
            raise ImportError(
                "wandb not installed. Install with: pip install wandb"
            )

        api_key = os.environ.get("WANDB_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "WANDB_API_KEY not found in environment.\n"
                "  1. Copy .env.example → .env\n"
                "  2. Set WANDB_API_KEY=<your_key> in .env\n"
                "  3. Get your key at https://wandb.ai/authorize"
            )
        wandb.login(key=api_key, relogin=False)

        self.wandb = wandb
        self.run = wandb.init(
            project  = project  or os.environ.get("WANDB_PROJECT", "supertonic-hindi-tts"),
            entity   = entity   or os.environ.get("WANDB_ENTITY"),
            name     = name     or os.environ.get("WANDB_RUN_NAME"),
            config   = config   or {},
            tags     = tags     or [],
            notes    = notes,
            resume   = "allow" if resume else None,
            id       = run_id,
        )
        self._logger = logging.getLogger(__name__)
        self._logger.info(f"W&B run: {self.run.url}")

    # ── Core scalars ─────────────────────────────────────────────────────────

    def log(self, metrics: Dict[str, Any], step: int) -> None:
        """Log arbitrary scalar metrics at a given step."""
        self.wandb.log(metrics, step=step)

    # ── Audio ─────────────────────────────────────────────────────────────────

    def log_audio(
        self,
        tag: str,
        waveform: np.ndarray,
        step: int,
        sr: int = 44100,
        caption: str = "",
    ) -> None:
        """Log a waveform as a W&B Audio object.

        Args:
            tag:      W&B key (e.g. "val/reconstructed_audio")
            waveform: (T,) or (1, T) float32 numpy array, range [-1, 1]
            step:     global training step
            sr:       sample rate
            caption:  optional text caption shown in the UI
        """
        if waveform.ndim > 1:
            waveform = waveform.squeeze()
        audio_obj = self.wandb.Audio(waveform, sample_rate=sr, caption=caption)
        self.wandb.log({tag: audio_obj}, step=step)

    def log_audio_batch(
        self,
        tag: str,
        waveforms: List[np.ndarray],
        step: int,
        sr: int = 44100,
        captions: Optional[List[str]] = None,
    ) -> None:
        """Log multiple audio samples in a single W&B panel."""
        if captions is None:
            captions = [f"sample_{i}" for i in range(len(waveforms))]
        audio_objs = [
            self.wandb.Audio(w.squeeze(), sample_rate=sr, caption=c)
            for w, c in zip(waveforms, captions)
        ]
        self.wandb.log({tag: audio_objs}, step=step)

    # ── Mel spectrogram plots ─────────────────────────────────────────────────

    def log_mel_spectrogram(
        self,
        tag: str,
        mel: np.ndarray,
        step: int,
        title: str = "Mel Spectrogram",
        caption: str = "",
    ) -> None:
        """Render and log a mel spectrogram as a W&B Image.

        Args:
            mel:  (n_mels, T) numpy array (log-scale recommended)
            step: global training step
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 4))
        im = ax.imshow(mel, aspect="auto", origin="lower",
                       interpolation="none", cmap="magma")
        ax.set_title(title)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Mel bin")
        plt.colorbar(im, ax=ax, format="%+.1f dB")
        plt.tight_layout()

        img = self.wandb.Image(fig, caption=caption)
        self.wandb.log({tag: img}, step=step)
        plt.close(fig)

    def log_mel_comparison(
        self,
        tag: str,
        mel_gt: np.ndarray,
        mel_pred: np.ndarray,
        step: int,
        caption: str = "Ground truth (top) vs Generated (bottom)",
    ) -> None:
        """Side-by-side GT and generated mel spectrogram."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        for ax, mel, title in zip(
            axes,
            [mel_gt, mel_pred],
            ["Ground Truth", "Generated"],
        ):
            im = ax.imshow(mel, aspect="auto", origin="lower",
                           interpolation="none", cmap="magma")
            ax.set_title(title)
            ax.set_ylabel("Mel bin")
            plt.colorbar(im, ax=ax, format="%+.1f dB")
        axes[-1].set_xlabel("Frame")
        plt.suptitle(caption, fontsize=10)
        plt.tight_layout()

        img = self.wandb.Image(fig, caption=caption)
        self.wandb.log({tag: img}, step=step)
        plt.close(fig)

    # ── Loss curve plots ─────────────────────────────────────────────────────

    def log_loss_curves(
        self,
        tag: str,
        train_losses: List[float],
        val_losses: Optional[List[float]] = None,
        steps: Optional[List[int]] = None,
        step: int = 0,
    ) -> None:
        """Plot and log loss curves as a W&B Image."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 4))
        xs = steps if steps else list(range(len(train_losses)))
        ax.plot(xs, train_losses, label="Train", alpha=0.8)
        if val_losses:
            val_xs = steps if steps else list(range(len(val_losses)))
            ax.plot(val_xs, val_losses, label="Val", alpha=0.8)
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title(tag)
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()

        img = self.wandb.Image(fig)
        self.wandb.log({tag: img}, step=step)
        plt.close(fig)

    # ── Duration scatter ─────────────────────────────────────────────────────

    def log_duration_scatter(
        self,
        gt_durations: List[float],
        pred_durations: List[float],
        step: int,
    ) -> None:
        """Scatter plot of predicted vs ground-truth durations."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(gt_durations, pred_durations, alpha=0.4, s=8)
        lim = max(max(gt_durations), max(pred_durations)) * 1.05
        ax.plot([0, lim], [0, lim], "r--", alpha=0.5, label="y=x")
        ax.set_xlabel("Ground truth duration (s)")
        ax.set_ylabel("Predicted duration (s)")
        ax.set_title("Duration predictor: GT vs Predicted")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()

        self.wandb.log({"duration/scatter": self.wandb.Image(fig)}, step=step)
        plt.close(fig)

    # ── Tables ────────────────────────────────────────────────────────────────

    def log_evaluation_table(
        self,
        rows: List[Dict[str, Any]],
        step: int,
        tag: str = "evaluation/per_sample",
    ) -> None:
        """Log per-sample evaluation results as a searchable W&B Table.

        Each row dict may contain: text, speaker_id, gender, duration,
        wer, cer, utmos, pesq, stoi, sim, rtf, audio_path.
        """
        if not rows:
            return
        cols = list(rows[0].keys())
        table = self.wandb.Table(columns=cols)
        for row in rows:
            table.add_data(*[row.get(c, "") for c in cols])
        self.wandb.log({tag: table}, step=step)

    def log_per_speaker_table(
        self,
        speaker_metrics: Dict[str, Dict[str, float]],
        step: int,
    ) -> None:
        """Log per-speaker aggregate metrics as a W&B Table."""
        if not speaker_metrics:
            return
        sample_spk = next(iter(speaker_metrics.values()))
        cols = ["speaker_id"] + list(sample_spk.keys())
        table = self.wandb.Table(columns=cols)
        for spk_id, metrics in speaker_metrics.items():
            table.add_data(spk_id, *[metrics.get(c, 0.0) for c in cols[1:]])
        self.wandb.log({"evaluation/per_speaker": table}, step=step)

    # ── Model artifacts ───────────────────────────────────────────────────────

    def log_model_artifact(
        self,
        checkpoint_path: str,
        name: str,
        artifact_type: str = "model",
        metadata: Optional[Dict] = None,
    ) -> None:
        """Upload a checkpoint as a versioned W&B Artifact."""
        art = self.wandb.Artifact(
            name=name,
            type=artifact_type,
            metadata=metadata or {},
        )
        art.add_file(checkpoint_path)
        self.run.log_artifact(art)
        self._logger.info(f"W&B artifact logged: {name} ← {checkpoint_path}")

    # ── Model summary ────────────────────────────────────────────────────────

    def log_model_summary(self, model_info: Dict[str, Any]) -> None:
        """Log model parameter counts to the W&B run summary."""
        for k, v in model_info.items():
            self.run.summary[k] = v

    # ── Gradient histograms ──────────────────────────────────────────────────

    def log_gradient_histogram(
        self,
        model,
        step: int,
        tag_prefix: str = "gradients",
    ) -> None:
        """Log gradient norm histograms for named parameter groups."""
        for name, param in model.named_parameters():
            if param.grad is not None:
                self.wandb.log(
                    {f"{tag_prefix}/{name}": self.wandb.Histogram(
                        param.grad.detach().cpu().float().numpy()
                    )},
                    step=step,
                )

    # ── Config update ────────────────────────────────────────────────────────

    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update the W&B run config mid-training."""
        self.wandb.config.update(updates, allow_val_change=True)

    # ── Alerts ───────────────────────────────────────────────────────────────

    def alert(self, title: str, text: str, level: str = "INFO") -> None:
        """Send a W&B alert (shows in Slack/email if configured)."""
        self.wandb.alert(title=title, text=text, level=level)

    # ── Finish ───────────────────────────────────────────────────────────────

    def finish(self) -> None:
        """Mark the W&B run as complete."""
        self.run.finish()


# ─────────────────────────────────────────────────────────────────────────────
# Unified experiment tracker (TB + W&B)
# ─────────────────────────────────────────────────────────────────────────────

class ExperimentTracker:
    """
    Unified logger that writes to TensorBoard and/or Weights & Biases.

    Usage:
        tracker = ExperimentTracker(
            log_dir      = "outputs/autoencoder/logs",
            stage        = "autoencoder",
            config       = config,            # full YAML config dict
            use_wandb    = True,
            wandb_tags   = ["hindi", "v1"],
        )
        tracker.log_scalars({"train/loss": 0.42, "train/lr": 1e-4}, step=100)
        tracker.log_audio("val/reconstructed", waveform, step=100)
        tracker.close()
    """

    def __init__(
        self,
        log_dir:       str,
        stage:         str = "train",
        config:        Optional[Dict] = None,
        use_tensorboard: bool = True,
        use_wandb:     bool = True,
        wandb_project: Optional[str] = None,
        wandb_entity:  Optional[str] = None,
        wandb_name:    Optional[str] = None,
        wandb_tags:    Optional[List[str]] = None,
        wandb_notes:   Optional[str] = None,
        wandb_resume:  bool = False,
        wandb_run_id:  Optional[str] = None,
    ):
        self.stage   = stage
        self.tracker = MetricsTracker()
        self.tb      = None
        self.wb      = None

        if use_tensorboard:
            try:
                self.tb = TensorboardLogger(log_dir)
            except Exception as e:
                logging.getLogger(__name__).warning(f"TensorBoard init failed: {e}")

        if use_wandb:
            try:
                self.wb = WandbLogger(
                    project  = wandb_project,
                    entity   = wandb_entity,
                    name     = wandb_name or f"{stage}-{datetime.now().strftime('%m%d-%H%M')}",
                    config   = config,
                    tags     = (wandb_tags or []) + [stage, "hindi-tts"],
                    notes    = wandb_notes,
                    resume   = wandb_resume,
                    run_id   = wandb_run_id,
                )
            except Exception as e:
                logging.getLogger(__name__).warning(
                    f"W&B init failed ({e}). Continuing without W&B."
                )

    # ── Scalars ──────────────────────────────────────────────────────────────

    def log_scalars(self, metrics: Dict[str, float], step: int) -> None:
        """Log scalar metrics to all active backends."""
        self.tracker.update(**metrics)
        if self.tb:
            self.tb.log_scalars(metrics, step)
        if self.wb:
            self.wb.log(metrics, step)

    # ── Audio ─────────────────────────────────────────────────────────────────

    def log_audio(
        self,
        tag: str,
        waveform: np.ndarray,
        step: int,
        sr: int = 44100,
        caption: str = "",
    ) -> None:
        if self.tb:
            self.tb.log_audio(tag, waveform, step, sr)
        if self.wb:
            self.wb.log_audio(tag, waveform, step, sr, caption)

    def log_audio_batch(
        self,
        tag: str,
        waveforms: List[np.ndarray],
        step: int,
        sr: int = 44100,
        captions: Optional[List[str]] = None,
    ) -> None:
        if self.wb:
            self.wb.log_audio_batch(tag, waveforms, step, sr, captions)

    # ── Mel spectrograms ─────────────────────────────────────────────────────

    def log_mel(
        self,
        tag: str,
        mel: np.ndarray,
        step: int,
        title: str = "",
        caption: str = "",
    ) -> None:
        if self.wb:
            self.wb.log_mel_spectrogram(tag, mel, step, title or tag, caption)
        if self.tb:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.imshow(mel, aspect="auto", origin="lower", cmap="magma")
            ax.set_title(title or tag)
            plt.tight_layout()
            self.tb.log_figure(tag, fig, step)
            plt.close(fig)

    def log_mel_comparison(
        self,
        tag: str,
        mel_gt: np.ndarray,
        mel_pred: np.ndarray,
        step: int,
        caption: str = "",
    ) -> None:
        if self.wb:
            self.wb.log_mel_comparison(tag, mel_gt, mel_pred, step, caption)

    # ── Duration scatter ─────────────────────────────────────────────────────

    def log_duration_scatter(
        self,
        gt_durations: List[float],
        pred_durations: List[float],
        step: int,
    ) -> None:
        if self.wb:
            self.wb.log_duration_scatter(gt_durations, pred_durations, step)

    # ── Tables ────────────────────────────────────────────────────────────────

    def log_evaluation_table(
        self,
        rows: List[Dict[str, Any]],
        step: int,
        tag: str = "evaluation/per_sample",
    ) -> None:
        if self.wb:
            self.wb.log_evaluation_table(rows, step, tag)

    def log_per_speaker_table(
        self,
        speaker_metrics: Dict[str, Dict[str, float]],
        step: int,
    ) -> None:
        if self.wb:
            self.wb.log_per_speaker_table(speaker_metrics, step)

    # ── Gradient histograms ──────────────────────────────────────────────────

    def log_gradient_histogram(self, model, step: int, tag_prefix: str = "gradients") -> None:
        if self.wb:
            self.wb.log_gradient_histogram(model, step, tag_prefix)

    # ── Model artifacts ───────────────────────────────────────────────────────

    def log_model_artifact(
        self,
        checkpoint_path: str,
        name: str,
        metadata: Optional[Dict] = None,
    ) -> None:
        if self.wb:
            self.wb.log_model_artifact(checkpoint_path, name, metadata=metadata)

    def log_model_summary(self, model_info: Dict[str, Any]) -> None:
        if self.wb:
            self.wb.log_model_summary(model_info)

    # ── W&B alerts ───────────────────────────────────────────────────────────

    def alert(self, title: str, text: str) -> None:
        if self.wb:
            self.wb.alert(title, text)

    # ── Misc ─────────────────────────────────────────────────────────────────

    def get_summary(self) -> Dict[str, Any]:
        return self.tracker.get_summary()

    @property
    def wandb_url(self) -> Optional[str]:
        return self.wb.run.url if self.wb else None

    def close(self) -> None:
        if self.tb:
            self.tb.close()
        if self.wb:
            self.wb.finish()
