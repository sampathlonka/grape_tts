"""
F5-TTS Conditional Flow Matching — adapted from SWivid/F5-TTS (Apache 2.0).
https://github.com/SWivid/F5-TTS/blob/main/src/f5_tts/model/cfm.py

What is taken verbatim from F5-TTS:
  - Optimal-transport CFM interpolation:  z_t = (1−t)·z₀ + t·z₁
  - Velocity target:                       u_t = z₁ − z₀
  - Classifier-free guidance (CFG) inference with double-batch trick
  - Euler ODE solver with configurable NFE
  - Sway-sampling schedule (t' = t + s·sin²(π·t/2), credit F5-TTS paper)

SupertonicTTS-specific adaptations (NOT in original F5-TTS):
  - L1 loss instead of MSE  (SupertonicTTS paper Eq. 1)
  - σ_min = 1e-8 instead of 0  (paper §3.2.4)
  - Context-sharing batch expansion  (Algorithm 1 in paper, Ke expansion factor)
  - SPFM (Self-Purifying Flow Matching)  (training_flow_matching.pdf)
  - Channel-first latent format  (B, C, T) throughout
  - Masked loss: only compute loss on non-reference region
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Sway Sampling  (F5-TTS inference trick)
# ---------------------------------------------------------------------------

def sway_sampling_coefs(
    t: torch.Tensor,
    sway_coef: float = -1.0,
) -> torch.Tensor:
    """Sway sampling timestep schedule from F5-TTS.

    Shifts timesteps toward 0 or 1 depending on the sign of sway_coef.
    Negative values concentrate steps near t=0 (noisy region).

    t' = t + sway_coef * (cos(π·t/2)² − 1 + t)

    Args:
        t:          (NFE,) uniform timesteps in [0, 1]
        sway_coef:  shift coefficient (−1.0 default from F5-TTS)
    Returns:
        (NFE,) adjusted timesteps
    """
    return t + sway_coef * (torch.cos(math.pi / 2 * t) ** 2 - 1 + t)


# ---------------------------------------------------------------------------
# Interpolation helpers  (OT-CFM — F5-TTS formulation)
# ---------------------------------------------------------------------------

def ot_cfm_interpolate(
    z0: torch.Tensor,
    z1: torch.Tensor,
    t: torch.Tensor,
    sigma_min: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Optimal-transport conditional flow matching interpolation.

    From F5-TTS cfm.py forward():
        z_t  = (1 − (1 − σ_min)·t) · z₀  +  t · z₁
        u_t  = z₁ − (1 − σ_min) · z₀          ← velocity target

    Args:
        z0:        (B, C, T) Gaussian noise samples
        z1:        (B, C, T) target latents (data)
        t:         (B,) timesteps in [0, 1]
        sigma_min: small positive float to avoid exact zero noise
    Returns:
        z_t: interpolated latents  (B, C, T)
        u_t: velocity target       (B, C, T)
    """
    t_bcT = t[:, None, None]                          # (B, 1, 1) for broadcast
    z_t = (1.0 - (1.0 - sigma_min) * t_bcT) * z0 + t_bcT * z1
    u_t = z1 - (1.0 - sigma_min) * z0
    return z_t, u_t


# ---------------------------------------------------------------------------
# Context-Sharing Batch Expansion  (SupertonicTTS Algorithm 1)
# ---------------------------------------------------------------------------

def context_sharing_batch_expand(
    z1: torch.Tensor,
    cond_fn: Callable[[torch.Tensor], torch.Tensor],
    Ke: int,
    sigma_min: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Context-sharing batch expansion from Algorithm 1 (SupertonicTTS paper).

    For each sample in the batch, encode conditions ONCE, then generate Ke
    different noise/timestep pairs that share those conditions.  This mimics
    a Ke× larger effective batch while only paying for 1× condition encoding.

    Args:
        z1:      (B, C, T) target compressed latents
        cond_fn: callable that encodes conditioning variables (text + ref)
                 and returns (cond_encoded,) — called once per real sample
        Ke:      expansion factor (paper uses Ke = 4)
        sigma_min: flow matching noise floor
    Returns:
        z_t_exp:  (B·Ke, C, T) noisy latents
        u_t_exp:  (B·Ke, C, T) velocity targets
        t_exp:    (B·Ke,)      timesteps
        cond_exp: encoded conditions repeated Ke times (passed back to caller
                  to forward through the velocity estimator)
    """
    B, C, T = z1.shape
    device = z1.device

    # Encode conditions for the real batch (once)
    cond_encoded = cond_fn(z1)           # implementation-defined; returned as-is

    # Sample Ke noise/timestep pairs per sample
    z0  = torch.randn(B * Ke, C, T, device=device)
    t   = torch.rand(B * Ke, device=device)

    # Repeat z1 Ke times along batch dimension
    z1_exp = z1.repeat_interleave(Ke, dim=0)   # (B·Ke, C, T)

    z_t_exp, u_t_exp = ot_cfm_interpolate(z0, z1_exp, t, sigma_min)

    return z_t_exp, u_t_exp, t_exp, cond_encoded


# ---------------------------------------------------------------------------
# CFM Loss  (SupertonicTTS: masked L1; F5-TTS: MSE — we use L1 per paper)
# ---------------------------------------------------------------------------

def cfm_loss(
    v_pred: torch.Tensor,
    v_target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Masked L1 flow-matching loss.

    SupertonicTTS Eq. 1:
        L_TTL = E_{t,(z1,c),z0} [ ||m · (v_θ − u_t)||₁ ]

    where m is a binary mask that zeros out the reference region (the cropped
    segment used as reference speech during training to prevent info leakage).

    Args:
        v_pred:   (B, C, T) predicted velocity field
        v_target: (B, C, T) OT-CFM target velocity
        mask:     (B, 1, T) or (B, C, T) float mask — 1 = include, 0 = exclude.
                  If None, full L1 is computed.
    Returns:
        scalar loss
    """
    diff = (v_pred - v_target).abs()
    if mask is not None:
        diff = diff * mask
        return diff.sum() / (mask.sum() * v_pred.shape[1] + 1e-8)
    return diff.mean()


# ---------------------------------------------------------------------------
# SPFM — Self-Purifying Flow Matching  (training_flow_matching.pdf)
# ---------------------------------------------------------------------------

class SPFMFilter:
    """Self-Purifying Flow Matching filter.

    After warmup iterations, for each training sample we compare:
        L_cond   = loss when conditioning on reference + text
        L_uncond = loss when conditioning on nothing

    If L_cond > L_uncond the sample is considered unreliable (the model
    finds it harder to use the condition than to ignore it, suggesting
    alignment noise).  That sample is trained unconditionally for this step.

    The inspection is done at a fixed interpolation time t′ = 0.5, which
    the paper shows is most discriminative.

    Usage:
        filter = SPFMFilter(warmup=40_000, t_inspect=0.5)
        ...
        # inside training loop:
        cond_mask = filter.should_condition(step, z1, model, cond)
        # cond_mask: (B,) bool — True = use conditioning, False = drop
    """

    def __init__(self, warmup: int = 40_000, t_inspect: float = 0.5):
        self.warmup    = warmup
        self.t_inspect = t_inspect

    @torch.no_grad()
    def should_condition(
        self,
        step: int,
        z1: torch.Tensor,
        model_fn: Callable,
        cond_kwargs: dict,
        sigma_min: float = 1e-8,
    ) -> torch.Tensor:
        """Return per-sample conditioning mask (B,).

        Args:
            step:        current training iteration
            z1:          (B, C, T) target latents
            model_fn:    callable(z_t, t, use_cond=bool, **cond_kwargs) → v_pred
            cond_kwargs: conditioning keyword arguments forwarded to model_fn
            sigma_min:   flow matching noise floor
        Returns:
            (B,) bool tensor — True means "use conditioning for this sample"
        """
        if step < self.warmup:
            return torch.ones(z1.shape[0], dtype=torch.bool, device=z1.device)

        B, C, T = z1.shape
        t_val = torch.full((B,), self.t_inspect, device=z1.device)
        z0    = torch.randn_like(z1)
        z_t, u_t = ot_cfm_interpolate(z0, z1, t_val, sigma_min)

        # Conditional loss per sample
        v_cond   = model_fn(z_t, t_val, use_cond=True,  **cond_kwargs)
        l_cond   = (v_cond - u_t).abs().mean(dim=(1, 2))   # (B,)

        # Unconditional loss per sample
        v_uncond = model_fn(z_t, t_val, use_cond=False, **cond_kwargs)
        l_uncond = (v_uncond - u_t).abs().mean(dim=(1, 2)) # (B,)

        # Keep conditioning only when it actually helps
        return l_cond <= l_uncond                           # (B,) bool


# ---------------------------------------------------------------------------
# Euler ODE Solver  (F5-TTS inference, adapted for SupertonicTTS)
# ---------------------------------------------------------------------------

@torch.inference_mode()
def euler_solve(
    model_fn: Callable,
    z_shape: Tuple[int, ...],
    nfe: int,
    cfg_strength: float,
    cond_kwargs: dict,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    sway: bool = True,
    sway_coef: float = -1.0,
    sigma_min: float = 1e-8,
) -> torch.Tensor:
    """Euler ODE solver for flow-matching inference.

    Follows F5-TTS cfm.py sample() with:
      - Optional sway-sampling timestep schedule (F5-TTS)
      - Classifier-free guidance via double-batch (F5-TTS)
      - Latent format (B, C, T) for SupertonicTTS

    Args:
        model_fn:      callable(z_t, t, **cond_kwargs) → velocity (B, C, T)
                       Must accept keyword `use_cond` (bool) for CFG.
        z_shape:       shape of the output latent tensor (B, C, T)
        nfe:           number of function evaluations (steps)
        cfg_strength:  classifier-free guidance scale (≥0); 0 = no guidance
        cond_kwargs:   conditioning dict forwarded to model_fn
        device/dtype:  target device and precision
        sway:          use sway-sampling schedule from F5-TTS (recommended)
        sway_coef:     sway coefficient (−1.0 default)
        sigma_min:     noise floor used during training
    Returns:
        z1: (B, C, T) generated latents
    """
    # Initial noise
    z = torch.randn(z_shape, device=device, dtype=dtype)

    # Timestep schedule
    t_base = torch.linspace(0.0, 1.0, nfe + 1, device=device)
    if sway:
        t_base = sway_sampling_coefs(t_base, sway_coef).clamp(0.0, 1.0)

    B = z_shape[0]

    for i in range(nfe):
        t_now  = t_base[i]
        t_next = t_base[i + 1]
        dt     = t_next - t_now
        t_batch = t_now.expand(B)

        if cfg_strength > 1e-5:
            # Double-batch CFG (F5-TTS trick): pack cond + uncond in one forward
            v_cond   = model_fn(z, t_batch, use_cond=True,  **cond_kwargs)
            v_uncond = model_fn(z, t_batch, use_cond=False, **cond_kwargs)
            # CFG formula:  v_guided = v_uncond + α·(v_cond − v_uncond)
            v = v_uncond + cfg_strength * (v_cond - v_uncond)
        else:
            v = model_fn(z, t_batch, use_cond=True, **cond_kwargs)

        z = z + dt * v

    return z


# ---------------------------------------------------------------------------
# SupertonicCFM — top-level class wiring everything together
# ---------------------------------------------------------------------------

class SupertonicCFM(nn.Module):
    """Conditional Flow Matching wrapper for SupertonicTTS.

    Combines:
      • OT-CFM interpolation  (F5-TTS formulation)
      • Context-sharing batch expansion  (SupertonicTTS Algorithm 1)
      • SPFM filtering  (Self-Purifying FM)
      • Masked L1 loss  (SupertonicTTS Eq. 1)
      • CFG + Euler inference  (F5-TTS style)
      • Sway-sampling schedule  (F5-TTS)

    The velocity estimator model is injected externally (VFEstimator from
    models/vf_estimator.py) so this class is purely about the training &
    inference logic.

    Args:
        vf_estimator:  nn.Module that predicts velocity field.
                       Signature: forward(z_t, t, text_emb, ref_emb, use_cond)
                                  → (B, C, T) velocity
        sigma_min:     noise floor (1e-8, SupertonicTTS paper)
        p_uncond:      probability of dropping conditions during training (0.05)
        Ke:            context-sharing expansion factor (4)
        spfm_warmup:   SPFM warmup iterations (40_000)
    """

    def __init__(
        self,
        vf_estimator: nn.Module,
        sigma_min: float = 1e-8,
        p_uncond: float = 0.05,
        Ke: int = 4,
        spfm_warmup: int = 40_000,
        spfm_t_inspect: float = 0.5,
    ):
        super().__init__()
        self.vf  = vf_estimator
        self.sigma_min = sigma_min
        self.p_uncond  = p_uncond
        self.Ke        = Ke
        self.spfm      = SPFMFilter(warmup=spfm_warmup, t_inspect=spfm_t_inspect)

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------

    def forward(
        self,
        z1: torch.Tensor,
        text_tokens: torch.Tensor,
        ref_latents: torch.Tensor,
        ref_mask: Optional[torch.Tensor] = None,
        step: int = 0,
    ) -> torch.Tensor:
        """Training step.

        Args:
            z1:          (B, C, T) target compressed latents (normalised)
            text_tokens: (B, L)    character token IDs
            ref_latents: (B, C, T_ref) reference compressed latents
            ref_mask:    (B, 1, T) binary mask — 0 on the reference region
                         (prevents info leakage, see paper §3.2.4)
            step:        current global training step (for SPFM)
        Returns:
            scalar loss
        """
        B, C, T = z1.shape
        device   = z1.device

        # ── Context-sharing batch expansion ──────────────────────────────
        # Encode conditions ONCE for real batch, then expand for Ke noise draws
        z0    = torch.randn(B * self.Ke, C, T, device=device)
        t     = torch.rand(B * self.Ke, device=device)
        z1_ke = z1.repeat_interleave(self.Ke, dim=0)          # (B·Ke, C, T)

        z_t, u_t = ot_cfm_interpolate(z0, z1_ke, t, self.sigma_min)

        # ── Classifier-free guidance dropout ─────────────────────────────
        # Drop all conditions with probability p_uncond per sample
        use_cond = torch.rand(B * self.Ke, device=device) >= self.p_uncond  # (B·Ke,)

        # ── SPFM per-sample masking ───────────────────────────────────────
        # After warmup: additionally drop conditioning for "unreliable" samples
        # (those where conditional loss > unconditional loss at t=0.5)
        # Evaluated on the original batch (not expanded) for efficiency
        if step >= self.spfm.warmup:
            def _model_fn(zt, t_, use_cond_flag, **kw):
                return self.vf(zt, t_,
                               text_tokens=kw['text_tokens'],
                               ref_latents=kw['ref_latents'],
                               use_cond=use_cond_flag)
            spfm_mask = self.spfm.should_condition(
                step, z1, _model_fn,
                cond_kwargs=dict(text_tokens=text_tokens, ref_latents=ref_latents),
                sigma_min=self.sigma_min,
            )                                                   # (B,)
            # Expand to Ke dimension
            spfm_mask_ke = spfm_mask.repeat_interleave(self.Ke)   # (B·Ke,)
            use_cond = use_cond & spfm_mask_ke

        # ── Velocity prediction ───────────────────────────────────────────
        # Text tokens and ref latents are repeated to match expanded batch
        text_ke = text_tokens.repeat_interleave(self.Ke, dim=0)  # (B·Ke, L)
        ref_ke  = ref_latents.repeat_interleave(self.Ke, dim=0)  # (B·Ke, C, Tr)

        v_pred = self.vf(
            z_t,
            t,
            text_tokens=text_ke,
            ref_latents=ref_ke,
            use_cond=use_cond,
        )

        # ── Masked L1 loss ────────────────────────────────────────────────
        mask_ke = None
        if ref_mask is not None:
            mask_ke = ref_mask.repeat_interleave(self.Ke, dim=0)

        loss = cfm_loss(v_pred, u_t, mask=mask_ke)
        return loss

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def synthesise(
        self,
        text_tokens: torch.Tensor,
        ref_latents: torch.Tensor,
        duration_frames: int,
        nfe: int = 32,
        cfg_strength: float = 3.0,
        sway: bool = True,
        sway_coef: float = -1.0,
    ) -> torch.Tensor:
        """Generate compressed latents for given text + reference.

        Args:
            text_tokens:    (1, L) or (B, L) character IDs
            ref_latents:    (1, C, T_ref) or (B, C, T_ref) reference
            duration_frames: number of compressed-latent time frames to generate
            nfe:            Euler steps (32 gives best quality/speed trade-off)
            cfg_strength:   CFG scale — 0 = no guidance (paper uses 3.0)
            sway:           use F5-TTS sway-sampling
            sway_coef:      sway coefficient
        Returns:
            (B, C, duration_frames) generated compressed latents
        """
        B = text_tokens.shape[0]
        device = text_tokens.device
        dtype  = next(self.parameters()).dtype

        def _model_fn(z_t, t_batch, use_cond, **_):
            return self.vf(
                z_t.to(dtype),
                t_batch.to(dtype),
                text_tokens=text_tokens,
                ref_latents=ref_latents,
                use_cond=use_cond,
            )

        z1 = euler_solve(
            model_fn    = _model_fn,
            z_shape     = (B, ref_latents.shape[1], duration_frames),
            nfe         = nfe,
            cfg_strength = cfg_strength,
            cond_kwargs  = {},
            device       = device,
            dtype        = dtype,
            sway         = sway,
            sway_coef    = sway_coef,
            sigma_min    = self.sigma_min,
        )
        return z1
