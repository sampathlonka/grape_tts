"""
Shared training utilities for all SupertonicTTS training stages.

Automatically loads .env (WANDB_API_KEY etc.) before any W&B calls.
Returns ExperimentTracker from setup_training() — used by all 3 stages.
"""

from __future__ import annotations

import logging
import os
import random
import signal
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

# .env is auto-loaded when logging_utils is imported
from supertonic_hindi_tts.utils.logging_utils import ExperimentTracker, setup_logger

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Running average
# ─────────────────────────────────────────────────────────────────────────────

class AverageMeter:
    """Tracks running average for a single metric."""

    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self):
        self.val   = 0.0
        self.avg   = 0.0
        self.sum   = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count


# ─────────────────────────────────────────────────────────────────────────────
# Signal handler
# ─────────────────────────────────────────────────────────────────────────────

class GracefulInterruptHandler:
    def __init__(self):
        self.interrupted = False
        signal.signal(signal.SIGINT,  self._handle)
        signal.signal(signal.SIGTERM, self._handle)

    def _handle(self, signum, frame):
        logger.warning("Interrupt signal received — wrapping up current step …")
        self.interrupted = True


# ─────────────────────────────────────────────────────────────────────────────
# Setup helpers
# ─────────────────────────────────────────────────────────────────────────────

def setup_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        props  = torch.cuda.get_device_properties(0)
        logger.info(f"GPU : {props.name}")
        logger.info(f"VRAM: {props.total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        logger.warning("CUDA not available — running on CPU")
    return device


def setup_dgx_spark() -> None:
    """GB10-specific optimisations."""
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("DGX Spark GB10: high matmul precision enabled")


def load_config(config_path: str) -> Dict[str, Any]:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    logger.info(f"Config loaded: {config_path}")
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Master setup — returns ExperimentTracker (TB + W&B)
# ─────────────────────────────────────────────────────────────────────────────

def setup_training(
    config: Dict[str, Any],
    output_dir: Path,
    stage: str,
    wandb_tags: Optional[list] = None,
    wandb_notes: Optional[str] = None,
    wandb_resume: bool = False,
    wandb_run_id: Optional[str] = None,
) -> Tuple[torch.device, ExperimentTracker]:
    """
    Initialise seed, device, DGX opts, logging, TensorBoard, and W&B.

    Args:
        config:       full YAML config dict
        output_dir:   Path for checkpoints and logs
        stage:        one of "autoencoder" | "ttl" | "duration" | "evaluation"
        wandb_tags:   extra tags for the W&B run
        wandb_notes:  free-text notes for W&B
        wandb_resume: resume a previous W&B run
        wandb_run_id: specific W&B run ID to resume

    Returns:
        (device, tracker) where tracker has .log_scalars(), .log_audio(), etc.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"

    # Python logging
    setup_logger("supertonic", str(log_dir))
    logger.info(f"{'='*60}")
    logger.info(f" SupertonicTTS Hindi — Stage: {stage}")
    logger.info(f"{'='*60}")

    # Seed + device
    setup_seed(config.get("project", {}).get("seed", 42))
    device = setup_device()
    setup_dgx_spark()

    # W&B: pick up project/entity from env (set in .env) or config
    proj_cfg  = config.get("project", {})
    use_wandb = os.environ.get("WANDB_MODE", "online") != "disabled"

    tracker = ExperimentTracker(
        log_dir         = str(output_dir / "tensorboard"),
        stage           = stage,
        config          = config,
        use_tensorboard = True,
        use_wandb       = use_wandb,
        wandb_project   = os.environ.get("WANDB_PROJECT", proj_cfg.get("name", "supertonic-hindi-tts")),
        wandb_entity    = os.environ.get("WANDB_ENTITY"),
        wandb_name      = f"{stage}-{proj_cfg.get('name', 'hindi')}",
        wandb_tags      = wandb_tags,
        wandb_notes     = wandb_notes,
        wandb_resume    = wandb_resume,
        wandb_run_id    = wandb_run_id,
    )

    if tracker.wandb_url:
        logger.info(f"W&B run: {tracker.wandb_url}")

    return device, tracker


# ─────────────────────────────────────────────────────────────────────────────
# Optimiser / scheduler
# ─────────────────────────────────────────────────────────────────────────────

def create_optimizer(
    model: nn.Module,
    lr: float = 2e-4,
    weight_decay: float = 0.01,
    betas: Tuple[float, float] = (0.9, 0.999),
) -> AdamW:
    return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)


def create_scheduler(
    optimizer: AdamW,
    total_steps: int,
    warmup_steps: int = 0,
    decay_factor: float = 0.5,
    decay_interval: int = 300_000,
) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        decay_steps = (step - warmup_steps) // decay_interval
        return decay_factor ** decay_steps

    return LambdaLR(optimizer, lr_lambda)


def get_lr(optimizer: AdamW) -> float:
    return optimizer.param_groups[0]["lr"]


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[AdamW],
    step: int,
    path: Path,
    scheduler: Optional[LambdaLR] = None,
    best_loss: Optional[float] = None,
    extra: Optional[Dict] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {"model": model.state_dict(), "step": step}
    if optimizer:
        ckpt["optimizer"] = optimizer.state_dict()
    if scheduler:
        ckpt["scheduler"] = scheduler.state_dict()
    if best_loss is not None:
        ckpt["best_loss"] = best_loss
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, str(path))
    logger.info(f"Checkpoint saved: {path}  (step={step})")


def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[AdamW],
    path: str,
    device: torch.device,
    scheduler: Optional[LambdaLR] = None,
) -> Tuple[int, Optional[float]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    step      = ckpt.get("step", 0)
    best_loss = ckpt.get("best_loss", None)
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    logger.info(f"Checkpoint loaded: {path}  (step={step})")
    return step, best_loss


# ─────────────────────────────────────────────────────────────────────────────
# Misc
# ─────────────────────────────────────────────────────────────────────────────

def gradient_clip(model: nn.Module, max_norm: float = 1.0) -> float:
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm).item()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def log_metrics(tracker: ExperimentTracker, metrics: Dict[str, float], step: int) -> None:
    """Thin shim kept for backwards compatibility."""
    tracker.log_scalars(metrics, step)


@contextmanager
def disable_logging(disable: bool = True):
    if not disable:
        yield
        return
    old = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        yield
    finally:
        logging.disable(old)
