"""
Stage 3: Duration Predictor Training for SupertonicTTS

Simple L1 loss training for predicting total speech duration.
Based on paper Sections 3.3, 4.2.
"""
import os
import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np

from supertonic_hindi_tts.training.trainer_utils import (
    setup_training, create_optimizer, create_scheduler,
    save_checkpoint, load_checkpoint, AverageMeter,
    GracefulInterruptHandler, count_parameters, get_lr,
    load_config, gradient_clip
)


# ============================================================================
# Text Encoder
# ============================================================================

class TextEncoder(nn.Module):
    """Simple text encoder for phoneme sequences."""

    def __init__(self, vocab_size: int = 256, embed_dim: int = 256, hidden_dim: int = 512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Transformer-based encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=hidden_dim,
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, text_ids: torch.Tensor) -> torch.Tensor:
        """
        Encode text to embeddings.

        Args:
            text_ids: [B, T_text] token IDs

        Returns:
            [B, T_text, embed_dim] text embeddings
        """
        x = self.embedding(text_ids)
        x = self.transformer(x)
        return x


# ============================================================================
# Duration Predictor Model
# ============================================================================

class DurationPredictor(nn.Module):
    """Predicts total speech duration from text."""

    def __init__(
        self,
        vocab_size: int = 256,
        text_embed_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 2
    ):
        super().__init__()

        # Text encoder
        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            embed_dim=text_embed_dim,
            hidden_dim=hidden_dim
        )

        # Duration prediction head
        # Takes aggregated text embedding and predicts duration
        layers = []
        input_dim = text_embed_dim

        for i in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(0.1))
            input_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, 1))  # Single output: duration in seconds
        layers.append(nn.ReLU())  # Duration is non-negative

        self.duration_head = nn.Sequential(*layers)

    def forward(self, text_ids: torch.Tensor) -> torch.Tensor:
        """
        Predict duration from text.

        Args:
            text_ids: [B, T_text] token IDs

        Returns:
            [B, 1] predicted duration in seconds
        """
        # Encode text
        text_embed = self.text_encoder(text_ids)  # [B, T_text, text_embed_dim]

        # Aggregate text embeddings
        text_agg = text_embed.mean(dim=1)  # [B, text_embed_dim]

        # Predict duration
        duration = self.duration_head(text_agg)  # [B, 1]

        return duration


# ============================================================================
# Duration Dataset
# ============================================================================

class DurationDataset(Dataset):
    """Dataset with text and corresponding durations."""

    def __init__(
        self,
        num_samples: int = 1000,
        min_duration: float = 0.5,
        max_duration: float = 10.0
    ):
        self.num_samples = num_samples
        self.min_duration = min_duration
        self.max_duration = max_duration

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random text (token IDs)
        text_len = np.random.randint(5, 100)
        text_ids = torch.randint(0, 256, (text_len,))

        # Generate duration (roughly proportional to text length)
        duration = np.random.uniform(self.min_duration, self.max_duration)
        # Add correlation with text length for realism
        duration = duration * (1.0 + text_len / 100.0)
        duration = np.clip(duration, self.min_duration, self.max_duration)

        return {
            'text_ids': text_ids,
            'duration': torch.tensor(duration, dtype=torch.float32)
        }


def collate_duration_batch(batch):
    """Custom collate function for duration dataset."""
    text_ids = []
    durations = []

    max_text_len = max(item['text_ids'].shape[0] for item in batch)

    for item in batch:
        # Pad text
        text = item['text_ids']
        pad_len = max_text_len - text.shape[0]
        text_padded = F.pad(text.unsqueeze(0), (0, pad_len)).squeeze(0)
        text_ids.append(text_padded)

        durations.append(item['duration'])

    return {
        'text_ids': torch.stack(text_ids),  # [B, T_text]
        'duration': torch.stack(durations).unsqueeze(1)  # [B, 1]
    }


# ============================================================================
# Training
# ============================================================================

class DurationPredictorTrainer:
    """Trains the duration predictor."""

    def __init__(self, config: Dict, device: torch.device, output_dir: str, tracker):
        self.config = config
        self.device = device
        self.output_dir = output_dir
        self.tracker = tracker

        # Model
        self.model = DurationPredictor(
            vocab_size=config.get("vocab_size", 256),
            text_embed_dim=config.get("text_embed_dim", 256),
            hidden_dim=config.get("hidden_dim", 512),
            num_layers=config.get("num_layers", 2)
        ).to(device)

        # Optimizer
        self.optimizer = create_optimizer(
            self.model,
            lr=config.get("lr", 5e-4),
            weight_decay=config.get("weight_decay", 0.01)
        )

        # Scheduler (minimal since training is short)
        total_steps = config.get("num_iterations", 3000)
        self.scheduler = create_scheduler(
            self.optimizer,
            total_steps=total_steps,
            decay_interval=10000
        )

        # Loss function (L1 for duration)
        self.criterion = nn.L1Loss()

        # Mixed precision
        self.use_amp = config.get("use_amp", True)
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        # Log model summary
        param_count = count_parameters(self.model)
        logging.info(f"Model parameters: {param_count:,}")
        self.tracker.log_model_summary({"duration_predictor/total_params": param_count})

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[float, float]:
        """
        Single training step.

        Args:
            batch: Batch dictionary with 'text_ids' and 'duration'

        Returns:
            Tuple of (loss, grad_norm)
        """
        # Move to device
        text_ids = batch['text_ids'].to(self.device)  # [B, T_text]
        durations = batch['duration'].to(self.device)  # [B, 1]

        # Forward pass
        self.optimizer.zero_grad()

        with torch.autocast("cuda", torch.bfloat16) if self.use_amp else torch.no_grad():
            pred_durations = self.model(text_ids)  # [B, 1]
            loss = self.criterion(pred_durations, durations)

        if self.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = gradient_clip(self.model, max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            grad_norm = gradient_clip(self.model, max_norm=1.0)
            self.optimizer.step()

        self.scheduler.step()

        return loss.item(), grad_norm

    def train(self, train_loader: DataLoader):
        """Main training loop."""
        self.model.train()

        interrupt_handler = GracefulInterruptHandler()
        best_loss = float('inf')
        loss_meter = AverageMeter()

        num_iterations = self.config.get("num_iterations", 3000)
        log_every = self.config.get("log_every", 100)
        val_every = self.config.get("val_every", 500)
        ckpt_interval = self.config.get("ckpt_interval", 1000)

        step = 0
        best_ckpt_path = None

        with tqdm(total=num_iterations, desc="Training Duration Predictor") as pbar:
            while step < num_iterations:
                for batch in train_loader:
                    if interrupt_handler.interrupted:
                        logging.info("Training interrupted by user")
                        break

                    loss, grad_norm = self.train_step(batch)
                    loss_meter.update(loss)

                    # Log training metrics every log_every steps
                    if (step + 1) % log_every == 0:
                        pbar.update(log_every)
                        current_lr = get_lr(self.optimizer)

                        logging.info(
                            f"Step {step + 1} | "
                            f"Loss: {loss_meter.avg:.6f} | "
                            f"LR: {current_lr:.2e}"
                        )

                        # Log to wandb/tracker
                        self.tracker.log(
                            {
                                "train/dur_loss": loss_meter.avg,
                                "train/lr": current_lr,
                                "train/grad_norm": grad_norm,
                            },
                            step=step + 1
                        )

                    # Validation
                    if (step + 1) % val_every == 0:
                        avg_val_loss, gt_durations, pred_durations = self.validate()
                        logging.info(f"Validation loss: {avg_val_loss:.6f}")

                        # Log validation metrics
                        self.tracker.log(
                            {"val/dur_loss": avg_val_loss},
                            step=step + 1
                        )

                        # Log duration scatter plot
                        gt_list = [float(x) for x in gt_durations]
                        pred_list = [float(x) for x in pred_durations]
                        self.tracker.log_duration_scatter(gt_list, pred_list, step=step + 1)

                        if avg_val_loss < best_loss:
                            best_loss = avg_val_loss
                            best_ckpt_path = self.save_checkpoint(step + 1, best_loss=best_loss, tag="best")

                    # Checkpoint
                    if (step + 1) % ckpt_interval == 0:
                        self.save_checkpoint(step + 1)

                    step += 1
                    if step >= num_iterations:
                        break

            pbar.close()

        logging.info(f"Training complete. Best loss: {best_loss:.6f}")
        final_ckpt = self.save_checkpoint(step, best_loss=best_loss, tag="final")

        # Log final model artifact
        if best_ckpt_path:
            self.tracker.log_model_artifact(best_ckpt_path, "best-duration-predictor")
        self.tracker.log_model_artifact(final_ckpt, "final-duration-predictor")

    def validate(self) -> Tuple[float, List[float], List[float]]:
        """
        Validation step.

        Returns:
            Tuple of (avg_loss, gt_durations_list, pred_durations_list)
        """
        self.model.eval()

        val_loss = AverageMeter()
        gt_durations_all = []
        pred_durations_all = []

        with torch.no_grad():
            # Validate on a few batches
            dataset = DurationDataset(num_samples=100)
            loader = DataLoader(
                dataset,
                batch_size=32,
                collate_fn=collate_duration_batch
            )

            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                pred_durations = self.model(batch['text_ids'])  # [B, 1]
                loss = self.criterion(pred_durations, batch['duration'])
                val_loss.update(loss.item())

                # Collect predictions and ground truth
                gt_durations_all.extend(batch['duration'].squeeze().cpu().numpy().tolist())
                pred_durations_all.extend(pred_durations.squeeze().detach().cpu().numpy().tolist())

        self.model.train()
        return val_loss.avg, gt_durations_all, pred_durations_all

    def save_checkpoint(self, step: int, best_loss: float = None, tag: str = "") -> str:
        """
        Save checkpoint.

        Args:
            step: Training step
            best_loss: Best validation loss
            tag: Checkpoint tag

        Returns:
            Path to saved checkpoint
        """
        ckpt_dir = os.path.join(self.output_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)

        ckpt_name = f"duration_step_{step}" + (f"_{tag}" if tag else "") + ".pt"
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)

        save_checkpoint(
            self.model,
            self.optimizer,
            step,
            ckpt_path,
            scheduler=self.scheduler,
            best_loss=best_loss
        )

        return ckpt_path


def main():
    parser = argparse.ArgumentParser(description="Train duration predictor")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    parser.add_argument("--resume", type=str, help="Checkpoint to resume from")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    if args.output_dir:
        config["output_dir"] = args.output_dir

    # Setup training with new signature returning (device, tracker)
    device, tracker = setup_training(config, args.output_dir, stage="duration", wandb_tags=["stage3"])

    # Create trainer
    trainer = DurationPredictorTrainer(config, device, args.output_dir, tracker)

    # Load checkpoint if resuming
    if args.resume:
        load_checkpoint(trainer.model, args.resume, trainer.optimizer, trainer.scheduler, device)

    # Create dataset
    dataset = DurationDataset(
        num_samples=config.get("num_samples", 5000),
        min_duration=config.get("min_duration", 0.5),
        max_duration=config.get("max_duration", 10.0)
    )

    train_loader = DataLoader(
        dataset,
        batch_size=config.get("batch_size", 128),
        shuffle=True,
        collate_fn=collate_duration_batch,
        num_workers=0
    )

    # Train
    trainer.train(train_loader)
    tracker.close()

    logging.info("Stage 3 (Duration Predictor) training complete")


if __name__ == "__main__":
    main()
