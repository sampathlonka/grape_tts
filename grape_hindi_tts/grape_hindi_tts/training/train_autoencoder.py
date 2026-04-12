"""
Stage 1: Speech Autoencoder Training for SupertonicTTS

GAN training with generator (encoder+decoder) and discriminator (MPD+MRD).
Based on paper Section 4.2 and Appendix B.1.
"""
import os
import argparse
import logging
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import librosa
import numpy as np

from trainer_utils import (
    setup_training, create_optimizer, create_scheduler,
    save_checkpoint, load_checkpoint, AverageMeter,
    GracefulInterruptHandler, count_parameters, get_lr,
    load_config, gradient_clip, log_metrics
)


# ============================================================================
# Mel Spectrogram Processor
# ============================================================================

class MelSpectrogramProcessor:
    """Compute mel spectrograms with configurable parameters."""

    def __init__(
        self,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 228,
        sample_rate: int = 44100
    ):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.sample_rate = sample_rate

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Compute mel spectrogram.

        Args:
            audio: [B, T] audio tensor

        Returns:
            [B, n_mels, time_steps] mel spectrogram
        """
        # Convert to numpy for librosa
        if audio.is_cuda:
            audio_np = audio.cpu().numpy()
        else:
            audio_np = audio.numpy()

        mel_specs = []
        for batch_item in audio_np:
            mel_spec = librosa.feature.melspectrogram(
                y=batch_item,
                sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels
            )
            # Log scale
            mel_spec = np.log(np.maximum(mel_spec, 1e-9))
            mel_specs.append(mel_spec)

        mel_specs = torch.from_numpy(np.stack(mel_specs)).float()
        return mel_specs


# ============================================================================
# Multi-Resolution Spectral Loss
# ============================================================================

class MultiResolutionSpectralLoss(nn.Module):
    """Multi-resolution spectral L1 loss."""

    def __init__(self, fft_sizes: list = None):
        super().__init__()
        self.fft_sizes = fft_sizes or [1024, 2048, 4096]
        self.mel_processor = MelSpectrogramProcessor()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-resolution spectral loss.

        Args:
            pred: [B, T] predicted audio
            target: [B, T] target audio

        Returns:
            Scalar loss
        """
        loss = 0.0

        # Compute mel spectrograms at different resolutions
        for fft_size in self.fft_sizes:
            mel_proc = MelSpectrogramProcessor(n_fft=fft_size, hop_length=fft_size // 4)

            pred_mel = mel_proc(pred)
            target_mel = mel_proc(target)

            # Pad to same length
            if pred_mel.shape[-1] < target_mel.shape[-1]:
                pred_mel = F.pad(pred_mel, (0, target_mel.shape[-1] - pred_mel.shape[-1]))
            elif target_mel.shape[-1] < pred_mel.shape[-1]:
                target_mel = F.pad(target_mel, (0, pred_mel.shape[-1] - target_mel.shape[-1]))

            loss += F.l1_loss(pred_mel, target_mel)

        return loss / len(self.fft_sizes)


# ============================================================================
# Generator: Encoder + Decoder
# ============================================================================

class ResidualBlock(nn.Module):
    """Residual block with dilated convolution."""

    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size=3,
            dilation=dilation, padding=dilation
        )
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size=3,
            dilation=dilation, padding=dilation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        return x + residual


class Encoder(nn.Module):
    """Speech encoder."""

    def __init__(self, latent_dim: int = 128, channels: int = 256):
        super().__init__()
        self.latent_dim = latent_dim

        # Downsampling blocks
        self.conv1 = nn.Conv1d(1, channels, kernel_size=7, stride=1, padding=3)
        self.conv2 = nn.Conv1d(channels, channels * 2, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv1d(channels * 2, channels * 4, kernel_size=5, stride=2, padding=2)
        self.conv4 = nn.Conv1d(channels * 4, channels * 8, kernel_size=5, stride=2, padding=2)

        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(channels * 8, dilation=1),
            ResidualBlock(channels * 8, dilation=3),
            ResidualBlock(channels * 8, dilation=5),
        ])

        # Latent projection
        self.latent_proj = nn.Conv1d(channels * 8, latent_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode audio to latent.

        Args:
            x: [B, 1, T] audio waveform

        Returns:
            [B, latent_dim, T'] latent code
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        for res_block in self.res_blocks:
            x = res_block(x)

        x = self.latent_proj(x)
        return x


class Decoder(nn.Module):
    """Speech decoder."""

    def __init__(self, latent_dim: int = 128, channels: int = 256):
        super().__init__()

        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(channels * 8, dilation=1),
            ResidualBlock(channels * 8, dilation=3),
            ResidualBlock(channels * 8, dilation=5),
        ])

        # Upsampling blocks
        self.latent_proj = nn.Conv1d(latent_dim, channels * 8, kernel_size=1)
        self.conv1 = nn.ConvTranspose1d(
            channels * 8, channels * 4, kernel_size=5,
            stride=2, padding=2, output_padding=1
        )
        self.conv2 = nn.ConvTranspose1d(
            channels * 4, channels * 2, kernel_size=5,
            stride=2, padding=2, output_padding=1
        )
        self.conv3 = nn.ConvTranspose1d(
            channels * 2, channels, kernel_size=5,
            stride=2, padding=2, output_padding=1
        )
        self.conv4 = nn.Conv1d(channels, 1, kernel_size=7, stride=1, padding=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to audio.

        Args:
            x: [B, latent_dim, T'] latent code

        Returns:
            [B, 1, T] audio waveform
        """
        x = self.latent_proj(x)

        for res_block in self.res_blocks:
            x = res_block(x)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.tanh(self.conv4(x))

        return x


class AutoencoderGenerator(nn.Module):
    """Generator: Encoder + Decoder."""

    def __init__(self, latent_dim: int = 128, channels: int = 256):
        super().__init__()
        self.encoder = Encoder(latent_dim, channels)
        self.decoder = Decoder(latent_dim, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)


# ============================================================================
# Multi-Period Discriminator (MPD)
# ============================================================================

class MPDBlock(nn.Module):
    """Multi-Period Discriminator block."""

    def __init__(self, period: int, channels: int = 32):
        super().__init__()
        self.period = period

        self.conv_blocks = nn.ModuleList([
            nn.Conv2d(1, channels, kernel_size=(5, 1), padding=(2, 0)),
            nn.Conv2d(channels, channels * 2, kernel_size=(5, 1), stride=(2, 1), padding=(2, 0)),
            nn.Conv2d(channels * 2, channels * 4, kernel_size=(5, 1), stride=(2, 1), padding=(2, 0)),
            nn.Conv2d(channels * 4, channels * 8, kernel_size=(5, 1), stride=(2, 1), padding=(2, 0)),
        ])

        self.output_conv = nn.Conv2d(channels * 8, 1, kernel_size=(3, 1), padding=(1, 0))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, list]:
        """
        Forward pass.

        Args:
            x: [B, 1, T] audio

        Returns:
            Discriminator output and list of intermediate features
        """
        # Reshape for period
        B, C, T = x.shape
        if T % self.period != 0:
            x = F.pad(x, (0, self.period - T % self.period))
            T = x.shape[-1]

        x = x.view(B, C, T // self.period, self.period)

        features = []
        for conv in self.conv_blocks:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            features.append(x)

        x = self.output_conv(x)
        features.append(x)

        return x, features


class MultiPeriodDiscriminator(nn.Module):
    """Multi-Period Discriminator."""

    def __init__(self, periods: list = None):
        super().__init__()
        self.periods = periods or [2, 3, 5, 7, 11]
        self.discriminators = nn.ModuleList([
            MPDBlock(period) for period in self.periods
        ])

    def forward(self, x: torch.Tensor) -> Tuple[list, list]:
        """
        Forward pass.

        Args:
            x: [B, 1, T] audio

        Returns:
            List of discriminator outputs and feature lists
        """
        outputs = []
        features = []

        for disc in self.discriminators:
            out, feats = disc(x)
            outputs.append(out)
            features.append(feats)

        return outputs, features


# ============================================================================
# Multi-Resolution Discriminator (MRD)
# ============================================================================

class MRDBlock(nn.Module):
    """Multi-Resolution Discriminator block."""

    def __init__(self, channels: int = 32):
        super().__init__()
        self.conv_blocks = nn.ModuleList([
            nn.Conv1d(1, channels, kernel_size=15, stride=1, padding=7),
            nn.Conv1d(channels, channels * 2, kernel_size=41, stride=2, padding=20, groups=4),
            nn.Conv1d(channels * 2, channels * 4, kernel_size=41, stride=2, padding=20, groups=16),
            nn.Conv1d(channels * 4, channels * 8, kernel_size=5, stride=1, padding=2),
        ])

        self.output_conv = nn.Conv1d(channels * 8, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, list]:
        """Forward pass."""
        features = []
        for conv in self.conv_blocks:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            features.append(x)

        x = self.output_conv(x)
        features.append(x)

        return x, features


class MultiResolutionDiscriminator(nn.Module):
    """Multi-Resolution Discriminator."""

    def __init__(self, resolutions: list = None):
        super().__init__()
        self.resolutions = resolutions or [8, 16, 32]
        self.discriminators = nn.ModuleList([
            MRDBlock() for _ in self.resolutions
        ])

    def forward(self, x: torch.Tensor) -> Tuple[list, list]:
        """Forward pass."""
        outputs = []
        features = []

        for i, disc in enumerate(self.discriminators):
            # Resample audio to different resolution
            if self.resolutions[i] != 1:
                x_resampled = F.interpolate(
                    x, scale_factor=1.0 / self.resolutions[i], mode='linear'
                )
            else:
                x_resampled = x

            out, feats = disc(x_resampled)
            outputs.append(out)
            features.append(feats)

        return outputs, features


# ============================================================================
# Losses
# ============================================================================

class LSGANGeneratorLoss(nn.Module):
    """Least-squares GAN generator loss."""

    def forward(self, disc_outputs: list) -> torch.Tensor:
        """
        Compute generator loss.

        Args:
            disc_outputs: List of discriminator outputs

        Returns:
            Generator loss
        """
        loss = 0.0
        for output in disc_outputs:
            loss += torch.mean((output - 1) ** 2)
        return loss / len(disc_outputs)


class LSGANDiscriminatorLoss(nn.Module):
    """Least-squares GAN discriminator loss."""

    def forward(self, real_outputs: list, fake_outputs: list) -> torch.Tensor:
        """
        Compute discriminator loss.

        Args:
            real_outputs: Discriminator outputs on real audio
            fake_outputs: Discriminator outputs on generated audio

        Returns:
            Discriminator loss
        """
        loss = 0.0
        for real_out, fake_out in zip(real_outputs, fake_outputs):
            loss += torch.mean((real_out - 1) ** 2) + torch.mean(fake_out ** 2)
        return loss / len(real_outputs)


class FeatureMatchingLoss(nn.Module):
    """Feature matching loss between discriminators."""

    def forward(self, real_features: list, fake_features: list) -> torch.Tensor:
        """
        Compute feature matching loss.

        Args:
            real_features: Features from discriminator on real audio
            fake_features: Features from discriminator on generated audio

        Returns:
            Feature matching loss
        """
        loss = 0.0
        for real_feat_list, fake_feat_list in zip(real_features, fake_features):
            for real_feat, fake_feat in zip(real_feat_list, fake_feat_list):
                loss += F.l1_loss(real_feat, fake_feat.detach())
        return loss


# ============================================================================
# Dummy Dataset
# ============================================================================

class AudioDataset(Dataset):
    """Dummy audio dataset for demonstration."""

    def __init__(self, num_samples: int = 1000, sample_rate: int = 44100):
        self.num_samples = num_samples
        self.sample_rate = sample_rate
        self.duration = 0.19  # ~0.19s for 8192 samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random audio
        num_frames = int(self.sample_rate * self.duration)
        audio = torch.randn(num_frames) * 0.1
        return audio.unsqueeze(0)  # [1, T]


# ============================================================================
# Training
# ============================================================================

class AutoencoderTrainer:
    """Trains the speech autoencoder."""

    def __init__(self, config: Dict, device: torch.device, tracker):
        self.config = config
        self.device = device
        self.tracker = tracker

        # Models
        self.generator = AutoencoderGenerator(
            latent_dim=config.get("latent_dim", 128),
            channels=config.get("channels", 256)
        ).to(device)

        self.mpd = MultiPeriodDiscriminator().to(device)
        self.mrd = MultiResolutionDiscriminator().to(device)

        # Optimizers
        self.gen_optimizer = create_optimizer(
            self.generator,
            lr=config.get("gen_lr", 2e-4),
            weight_decay=config.get("weight_decay", 0.01)
        )

        self.disc_optimizer = create_optimizer(
            nn.ModuleList([self.mpd, self.mrd]),
            lr=config.get("disc_lr", 2e-4),
            weight_decay=config.get("weight_decay", 0.01)
        )

        # Schedulers
        total_steps = config.get("num_iterations", 1500000)
        self.gen_scheduler = create_scheduler(
            self.gen_optimizer,
            total_steps=total_steps,
            decay_interval=config.get("decay_interval", 300000)
        )
        self.disc_scheduler = create_scheduler(
            self.disc_optimizer,
            total_steps=total_steps,
            decay_interval=config.get("decay_interval", 300000)
        )

        # Loss functions
        self.spec_loss = MultiResolutionSpectralLoss().to(device)
        self.gen_loss = LSGANGeneratorLoss()
        self.disc_loss = LSGANDiscriminatorLoss()
        self.fm_loss = FeatureMatchingLoss()

        # Loss weights
        self.lambda_recon = config.get("lambda_recon", 45)
        self.lambda_adv = config.get("lambda_adv", 1)
        self.lambda_fm = config.get("lambda_fm", 0.1)

        # Mixed precision
        self.use_amp = config.get("use_amp", True)
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        # Log model summary
        gen_params = count_parameters(self.generator)
        enc_params = count_parameters(self.generator.encoder)
        dec_params = count_parameters(self.generator.decoder)

        self.tracker.log_model_summary({
            "autoencoder/total_params": gen_params,
            "autoencoder/encoder_params": enc_params,
            "autoencoder/decoder_params": dec_params,
            "mpd/params": count_parameters(self.mpd),
            "mrd/params": count_parameters(self.mrd),
        })

        logging.info(f"Generator parameters: {gen_params:,}")
        logging.info(f"MPD parameters: {count_parameters(self.mpd):,}")
        logging.info(f"MRD parameters: {count_parameters(self.mrd):,}")

    def train_step(self, batch: torch.Tensor) -> Dict[str, float]:
        """
        Single training step.

        Args:
            batch: [B, 1, T] audio batch

        Returns:
            Dictionary of losses
        """
        batch = batch.to(self.device)

        # ========== Discriminator Update ==========
        self.disc_optimizer.zero_grad()

        with torch.autocast("cuda", torch.bfloat16) if self.use_amp else torch.no_grad():
            # Generate fake audio
            with torch.no_grad():
                fake_audio = self.generator(batch)

            # MPD
            real_mpd_out, real_mpd_feats = self.mpd(batch)
            fake_mpd_out, _ = self.mpd(fake_audio)

            # MRD
            real_mrd_out, real_mrd_feats = self.mrd(batch)
            fake_mrd_out, _ = self.mrd(fake_audio)

            # Discriminator loss
            mpd_loss = self.disc_loss(real_mpd_out, fake_mpd_out)
            mrd_loss = self.disc_loss(real_mrd_out, fake_mrd_out)
            total_disc_loss = mpd_loss + mrd_loss

        if self.use_amp:
            self.scaler.scale(total_disc_loss).backward()
            self.scaler.unscale_(self.disc_optimizer)
            gradient_clip(nn.ModuleList([self.mpd, self.mrd]), max_norm=1.0)
            self.scaler.step(self.disc_optimizer)
            self.scaler.update()
        else:
            total_disc_loss.backward()
            gradient_clip(nn.ModuleList([self.mpd, self.mrd]), max_norm=1.0)
            self.disc_optimizer.step()

        # ========== Generator Update ==========
        self.gen_optimizer.zero_grad()

        with torch.autocast("cuda", torch.bfloat16) if self.use_amp else torch.no_grad():
            # Generate fake audio
            fake_audio = self.generator(batch)

            # Reconstruction loss
            recon_loss = self.spec_loss(fake_audio, batch)

            # Adversarial loss
            fake_mpd_out, fake_mpd_feats = self.mpd(fake_audio)
            fake_mrd_out, fake_mrd_feats = self.mrd(fake_audio)

            adv_loss = self.gen_loss(fake_mpd_out) + self.gen_loss(fake_mrd_out)

            # Feature matching loss
            real_mpd_out, real_mpd_feats = self.mpd(batch)
            real_mrd_out, real_mrd_feats = self.mrd(batch)

            fm_loss = (
                self.fm_loss(real_mpd_feats, fake_mpd_feats) +
                self.fm_loss(real_mrd_feats, fake_mrd_feats)
            )

            # Total generator loss
            total_gen_loss = (
                self.lambda_recon * recon_loss +
                self.lambda_adv * adv_loss +
                self.lambda_fm * fm_loss
            )

        if self.use_amp:
            self.scaler.scale(total_gen_loss).backward()
            self.scaler.unscale_(self.gen_optimizer)
            grad_norm = gradient_clip(self.generator, max_norm=1.0)
            self.scaler.step(self.gen_optimizer)
            self.scaler.update()
        else:
            total_gen_loss.backward()
            grad_norm = gradient_clip(self.generator, max_norm=1.0)
            self.gen_optimizer.step()

        return {
            "gen_loss": total_gen_loss.item(),
            "recon_loss": recon_loss.item(),
            "adv_loss": adv_loss.item(),
            "fm_loss": fm_loss.item(),
            "disc_loss": total_disc_loss.item(),
            "grad_norm": grad_norm,
        }

    def train(self, train_loader: DataLoader):
        """Main training loop."""
        self.generator.train()
        self.mpd.train()
        self.mrd.train()

        interrupt_handler = GracefulInterruptHandler()
        best_loss = float('inf')

        num_iterations = self.config.get("num_iterations", 1500000)
        log_every = self.config.get("log_every", 100)
        val_every = self.config.get("val_every", 10000)
        ckpt_interval = self.config.get("ckpt_interval", 50000)

        step = 0
        epoch = 0

        with tqdm(total=num_iterations, desc="Training Autoencoder") as pbar:
            while step < num_iterations:
                for batch in train_loader:
                    if interrupt_handler.interrupted:
                        logging.info("Training interrupted by user")
                        break

                    losses = self.train_step(batch)

                    # Update schedulers
                    self.gen_scheduler.step()
                    self.disc_scheduler.step()

                    # Log metrics every log_every steps
                    if (step + 1) % log_every == 0:
                        current_lr = get_lr(self.gen_optimizer)
                        log_dict = {
                            "train/g_loss": losses["gen_loss"],
                            "train/recon_loss": losses["recon_loss"],
                            "train/adv_loss": losses["adv_loss"],
                            "train/fm_loss": losses["fm_loss"],
                            "train/d_loss": losses["disc_loss"],
                            "train/grad_norm": losses["grad_norm"],
                            "train/lr": current_lr,
                        }
                        self.tracker.log_scalars(log_dict, step=step + 1)

                        pbar.update(log_every)
                        logging.info(
                            f"Step {step + 1} | "
                            f"Gen Loss: {losses['gen_loss']:.4f} | "
                            f"Disc Loss: {losses['disc_loss']:.4f} | "
                            f"LR: {current_lr:.2e}"
                        )

                    # Validation every val_every steps
                    if (step + 1) % val_every == 0:
                        val_recon_loss, val_g_loss = self.validate()
                        logging.info(f"Validation - Recon Loss: {val_recon_loss:.4f}, G Loss: {val_g_loss:.4f}")

                        # Log validation metrics
                        val_dict = {
                            "val/recon_loss": val_recon_loss,
                            "val/g_loss": val_g_loss,
                        }
                        self.tracker.log_scalars(val_dict, step=step + 1)

                        # Log reconstructed audio sample
                        with torch.no_grad():
                            sample_batch = next(iter(train_loader))
                            sample_audio = sample_batch[0:1].to(self.device)
                            recon_audio = self.generator(sample_audio)

                            recon_np = recon_audio[0, 0].cpu().numpy()
                            self.tracker.log_audio(
                                "val/reconstructed_audio",
                                recon_np,
                                step=step + 1,
                                sr=44100,
                                caption="Reconstructed audio sample"
                            )

                            # Log mel spectrogram comparison
                            mel_proc = MelSpectrogramProcessor()
                            mel_gt = mel_proc(sample_audio)
                            mel_pred = mel_proc(recon_audio)

                            mel_gt_np = mel_gt[0].cpu().numpy()
                            mel_pred_np = mel_pred[0].cpu().numpy()

                            self.tracker.log_mel_comparison(
                                "val/mel_comparison",
                                mel_gt_np,
                                mel_pred_np,
                                step=step + 1
                            )

                        if val_g_loss < best_loss:
                            best_loss = val_g_loss
                            self.save_checkpoint(step + 1, best_loss=val_g_loss, tag="best")

                    # Checkpoint every ckpt_interval steps
                    if (step + 1) % ckpt_interval == 0:
                        self.save_checkpoint(step + 1)

                    step += 1
                    if step >= num_iterations:
                        break

                epoch += 1

            pbar.close()

        logging.info(f"Training complete. Best loss: {best_loss:.4f}")
        self.save_checkpoint(step, best_loss=best_loss, tag="final")

    def validate(self) -> Tuple[float, float]:
        """Validation step."""
        self.generator.eval()
        self.mpd.eval()
        self.mrd.eval()

        val_recon_meter = AverageMeter()
        val_g_meter = AverageMeter()

        with torch.no_grad():
            # Validate on a few batches
            dataset = AudioDataset(num_samples=10)
            loader = DataLoader(dataset, batch_size=32)

            for batch in loader:
                batch = batch.to(self.device)
                fake_audio = self.generator(batch)

                recon_loss = self.spec_loss(fake_audio, batch)

                # Compute adversarial loss on validation
                fake_mpd_out, _ = self.mpd(fake_audio)
                fake_mrd_out, _ = self.mrd(fake_audio)
                adv_loss = self.gen_loss(fake_mpd_out) + self.gen_loss(fake_mrd_out)

                g_loss = self.lambda_recon * recon_loss + self.lambda_adv * adv_loss

                val_recon_meter.update(recon_loss.item())
                val_g_meter.update(g_loss.item())

        self.generator.train()
        self.mpd.train()
        self.mrd.train()

        return val_recon_meter.avg, val_g_meter.avg

    def save_checkpoint(self, step: int, best_loss: float = None, tag: str = ""):
        """Save checkpoint and log artifact."""
        ckpt_dir = Path(self.tracker.output_dir) / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        ckpt_name = f"autoencoder_step_{step}" + (f"_{tag}" if tag else "") + ".pt"
        ckpt_path = ckpt_dir / ckpt_name

        save_checkpoint(
            self.generator,
            self.gen_optimizer,
            step,
            str(ckpt_path),
            scheduler=self.gen_scheduler,
            best_loss=best_loss
        )

        # Also save discriminators
        disc_ckpt_path = str(ckpt_path).replace("autoencoder", "discriminators")
        torch.save({
            'mpd': self.mpd.state_dict(),
            'mrd': self.mrd.state_dict(),
            'step': step,
        }, disc_ckpt_path)

        # Log model artifact if it's a best checkpoint
        if tag == "best" and best_loss is not None:
            self.tracker.log_model_artifact(
                str(ckpt_path),
                name="best-autoencoder",
                metadata={"step": step, "val_loss": best_loss}
            )


def main():
    parser = argparse.ArgumentParser(description="Train autoencoder")
    parser.add_argument("--config", type=str, required=True, help="Config file path")
    parser.add_argument("--resume", type=str, help="Checkpoint to resume from")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    if args.output_dir:
        config["output_dir"] = args.output_dir

    # Setup training with new ExperimentTracker
    device, tracker = setup_training(
        config,
        output_dir=config.get("output_dir", "./outputs"),
        stage="autoencoder",
        wandb_tags=["stage1", "autoencoder"]
    )

    # Create trainer
    trainer = AutoencoderTrainer(config, device, tracker)

    # Load checkpoint if resuming
    if args.resume:
        load_checkpoint(trainer.generator, args.resume, trainer.gen_optimizer, trainer.gen_scheduler, device)

    # Create dataset
    dataset = AudioDataset(num_samples=config.get("num_samples", 10000))
    train_loader = DataLoader(
        dataset,
        batch_size=config.get("batch_size", 128),
        shuffle=True,
        num_workers=0
    )

    # Train
    trainer.train(train_loader)
    tracker.close()

    logging.info("Stage 1 (Autoencoder) training complete")


if __name__ == "__main__":
    main()
