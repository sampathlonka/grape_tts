"""
Loss functions for Speech Autoencoder training.

Includes spectral reconstruction, adversarial, feature matching,
flow matching, and duration losses.
"""

from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def spectral_reconstruction_loss(
    y_hat: torch.Tensor,
    y: torch.Tensor,
    fft_sizes: List[int] = None,
    mel_bands: List[int] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Multi-resolution spectral reconstruction loss using STFT.

    Computes L1 loss between log-magnitude spectrograms at multiple
    resolutions (FFT sizes) to capture both global and local structure.

    Args:
        y_hat: Reconstructed waveform of shape (B, T)
        y: Target waveform of shape (B, T)
        fft_sizes: List of FFT sizes (default: [1024, 2048, 4096])
        mel_bands: List of mel bands per FFT size (default: [64, 128, 128])
        reduction: Reduction method - 'mean' or 'sum' (default: 'mean')

    Returns:
        Scalar loss tensor
    """
    if fft_sizes is None:
        fft_sizes = [1024, 2048, 4096]
    if mel_bands is None:
        mel_bands = [64, 128, 128]

    assert len(fft_sizes) == len(
        mel_bands
    ), "fft_sizes and mel_bands must have same length"

    loss = 0.0
    num_resolutions = len(fft_sizes)

    for fft_size, n_mels in zip(fft_sizes, mel_bands):
        # Ensure minimum hop_length
        hop_size = fft_size // 4

        # Window for STFT
        window = torch.hann_window(fft_size, device=y.device)

        # Compute STFT for both signals
        # Output shape: (B, n_fft//2 + 1, n_frames, 2) [real and imag parts]
        spec_hat = torch.stft(
            y_hat,
            n_fft=fft_size,
            hop_length=hop_size,
            win_length=fft_size,
            window=window,
            return_complex=False,
        )
        spec_y = torch.stft(
            y,
            n_fft=fft_size,
            hop_length=hop_size,
            win_length=fft_size,
            window=window,
            return_complex=False,
        )

        # Convert to magnitude (L2 norm of real and imaginary parts)
        # Shape: (B, n_fft//2 + 1, n_frames)
        magnitude_hat = torch.sqrt(
            spec_hat[:, :, :, 0] ** 2 + spec_hat[:, :, :, 1] ** 2 + 1e-9
        )
        magnitude_y = torch.sqrt(
            spec_y[:, :, :, 0] ** 2 + spec_y[:, :, :, 1] ** 2 + 1e-9
        )

        # Log magnitude for perceptual loss (log compression)
        log_mag_hat = torch.log(magnitude_hat + 1e-9)
        log_mag_y = torch.log(magnitude_y + 1e-9)

        # L1 loss on log magnitude spectrograms
        # Shape of log_mag_*: (B, n_fft//2 + 1, n_frames)
        lin_loss = F.l1_loss(magnitude_hat, magnitude_y, reduction=reduction)
        log_loss = F.l1_loss(log_mag_hat, log_mag_y, reduction=reduction)

        # Weight: contributions from all resolutions
        loss = loss + lin_loss + log_loss

    # Average over number of resolutions
    loss = loss / (2 * num_resolutions)

    return loss


def adversarial_loss_generator(
    disc_outputs: List[torch.Tensor], reduction: str = "mean"
) -> torch.Tensor:
    """
    Adversarial loss for generator (LS-GAN style).

    Encourages discriminator to believe generated samples are real.
    Formula: L_adv(G;D) = E[(D(G(x)) - 1)^2]

    Args:
        disc_outputs: List of discriminator outputs (one per sub-discriminator)
                     Each of shape (B, ...)
        reduction: Reduction method - 'mean' or 'sum' (default: 'mean')

    Returns:
        Scalar loss tensor
    """
    loss = 0.0

    for output in disc_outputs:
        # Discriminator output should be close to 1 (real)
        # Loss = E[(D(x_fake) - 1)^2]
        loss = loss + F.mse_loss(output, torch.ones_like(output), reduction=reduction)

    # Average over sub-discriminators
    loss = loss / len(disc_outputs)

    return loss


def adversarial_loss_discriminator(
    real_outputs: List[torch.Tensor],
    fake_outputs: List[torch.Tensor],
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Adversarial loss for discriminator (LS-GAN style with hinge-like form).

    Encourages discriminator to output 1 for real samples and 0 for fake.
    Formula: L_adv(D;G) = E[(D(x_real) - 1)^2] + E[D(x_fake)^2]

    This is equivalent to the hinge-like form in the paper equations 4,5:
    L_adv(D;G) = E[(D(x_fake) + 1)^2 + (D(x_real) - 1)^2]

    Args:
        real_outputs: List of discriminator outputs for real samples
        fake_outputs: List of discriminator outputs for fake samples
        reduction: Reduction method - 'mean' or 'sum' (default: 'mean')

    Returns:
        Scalar loss tensor
    """
    loss_real = 0.0
    loss_fake = 0.0

    for real_out, fake_out in zip(real_outputs, fake_outputs):
        # Real: discriminator should output 1
        loss_real = loss_real + F.mse_loss(
            real_out, torch.ones_like(real_out), reduction=reduction
        )

        # Fake: discriminator should output 0
        loss_fake = loss_fake + F.mse_loss(
            fake_out, torch.zeros_like(fake_out), reduction=reduction
        )

    # Average over sub-discriminators
    num_discs = len(real_outputs)
    loss = (loss_real + loss_fake) / num_discs

    return loss


def feature_matching_loss(
    real_features: List[torch.Tensor],
    fake_features: List[torch.Tensor],
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Feature matching loss between real and fake samples.

    Encourages generator to produce intermediate discriminator features
    that match those of real samples. This provides richer gradients
    than just adversarial loss.

    Formula: L_fm = E[||D^(l)(x_real) - D^(l)(x_fake)||_1]
    where D^(l) are intermediate features at layer l.

    Args:
        real_features: List of real intermediate features from discriminator
                      Each of shape (B, C, H, W) or similar
        fake_features: List of fake intermediate features from discriminator
        reduction: Reduction method - 'mean' or 'sum' (default: 'mean')

    Returns:
        Scalar loss tensor
    """
    loss = 0.0

    for real_feat, fake_feat in zip(real_features, fake_features):
        # L1 loss between features
        loss = loss + F.l1_loss(real_feat, fake_feat, reduction=reduction)

    # Average over feature layers
    if len(real_features) > 0:
        loss = loss / len(real_features)
    else:
        loss = torch.tensor(0.0, device=real_features[0].device)

    return loss


def flow_matching_loss(
    predicted_velocity: torch.Tensor,
    target_velocity: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Flow matching loss for continuous latent trajectory matching.

    Used in flow-based generative models. Matches predicted velocity field
    to target velocity (difference between start and end states).

    Formula: L_flow = ||v_θ(t, x_t) - (x_1 - x_0)||^2

    Args:
        predicted_velocity: Predicted velocity from model
                          Shape: (B, T, latent_dim) or (B, latent_dim, T)
        target_velocity: Target velocity (x_1 - x_0)
                        Same shape as predicted_velocity
        mask: Optional mask for valid time steps
             Shape: (B, T) or (B, 1, T)
        reduction: Reduction method - 'mean' or 'sum' (default: 'mean')

    Returns:
        Scalar loss tensor
    """
    # MSE loss between predicted and target velocities
    loss = F.mse_loss(predicted_velocity, target_velocity, reduction="none")

    # Apply mask if provided
    if mask is not None:
        # Ensure mask has correct shape
        if mask.dim() == 2:
            mask = mask.unsqueeze(1)  # (B, T) → (B, 1, T)
        elif mask.dim() == 3 and mask.shape[1] == 1:
            pass  # Already (B, 1, T) or similar
        else:
            raise ValueError(f"Unexpected mask shape: {mask.shape}")

        # Expand mask to match loss shape
        mask = mask.float()
        loss = loss * mask

    # Reduce
    if reduction == "mean":
        if mask is not None:
            # Average only over valid positions
            loss = loss.sum() / mask.sum().clamp(min=1.0)
        else:
            loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    else:
        raise ValueError(f"Unknown reduction: {reduction}")

    return loss


def duration_loss(
    predicted_duration: torch.Tensor,
    target_duration: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    L1 loss for predicted phoneme/duration alignment.

    Used in TTS models to align predicted durations to ground truth.

    Args:
        predicted_duration: Predicted durations, shape (B, num_phonemes) or (B, num_phonemes, 1)
        target_duration: Target durations, same shape as predicted
        reduction: Reduction method - 'mean' or 'sum' (default: 'mean')

    Returns:
        Scalar loss tensor
    """
    # L1 loss (MAE)
    loss = F.l1_loss(predicted_duration, target_duration, reduction=reduction)
    return loss


# Combined loss utilities
def combined_generator_loss(
    reconstructed_waveform: torch.Tensor,
    target_waveform: torch.Tensor,
    disc_outputs: List[torch.Tensor],
    real_features: List[torch.Tensor],
    fake_features: List[torch.Tensor],
    lambda_adv: float = 1.0,
    lambda_feat: float = 10.0,
    lambda_spec: float = 1.0,
) -> Tuple[torch.Tensor, dict]:
    """
    Combined generator loss with spectral, adversarial, and feature matching.

    Args:
        reconstructed_waveform: Generator output, shape (B, T)
        target_waveform: Target waveform, shape (B, T)
        disc_outputs: Discriminator outputs for fake samples
        real_features: Real features from discriminator
        fake_features: Fake features from discriminator
        lambda_adv: Weight for adversarial loss (default: 1.0)
        lambda_feat: Weight for feature matching loss (default: 10.0)
        lambda_spec: Weight for spectral loss (default: 1.0)

    Returns:
        Tuple of:
            - total_loss: Combined loss
            - loss_dict: Dictionary of individual loss components
    """
    # Spectral reconstruction loss
    spec_loss = spectral_reconstruction_loss(reconstructed_waveform, target_waveform)

    # Adversarial loss
    adv_loss = adversarial_loss_generator(disc_outputs)

    # Feature matching loss
    feat_loss = feature_matching_loss(real_features, fake_features)

    # Combined loss
    total_loss = (
        lambda_spec * spec_loss + lambda_adv * adv_loss + lambda_feat * feat_loss
    )

    loss_dict = {
        "loss_generator": total_loss.item(),
        "loss_spec": spec_loss.item(),
        "loss_adv_gen": adv_loss.item(),
        "loss_feat_matching": feat_loss.item(),
    }

    return total_loss, loss_dict


def combined_discriminator_loss(
    real_outputs: List[torch.Tensor],
    fake_outputs: List[torch.Tensor],
    lambda_disc: float = 1.0,
) -> Tuple[torch.Tensor, dict]:
    """
    Combined discriminator loss.

    Args:
        real_outputs: Discriminator outputs for real samples
        fake_outputs: Discriminator outputs for fake samples
        lambda_disc: Weight for discriminator loss (default: 1.0)

    Returns:
        Tuple of:
            - total_loss: Combined loss
            - loss_dict: Dictionary of loss components
    """
    disc_loss = adversarial_loss_discriminator(real_outputs, fake_outputs)
    total_loss = lambda_disc * disc_loss

    loss_dict = {
        "loss_discriminator": total_loss.item(),
        "loss_adv_disc": disc_loss.item(),
    }

    return total_loss, loss_dict


if __name__ == "__main__":
    print("Testing loss functions...")

    # Create dummy tensors
    B, T = 4, 8192  # Batch size, time steps
    y_hat = torch.randn(B, T)
    y = torch.randn(B, T)

    # Test spectral reconstruction loss
    print("Testing spectral_reconstruction_loss...")
    spec_loss = spectral_reconstruction_loss(y_hat, y)
    print(f"✓ Spectral loss: {spec_loss.item():.4f}")

    # Test adversarial loss generator
    print("\nTesting adversarial_loss_generator...")
    disc_outputs = [torch.randn(B, 1) for _ in range(5)]
    adv_loss_gen = adversarial_loss_generator(disc_outputs)
    print(f"✓ Adversarial loss (generator): {adv_loss_gen.item():.4f}")

    # Test adversarial loss discriminator
    print("\nTesting adversarial_loss_discriminator...")
    real_outputs = [torch.randn(B, 1) for _ in range(5)]
    fake_outputs = [torch.randn(B, 1) for _ in range(5)]
    adv_loss_disc = adversarial_loss_discriminator(real_outputs, fake_outputs)
    print(f"✓ Adversarial loss (discriminator): {adv_loss_disc.item():.4f}")

    # Test feature matching loss
    print("\nTesting feature_matching_loss...")
    real_feats = [torch.randn(B, 64, 32, 32) for _ in range(3)]
    fake_feats = [torch.randn(B, 64, 32, 32) for _ in range(3)]
    feat_loss = feature_matching_loss(real_feats, fake_feats)
    print(f"✓ Feature matching loss: {feat_loss.item():.4f}")

    # Test flow matching loss
    print("\nTesting flow_matching_loss...")
    pred_vel = torch.randn(B, 100, 24)
    target_vel = torch.randn(B, 100, 24)
    mask = torch.ones(B, 100)
    flow_loss = flow_matching_loss(pred_vel, target_vel, mask=mask)
    print(f"✓ Flow matching loss: {flow_loss.item():.4f}")

    # Test duration loss
    print("\nTesting duration_loss...")
    pred_dur = torch.randn(10, 50)
    target_dur = torch.randn(10, 50)
    dur_loss = duration_loss(pred_dur, target_dur)
    print(f"✓ Duration loss: {dur_loss.item():.4f}")

    # Test combined losses
    print("\nTesting combined_generator_loss...")
    total_gen_loss, gen_loss_dict = combined_generator_loss(
        y_hat,
        y,
        disc_outputs,
        real_feats,
        fake_feats,
    )
    print(f"✓ Combined generator loss: {total_gen_loss.item():.4f}")
    for key, val in gen_loss_dict.items():
        print(f"  {key}: {val:.4f}")

    print("\nTesting combined_discriminator_loss...")
    total_disc_loss, disc_loss_dict = combined_discriminator_loss(
        real_outputs, fake_outputs
    )
    print(f"✓ Combined discriminator loss: {total_disc_loss.item():.4f}")
    for key, val in disc_loss_dict.items():
        print(f"  {key}: {val:.4f}")

    print("\nAll tests passed!")
