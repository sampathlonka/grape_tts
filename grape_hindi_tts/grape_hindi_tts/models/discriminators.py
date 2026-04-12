"""
GAN Discriminators for Speech Autoencoder training.

Includes MultiPeriodDiscriminator (MPD) and MultiResolutionDiscriminator (MRD)
from HiFi-GAN paper, adapted for latent-space discriminators.
"""

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MultiPeriodDiscriminator(nn.Module):
    """
    Multi-Period Discriminator from HiFi-GAN.

    Applies waveform discriminators at multiple periods to capture
    different temporal patterns. Each sub-discriminator:
    - Reshapes input to 2D with period p
    - Applies 6 Conv2D layers with increasing channels [16, 64, 256, 512, 512, 1]
    - Uses stride=(1,2) for temporal downsampling

    Args:
        periods: List of periods to use (default: [2, 3, 5, 7, 11])
        kernel_size: 2D kernel size for convolutions (default: (5, 5))
        stride: 2D stride for convolutions (default: (1, 2))
    """

    def __init__(
        self,
        periods: List[int] = None,
        kernel_size: Tuple[int, int] = (5, 5),
        stride: Tuple[int, int] = (1, 2),
    ):
        super().__init__()
        if periods is None:
            periods = [2, 3, 5, 7, 11]

        self.periods = periods
        self.kernel_size = kernel_size
        self.stride = stride

        # Channel progression
        channels = [1, 16, 64, 256, 512, 512, 1]

        self.sub_discriminators = nn.ModuleList()

        for period in periods:
            # Build discriminator for this period
            # Input: (B, 1, period, -1) after reshaping
            layers = []
            for i in range(len(channels) - 1):
                in_ch = channels[i]
                out_ch = channels[i + 1]

                if i < len(channels) - 2:
                    # Conv2D layer with stride and padding
                    layers.append(
                        nn.Conv2d(
                            in_ch,
                            out_ch,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=(kernel_size[0] // 2, kernel_size[1] // 2),
                        )
                    )
                    layers.append(nn.LeakyReLU(0.1))
                else:
                    # Last layer: no activation
                    layers.append(
                        nn.Conv2d(
                            in_ch,
                            out_ch,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=(kernel_size[0] // 2, kernel_size[1] // 2),
                        )
                    )

            self.sub_discriminators.append(nn.Sequential(*layers))

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass through all period discriminators.

        Args:
            x: Input waveform of shape (B, 1, T) or (B, T)

        Returns:
            Tuple of:
                - scores: List of discriminator outputs (one per period)
                - features: List of intermediate features before final conv
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, T) → (B, 1, T)

        scores = []
        features = []

        for i, (period, disc) in enumerate(zip(self.periods, self.sub_discriminators)):
            # Reshape to (B, 1, period, -1)
            # Pad if necessary to make length divisible by period
            if x.shape[-1] % period != 0:
                pad_amount = period - (x.shape[-1] % period)
                x_padded = F.pad(x, (0, pad_amount))
            else:
                x_padded = x

            B, C, T = x_padded.shape
            x_reshaped = x_padded.view(B, C, period, T // period)

            # Forward through discriminator
            feat = x_reshaped
            for j, layer in enumerate(disc):
                if j < len(list(disc)) - 1:
                    # Store features before final conv
                    features.append(feat)
                feat = layer(feat)

            scores.append(feat)

        return scores, features


class MultiResolutionDiscriminator(nn.Module):
    """
    Multi-Resolution Discriminator from HiFi-GAN.

    Applies discriminators to multi-resolution spectrograms.
    Each sub-discriminator:
    - Computes STFT with different FFT sizes
    - Converts to log-magnitude spectrogram
    - Applies 6 Conv2D layers

    Args:
        fft_sizes: List of FFT sizes (default: [512, 1024, 2048])
        hop_sizes: List of hop sizes (default: [50, 100, 200])
        win_sizes: List of window sizes (default: [240, 480, 960])
        mel_bands: List of mel bands for each FFT (default: [64, 128, 128])
    """

    def __init__(
        self,
        fft_sizes: List[int] = None,
        hop_sizes: List[int] = None,
        win_sizes: List[int] = None,
        mel_bands: List[int] = None,
    ):
        super().__init__()
        if fft_sizes is None:
            fft_sizes = [512, 1024, 2048]
        if hop_sizes is None:
            hop_sizes = [50, 100, 200]
        if win_sizes is None:
            win_sizes = [240, 480, 960]
        if mel_bands is None:
            mel_bands = [64, 128, 128]

        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_sizes = win_sizes
        self.mel_bands = mel_bands

        # Create sub-discriminators
        # Each operates on spectrograms with different resolutions
        self.sub_discriminators = nn.ModuleList()
        channels = [1, 16, 16, 16, 16, 16, 1]

        for i in range(len(fft_sizes)):
            layers = []
            for j in range(len(channels) - 1):
                in_ch = channels[j]
                out_ch = channels[j + 1]

                if j < len(channels) - 2:
                    # Conv2D with kernel=5x5
                    layers.append(
                        nn.Conv2d(
                            in_ch,
                            out_ch,
                            kernel_size=(5, 5),
                            stride=(2, 2),
                            padding=(2, 2),
                        )
                    )
                    layers.append(nn.LeakyReLU(0.1))
                else:
                    # Last layer
                    layers.append(
                        nn.Conv2d(
                            in_ch,
                            out_ch,
                            kernel_size=(5, 5),
                            stride=(2, 2),
                            padding=(2, 2),
                        )
                    )

            self.sub_discriminators.append(nn.Sequential(*layers))

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass through all resolution discriminators.

        Args:
            x: Input waveform of shape (B, T) or (B, 1, T)

        Returns:
            Tuple of:
                - scores: List of discriminator outputs (one per FFT size)
                - features: List of intermediate features
        """
        if x.dim() == 3:
            x = x.squeeze(1)  # (B, 1, T) → (B, T)

        scores = []
        features = []

        for i, (fft_size, hop_size, win_size, disc) in enumerate(
            zip(self.fft_sizes, self.hop_sizes, self.win_sizes, self.sub_discriminators)
        ):
            # Compute STFT
            # (B, T) → (B, fft_size//2 + 1, n_frames)
            window = torch.hann_window(win_size, device=x.device)
            spec = torch.stft(
                x,
                n_fft=fft_size,
                hop_length=hop_size,
                win_length=win_size,
                window=window,
                return_complex=False,
            )

            # Convert to log magnitude
            # spec shape: (B, fft_size//2 + 1, n_frames, 2)
            magnitude = torch.sqrt(spec[:, :, :, 0] ** 2 + spec[:, :, :, 1] ** 2 + 1e-9)
            log_magnitude = torch.log(magnitude + 1e-9)

            # Add channel dimension: (B, 1, freq, time)
            x_spec = log_magnitude.unsqueeze(1)

            # Forward through discriminator
            feat = x_spec
            for j, layer in enumerate(disc):
                if j < len(list(disc)) - 1:
                    features.append(feat)
                feat = layer(feat)

            scores.append(feat)

        return scores, features


class CompositeDiscriminator(nn.Module):
    """
    Composite discriminator combining MPD and MRD.

    Returns features and scores from both multi-period and multi-resolution
    discriminators for joint adversarial training.

    Args:
        use_mpd: Whether to include MultiPeriodDiscriminator (default: True)
        use_mrd: Whether to include MultiResolutionDiscriminator (default: True)
        periods: List of periods for MPD (default: [2, 3, 5, 7, 11])
        fft_sizes: List of FFT sizes for MRD (default: [512, 1024, 2048])
    """

    def __init__(
        self,
        use_mpd: bool = True,
        use_mrd: bool = True,
        periods: List[int] = None,
        fft_sizes: List[int] = None,
    ):
        super().__init__()
        self.use_mpd = use_mpd
        self.use_mrd = use_mrd

        if periods is None:
            periods = [2, 3, 5, 7, 11]
        if fft_sizes is None:
            fft_sizes = [512, 1024, 2048]

        if use_mpd:
            self.mpd = MultiPeriodDiscriminator(periods=periods)
        else:
            self.mpd = None

        if use_mrd:
            self.mrd = MultiResolutionDiscriminator(fft_sizes=fft_sizes)
        else:
            self.mrd = None

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], List[torch.Tensor]]:
        """
        Forward pass through composite discriminator.

        Args:
            x: Input waveform of shape (B, T) or (B, 1, T)

        Returns:
            Tuple of:
                - sub_discriminators: List of (features, score) tuples from all sub-discriminators
                - all_features: Flattened list of all intermediate features
        """
        sub_discriminators = []
        all_features = []

        if self.use_mpd:
            mpd_scores, mpd_features = self.mpd(x)
            for score, feat in zip(mpd_scores, mpd_features):
                sub_discriminators.append((feat, score))
                all_features.extend(feat if isinstance(feat, list) else [feat])

        if self.use_mrd:
            mrd_scores, mrd_features = self.mrd(x)
            for score, feat in zip(mrd_scores, mrd_features):
                sub_discriminators.append((feat, score))
                all_features.extend(feat if isinstance(feat, list) else [feat])

        return sub_discriminators, all_features


if __name__ == "__main__":
    print("Testing MultiPeriodDiscriminator...")
    mpd = MultiPeriodDiscriminator()
    x = torch.randn(2, 8192)  # (B, T)
    scores, features = mpd(x)
    print(f"MPD scores: {len(scores)} outputs")
    for i, score in enumerate(scores):
        print(f"  Period {mpd.periods[i]}: {score.shape}")

    print("\nTesting MultiResolutionDiscriminator...")
    mrd = MultiResolutionDiscriminator()
    scores, features = mrd(x)
    print(f"MRD scores: {len(scores)} outputs")
    for i, score in enumerate(scores):
        print(f"  FFT {mrd.fft_sizes[i]}: {score.shape}")

    print("\nTesting CompositeDiscriminator...")
    disc = CompositeDiscriminator()
    sub_discs, all_feats = disc(x)
    print(f"Composite: {len(sub_discs)} sub-discriminators")
    print(f"Total features: {len(all_feats)}")

    print("\nAll tests passed!")
