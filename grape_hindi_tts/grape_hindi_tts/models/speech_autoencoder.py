"""
Speech Autoencoder implementation for SupertonicTTS.

Vocos-based architecture with ConvNeXt blocks for encoding/decoding.
- Encodes 228-dim mel spectrograms into 24-dim continuous latents
- Decodes latents back to waveform
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from convnext import ConvNeXtBlock, CausalConvNeXtBlock, CausalConv1d


class LatentEncoder(nn.Module):
    """
    Encodes mel spectrograms into continuous latent representations.

    Architecture:
    - Conv1d (228 → 512) + BatchNorm
    - 10 ConvNeXt blocks (hidden=512, intermediate=2048, kernel=7)
    - Linear (512 → 24) + LayerNorm
    - Output: 24-dim latents with same temporal length as input mel

    Args:
        mel_dim: Dimension of input mel spectrogram (default: 228)
        hidden_dim: Hidden dimension (default: 512)
        latent_dim: Output latent dimension (default: 24)
        num_blocks: Number of ConvNeXt blocks (default: 10)
        intermediate_dim: Intermediate dimension in feed-forward (default: 2048)
        kernel_size: Kernel size for ConvNeXt blocks (default: 7)
        dropout: Dropout rate (default: 0.0)
    """

    def __init__(
        self,
        mel_dim: int = 228,
        hidden_dim: int = 512,
        latent_dim: int = 24,
        num_blocks: int = 10,
        intermediate_dim: int = 2048,
        kernel_size: int = 7,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.mel_dim = mel_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Initial projection from mel to hidden dimension
        # Input: (B, mel_dim, T) → (B, hidden_dim, T)
        self.input_conv = nn.Conv1d(mel_dim, hidden_dim, kernel_size=1, bias=True)
        self.input_norm = nn.BatchNorm1d(hidden_dim)

        # Stack of ConvNeXt blocks
        # (B, hidden_dim, T) → (B, hidden_dim, T)
        self.blocks = nn.ModuleList(
            [
                ConvNeXtBlock(
                    hidden_dim=hidden_dim,
                    intermediate_dim=intermediate_dim,
                    kernel_size=kernel_size,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )

        # Final projection to latent dimension
        # (B, hidden_dim, T) → (B, latent_dim, T)
        self.output_linear = nn.Linear(hidden_dim, latent_dim)
        self.output_norm = nn.LayerNorm(latent_dim)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Encode mel spectrogram to latent representation.

        Args:
            mel: Mel spectrogram tensor of shape (B, T, mel_dim) or (B, mel_dim, T)

        Returns:
            Latent tensor of shape (B, T, latent_dim)
        """
        # Handle both (B, T, mel_dim) and (B, mel_dim, T) formats
        if mel.dim() == 3 and mel.shape[-1] == self.mel_dim:
            # Format: (B, T, mel_dim) → transpose to (B, mel_dim, T)
            mel = mel.transpose(1, 2)

        # Initial projection
        # (B, mel_dim, T) → (B, hidden_dim, T)
        x = self.input_conv(mel)
        x = self.input_norm(x)

        # ConvNeXt blocks
        for block in self.blocks:
            x = block(x)

        # Transpose for linear projection: (B, hidden_dim, T) → (B, T, hidden_dim)
        x = x.transpose(1, 2)

        # Final linear projection to latent dimension
        # (B, T, hidden_dim) → (B, T, latent_dim)
        latents = self.output_linear(x)
        latents = self.output_norm(latents)

        return latents


class LatentDecoder(nn.Module):
    """
    Decodes continuous latents back to waveform.

    Architecture:
    - CausalConv1d (24 → 512) + BatchNorm
    - 10 Dilated CausalConvNeXt blocks with dilation_rates = [1, 2, 4, 1, 2, 4, 1, 1, 1, 1]
    - BatchNorm
    - CausalConv1d (512 → 512, kernel=3) → hidden representation
    - Linear (512 → 512) + PReLU + Linear (512 → hop_length) → frame-level output
    - Flatten/reshape to produce waveform

    Args:
        latent_dim: Input latent dimension (default: 24)
        hidden_dim: Hidden dimension (default: 512)
        hop_length: Hop length for frame-to-sample conversion (default: 256)
        num_blocks: Number of ConvNeXt blocks (default: 10)
        intermediate_dim: Intermediate dimension in feed-forward (default: 2048)
        kernel_size: Kernel size for ConvNeXt blocks (default: 7)
        dilation_rates: List of dilation rates for each block (default: [1,2,4,1,2,4,1,1,1,1])
        dropout: Dropout rate (default: 0.0)
    """

    def __init__(
        self,
        latent_dim: int = 24,
        hidden_dim: int = 512,
        hop_length: int = 256,
        num_blocks: int = 10,
        intermediate_dim: int = 2048,
        kernel_size: int = 7,
        dilation_rates: Optional[list] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.hop_length = hop_length

        if dilation_rates is None:
            dilation_rates = [1, 2, 4, 1, 2, 4, 1, 1, 1, 1]

        assert (
            len(dilation_rates) == num_blocks
        ), f"dilation_rates length {len(dilation_rates)} != num_blocks {num_blocks}"

        # Initial causal convolution from latent to hidden dimension
        # Input: (B, latent_dim, T)
        # Output: (B, hidden_dim, T)
        self.input_conv = CausalConv1d(
            latent_dim, hidden_dim, kernel_size=1, dilation=1, bias=True
        )
        self.input_norm = nn.BatchNorm1d(hidden_dim)

        # Stack of Dilated CausalConvNeXt blocks
        # (B, hidden_dim, T) → (B, hidden_dim, T)
        self.blocks = nn.ModuleList(
            [
                CausalConvNeXtBlock(
                    hidden_dim=hidden_dim,
                    intermediate_dim=intermediate_dim,
                    kernel_size=kernel_size,
                    dilation=dilation_rates[i],
                    dropout=dropout,
                )
                for i in range(num_blocks)
            ]
        )

        # Final batch norm before output projection
        self.pre_output_norm = nn.BatchNorm1d(hidden_dim)

        # Second causal convolution to hidden representation
        # Input: (B, hidden_dim, T)
        # Output: (B, hidden_dim, T)
        self.hidden_conv = CausalConv1d(
            hidden_dim, hidden_dim, kernel_size=3, dilation=1, bias=True
        )

        # Frame-to-sample output projection
        # (B, T, hidden_dim) → (B, T, hop_length) → (B, T*hop_length)
        self.output_linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.output_prelu = nn.PReLU(hidden_dim)
        self.output_linear2 = nn.Linear(hidden_dim, hop_length)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to waveform.

        Args:
            latents: Latent tensor of shape (B, T, latent_dim) or (B, latent_dim, T)

        Returns:
            Waveform tensor of shape (B, T * hop_length)
        """
        # Handle both (B, T, latent_dim) and (B, latent_dim, T) formats
        if latents.dim() == 3 and latents.shape[-1] == self.latent_dim:
            # Format: (B, T, latent_dim) → transpose to (B, latent_dim, T)
            latents = latents.transpose(1, 2)

        # Initial causal convolution
        # (B, latent_dim, T) → (B, hidden_dim, T)
        x = self.input_conv(latents)
        x = self.input_norm(x)

        # Dilated causal ConvNeXt blocks
        for block in self.blocks:
            x = block(x)

        # Pre-output normalization
        x = self.pre_output_norm(x)

        # Second causal convolution to hidden representation
        # (B, hidden_dim, T) → (B, hidden_dim, T)
        x = self.hidden_conv(x)

        # Transpose for linear projection: (B, hidden_dim, T) → (B, T, hidden_dim)
        x = x.transpose(1, 2)

        # Frame-to-sample output projection
        # (B, T, hidden_dim) → (B, T, hidden_dim) → (B, T, hop_length)
        x = self.output_linear1(x)
        x = self.output_prelu(x)
        x = self.output_linear2(x)

        # Flatten frames into waveform samples
        # (B, T, hop_length) → (B, T*hop_length)
        B, T, H = x.shape
        waveform = x.reshape(B, T * H)

        return waveform


class SpeechAutoencoder(nn.Module):
    """
    Complete Speech Autoencoder combining encoder and decoder.

    Encodes mel spectrograms to 24-dim latents and reconstructs waveform.

    Args:
        mel_dim: Dimension of input mel spectrogram (default: 228)
        latent_dim: Output latent dimension (default: 24)
        hidden_dim: Hidden dimension in encoder/decoder (default: 512)
        hop_length: Hop length for frame-to-sample conversion (default: 256)
        num_encoder_blocks: Number of ConvNeXt blocks in encoder (default: 10)
        num_decoder_blocks: Number of ConvNeXt blocks in decoder (default: 10)
        intermediate_dim: Intermediate dimension in feed-forward (default: 2048)
        kernel_size: Kernel size for ConvNeXt blocks (default: 7)
        dropout: Dropout rate (default: 0.0)
    """

    def __init__(
        self,
        mel_dim: int = 228,
        latent_dim: int = 24,
        hidden_dim: int = 512,
        hop_length: int = 256,
        num_encoder_blocks: int = 10,
        num_decoder_blocks: int = 10,
        intermediate_dim: int = 2048,
        kernel_size: int = 7,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.mel_dim = mel_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.hop_length = hop_length

        self.encoder = LatentEncoder(
            mel_dim=mel_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_blocks=num_encoder_blocks,
            intermediate_dim=intermediate_dim,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        self.decoder = LatentDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            hop_length=hop_length,
            num_blocks=num_decoder_blocks,
            intermediate_dim=intermediate_dim,
            kernel_size=kernel_size,
            dropout=dropout,
        )

    def encode(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Encode mel spectrogram to latent representation.

        Args:
            mel: Mel spectrogram of shape (B, T, mel_dim) or (B, mel_dim, T)

        Returns:
            Latents of shape (B, T, latent_dim)
        """
        return self.encoder(mel)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode latents to waveform.

        Args:
            latents: Latent tensor of shape (B, T, latent_dim) or (B, latent_dim, T)

        Returns:
            Waveform of shape (B, T * hop_length)
        """
        return self.decoder(latents)

    def forward(
        self, mel: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: encode mel to latents and decode back to waveform.

        Args:
            mel: Mel spectrogram of shape (B, T, mel_dim) or (B, mel_dim, T)

        Returns:
            Tuple of:
                - reconstructed_waveform: (B, T * hop_length)
                - latents: (B, T, latent_dim)
        """
        latents = self.encode(mel)
        reconstructed_waveform = self.decode(latents)
        return reconstructed_waveform, latents


if __name__ == "__main__":
    print("Testing Speech Autoencoder...")

    # Create autoencoder
    autoencoder = SpeechAutoencoder(
        mel_dim=228,
        latent_dim=24,
        hidden_dim=512,
        hop_length=256,
        num_encoder_blocks=10,
        num_decoder_blocks=10,
    )

    # Test with random mel spectrogram
    # Shape: (B=2, T=100, mel_dim=228)
    mel = torch.randn(2, 100, 228)
    print(f"Input mel shape: {mel.shape}")

    # Forward pass
    waveform, latents = autoencoder(mel)
    print(f"Output waveform shape: {waveform.shape}")
    print(f"Output latents shape: {latents.shape}")

    # Expected shapes
    assert waveform.shape == (2, 100 * 256), f"Unexpected waveform shape: {waveform.shape}"
    assert latents.shape == (2, 100, 24), f"Unexpected latents shape: {latents.shape}"

    # Test encode only
    latents2 = autoencoder.encode(mel)
    assert latents2.shape == latents.shape
    print(f"✓ Encode: {mel.shape} → {latents2.shape}")

    # Test decode only
    waveform2 = autoencoder.decode(latents)
    assert waveform2.shape == waveform.shape
    print(f"✓ Decode: {latents.shape} → {waveform2.shape}")

    # Test with (B, mel_dim, T) format
    mel_alt = mel.transpose(1, 2)
    waveform3, latents3 = autoencoder(mel_alt)
    assert waveform3.shape == waveform.shape
    assert latents3.shape == latents.shape
    print(f"✓ Alternative format: {mel_alt.shape} → latents {latents3.shape}, waveform {waveform3.shape}")

    print("\nAll tests passed!")
