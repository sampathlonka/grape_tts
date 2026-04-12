"""
Vector Field Estimator for SupertonicTTS Text-to-Latent module.

Predicts velocity field for flow matching using text and speaker conditioning.

Architecture:
- Main block (repeated Nm=4 times):
  * 4 Dilated ConvNeXt blocks (dilations=[1,2,4,8])
  * 2 Standard ConvNeXt blocks
  * TimeCondBlock: projects time embedding to channel dim
  * TextCondBlock: cross-attention with text embeddings
  * RefCondBlock: cross-attention with reference embeddings
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple

from .attention import MultiHeadCrossAttention
from .convnext import ConvNeXtStack


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding."""

    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Generate sinusoidal time embeddings.

        Args:
            timesteps: (batch,) tensor of timesteps

        Returns:
            embeddings: (batch, dim) time embeddings
        """
        batch = timesteps.shape[0]
        device = timesteps.device

        half_dim = self.dim // 2
        freqs = torch.arange(
            half_dim, dtype=torch.float32, device=device
        ) / half_dim
        freqs = self.max_period ** -freqs

        # Outer product of timesteps and frequencies
        t_emb = torch.outer(timesteps, freqs)  # (batch, half_dim)

        # Cat sin and cos
        t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)

        return t_emb


class TimeCondBlock(nn.Module):
    """Time conditioning block.

    Projects time embedding to channel dimension and adds globally.
    """

    def __init__(self, time_dim: int, channel_dim: int):
        super().__init__()
        self.time_dim = time_dim
        self.channel_dim = channel_dim

        self.fc1 = nn.Linear(time_dim, time_dim * 2)
        self.fc2 = nn.Linear(time_dim * 2, channel_dim)
        self.silu = nn.SiLU()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, channel, seq_len) feature tensor
            t_emb: (batch, time_dim) time embedding

        Returns:
            output: (batch, channel, seq_len) time-conditioned features
        """
        # Project time embedding to channel dimension
        t_proj = self.fc1(t_emb)
        t_proj = self.silu(t_proj)
        t_proj = self.fc2(t_proj)  # (batch, channel)

        # Add to features (broadcast over sequence)
        t_proj = t_proj.unsqueeze(-1)  # (batch, channel, 1)
        return x + t_proj


class TextCondBlock(nn.Module):
    """Text conditioning block using cross-attention."""

    def __init__(
        self,
        channel_dim: int,
        text_dim: int,
        n_heads: int = 4,
    ):
        super().__init__()
        self.channel_dim = channel_dim
        self.text_dim = text_dim

        self.norm_x = nn.LayerNorm(channel_dim)
        self.norm_text = nn.LayerNorm(text_dim)

        self.cross_attn = MultiHeadCrossAttention(
            dim=channel_dim,
            n_heads=n_heads,
            use_larope=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, channel, seq_len) feature tensor
            text_embeddings: (batch, text_len, text_dim) text embeddings

        Returns:
            output: (batch, channel, seq_len) text-conditioned features
        """
        batch, channel, seq_len = x.shape

        # Move x to (batch, seq_len, channel)
        x_t = x.permute(0, 2, 1)

        # Normalize
        x_norm = self.norm_x(x_t)
        text_norm = self.norm_text(text_embeddings)

        # Cross-attention
        attn_out = self.cross_attn(x_norm, text_norm, text_norm)

        # Move back to (batch, channel, seq_len)
        attn_out = attn_out.permute(0, 2, 1)

        return x + attn_out


class RefCondBlock(nn.Module):
    """Reference conditioning block using cross-attention."""

    def __init__(
        self,
        channel_dim: int,
        ref_dim: int,
        n_heads: int = 4,
    ):
        super().__init__()
        self.channel_dim = channel_dim
        self.ref_dim = ref_dim

        self.norm_x = nn.LayerNorm(channel_dim)
        self.norm_ref = nn.LayerNorm(ref_dim)

        self.cross_attn = MultiHeadCrossAttention(
            dim=channel_dim,
            n_heads=n_heads,
            use_larope=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        ref_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, channel, seq_len) feature tensor
            ref_embeddings: (batch, num_refs, ref_dim) reference embeddings

        Returns:
            output: (batch, channel, seq_len) reference-conditioned features
        """
        batch, channel, seq_len = x.shape

        # Move x to (batch, seq_len, channel)
        x_t = x.permute(0, 2, 1)

        # Normalize
        x_norm = self.norm_x(x_t)
        ref_norm = self.norm_ref(ref_embeddings)

        # Cross-attention
        attn_out = self.cross_attn(x_norm, ref_norm, ref_embeddings)

        # Move back to (batch, channel, seq_len)
        attn_out = attn_out.permute(0, 2, 1)

        return x + attn_out


class VFEstimatorMainBlock(nn.Module):
    """Main processing block for VF Estimator.

    Combines ConvNeXt processing with time, text, and reference conditioning.
    """

    def __init__(
        self,
        channel_dim: int,
        intermediate_dim: int,
        kernel_size: int = 5,
        time_dim: int = 64,
        text_dim: int = 512,
        ref_dim: int = 128,
        n_heads: int = 4,
    ):
        super().__init__()

        # Dilated ConvNeXt blocks
        self.dilated_convnext = ConvNeXtStack(
            hidden=channel_dim,
            intermediate=intermediate_dim,
            kernel_size=kernel_size,
            num_blocks=4,
            dilations=[1, 2, 4, 8],
        )

        # Standard ConvNeXt blocks
        self.convnext = ConvNeXtStack(
            hidden=channel_dim,
            intermediate=intermediate_dim,
            kernel_size=kernel_size,
            num_blocks=2,
        )

        # Conditioning blocks
        self.time_cond = TimeCondBlock(time_dim, channel_dim)
        self.text_cond = TextCondBlock(channel_dim, text_dim, n_heads)
        self.ref_cond = RefCondBlock(channel_dim, ref_dim, n_heads)

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        text_embeddings: torch.Tensor,
        ref_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, channel, seq_len) feature tensor
            t_emb: (batch, time_dim) time embedding
            text_embeddings: (batch, text_len, text_dim) text embeddings
            ref_embeddings: (batch, num_refs, ref_dim) reference embeddings

        Returns:
            output: (batch, channel, seq_len) processed features
        """
        # Dilated ConvNeXt
        x = self.dilated_convnext(x)

        # Standard ConvNeXt
        x = self.convnext(x)

        # Time conditioning
        x = self.time_cond(x, t_emb)

        # Text conditioning
        x = self.text_cond(x, text_embeddings)

        # Reference conditioning
        x = self.ref_cond(x, ref_embeddings)

        return x


class VFEstimator(nn.Module):
    """Vector Field Estimator for flow matching.

    Predicts velocity field for denoising latents.
    """

    def __init__(
        self,
        latent_channels: int = 144,
        channel_dim: int = 256,
        intermediate_dim: int = 1024,
        kernel_size: int = 5,
        time_dim: int = 64,
        text_dim: int = 512,
        ref_dim: int = 128,
        n_heads: int = 4,
        num_main_blocks: int = 4,
    ):
        super().__init__()

        self.latent_channels = latent_channels
        self.channel_dim = channel_dim
        self.time_dim = time_dim

        # Input projection: 144 -> 256
        self.input_proj = nn.Linear(latent_channels, channel_dim)

        # Time embedding
        self.time_embedding = TimeEmbedding(time_dim)

        # Main blocks
        self.main_blocks = nn.ModuleList(
            [
                VFEstimatorMainBlock(
                    channel_dim=channel_dim,
                    intermediate_dim=intermediate_dim,
                    kernel_size=kernel_size,
                    time_dim=time_dim,
                    text_dim=text_dim,
                    ref_dim=ref_dim,
                    n_heads=n_heads,
                )
                for _ in range(num_main_blocks)
            ]
        )

        # Final ConvNeXt blocks
        self.final_convnext = ConvNeXtStack(
            hidden=channel_dim,
            intermediate=intermediate_dim,
            kernel_size=kernel_size,
            num_blocks=4,
        )

        # Output projection: 256 -> 144
        self.output_proj = nn.Linear(channel_dim, latent_channels)

    def forward(
        self,
        noisy_latents: torch.Tensor,
        text_embeddings: torch.Tensor,
        ref_keys: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            noisy_latents: (batch, 144, seq_len) noisy compressed latents
            text_embeddings: (batch, text_len, 512) text embeddings
            ref_keys: (batch, num_refs, ref_dim) reference key vectors
            timestep: (batch,) timestep values in [0, 1]

        Returns:
            velocity: (batch, 144, seq_len) predicted velocity field
        """
        batch, latent_channels, seq_len = noisy_latents.shape

        # Project input
        x = noisy_latents.permute(0, 2, 1)  # (batch, seq_len, 144)
        x = self.input_proj(x)  # (batch, seq_len, 256)
        x = x.permute(0, 2, 1)  # (batch, 256, seq_len)

        # Time embedding
        t_emb = self.time_embedding(timestep)  # (batch, time_dim)

        # Main blocks
        for main_block in self.main_blocks:
            x = main_block(x, t_emb, text_embeddings, ref_keys)

        # Final ConvNeXt
        x = self.final_convnext(x)

        # Output projection
        x = x.permute(0, 2, 1)  # (batch, seq_len, 256)
        x = self.output_proj(x)  # (batch, seq_len, 144)
        x = x.permute(0, 2, 1)  # (batch, 144, seq_len)

        return x
