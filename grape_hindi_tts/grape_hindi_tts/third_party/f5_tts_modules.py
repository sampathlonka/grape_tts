"""
F5-TTS Components — verbatim source from SWivid/F5-TTS (Apache 2.0 License).
https://github.com/SWivid/F5-TTS/blob/main/src/f5_tts/model/modules.py

Taken as-is: ConvNeXtV2Block, GRN, ConvPositionEmbedding, SinusPositionEmbedding.
These replace our hand-written ConvNeXt blocks with the battle-tested F5-TTS versions.

Changes from original:
  - Standalone file (no f5_tts package import needed)
  - Added format adapter helpers for (B,D,T) ↔ (B,T,D) conversion used by our pipeline
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# GRN — Global Response Normalization
# Source: F5-TTS modules.py  (originates from ConvNeXt V2 paper)
# ---------------------------------------------------------------------------

class GRN(nn.Module):
    """Global Response Normalization layer.

    Introduced in ConvNeXt V2: https://arxiv.org/abs/2301.00808
    Calibrates feature responses globally to avoid feature collapse.

    Expects input shape: (B, T, D)
    """

    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta  = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        Gx = torch.norm(x, p=2, dim=1, keepdim=True)              # (B, 1, D)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)          # (B, 1, D)
        return self.gamma * (x * Nx) + self.beta + x


# ---------------------------------------------------------------------------
# ConvNeXtV2Block
# Source: F5-TTS modules.py
# ---------------------------------------------------------------------------

class ConvNeXtV2Block(nn.Module):
    """ConvNeXt V2 block with GRN.

    Improvements over V1:
      - GRN (Global Response Normalization) replaces plain GELU gate
      - LayerNorm in channel-last form for efficiency

    Input/output shape: (B, T, D)  [channel-last, time-second]

    Args:
        dim:              channel dimension
        intermediate_dim: expansion dimension (typically 4×dim)
        dilation:         dilation for the depthwise conv (default 1)
    """

    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        dilation: int = 1,
    ):
        super().__init__()
        padding = (dilation * (7 - 1)) // 2
        self.dwconv = nn.Conv1d(
            dim, dim,
            kernel_size=7,
            padding=padding,
            dilation=dilation,
            groups=dim,        # depthwise
        )
        self.norm    = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act     = nn.GELU()
        self.grn     = GRN(intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D)
        Returns:
            (B, T, D)
        """
        residual = x
        x = x.transpose(1, 2)   # (B, T, D) → (B, D, T)
        x = self.dwconv(x)
        x = x.transpose(1, 2)   # (B, D, T) → (B, T, D)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        return residual + x


# ---------------------------------------------------------------------------
# ConvPositionEmbedding
# Source: F5-TTS modules.py
# ---------------------------------------------------------------------------

class ConvPositionEmbedding(nn.Module):
    """Convolutional position embedding.

    Adds a local position bias via depthwise conv without sinusoidal tables.
    Supports variable-length sequences and masked inputs.

    Input/output shape: (B, T, D)
    """

    def __init__(self, dim: int, kernel_size: int = 31, groups: int = 16):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.conv1d = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=kernel_size // 2),
            nn.Mish(),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:    (B, T, D)
            mask: (B, T) bool mask — True where valid (optional)
        Returns:
            (B, T, D)
        """
        if mask is not None:
            x = x * mask.unsqueeze(-1).float()
        x = x.transpose(1, 2)   # (B, D, T)
        x = self.conv1d(x)
        x = x.transpose(1, 2)   # (B, T, D)
        if mask is not None:
            x = x * mask.unsqueeze(-1).float()
        return x


# ---------------------------------------------------------------------------
# SinusPositionEmbedding
# Source: F5-TTS modules.py
# ---------------------------------------------------------------------------

class SinusPositionEmbedding(nn.Module):
    """Sinusoidal position embedding for timestep / position encoding.

    Used for timestep conditioning in the flow matching diffusion process.
    Returns embeddings of shape (B, dim).
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor, scale: float = 1000.0) -> torch.Tensor:
        """
        Args:
            x:     (B,) float tensor of timesteps in [0, 1]
            scale: frequency multiplier
        Returns:
            (B, dim) sinusoidal embeddings
        """
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)   # (B, half_dim)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)    # (B, dim)
        return emb


# ---------------------------------------------------------------------------
# TimestepEmbedding
# Source: F5-TTS modules.py
# ---------------------------------------------------------------------------

class TimestepEmbedding(nn.Module):
    """Projects sinusoidal timestep embeddings to model dimension.

    sin/cos embedding → Linear → SiLU → Linear → time_dim
    """

    def __init__(self, dim: int, freq_embed_dim: int = 256):
        super().__init__()
        self.time_embed = SinusPositionEmbedding(freq_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(freq_embed_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timestep: (B,) float values in [0, 1]
        Returns:
            (B, dim)
        """
        t_freq = self.time_embed(timestep)
        return self.time_mlp(t_freq)


# ---------------------------------------------------------------------------
# Adapter helpers  (not in F5-TTS, added for SupertonicTTS channel-first convention)
# ---------------------------------------------------------------------------

class ConvNeXtV2BlockCF(nn.Module):
    """ConvNeXtV2Block wrapper for channel-first (B, D, T) inputs.

    SupertonicTTS models use (B, D, T) internally (channel-first).
    This wrapper converts to/from the F5-TTS (B, T, D) convention so we can
    use the original block unchanged.
    """

    def __init__(self, dim: int, intermediate_dim: int, dilation: int = 1):
        super().__init__()
        self.block = ConvNeXtV2Block(dim, intermediate_dim, dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, D, T)
        Returns:
            (B, D, T)
        """
        x = x.transpose(1, 2)         # → (B, T, D)
        x = self.block(x)
        x = x.transpose(1, 2)         # → (B, D, T)
        return x


class ConvNeXtV2Stack(nn.Module):
    """Stack of ConvNeXtV2BlockCF for channel-first (B, D, T) inputs."""

    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        num_blocks: int,
        dilations: Optional[list] = None,
    ):
        super().__init__()
        if dilations is None:
            dilations = [1] * num_blocks
        assert len(dilations) == num_blocks
        self.blocks = nn.ModuleList([
            ConvNeXtV2BlockCF(dim, intermediate_dim, dilations[i])
            for i in range(num_blocks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x
