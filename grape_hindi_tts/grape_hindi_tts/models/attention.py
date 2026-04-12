"""
Attention mechanisms for SupertonicTTS Text-to-Latent module.
Implements LARoPE (Length-Aware Rotary Position Embedding) and multi-head attention.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply Rotary Position Embedding to query or key tensor.

    Args:
        x: (batch, n_heads, seq_len, head_dim) tensor
        cos: (seq_len, head_dim) precomputed cosine
        sin: (seq_len, head_dim) precomputed sine

    Returns:
        (batch, n_heads, seq_len, head_dim) rotated tensor
    """
    # x1, x2 are (batch, n_heads, seq_len, head_dim//2)
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]

    # Rotate
    cos = cos[None, None, :, :]  # (1, 1, seq_len, head_dim)
    sin = sin[None, None, :, :]  # (1, 1, seq_len, head_dim)

    out1 = x1 * cos[..., : x1.shape[-1]] - x2 * sin[..., : x2.shape[-1]]
    out2 = x1 * sin[..., : x1.shape[-1]] + x2 * cos[..., : x2.shape[-1]]

    return torch.cat([out1, out2], dim=-1)


class RoPEEmbedding(nn.Module):
    """Standard Rotary Position Embedding.

    Applies rotation based on absolute position p:
    R_θ(x, p) = [x1*cos(p*θ_j) - x2*sin(p*θ_j), x1*sin(p*θ_j) + x2*cos(p*θ_j)]
    """

    def __init__(self, head_dim: int, max_seq_len: int = 2048):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

        # θ_j = 10000^(-2j/d)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute cos and sin for positions 0..seq_len-1.

        Returns:
            cos: (seq_len, head_dim)
            sin: (seq_len, head_dim)
        """
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)

        # θ_j * p for all positions and frequencies
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, head_dim//2)

        # Duplicate to full head_dim
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, head_dim)

        return emb.cos(), emb.sin()


class LARoPEEmbedding(nn.Module):
    """Length-Aware Rotary Position Embedding.

    Modifies RoPE to normalize positions by sequence length:
    R'_θ(x, p, L) with rotation angle γ * (p/L) * θ_j where γ=10

    Creates diagonal bias in cross-attention by scaling positions.
    """

    def __init__(self, head_dim: int, max_seq_len: int = 2048, gamma: float = 10.0):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.gamma = gamma

        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(
        self,
        query_len: int,
        key_len: int,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute cos/sin for queries and keys with length-aware scaling.

        Args:
            query_len: length of query sequence
            key_len: length of key sequence
            device: torch device

        Returns:
            q_cos: (query_len, head_dim) cosine for queries
            q_sin: (query_len, head_dim) sine for queries
            k_cos: (key_len, head_dim) cosine for keys
            k_sin: (key_len, head_dim) sine for keys
        """
        # Normalize positions by sequence length
        q_t = torch.arange(query_len, device=device, dtype=self.inv_freq.dtype)
        k_t = torch.arange(key_len, device=device, dtype=self.inv_freq.dtype)

        # Length-aware scaling: γ * (p/L) * θ_j
        q_freqs = torch.outer(
            self.gamma * (q_t / query_len),
            self.inv_freq
        )  # (query_len, head_dim//2)
        k_freqs = torch.outer(
            self.gamma * (k_t / key_len),
            self.inv_freq
        )  # (key_len, head_dim//2)

        # Duplicate to full head_dim
        q_emb = torch.cat([q_freqs, q_freqs], dim=-1)
        k_emb = torch.cat([k_freqs, k_freqs], dim=-1)

        return q_emb.cos(), q_emb.sin(), k_emb.cos(), k_emb.sin()


class MultiHeadSelfAttention(nn.Module):
    """Standard multi-head self-attention with RoPE.

    Q, K, V all come from the same source.
    Uses standard RoPE based on absolute positions.
    """

    def __init__(self, dim: int, n_heads: int = 4, max_seq_len: int = 2048):
        super().__init__()
        assert dim % n_heads == 0, f"dim {dim} must be divisible by n_heads {n_heads}"

        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv_proj = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)

        self.rope = RoPEEmbedding(self.head_dim, max_seq_len)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, seq_len, dim) input tensor
            mask: (batch, seq_len) or (batch, 1, seq_len, seq_len) optional attention mask
            return_attention_weights: if True, also return attention weights

        Returns:
            output: (batch, seq_len, dim)
            attn_weights: (batch, n_heads, seq_len, seq_len) if return_attention_weights
        """
        batch, seq_len, dim = x.shape

        # Project to Q, K, V
        qkv = self.qkv_proj(x)  # (batch, seq_len, 3*dim)
        qkv = qkv.reshape(batch, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, n_heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply RoPE
        cos, sin = self.rope(seq_len, x.device)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Compute attention
        # (batch, n_heads, seq_len, head_dim) @ (batch, n_heads, head_dim, seq_len)
        # -> (batch, n_heads, seq_len, seq_len)
        scores = (q @ k.transpose(-2, -1)) * self.scale

        # Apply mask if provided
        if mask is not None:
            if mask.dim() == 2:
                # (batch, seq_len) -> expand to attention shape
                mask = mask[:, None, None, :]
            scores = scores.masked_fill(~mask, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, 0.0)  # Handle -inf -> 0

        # Apply attention to values
        out = attn_weights @ v  # (batch, n_heads, seq_len, head_dim)
        out = out.permute(0, 2, 1, 3)  # (batch, seq_len, n_heads, head_dim)
        out = out.reshape(batch, seq_len, dim)

        # Output projection
        out = self.out_proj(out)

        if return_attention_weights:
            return out, attn_weights
        return out


class MultiHeadCrossAttention(nn.Module):
    """Multi-head cross-attention with optional LARoPE.

    Q comes from one source, K and V from another.
    Supports Length-Aware RoPE for normalized position scaling.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int = 4,
        max_seq_len: int = 2048,
        use_larope: bool = False,
        gamma: float = 10.0
    ):
        super().__init__()
        assert dim % n_heads == 0, f"dim {dim} must be divisible by n_heads {n_heads}"

        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.use_larope = use_larope

        # Separate projections for Q, K, V
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        if use_larope:
            self.rope = LARoPEEmbedding(self.head_dim, max_seq_len, gamma)
        else:
            self.rope_q = RoPEEmbedding(self.head_dim, max_seq_len)
            self.rope_k = RoPEEmbedding(self.head_dim, max_seq_len)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            query: (batch, query_len, dim)
            key: (batch, key_len, dim)
            value: (batch, key_len, dim)
            mask: (batch, 1, query_len, key_len) or (batch, key_len) optional mask
            return_attention_weights: if True, also return attention weights

        Returns:
            output: (batch, query_len, dim)
            attn_weights: (batch, n_heads, query_len, key_len) if return_attention_weights
        """
        batch, query_len, dim = query.shape
        _, key_len, _ = key.shape

        # Project to Q, K, V
        q = self.q_proj(query).reshape(batch, query_len, self.n_heads, self.head_dim)
        k = self.k_proj(key).reshape(batch, key_len, self.n_heads, self.head_dim)
        v = self.v_proj(value).reshape(batch, key_len, self.n_heads, self.head_dim)

        # Transpose to (batch, n_heads, seq_len, head_dim)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # Apply RoPE
        if self.use_larope:
            q_cos, q_sin, k_cos, k_sin = self.rope(query_len, key_len, query.device)
            q = apply_rope(q, q_cos, q_sin)
            k = apply_rope(k, k_cos, k_sin)
        else:
            q_cos, q_sin = self.rope_q(query_len, query.device)
            k_cos, k_sin = self.rope_k(key_len, query.device)
            q = apply_rope(q, q_cos, q_sin)
            k = apply_rope(k, k_cos, k_sin)

        # Compute attention scores
        scores = (q @ k.transpose(-2, -1)) * self.scale

        # Apply mask if provided
        if mask is not None:
            if mask.dim() == 2:
                # (batch, key_len) -> (batch, 1, 1, key_len)
                mask = mask[:, None, None, :]
            scores = scores.masked_fill(~mask, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, 0.0)

        # Apply attention to values
        out = attn_weights @ v  # (batch, n_heads, query_len, head_dim)
        out = out.permute(0, 2, 1, 3)  # (batch, query_len, n_heads, head_dim)
        out = out.reshape(batch, query_len, dim)

        # Output projection
        out = self.out_proj(out)

        if return_attention_weights:
            return out, attn_weights
        return out
