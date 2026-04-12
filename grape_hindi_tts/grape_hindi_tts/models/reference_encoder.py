"""
Reference Encoder for SupertonicTTS Text-to-Latent module.

Extracts speaker characteristics from reference speech latents.

Architecture:
1. Linear: 144 -> 128
2. 6 ConvNeXt blocks (hidden=128, intermediate=512, kernel=5)
3. 2 Cross-Attention layers with learnable query vectors
"""

import torch
import torch.nn as nn
from typing import Tuple

from .attention import MultiHeadCrossAttention
from .convnext import ConvNeXtStack


class ReferenceEncoder(nn.Module):
    """Reference Encoder for extracting speaker characteristics.

    Takes compressed latents and returns reference key/value vectors
    for conditioning the text encoder and vector field estimator.
    """

    def __init__(
        self,
        latent_channels: int = 144,
        convnext_hidden: int = 128,
        convnext_intermediate: int = 512,
        convnext_kernel: int = 5,
        num_convnext: int = 6,
        num_cross_attn: int = 2,
        n_heads: int = 4,
        num_query_vectors: int = 50,
        query_dim: int = 128,
    ):
        super().__init__()

        self.latent_channels = latent_channels
        self.convnext_hidden = convnext_hidden
        self.num_query_vectors = num_query_vectors
        self.query_dim = query_dim

        # Initial linear projection: 144 -> 128
        self.input_proj = nn.Linear(latent_channels, convnext_hidden)

        # ConvNeXt blocks
        self.convnext = ConvNeXtStack(
            hidden=convnext_hidden,
            intermediate=convnext_intermediate,
            kernel_size=convnext_kernel,
            num_blocks=num_convnext,
        )

        # Learnable query vectors for cross-attention
        self.learnable_queries = nn.Parameter(
            torch.randn(1, num_query_vectors, query_dim)
        )

        # Cross-Attention layers
        self.cross_attn_layers = nn.ModuleList()
        for i in range(num_cross_attn):
            # First layer: ConvNeXt output as K,V
            # Second layer: first layer output as K,V
            self.cross_attn_layers.append(
                nn.ModuleDict({
                    'norm_q': nn.LayerNorm(query_dim),
                    'norm_kv': nn.LayerNorm(convnext_hidden if i == 0 else query_dim),
                    'cross_attn': MultiHeadCrossAttention(
                        dim=query_dim,
                        n_heads=n_heads,
                        use_larope=True,
                    ),
                })
            )

    def forward(
        self,
        compressed_latents: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            compressed_latents: (batch, 144, T_compressed) compressed speech latents

        Returns:
            ref_keys: (batch, num_query_vectors, query_dim) reference keys
            ref_values: (batch, num_query_vectors, query_dim) reference values (same for now)
        """
        batch, latent_channels, seq_len = compressed_latents.shape

        # Project: (batch, 144, T) -> (batch, T, 128) -> (batch, 128, T)
        x = compressed_latents.permute(0, 2, 1)  # (batch, T, 144)
        x = self.input_proj(x)  # (batch, T, 128)
        x = x.permute(0, 2, 1)  # (batch, 128, T)

        # ConvNeXt processing
        x = self.convnext(x)  # (batch, 128, T)
        x = x.permute(0, 2, 1)  # (batch, T, 128)

        # Cross-Attention layers
        # Start with learnable queries
        queries = self.learnable_queries.expand(batch, -1, -1)  # (batch, num_queries, query_dim)

        for i, layer_dict in enumerate(self.cross_attn_layers):
            # Prepare K,V
            if i == 0:
                # First layer: ConvNeXt output as K,V
                kv = x  # (batch, T, 128)
            else:
                # Second layer: previous layer output as K,V
                kv = queries  # (batch, num_queries, query_dim)

            # Normalize
            q_norm = layer_dict['norm_q'](queries)
            kv_norm = layer_dict['norm_kv'](kv)

            # Cross-attention
            queries = layer_dict['cross_attn'](q_norm, kv_norm, kv_norm)

        # Reference keys and values are the same in this implementation
        ref_keys = queries
        ref_values = queries

        return ref_keys, ref_values
