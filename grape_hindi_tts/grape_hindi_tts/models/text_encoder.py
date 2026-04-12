"""
Text Encoder for SupertonicTTS Text-to-Latent module.

Architecture:
1. Character embedding: vocab_size -> 128 dim
2. 6 ConvNeXt blocks (hidden=128, intermediate=512, kernel=5)
3. 4 Self-Attention blocks with RoPE
4. 2 Cross-Attention layers with reference features (LARoPE)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .attention import MultiHeadSelfAttention, MultiHeadCrossAttention
from .convnext import ConvNeXtStack


class TextEncoder(nn.Module):
    """Text Encoder for speaker-adaptive text representations.

    Takes text tokens and reference speech features, outputs speaker-adapted
    text embeddings for use in the VF Estimator.
    """

    def __init__(
        self,
        vocab_size: int = 256,
        char_embed_dim: int = 128,
        convnext_hidden: int = 128,
        convnext_intermediate: int = 512,
        convnext_kernel: int = 5,
        num_convnext: int = 6,
        transformer_hidden: int = 512,
        num_self_attn: int = 4,
        num_cross_attn: int = 2,
        n_heads: int = 4,
        num_reference_keys: int = 50,
        reference_dim: int = 128,
    ):
        super().__init__()

        self.char_embed_dim = char_embed_dim
        self.transformer_hidden = transformer_hidden
        self.num_reference_keys = num_reference_keys
        self.reference_dim = reference_dim

        # Character embedding
        self.char_embedding = nn.Embedding(vocab_size, char_embed_dim)

        # ConvNeXt blocks (input: char_embed_dim, output: char_embed_dim)
        self.convnext = ConvNeXtStack(
            hidden=convnext_hidden,
            intermediate=convnext_intermediate,
            kernel_size=convnext_kernel,
            num_blocks=num_convnext,
        )

        # Project from char_embed_dim to transformer_hidden
        self.to_transformer = nn.Linear(char_embed_dim, transformer_hidden)

        # Self-Attention blocks (Transformer encoder style)
        self.self_attn_layers = nn.ModuleList(
            [
                nn.ModuleDict({
                    'norm': nn.LayerNorm(transformer_hidden),
                    'attn': MultiHeadSelfAttention(
                        dim=transformer_hidden,
                        n_heads=n_heads,
                    ),
                    'ffn': nn.Sequential(
                        nn.Linear(transformer_hidden, transformer_hidden * 4),
                        nn.GELU(),
                        nn.Linear(transformer_hidden * 4, transformer_hidden),
                    ),
                    'norm_ffn': nn.LayerNorm(transformer_hidden),
                })
                for _ in range(num_self_attn)
            ]
        )

        # Learnable reference key vectors (num_reference_keys, reference_dim)
        self.learnable_ref_keys = nn.Parameter(
            torch.randn(1, num_reference_keys, reference_dim)
        )

        # Cross-Attention layers
        self.cross_attn_layers = nn.ModuleList()
        for i in range(num_cross_attn):
            # First layer: uses learnable ref keys as K,V
            # Subsequent layers: use previous layer output as K,V
            self.cross_attn_layers.append(
                nn.ModuleDict({
                    'norm_q': nn.LayerNorm(transformer_hidden),
                    'norm_kv': nn.LayerNorm(reference_dim if i == 0 else transformer_hidden),
                    'cross_attn': MultiHeadCrossAttention(
                        dim=transformer_hidden,
                        n_heads=n_heads,
                        use_larope=True,
                    ),
                    'ffn': nn.Sequential(
                        nn.Linear(transformer_hidden, transformer_hidden * 4),
                        nn.GELU(),
                        nn.Linear(transformer_hidden * 4, transformer_hidden),
                    ),
                    'norm_ffn': nn.LayerNorm(transformer_hidden),
                })
            )

    def forward(
        self,
        text_tokens: torch.Tensor,
        reference_features: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            text_tokens: (batch, text_len) LongTensor of character indices
            reference_features: (batch, num_ref_keys, ref_dim) speaker reference features
            text_mask: (batch, text_len) BoolTensor, True for valid tokens

        Returns:
            text_embeddings: (batch, text_len, transformer_hidden)
        """
        batch, text_len = text_tokens.shape

        # Character embedding
        x = self.char_embedding(text_tokens)  # (batch, text_len, char_embed_dim)

        # ConvNeXt processing
        x = x.permute(0, 2, 1)  # (batch, char_embed_dim, text_len)
        x = self.convnext(x)  # (batch, char_embed_dim, text_len)
        x = x.permute(0, 2, 1)  # (batch, text_len, char_embed_dim)

        # Project to transformer hidden dimension
        x = self.to_transformer(x)  # (batch, text_len, transformer_hidden)

        # Self-Attention blocks
        for layer_dict in self.self_attn_layers:
            # Pre-norm
            x_norm = layer_dict['norm'](x)
            x_attn = layer_dict['attn'](x_norm, mask=text_mask)
            x = x + x_attn

            # FFN
            x_norm = layer_dict['norm_ffn'](x)
            x_ffn = layer_dict['ffn'](x_norm)
            x = x + x_ffn

        # Cross-Attention layers
        for i, layer_dict in enumerate(self.cross_attn_layers):
            # Prepare K, V from reference features
            if i == 0:
                # First layer: use learnable reference keys as K,V
                # Broadcast learnable keys to batch size
                kv = self.learnable_ref_keys.expand(batch, -1, -1)  # (batch, num_ref_keys, ref_dim)
            else:
                # Subsequent layers: use previous layer output as K,V
                kv = x  # (batch, text_len, transformer_hidden)

            # Pre-norm
            q_norm = layer_dict['norm_q'](x)
            kv_norm = layer_dict['norm_kv'](kv)

            # Cross-attention
            x_cross = layer_dict['cross_attn'](q_norm, kv_norm, kv_norm)
            x = x + x_cross

            # FFN
            x_norm = layer_dict['norm_ffn'](x)
            x_ffn = layer_dict['ffn'](x_norm)
            x = x + x_ffn

        return x
