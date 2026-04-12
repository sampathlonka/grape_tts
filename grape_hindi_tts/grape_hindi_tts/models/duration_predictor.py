"""
Duration Predictor module for SupertonicTTS.

This lightweight module (~0.5M parameters) predicts total utterance duration
from speech reference and text input. It combines:
- DPReferenceEncoder: encodes compressed latent features from reference speech
- DPTextEncoder: encodes text tokens with utterance-level representation
- DurationEstimator: predicts duration from concatenated embeddings

Architecture details are based on the SupertonicTTS paper.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .convnext import ConvNeXtBlock
from .attention import MultiHeadSelfAttention, MultiHeadCrossAttention


class DPReferenceEncoder(nn.Module):
    """
    Duration Predictor Reference Encoder.

    A lightweight reference encoder that processes compressed latent features
    from reference speech to extract a 64-dimensional utterance-level embedding.

    Architecture:
    - Input: compressed latent features (variable length)
    - Linear projection: 144 → 64 dimensions
    - 4 ConvNeXt blocks (hidden=64, intermediate=256, kernel=5)
    - 2 Cross-Attention layers with learnable query vectors
    - Output: 64-dim reference embedding

    Args:
        in_channels: Input feature dimension (typically 144 from compressed latents)
        hidden_dim: Hidden dimension for ConvNeXt blocks (default: 64)
        intermediate_dim: Intermediate dimension for ConvNeXt MLPs (default: 256)
        kernel_size: Convolution kernel size (default: 5)
        num_convnext_blocks: Number of ConvNeXt blocks (default: 4)
        num_cross_attn_layers: Number of cross-attention layers (default: 2)
        num_query_vectors: Number of learnable query vectors per cross-attn layer (default: 8)
        query_dim: Dimension of query vectors (default: 16)
    """

    def __init__(
        self,
        in_channels: int = 144,
        hidden_dim: int = 64,
        intermediate_dim: int = 256,
        kernel_size: int = 5,
        num_convnext_blocks: int = 4,
        num_cross_attn_layers: int = 2,
        num_query_vectors: int = 8,
        query_dim: int = 16,
    ) -> None:
        super().__init__()

        # Initial projection to hidden dimension
        self.input_projection = nn.Linear(in_channels, hidden_dim)

        # ConvNeXt blocks for feature processing
        self.convnext_blocks = nn.ModuleList([
            ConvNeXtBlock(
                channels=hidden_dim,
                intermediate_channels=intermediate_dim,
                kernel_size=kernel_size,
                dropout=0.0
            )
            for _ in range(num_convnext_blocks)
        ])

        # Cross-attention layers with learnable query vectors
        self.cross_attn_layers = nn.ModuleList()
        self.query_vectors = nn.ParameterList()

        for _ in range(num_cross_attn_layers):
            # Learnable query vectors: [num_query_vectors, query_dim]
            query = nn.Parameter(torch.randn(num_query_vectors, query_dim))
            nn.init.xavier_uniform_(query)
            self.query_vectors.append(query)

            # Cross-attention layer
            # Use query_dim for Q, hidden_dim for K,V
            self.cross_attn_layers.append(
                MultiHeadCrossAttention(
                    query_dim=query_dim,
                    kv_dim=hidden_dim,
                    num_heads=min(4, query_dim),  # Heads limited by query_dim
                    dropout=0.0
                )
            )

        self.num_cross_attn_layers = num_cross_attn_layers
        self.output_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len, in_channels)
               containing compressed latent features from reference speech

        Returns:
            Tensor of shape (batch_size, output_dim) containing the
            64-dimensional reference embedding
        """
        # Project to hidden dimension
        # (batch, seq_len, in_channels) → (batch, seq_len, hidden_dim)
        x = self.input_projection(x)

        # Apply ConvNeXt blocks
        # Need to transpose for convolution: (batch, hidden_dim, seq_len)
        x = x.transpose(1, 2)
        for block in self.convnext_blocks:
            x = block(x)
        x = x.transpose(1, 2)  # Back to (batch, seq_len, hidden_dim)

        # Apply cross-attention layers with learnable queries
        for i, (attn_layer, query_vectors) in enumerate(
            zip(self.cross_attn_layers, self.query_vectors)
        ):
            # query_vectors: (num_query_vectors, query_dim)
            # Expand to batch: (batch, num_query_vectors, query_dim)
            batch_size = x.size(0)
            q = query_vectors.unsqueeze(0).expand(batch_size, -1, -1)

            # Cross-attention: query as Q, x (latent features) as K,V
            # Output: (batch, num_query_vectors, query_dim)
            attn_output = attn_layer(q, x, x)

            # Stack along channel dimension if not the first layer
            if i == 0:
                x = attn_output  # Start with first cross-attention output
            else:
                # Concatenate with previous outputs
                x = torch.cat([x, attn_output], dim=-1)

        # Global average pooling to get utterance-level embedding
        # (batch, num_queries, features) → (batch, features)
        x = x.mean(dim=1)

        return x


class DPTextEncoder(nn.Module):
    """
    Duration Predictor Text Encoder.

    Encodes text tokens into a 64-dimensional utterance-level embedding.
    Uses ConvNeXt blocks for sequence processing and self-attention for
    context modeling.

    Architecture:
    - Character embedding: vocab_size → 64
    - Learnable utterance token prepended
    - 6 ConvNeXt blocks (hidden=64, intermediate=256, kernel=5)
    - 2 Self-Attention blocks (256 filter channels, 2 heads, with RoPE)
    - Linear layer (64 → 64)
    - Extract first token (utterance token) → 64-dim text embedding

    Args:
        vocab_size: Size of character vocabulary
        embedding_dim: Character embedding dimension (default: 64)
        hidden_dim: Hidden dimension for ConvNeXt blocks (default: 64)
        intermediate_dim: Intermediate dimension for ConvNeXt MLPs (default: 256)
        kernel_size: Convolution kernel size (default: 5)
        num_convnext_blocks: Number of ConvNeXt blocks (default: 6)
        num_self_attn_blocks: Number of self-attention blocks (default: 2)
        num_attn_heads: Number of attention heads (default: 2)
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        hidden_dim: int = 64,
        intermediate_dim: int = 256,
        kernel_size: int = 5,
        num_convnext_blocks: int = 6,
        num_self_attn_blocks: int = 2,
        num_attn_heads: int = 2,
    ) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Character embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Learnable utterance token
        self.utterance_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        nn.init.xavier_uniform_(self.utterance_token)

        # ConvNeXt blocks
        self.convnext_blocks = nn.ModuleList([
            ConvNeXtBlock(
                channels=hidden_dim,
                intermediate_channels=intermediate_dim,
                kernel_size=kernel_size,
                dropout=0.0
            )
            for _ in range(num_convnext_blocks)
        ])

        # Self-attention blocks
        self.self_attn_blocks = nn.ModuleList([
            MultiHeadSelfAttention(
                hidden_channels=hidden_dim,
                filter_channels=256,
                num_heads=num_attn_heads,
                dropout=0.0,
                use_rope=True
            )
            for _ in range(num_self_attn_blocks)
        ])

        # Output linear layer
        self.output_layer = nn.Linear(hidden_dim, hidden_dim)

        self.output_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, seq_len) containing
               character token indices

        Returns:
            Tensor of shape (batch_size, output_dim) containing the
            64-dimensional text embedding extracted from the utterance token
        """
        batch_size = x.size(0)

        # Embed characters
        # (batch, seq_len) → (batch, seq_len, embedding_dim)
        x = self.embedding(x)

        # Prepend learnable utterance token
        # (batch, 1, embedding_dim)
        utterance_token = self.utterance_token.expand(batch_size, -1, -1)
        # (batch, seq_len+1, embedding_dim)
        x = torch.cat([utterance_token, x], dim=1)

        # Apply ConvNeXt blocks
        # Transpose for convolution: (batch, embedding_dim, seq_len+1)
        x = x.transpose(1, 2)
        for block in self.convnext_blocks:
            x = block(x)
        x = x.transpose(1, 2)  # Back to (batch, seq_len+1, hidden_dim)

        # Apply self-attention blocks
        for attn_block in self.self_attn_blocks:
            x = attn_block(x)

        # Apply output linear layer
        x = self.output_layer(x)

        # Extract utterance token (first position)
        # (batch, 1, embedding_dim) → (batch, embedding_dim)
        utterance_embedding = x[:, 0, :]

        return utterance_embedding


class DurationEstimator(nn.Module):
    """
    Duration Estimator.

    Predicts scalar duration value from concatenated reference and text embeddings.

    Architecture:
    - Input: ref_embedding (64) + text_embedding (64) = 128 dimensions
    - Linear: 128 → 164
    - PReLU activation
    - Linear: 164 → 1
    - Output: scalar duration value

    Args:
        input_dim: Concatenated embedding dimension (default: 128 from 64+64)
        hidden_dim: Hidden layer dimension (default: 164)
    """

    def __init__(self, input_dim: int = 128, hidden_dim: int = 164) -> None:
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.PReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_dim) containing
               concatenated reference and text embeddings

        Returns:
            Tensor of shape (batch_size, 1) containing predicted duration values
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


class DurationPredictor(nn.Module):
    """
    Complete Duration Predictor module.

    Combines reference encoder, text encoder, and duration estimator to predict
    total utterance duration from compressed latent features and text tokens.

    This is a lightweight module (~0.5M parameters) designed for efficient
    duration estimation in the SupertonicTTS pipeline.

    Args:
        vocab_size: Size of character vocabulary
        reference_feature_dim: Dimension of compressed latent features (default: 144)
        embedding_dim: Character embedding dimension (default: 64)
        hidden_dim: Hidden dimension for encoders (default: 64)
        ref_encoder_intermediate: Intermediate dimension for reference encoder (default: 256)
        text_encoder_intermediate: Intermediate dimension for text encoder (default: 256)
        kernel_size: Convolution kernel size (default: 5)
        num_ref_convnext_blocks: Number of ConvNeXt blocks in reference encoder (default: 4)
        num_text_convnext_blocks: Number of ConvNeXt blocks in text encoder (default: 6)
        num_cross_attn_layers: Number of cross-attention layers in reference encoder (default: 2)
        num_self_attn_blocks: Number of self-attention blocks in text encoder (default: 2)
        num_attn_heads: Number of attention heads (default: 2)
    """

    def __init__(
        self,
        vocab_size: int,
        reference_feature_dim: int = 144,
        embedding_dim: int = 64,
        hidden_dim: int = 64,
        ref_encoder_intermediate: int = 256,
        text_encoder_intermediate: int = 256,
        kernel_size: int = 5,
        num_ref_convnext_blocks: int = 4,
        num_text_convnext_blocks: int = 6,
        num_cross_attn_layers: int = 2,
        num_self_attn_blocks: int = 2,
        num_attn_heads: int = 2,
    ) -> None:
        super().__init__()

        # Reference encoder
        self.reference_encoder = DPReferenceEncoder(
            in_channels=reference_feature_dim,
            hidden_dim=hidden_dim,
            intermediate_dim=ref_encoder_intermediate,
            kernel_size=kernel_size,
            num_convnext_blocks=num_ref_convnext_blocks,
            num_cross_attn_layers=num_cross_attn_layers,
        )

        # Text encoder
        self.text_encoder = DPTextEncoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            intermediate_dim=text_encoder_intermediate,
            kernel_size=kernel_size,
            num_convnext_blocks=num_text_convnext_blocks,
            num_self_attn_blocks=num_self_attn_blocks,
            num_attn_heads=num_attn_heads,
        )

        # Duration estimator
        # Concatenates reference embedding (hidden_dim) + text embedding (hidden_dim)
        estimator_input_dim = hidden_dim + hidden_dim
        self.duration_estimator = DurationEstimator(
            input_dim=estimator_input_dim,
            hidden_dim=164,
        )

    def forward(
        self,
        compressed_latents_ref: torch.Tensor,
        text_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            compressed_latents_ref: Compressed latent features from reference speech
                of shape (batch_size, seq_len, reference_feature_dim)
            text_tokens: Text token indices of shape (batch_size, seq_len)

        Returns:
            Tensor of shape (batch_size, 1) containing predicted duration values
        """
        # Encode reference speech
        ref_embedding = self.reference_encoder(compressed_latents_ref)

        # Encode text
        text_embedding = self.text_encoder(text_tokens)

        # Concatenate embeddings
        combined_embedding = torch.cat([ref_embedding, text_embedding], dim=-1)

        # Predict duration
        duration = self.duration_estimator(combined_embedding)

        return duration

    def compute_loss(
        self,
        predicted_duration: torch.Tensor,
        target_duration: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute L1 loss for duration prediction.

        Args:
            predicted_duration: Model predictions of shape (batch_size, 1)
            target_duration: Ground truth durations of shape (batch_size, 1) or (batch_size,)

        Returns:
            Scalar loss value
        """
        if target_duration.dim() == 1:
            target_duration = target_duration.unsqueeze(1)

        return F.l1_loss(predicted_duration, target_duration)
