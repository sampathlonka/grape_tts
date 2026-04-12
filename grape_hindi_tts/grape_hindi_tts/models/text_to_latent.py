"""
Text-to-Latent module for SupertonicTTS.

Complete module combining TextEncoder, ReferenceEncoder, and VFEstimator
for flow-matching based latent prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

from .text_encoder import TextEncoder
from .reference_encoder import ReferenceEncoder
from .vf_estimator import VFEstimator


class TextToLatent(nn.Module):
    """Complete Text-to-Latent module for SupertonicTTS.

    Combines:
    1. TextEncoder: processes text tokens with speaker conditioning
    2. ReferenceEncoder: extracts speaker characteristics
    3. VFEstimator: predicts velocity field for flow matching

    This module handles the full pipeline from text and reference speech
    to velocity predictions for latent denoising.
    """

    def __init__(
        self,
        # Text encoding
        vocab_size: int = 256,
        text_char_embed_dim: int = 128,
        text_convnext_hidden: int = 128,
        text_convnext_intermediate: int = 512,
        text_transformer_hidden: int = 512,
        text_n_self_attn: int = 4,
        text_n_cross_attn: int = 2,
        text_n_heads: int = 4,
        text_num_ref_keys: int = 50,
        text_ref_dim: int = 128,
        # Reference encoding
        ref_convnext_hidden: int = 128,
        ref_convnext_intermediate: int = 512,
        ref_num_cross_attn: int = 2,
        ref_n_heads: int = 4,
        ref_num_query_vectors: int = 50,
        ref_query_dim: int = 128,
        # VF Estimator
        vf_channel_dim: int = 256,
        vf_intermediate_dim: int = 1024,
        vf_time_dim: int = 64,
        vf_n_heads: int = 4,
        vf_num_main_blocks: int = 4,
        # Latent compression
        latent_channels: int = 24,
        compression_ratio: int = 6,
        # Classifier-free guidance
        use_cfg: bool = False,
        cfg_uncond_prob: float = 0.1,
    ):
        super().__init__()

        self.latent_channels = latent_channels
        self.compression_ratio = compression_ratio
        self.compressed_channels = latent_channels * compression_ratio  # 144
        self.use_cfg = use_cfg
        self.cfg_uncond_prob = cfg_uncond_prob

        # Text Encoder
        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            char_embed_dim=text_char_embed_dim,
            convnext_hidden=text_convnext_hidden,
            convnext_intermediate=text_convnext_intermediate,
            transformer_hidden=text_transformer_hidden,
            num_self_attn=text_n_self_attn,
            num_cross_attn=text_n_cross_attn,
            n_heads=text_n_heads,
            num_reference_keys=text_num_ref_keys,
            reference_dim=text_ref_dim,
        )

        # Reference Encoder
        self.reference_encoder = ReferenceEncoder(
            latent_channels=self.compressed_channels,
            convnext_hidden=ref_convnext_hidden,
            convnext_intermediate=ref_convnext_intermediate,
            num_cross_attn=ref_num_cross_attn,
            n_heads=ref_n_heads,
            num_query_vectors=ref_num_query_vectors,
            query_dim=ref_query_dim,
        )

        # VF Estimator
        self.vf_estimator = VFEstimator(
            latent_channels=self.compressed_channels,
            channel_dim=vf_channel_dim,
            intermediate_dim=vf_intermediate_dim,
            time_dim=vf_time_dim,
            text_dim=text_transformer_hidden,
            ref_dim=ref_query_dim,
            n_heads=vf_n_heads,
            num_main_blocks=vf_num_main_blocks,
        )

        # Learnable parameters for classifier-free guidance
        if use_cfg:
            self.uncond_text_embed = nn.Parameter(
                torch.randn(1, 1, text_transformer_hidden)
            )
            self.uncond_ref_keys = nn.Parameter(
                torch.randn(1, ref_num_query_vectors, ref_query_dim)
            )

    def compress_latents(
        self,
        latents: torch.Tensor,
        Kc: int = 6,
    ) -> torch.Tensor:
        """Compress latents by reshaping temporal dimension.

        Reshapes (B, C, T) -> (B, Kc*C, T//Kc)
        This groups Kc consecutive timesteps and flattens channel dimension.

        Args:
            latents: (batch, latent_channels, time) uncompressed latents
            Kc: compression factor (default 6)

        Returns:
            compressed: (batch, latent_channels*Kc, time//Kc)
        """
        batch, channels, time = latents.shape
        assert time % Kc == 0, f"Time dimension {time} must be divisible by Kc {Kc}"

        # Reshape: (B, C, T) -> (B, C, T//Kc, Kc) -> (B, C*Kc, T//Kc)
        compressed = latents.reshape(batch, channels, time // Kc, Kc)
        compressed = compressed.permute(0, 1, 3, 2).reshape(
            batch, channels * Kc, time // Kc
        )

        return compressed

    def decompress_latents(
        self,
        compressed: torch.Tensor,
        Kc: int = 6,
    ) -> torch.Tensor:
        """Decompress latents (inverse of compress_latents).

        Reshapes (B, Kc*C, T//Kc) -> (B, C, T)

        Args:
            compressed: (batch, latent_channels*Kc, time//Kc)
            Kc: compression factor (default 6)

        Returns:
            latents: (batch, latent_channels, time)
        """
        batch, channels, time_compressed = compressed.shape
        assert channels % Kc == 0, f"Channels {channels} must be divisible by Kc {Kc}"

        original_channels = channels // Kc

        # Reshape: (B, C*Kc, T//Kc) -> (B, C, Kc, T//Kc) -> (B, C, T)
        latents = compressed.reshape(batch, original_channels, Kc, time_compressed)
        latents = latents.permute(0, 1, 3, 2).reshape(
            batch, original_channels, time_compressed * Kc
        )

        return latents

    def forward(
        self,
        noisy_latents: torch.Tensor,
        compressed_latents_ref: torch.Tensor,
        text_tokens: torch.Tensor,
        timestep: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        cfg_scale: float = 1.0,
    ) -> torch.Tensor:
        """Forward pass for training/inference.

        Args:
            noisy_latents: (batch, 24, seq_len) noisy uncompressed latents
                          or (batch, 144, seq_len//6) pre-compressed noisy latents
            compressed_latents_ref: (batch, 144, T_ref//6) reference latents (pre-compressed)
            text_tokens: (batch, text_len) character token indices
            timestep: (batch,) or scalar, timestep in [0, 1]
            text_mask: (batch, text_len) optional mask for valid text positions
            cfg_scale: classifier-free guidance scale (1.0 = no guidance)

        Returns:
            velocity: (batch, 144, seq_len_compressed) predicted velocity
        """
        batch = noisy_latents.shape[0]

        # Ensure latents are compressed
        if noisy_latents.shape[1] == self.latent_channels:
            # Compress if needed
            noisy_latents_compressed = self.compress_latents(
                noisy_latents, Kc=self.compression_ratio
            )
        else:
            # Already compressed
            noisy_latents_compressed = noisy_latents

        # Ensure timestep is proper shape
        if timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)
        if timestep.shape[0] == 1:
            timestep = timestep.expand(batch)

        # Encode reference speech features
        ref_keys, ref_values = self.reference_encoder(compressed_latents_ref)

        # Encode text
        text_embeddings = self.text_encoder(text_tokens, ref_keys, text_mask)

        # Classifier-free guidance
        if self.use_cfg and cfg_scale != 1.0:
            # Get unconditional embeddings
            # Use learnable unconditional parameters
            uncond_text_emb = self.uncond_text_embed.expand(batch, -1, -1)
            uncond_ref_keys = self.uncond_ref_keys.expand(batch, -1, -1)

            # Predict with conditions
            velocity_cond = self.vf_estimator(
                noisy_latents_compressed,
                text_embeddings,
                ref_keys,
                timestep,
            )

            # Predict without conditions
            velocity_uncond = self.vf_estimator(
                noisy_latents_compressed,
                uncond_text_emb,
                uncond_ref_keys,
                timestep,
            )

            # Interpolate: v = v_uncond + scale * (v_cond - v_uncond)
            velocity = velocity_uncond + cfg_scale * (velocity_cond - velocity_uncond)
        else:
            # Standard forward pass
            velocity = self.vf_estimator(
                noisy_latents_compressed,
                text_embeddings,
                ref_keys,
                timestep,
            )

        return velocity

    def inference(
        self,
        text_tokens: torch.Tensor,
        compressed_latents_ref: torch.Tensor,
        num_inference_steps: int = 50,
        cfg_scale: float = 1.0,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Inference using Euler ODE solver.

        Args:
            text_tokens: (batch, text_len) character indices
            compressed_latents_ref: (batch, 144, T_ref//6) reference latents
            num_inference_steps: number of ODE solver steps
            cfg_scale: classifier-free guidance scale
            text_mask: optional text mask

        Returns:
            latents: (batch, 144, T) denoised compressed latents
        """
        batch = compressed_latents_ref.shape[0]
        seq_len_compressed = compressed_latents_ref.shape[-1]

        # Initialize with noise
        with torch.no_grad():
            noisy_latents = torch.randn(
                batch, self.compressed_channels, seq_len_compressed,
                device=compressed_latents_ref.device,
                dtype=compressed_latents_ref.dtype,
            )

            # Prepare timesteps
            timesteps = torch.linspace(1.0, 0.0, num_inference_steps + 1)
            dt = 1.0 / num_inference_steps

            # ODE solver loop
            x = noisy_latents
            for i in range(num_inference_steps):
                t = timesteps[i]
                t_batch = torch.full(
                    (batch,), t, device=compressed_latents_ref.device,
                    dtype=compressed_latents_ref.dtype,
                )

                # Predict velocity
                velocity = self.forward(
                    x,
                    compressed_latents_ref,
                    text_tokens,
                    t_batch,
                    text_mask=text_mask,
                    cfg_scale=cfg_scale,
                )

                # Euler step
                x = x + velocity * dt

        return x

    def sample_training_timesteps(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Sample random timesteps for training.

        Args:
            batch_size: number of samples
            device: torch device
            dtype: torch dtype

        Returns:
            timesteps: (batch_size,) uniformly sampled in [0, 1]
        """
        return torch.rand(batch_size, device=device, dtype=dtype)
