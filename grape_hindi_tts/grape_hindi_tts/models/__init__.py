"""
SupertonicTTS Models Package

Architecture components — two sources:
  • F5-TTS (SWivid/F5-TTS, Apache 2.0):
      - ConvNeXtV2Block with GRN  ← replaces our hand-written V1 block
      - ConvPositionEmbedding / TimestepEmbedding
  • SupertonicTTS paper (Supertone Inc., 2025):
      - LARoPE, cross-attention encoders, VF estimator
      - Duration predictor, speech autoencoder, GAN discriminators
"""

# ── F5-TTS building blocks (re-exported for convenience) ────────────────────
from supertonic_hindi_tts.third_party.f5_tts_modules import (
    GRN,
    ConvNeXtV2Block,
    ConvNeXtV2BlockCF,
    ConvNeXtV2Stack,
    ConvPositionEmbedding,
    TimestepEmbedding,
)

# ── ConvNeXt with backwards-compatible aliases ───────────────────────────────
from .convnext import (
    ConvNeXtBlock,           # alias → ConvNeXtV2BlockCF  (channel-first)
    DilatedConvNeXtBlock,
    CausalConvNeXtBlock,
    CausalConv1d,
    ConvNeXtStack,
)

# ── Attention ─────────────────────────────────────────────────────────────────
from .attention import (
    RoPEEmbedding,
    LARoPEEmbedding,
    MultiHeadSelfAttention,
    MultiHeadCrossAttention,
)

# ── Encoder modules ───────────────────────────────────────────────────────────
from .text_encoder import TextEncoder
from .reference_encoder import ReferenceEncoder

# ── VF Estimator ──────────────────────────────────────────────────────────────
from .vf_estimator import (
    VFEstimator,
    TimeCondBlock,
    TextCondBlock,
    RefCondBlock,
)

# ── Complete Text-to-Latent module ────────────────────────────────────────────
from .text_to_latent import TextToLatentModule

# ── Speech Autoencoder ────────────────────────────────────────────────────────
from .speech_autoencoder import (
    SpeechAutoencoder,
    LatentEncoder,
    LatentDecoder,
)

# ── GAN Discriminators ────────────────────────────────────────────────────────
from .discriminators import (
    MultiPeriodDiscriminator,
    MultiResolutionDiscriminator,
    CompositeDiscriminator,
)

# ── Duration Predictor ────────────────────────────────────────────────────────
from .duration_predictor import DurationPredictor

# ── Loss functions ────────────────────────────────────────────────────────────
from .losses import (
    spectral_reconstruction_loss,
    adversarial_loss_generator,
    adversarial_loss_discriminator,
    feature_matching_loss,
    flow_matching_loss,
    duration_loss,
)

__all__ = [
    # F5-TTS components
    "GRN",
    "ConvNeXtV2Block",
    "ConvNeXtV2BlockCF",
    "ConvNeXtV2Stack",
    "ConvPositionEmbedding",
    "TimestepEmbedding",
    # ConvNeXt (aliases)
    "ConvNeXtBlock",
    "DilatedConvNeXtBlock",
    "CausalConvNeXtBlock",
    "CausalConv1d",
    "ConvNeXtStack",
    # Attention
    "RoPEEmbedding",
    "LARoPEEmbedding",
    "MultiHeadSelfAttention",
    "MultiHeadCrossAttention",
    # Encoders
    "TextEncoder",
    "ReferenceEncoder",
    # VF Estimator
    "VFEstimator",
    "TimeCondBlock",
    "TextCondBlock",
    "RefCondBlock",
    # Top-level modules
    "TextToLatentModule",
    "SpeechAutoencoder",
    "LatentEncoder",
    "LatentDecoder",
    "DurationPredictor",
    # Discriminators
    "MultiPeriodDiscriminator",
    "MultiResolutionDiscriminator",
    "CompositeDiscriminator",
    # Losses
    "spectral_reconstruction_loss",
    "adversarial_loss_generator",
    "adversarial_loss_discriminator",
    "feature_matching_loss",
    "flow_matching_loss",
    "duration_loss",
]
