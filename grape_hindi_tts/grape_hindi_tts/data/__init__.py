"""
SupertonicTTS Data Processing Pipeline

Complete Hindi text and audio processing for TTS training.
Includes text normalization, audio processing, dataset classes, and data preparation utilities.
"""

from .hindi_text_processor import HindiTextProcessor
from .audio_processor import AudioProcessor
from .dataset import (
    HindiTTSDataset,
    AutoencoderDataset,
    TTLDataset,
    collate_tts_batch,
    collate_autoencoder_batch,
    collate_ttl_batch,
)
from .prepare_dataset import DatasetPreparer
from .precompute_latents import LatentPrecomputer, LatentNormalizer

__all__ = [
    # Text processing
    "HindiTextProcessor",
    # Audio processing
    "AudioProcessor",
    # Datasets
    "HindiTTSDataset",
    "AutoencoderDataset",
    "TTLDataset",
    # Collate functions
    "collate_tts_batch",
    "collate_autoencoder_batch",
    "collate_ttl_batch",
    # Data preparation
    "DatasetPreparer",
    "LatentPrecomputer",
    "LatentNormalizer",
]

__version__ = "0.1.0"
