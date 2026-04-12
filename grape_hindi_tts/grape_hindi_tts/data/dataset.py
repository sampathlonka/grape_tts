"""
PyTorch Dataset Classes for SupertonicTTS

Includes:
- HindiTTSDataset: TTS training dataset (audio + text pairs)
- AutoencoderDataset: Speech autoencoder training (audio only)
- TTLDataset: Text-to-latent training (latents + text)
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path

from hindi_text_processor import HindiTextProcessor
from audio_processor import AudioProcessor

logger = logging.getLogger(__name__)


class HindiTTSDataset(Dataset):
    """
    Hindi TTS Dataset for training text-to-speech models.

    Manifest format:
    {
        "audio_path": "path/to/audio.wav",
        "text": "hindi text",
        "speaker_id": "speaker_001",
        "duration": 2.5,
        "gender": "M"
    }
    """

    def __init__(
        self,
        manifest_path: str,
        audio_processor: Optional[AudioProcessor] = None,
        text_processor: Optional[HindiTextProcessor] = None,
        max_text_length: int = 256,
        normalize_text: bool = True,
        load_audio: bool = True,
    ):
        """
        Initialize HindiTTSDataset.

        Args:
            manifest_path: Path to JSON manifest file
            audio_processor: AudioProcessor instance (creates default if None)
            text_processor: HindiTextProcessor instance (creates default if None)
            max_text_length: Maximum text length in characters
            normalize_text: Whether to normalize text
            load_audio: Whether to load audio in __getitem__ or just path
        """
        self.manifest_path = Path(manifest_path)
        self.max_text_length = max_text_length
        self.normalize_text = normalize_text
        self.load_audio = load_audio

        # Initialize processors
        self.audio_processor = audio_processor or AudioProcessor()
        self.text_processor = text_processor or HindiTextProcessor()

        # Load manifest
        self.samples = self._load_manifest()

    def _load_manifest(self) -> List[Dict[str, Any]]:
        """Load and validate manifest file."""
        try:
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)

            if isinstance(manifest, list):
                samples = manifest
            elif isinstance(manifest, dict) and "samples" in manifest:
                samples = manifest["samples"]
            else:
                raise ValueError("Invalid manifest format")

            logger.info(f"Loaded {len(samples)} samples from manifest")
            return samples
        except Exception as e:
            logger.error(f"Error loading manifest {self.manifest_path}: {e}")
            raise

    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample.

        Returns:
            Dict with keys:
            - mel_spectrogram: (n_mels, time_steps) or None if load_audio=False
            - text_token_ids: (text_length,)
            - speaker_id: str
            - duration: float
            - audio_path: str
            - text: str (original text)
        """
        sample = self.samples[idx]

        # Get basic info
        audio_path = sample["audio_path"]
        text = sample["text"]
        speaker_id = sample.get("speaker_id", "unknown")
        duration = sample.get("duration", 0.0)
        gender = sample.get("gender", "")

        # Process text
        if self.normalize_text:
            text = self.text_processor.normalize_text(text)
        text_token_ids = self.text_processor.text_to_token_ids(
            text,
            add_special_tokens=True,
            max_length=self.max_text_length
        )

        # Load and process audio
        mel_spectrogram = None
        if self.load_audio:
            try:
                waveform, sr = self.audio_processor.process_audio_file(
                    audio_path,
                    normalize=True,
                    trim_silence=True
                )
                mel_spectrogram = self.audio_processor.compute_mel_spectrogram(waveform)
            except Exception as e:
                logger.warning(f"Error processing audio {audio_path}: {e}")
                mel_spectrogram = None

        return {
            "mel_spectrogram": mel_spectrogram,
            "text_token_ids": torch.tensor(text_token_ids, dtype=torch.long),
            "speaker_id": speaker_id,
            "duration": duration,
            "audio_path": audio_path,
            "text": text,
            "gender": gender,
        }


class AutoencoderDataset(Dataset):
    """
    Speech Autoencoder Dataset for training speech autoencoder.
    Uses random crops of audio.
    """

    def __init__(
        self,
        manifest_path: str,
        audio_processor: Optional[AudioProcessor] = None,
        segment_length: int = 32000,
        num_crops_per_sample: int = 3,
    ):
        """
        Initialize AutoencoderDataset.

        Args:
            manifest_path: Path to JSON manifest file (needs 'audio_path' and 'duration')
            audio_processor: AudioProcessor instance
            segment_length: Length of audio segments in samples
            num_crops_per_sample: Number of random crops per audio file
        """
        self.manifest_path = Path(manifest_path)
        self.audio_processor = audio_processor or AudioProcessor()
        self.segment_length = segment_length
        self.num_crops_per_sample = num_crops_per_sample

        # Load manifest
        self.samples = self._load_manifest()

        # Create list of (sample_idx, crop_idx) pairs
        self.sample_crop_pairs = []
        for i in range(len(self.samples)):
            for j in range(num_crops_per_sample):
                self.sample_crop_pairs.append((i, j))

    def _load_manifest(self) -> List[Dict[str, Any]]:
        """Load manifest file."""
        try:
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)

            if isinstance(manifest, list):
                samples = manifest
            elif isinstance(manifest, dict) and "samples" in manifest:
                samples = manifest["samples"]
            else:
                raise ValueError("Invalid manifest format")

            return samples
        except Exception as e:
            logger.error(f"Error loading manifest: {e}")
            raise

    def __len__(self) -> int:
        """Get total number of crops."""
        return len(self.sample_crop_pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a random crop from audio.

        Returns:
            Dict with keys:
            - audio_segment: (segment_length,)
            - mel_spectrogram: (n_mels, time_steps)
            - audio_path: str
        """
        sample_idx, crop_idx = self.sample_crop_pairs[idx]
        sample = self.samples[sample_idx]
        audio_path = sample["audio_path"]

        try:
            # Load audio
            waveform, sr = self.audio_processor.process_audio_file(
                audio_path,
                normalize=True,
                trim_silence=True
            )

            # Random crop
            if len(waveform) > self.segment_length:
                max_start = len(waveform) - self.segment_length
                start = np.random.randint(0, max_start)
                audio_segment = waveform[start:start + self.segment_length]
            else:
                # Pad if too short
                if len(waveform) < self.segment_length:
                    pad_amount = self.segment_length - len(waveform)
                    audio_segment = np.pad(waveform, (0, pad_amount), mode="constant")
                else:
                    audio_segment = waveform[:self.segment_length]

            # Compute mel
            mel_spectrogram = self.audio_processor.compute_mel_spectrogram(audio_segment)

            return {
                "audio_segment": torch.tensor(audio_segment, dtype=torch.float32),
                "mel_spectrogram": torch.tensor(mel_spectrogram, dtype=torch.float32),
                "audio_path": audio_path,
            }
        except Exception as e:
            logger.error(f"Error processing audio {audio_path}: {e}")
            raise


class TTLDataset(Dataset):
    """
    Text-to-Latent Dataset for training text-to-latent models.
    Uses precomputed latents from trained autoencoder.
    """

    def __init__(
        self,
        manifest_path: str,
        latent_dir: str,
        text_processor: Optional[HindiTextProcessor] = None,
        max_text_length: int = 256,
        normalize_text: bool = True,
    ):
        """
        Initialize TTLDataset.

        Args:
            manifest_path: Path to JSON manifest file
            latent_dir: Directory containing precomputed latent .npy files
            text_processor: HindiTextProcessor instance
            max_text_length: Maximum text length
            normalize_text: Whether to normalize text
        """
        self.manifest_path = Path(manifest_path)
        self.latent_dir = Path(latent_dir)
        self.max_text_length = max_text_length
        self.normalize_text = normalize_text

        # Initialize text processor
        self.text_processor = text_processor or HindiTextProcessor()

        # Load manifest
        self.samples = self._load_manifest()

    def _load_manifest(self) -> List[Dict[str, Any]]:
        """Load manifest file."""
        try:
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)

            if isinstance(manifest, list):
                samples = manifest
            elif isinstance(manifest, dict) and "samples" in manifest:
                samples = manifest["samples"]
            else:
                raise ValueError("Invalid manifest format")

            return samples
        except Exception as e:
            logger.error(f"Error loading manifest: {e}")
            raise

    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample with latent representation.

        Returns:
            Dict with keys:
            - latent: (latent_dim,) from autoencoder
            - text_token_ids: (text_length,)
            - text: str
            - speaker_id: str
            - audio_path: str
        """
        sample = self.samples[idx]

        # Get info
        audio_path = sample["audio_path"]
        text = sample["text"]
        speaker_id = sample.get("speaker_id", "unknown")

        # Get latent file path
        audio_stem = Path(audio_path).stem
        latent_path = self.latent_dir / f"{audio_stem}.npy"

        # Load latent
        try:
            latent = np.load(str(latent_path))
            latent = torch.tensor(latent, dtype=torch.float32)
        except FileNotFoundError:
            logger.warning(f"Latent file not found: {latent_path}")
            # Return zero latent as fallback
            latent = torch.zeros(512, dtype=torch.float32)  # Adjust dim as needed

        # Process text
        if self.normalize_text:
            text = self.text_processor.normalize_text(text)
        text_token_ids = self.text_processor.text_to_token_ids(
            text,
            add_special_tokens=True,
            max_length=self.max_text_length
        )

        return {
            "latent": latent,
            "text_token_ids": torch.tensor(text_token_ids, dtype=torch.long),
            "text": text,
            "speaker_id": speaker_id,
            "audio_path": audio_path,
        }


# Collate functions with padding

def collate_tts_batch(
    batch: List[Dict[str, Any]]
) -> Dict[str, torch.Tensor]:
    """
    Collate function for HindiTTSDataset.
    Pads sequences to max length in batch.
    """
    # Separate components
    mel_specs = [item["mel_spectrogram"] for item in batch]
    text_ids = [item["text_token_ids"] for item in batch]
    speakers = [item["speaker_id"] for item in batch]
    durations = torch.tensor([item["duration"] for item in batch], dtype=torch.float32)
    audio_paths = [item["audio_path"] for item in batch]

    # Pad text sequences
    max_text_len = max(len(ids) for ids in text_ids)
    text_ids_padded = []
    for ids in text_ids:
        padded = torch.cat([
            ids,
            torch.zeros(max_text_len - len(ids), dtype=torch.long)
        ])
        text_ids_padded.append(padded)
    text_ids_batch = torch.stack(text_ids_padded)

    # Pad mel spectrograms (pad time dimension)
    mel_specs_valid = [m for m in mel_specs if m is not None]
    if mel_specs_valid:
        max_time = max(m.shape[1] for m in mel_specs_valid)
        mel_specs_padded = []
        for m in mel_specs:
            if m is None:
                m = torch.zeros_like(mel_specs_valid[0])
            if m.shape[1] < max_time:
                m = torch.nn.functional.pad(
                    torch.tensor(m, dtype=torch.float32),
                    (0, max_time - m.shape[1])
                )
            mel_specs_padded.append(m)
        mel_specs_batch = torch.stack(mel_specs_padded)
    else:
        mel_specs_batch = None

    return {
        "mel_spectrogram": mel_specs_batch,
        "text_token_ids": text_ids_batch,
        "speaker_id": speakers,
        "duration": durations,
        "audio_path": audio_paths,
    }


def collate_autoencoder_batch(
    batch: List[Dict[str, Any]]
) -> Dict[str, torch.Tensor]:
    """Collate function for AutoencoderDataset."""
    audio_segments = torch.stack([item["audio_segment"] for item in batch])
    mel_specs = torch.stack([item["mel_spectrogram"] for item in batch])
    audio_paths = [item["audio_path"] for item in batch]

    return {
        "audio_segment": audio_segments,
        "mel_spectrogram": mel_specs,
        "audio_path": audio_paths,
    }


def collate_ttl_batch(
    batch: List[Dict[str, Any]]
) -> Dict[str, torch.Tensor]:
    """Collate function for TTLDataset."""
    latents = torch.stack([item["latent"] for item in batch])
    text_ids = [item["text_token_ids"] for item in batch]
    speakers = [item["speaker_id"] for item in batch]
    audio_paths = [item["audio_path"] for item in batch]

    # Pad text sequences
    max_text_len = max(len(ids) for ids in text_ids)
    text_ids_padded = []
    for ids in text_ids:
        padded = torch.cat([
            ids,
            torch.zeros(max_text_len - len(ids), dtype=torch.long)
        ])
        text_ids_padded.append(padded)
    text_ids_batch = torch.stack(text_ids_padded)

    return {
        "latent": latents,
        "text_token_ids": text_ids_batch,
        "speaker_id": speakers,
        "audio_path": audio_paths,
    }


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Create dummy manifest for testing
    test_manifest = [
        {
            "audio_path": "/path/to/audio1.wav",
            "text": "नमस्ते दुनिया",
            "speaker_id": "speaker_001",
            "duration": 2.5,
            "gender": "M"
        }
    ]

    manifest_path = "/tmp/test_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(test_manifest, f)

    print("Dataset classes created successfully:")
    print("- HindiTTSDataset")
    print("- AutoencoderDataset")
    print("- TTLDataset")
    print("\nCollate functions:")
    print("- collate_tts_batch")
    print("- collate_autoencoder_batch")
    print("- collate_ttl_batch")
