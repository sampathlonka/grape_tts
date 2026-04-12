"""
Precompute Latents Script for SupertonicTTS

Encodes audio through trained speech autoencoder and saves latent representations.
Also computes normalization statistics for latent features.
"""

import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class LatentPrecomputer:
    """Precompute latent representations from trained autoencoder."""

    def __init__(
        self,
        model_path: str,
        output_dir: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 32,
    ):
        """
        Initialize latent precomputer.

        Args:
            model_path: Path to trained autoencoder model
            output_dir: Directory to save latents
            device: Device to use (cuda/cpu)
            batch_size: Batch size for processing
        """
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.device = device
        self.batch_size = batch_size

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Statistics tracking
        self.stats = {
            "total_samples": 0,
            "processed_samples": 0,
            "failed_samples": 0,
            "latent_mean": None,
            "latent_std": None,
            "latent_min": None,
            "latent_max": None,
        }

        self.failed_samples = []

    def load_model(self, model: nn.Module) -> nn.Module:
        """
        Load trained autoencoder model.

        Args:
            model: Autoencoder model instance

        Returns:
            Loaded model on correct device
        """
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
            logger.info(f"Loaded model from {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

        model.to(self.device)
        model.eval()
        return model

    def encode_audio(
        self,
        model: nn.Module,
        mel_spectrogram: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Encode audio mel spectrogram to latent vector.

        Args:
            model: Autoencoder model (with .encode method)
            mel_spectrogram: Mel spectrogram (n_mels, time_steps)

        Returns:
            Latent vector (latent_dim,) or None if error
        """
        try:
            # Convert to tensor
            if isinstance(mel_spectrogram, np.ndarray):
                mel_tensor = torch.from_numpy(mel_spectrogram).float()
            else:
                mel_tensor = mel_spectrogram.float()

            # Add batch dimension if needed
            if mel_tensor.dim() == 2:
                mel_tensor = mel_tensor.unsqueeze(0)

            # Move to device
            mel_tensor = mel_tensor.to(self.device)

            # Encode
            with torch.no_grad():
                if hasattr(model, "encode"):
                    latent = model.encode(mel_tensor)
                else:
                    # Assume first call to model returns latent
                    latent = model(mel_tensor)

            # Convert to numpy
            latent = latent.squeeze(0).cpu().numpy()
            return latent
        except Exception as e:
            logger.warning(f"Error encoding audio: {e}")
            return None

    def precompute_latents_from_manifest(
        self,
        model: nn.Module,
        manifest_path: str,
        audio_processor,
    ) -> None:
        """
        Precompute latents for all samples in manifest.

        Args:
            model: Trained autoencoder model
            manifest_path: Path to manifest JSON file
            audio_processor: AudioProcessor instance for loading audio
        """
        # Load manifest
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)

            if isinstance(manifest, dict) and "samples" in manifest:
                samples = manifest["samples"]
            else:
                samples = manifest
        except Exception as e:
            logger.error(f"Error loading manifest: {e}")
            raise

        self.stats["total_samples"] = len(samples)

        logger.info(f"Processing {len(samples)} samples")

        # Process samples
        latents_list = []
        sample_info = []

        for sample in tqdm(samples, desc="Encoding"):
            audio_path = sample["audio_path"]

            try:
                # Load and process audio
                waveform, sr = audio_processor.process_audio_file(
                    audio_path,
                    normalize=True,
                    trim_silence=True
                )

                # Compute mel
                mel_spec = audio_processor.compute_mel_spectrogram(waveform)

                # Encode
                latent = self.encode_audio(model, mel_spec)

                if latent is None:
                    raise RuntimeError("Encoding returned None")

                # Save latent
                latent_filename = Path(audio_path).stem + ".npy"
                latent_path = self.output_dir / latent_filename
                np.save(str(latent_path), latent)

                latents_list.append(latent)
                sample_info.append({
                    "audio_path": audio_path,
                    "latent_path": str(latent_path),
                    "latent_shape": latent.shape,
                })

                self.stats["processed_samples"] += 1

            except Exception as e:
                logger.warning(f"Failed to encode {audio_path}: {e}")
                self.failed_samples.append((audio_path, str(e)))
                self.stats["failed_samples"] += 1

        # Compute statistics
        if latents_list:
            logger.info("Computing latent statistics...")
            latents_array = np.array(latents_list)  # (n_samples, latent_dim)

            self.stats["latent_mean"] = latents_array.mean(axis=0).tolist()
            self.stats["latent_std"] = latents_array.std(axis=0).tolist()
            self.stats["latent_min"] = float(latents_array.min())
            self.stats["latent_max"] = float(latents_array.max())

        # Save statistics
        self._save_statistics(sample_info)

    def _save_statistics(self, sample_info: List[Dict[str, Any]]) -> None:
        """Save latent statistics."""
        # Save sample mapping
        mapping_path = self.output_dir / "latent_mapping.json"
        with open(mapping_path, "w", encoding="utf-8") as f:
            json.dump(sample_info, f, indent=2)

        # Save statistics
        stats_path = self.output_dir / "latent_statistics.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(self.stats, f, indent=2)

        logger.info(f"Saved statistics to {stats_path}")

        # Print summary
        print("\n" + "=" * 70)
        print("Latent Precomputation Summary")
        print("=" * 70)
        print(f"Total samples:         {self.stats['total_samples']}")
        print(f"Successfully encoded:  {self.stats['processed_samples']}")
        print(f"Failed:                {self.stats['failed_samples']}")

        if self.stats["latent_mean"] is not None:
            latent_mean = np.array(self.stats["latent_mean"])
            latent_std = np.array(self.stats["latent_std"])
            print(f"\nLatent Statistics:")
            print(f"  Mean range: [{latent_mean.min():.6f}, {latent_mean.max():.6f}]")
            print(f"  Std range:  [{latent_std.min():.6f}, {latent_std.max():.6f}]")
            print(f"  Min value:  {self.stats['latent_min']:.6f}")
            print(f"  Max value:  {self.stats['latent_max']:.6f}")

        if self.failed_samples:
            print(f"\nFailed samples (first 10):")
            for audio_path, error in self.failed_samples[:10]:
                print(f"  {audio_path}: {error}")

        print("=" * 70)


class LatentNormalizer:
    """Normalize latent vectors using precomputed statistics."""

    def __init__(self, statistics_path: str):
        """
        Initialize normalizer.

        Args:
            statistics_path: Path to latent_statistics.json
        """
        with open(statistics_path, "r") as f:
            stats = json.load(f)

        self.mean = np.array(stats["latent_mean"])
        self.std = np.array(stats["latent_std"])

    def normalize(self, latent: np.ndarray) -> np.ndarray:
        """
        Normalize latent vector.

        Args:
            latent: Latent vector (latent_dim,)

        Returns:
            Normalized latent vector
        """
        return (latent - self.mean) / (self.std + 1e-8)

    def denormalize(self, latent: np.ndarray) -> np.ndarray:
        """
        Denormalize latent vector.

        Args:
            latent: Normalized latent vector

        Returns:
            Original-scale latent vector
        """
        return latent * (self.std + 1e-8) + self.mean


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Precompute latents from trained autoencoder")
    parser.add_argument("model_path", help="Path to trained autoencoder model")
    parser.add_argument("manifest_path", help="Path to dataset manifest")
    parser.add_argument("--output_dir", default="./precomputed_latents", help="Output directory")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Import required modules (these would come from your model implementation)
    # For demonstration, we show the structure
    logger.info("Note: This script requires your autoencoder model implementation")
    logger.info("Please load your model class and pass it to the precomputer")

    # Example usage:
    # from your_model import SpeechAutoencoder
    # from audio_processor import AudioProcessor
    #
    # model = SpeechAutoencoder(...)
    # audio_processor = AudioProcessor()
    #
    # precomputer = LatentPrecomputer(
    #     model_path=args.model_path,
    #     output_dir=args.output_dir,
    #     device=args.device,
    #     batch_size=args.batch_size,
    # )
    #
    # loaded_model = precomputer.load_model(model)
    # precomputer.precompute_latents_from_manifest(
    #     loaded_model,
    #     args.manifest_path,
    #     audio_processor
    # )


if __name__ == "__main__":
    main()
