"""
Utility script to precompute latents from audio using pretrained autoencoder.

Usage:
    python precompute_latents.py \
        --autoencoder_checkpoint ./path/to/autoencoder.pt \
        --audio_dir ./path/to/audio/files \
        --output_path ./latents_precomputed.pt
"""
import os
import argparse
import logging
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torchaudio
import numpy as np
from tqdm import tqdm

from train_autoencoder import AutoencoderGenerator


def setup_logging(output_dir: str = "."):
    """Setup logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
        ]
    )


def load_autoencoder(checkpoint_path: str, device: torch.device) -> AutoencoderGenerator:
    """Load pretrained autoencoder from checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Infer latent_dim from checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_state = checkpoint['model']

    # Create model
    latent_dim = 128  # Default, can be inferred from state dict
    for key in model_state.keys():
        if 'latent_proj' in key:
            # Extract latent_dim from latent projection layer
            if 'weight' in key:
                latent_dim = model_state[key].shape[0]
                break

    model = AutoencoderGenerator(latent_dim=latent_dim)
    model.load_state_dict(model_state)
    model = model.to(device)
    model.eval()

    logging.info(f"Loaded autoencoder from {checkpoint_path} (latent_dim={latent_dim})")
    return model


def load_audio(audio_path: str, sample_rate: int = 44100) -> torch.Tensor:
    """Load audio file and convert to mono."""
    waveform, sr = torchaudio.load(audio_path)

    # Resample if necessary
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    return waveform


def get_audio_files(directory: str) -> List[str]:
    """Get list of audio files from directory."""
    audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}
    audio_files = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if Path(file).suffix.lower() in audio_extensions:
                audio_files.append(os.path.join(root, file))

    return sorted(audio_files)


@torch.no_grad()
def precompute_latents(
    model: AutoencoderGenerator,
    audio_paths: List[str],
    device: torch.device,
    max_duration: float = 10.0,
    sample_rate: int = 44100,
    hop_length: int = 512
) -> List[torch.Tensor]:
    """
    Precompute latents for audio files.

    Args:
        model: Pretrained autoencoder
        audio_paths: List of audio file paths
        device: Device to compute on
        max_duration: Maximum audio duration in seconds
        sample_rate: Sample rate for audio
        hop_length: Hop length for latent frames

    Returns:
        List of latent tensors
    """
    latents_list = []
    max_frames = int(max_duration * sample_rate)

    for audio_path in tqdm(audio_paths, desc="Precomputing latents"):
        try:
            # Load audio
            waveform = load_audio(audio_path, sample_rate)

            # Truncate to max duration
            if waveform.shape[1] > max_frames:
                waveform = waveform[:, :max_frames]

            # Add batch dimension and move to device
            waveform = waveform.unsqueeze(0).to(device)

            # Encode
            latent = model.encode(waveform)  # [1, latent_dim, T']

            # Remove batch dimension
            latent = latent.squeeze(0).cpu()

            latents_list.append(latent)

        except Exception as e:
            logging.warning(f"Failed to process {audio_path}: {e}")
            continue

    logging.info(f"Precomputed {len(latents_list)} latents")
    return latents_list


def main():
    parser = argparse.ArgumentParser(description="Precompute latents from audio")
    parser.add_argument(
        "--autoencoder_checkpoint",
        type=str,
        required=True,
        help="Path to pretrained autoencoder checkpoint"
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        help="Directory containing audio files (if not provided, creates dummy latents)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output path for precomputed latents"
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=44100,
        help="Audio sample rate"
    )
    parser.add_argument(
        "--max_duration",
        type=float,
        default=10.0,
        help="Maximum audio duration in seconds"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10000,
        help="Number of dummy latents to generate (if no audio_dir)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for computation"
    )

    args = parser.parse_args()

    # Setup
    setup_logging()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    logging.info(f"Using device: {device}")

    # Load autoencoder
    model = load_autoencoder(args.autoencoder_checkpoint, device)

    # Precompute latents
    if args.audio_dir and os.path.isdir(args.audio_dir):
        logging.info(f"Loading audio files from {args.audio_dir}")
        audio_files = get_audio_files(args.audio_dir)

        if not audio_files:
            logging.warning(f"No audio files found in {args.audio_dir}")
            logging.info(f"Generating {args.num_samples} dummy latents instead")
            latents_list = [
                torch.randn(
                    int(args.max_duration * args.sample_rate / 512),
                    128  # latent_dim
                )
                for _ in range(args.num_samples)
            ]
        else:
            logging.info(f"Found {len(audio_files)} audio files")
            latents_list = precompute_latents(
                model,
                audio_files,
                device,
                max_duration=args.max_duration,
                sample_rate=args.sample_rate
            )
    else:
        logging.info(f"Generating {args.num_samples} dummy latents")
        latents_list = [
            torch.randn(
                int(args.max_duration * args.sample_rate / 512),
                128  # latent_dim
            )
            for _ in range(args.num_samples)
        ]

    # Save latents
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(latents_list, args.output_path)
    logging.info(f"Saved {len(latents_list)} latents to {args.output_path}")

    # Print statistics
    if latents_list:
        lengths = [z.shape[0] for z in latents_list]
        logging.info(f"Latent lengths: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.1f}")


if __name__ == "__main__":
    main()
