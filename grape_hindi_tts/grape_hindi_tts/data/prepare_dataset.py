"""
Dataset Preparation Script for Hindi TTS

Prepares raw audio and transcript data for SupertonicTTS training:
- Validates audio files
- Normalizes Hindi text
- Splits data into train/val/test
- Generates JSON manifest files
- Computes dataset statistics
- Creates processing report
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
import numpy as np
import librosa
from datetime import datetime
from collections import defaultdict

from hindi_text_processor import HindiTextProcessor
from audio_processor import AudioProcessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DatasetPreparer:
    """Prepare Hindi TTS dataset for training."""

    def __init__(
        self,
        audio_dir: str,
        transcript_file: str,
        output_dir: str,
        sample_rate: int = 44100,
        min_duration: float = 0.5,
        max_duration: float = 30.0,
        max_silence_ratio: float = 0.5,
    ):
        """
        Initialize dataset preparer.

        Args:
            audio_dir: Directory containing audio files
            transcript_file: File with audio_filename<tab>text<tab>speaker_id<tab>gender
            output_dir: Output directory for manifests and stats
            sample_rate: Target sample rate
            min_duration: Minimum audio duration in seconds
            max_duration: Maximum audio duration in seconds
            max_silence_ratio: Maximum allowed silence ratio
        """
        self.audio_dir = Path(audio_dir)
        self.transcript_file = Path(transcript_file)
        self.output_dir = Path(output_dir)
        self.sample_rate = sample_rate
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.max_silence_ratio = max_silence_ratio

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize processors
        self.text_processor = HindiTextProcessor()
        self.audio_processor = AudioProcessor(sample_rate=sample_rate)

        # Statistics tracking
        self.stats = {
            "total_samples": 0,
            "valid_samples": 0,
            "invalid_samples": 0,
            "skipped_reasons": defaultdict(int),
            "total_duration": 0.0,
            "mel_stats": None,
            "speaker_stats": {},
        }

        self.invalid_samples = []

    def _load_transcripts(self) -> Dict[str, Tuple[str, str, str]]:
        """
        Load transcripts from file.
        Format: audio_filename<tab>text<tab>speaker_id<tab>gender

        Returns:
            Dict mapping filename -> (text, speaker_id, gender)
        """
        transcripts = {}
        try:
            with open(self.transcript_file, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    parts = line.strip().split("\t")
                    if len(parts) < 3:
                        logger.warning(f"Line {line_num}: Invalid format, skipping")
                        continue

                    filename = parts[0].strip()
                    text = parts[1].strip()
                    speaker_id = parts[2].strip()
                    gender = parts[3].strip() if len(parts) > 3 else ""

                    # Validate text
                    if not text:
                        logger.warning(f"Line {line_num}: Empty text for {filename}")
                        continue

                    transcripts[filename] = (text, speaker_id, gender)

            logger.info(f"Loaded {len(transcripts)} transcripts")
            return transcripts
        except Exception as e:
            logger.error(f"Error loading transcripts: {e}")
            raise

    def _compute_silence_ratio(self, waveform: np.ndarray) -> float:
        """
        Estimate silence ratio using energy-based VAD.

        Args:
            waveform: Audio waveform

        Returns:
            Ratio of silent frames (0.0 to 1.0)
        """
        # Compute short-time energy
        frame_length = 512
        energy = []
        for i in range(0, len(waveform) - frame_length, frame_length):
            frame = waveform[i:i + frame_length]
            frame_energy = np.sqrt(np.mean(frame ** 2))
            energy.append(frame_energy)

        if not energy:
            return 1.0

        energy = np.array(energy)
        threshold = np.mean(energy) * 0.1
        silent_frames = np.sum(energy < threshold)
        silence_ratio = silent_frames / len(energy)

        return silence_ratio

    def _validate_audio(self, audio_path: Path) -> Tuple[bool, str, Optional[float], Optional[int]]:
        """
        Validate audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            Tuple of (is_valid, reason, duration, sample_rate)
        """
        try:
            # Load audio
            waveform, sr = librosa.load(str(audio_path), sr=None, mono=True)

            # Check duration
            duration = len(waveform) / sr
            if duration < self.min_duration:
                return False, f"Duration too short ({duration:.2f}s < {self.min_duration}s)", duration, sr

            if duration > self.max_duration:
                return False, f"Duration too long ({duration:.2f}s > {self.max_duration}s)", duration, sr

            # Check silence ratio
            silence_ratio = self._compute_silence_ratio(waveform)
            if silence_ratio > self.max_silence_ratio:
                return False, f"Too much silence ({silence_ratio:.2%})", duration, sr

            return True, "OK", duration, sr
        except Exception as e:
            return False, f"Load error: {str(e)}", None, None

    def prepare_dataset(
        self,
        train_split: float = 0.9,
        val_split: float = 0.05,
        test_split: float = 0.05,
        seed: int = 42,
    ) -> None:
        """
        Prepare complete dataset.

        Args:
            train_split: Train split ratio
            val_split: Validation split ratio
            test_split: Test split ratio
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)

        logger.info("=" * 60)
        logger.info("Dataset Preparation Pipeline")
        logger.info("=" * 60)

        # Load transcripts
        logger.info(f"Loading transcripts from {self.transcript_file}")
        transcripts = self._load_transcripts()

        # Validate audio files and build sample list
        logger.info(f"Validating audio files in {self.audio_dir}")
        valid_samples = []

        for filename, (text, speaker_id, gender) in transcripts.items():
            audio_path = self.audio_dir / filename
            self.stats["total_samples"] += 1

            # Check file exists
            if not audio_path.exists():
                reason = "File not found"
                self.stats["skipped_reasons"][reason] += 1
                self.invalid_samples.append((filename, reason))
                continue

            # Validate audio
            is_valid, reason, duration, sr = self._validate_audio(audio_path)

            if not is_valid:
                self.stats["skipped_reasons"][reason] += 1
                self.invalid_samples.append((filename, reason))
                logger.warning(f"Invalid: {filename} - {reason}")
                continue

            # Normalize text
            normalized_text = self.text_processor.normalize_text(text)

            # Add to valid samples
            valid_samples.append({
                "audio_path": str(audio_path),
                "filename": filename,
                "text": text,
                "normalized_text": normalized_text,
                "speaker_id": speaker_id,
                "gender": gender,
                "duration": duration,
                "sample_rate": sr,
            })

            self.stats["valid_samples"] += 1
            self.stats["total_duration"] += duration

        logger.info(f"Valid samples: {len(valid_samples)} / {self.stats['total_samples']}")

        # Split dataset
        logger.info("Splitting dataset...")
        train_samples, val_samples, test_samples = self._split_dataset(
            valid_samples,
            train_split,
            val_split,
            test_split
        )

        logger.info(f"Train: {len(train_samples)}")
        logger.info(f"Val: {len(val_samples)}")
        logger.info(f"Test: {len(test_samples)}")

        # Compute statistics
        logger.info("Computing statistics...")
        self._compute_statistics(valid_samples)

        # Save manifests
        logger.info("Saving manifest files...")
        self._save_manifests(train_samples, val_samples, test_samples)

        # Generate report
        self._generate_report()

    def _split_dataset(
        self,
        samples: List[Dict[str, Any]],
        train_split: float,
        val_split: float,
        test_split: float,
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Split dataset while maintaining speaker balance.

        Args:
            samples: List of valid samples
            train_split: Train ratio
            val_split: Validation ratio
            test_split: Test ratio

        Returns:
            Tuple of (train_samples, val_samples, test_samples)
        """
        # Group by speaker
        speaker_samples = defaultdict(list)
        for sample in samples:
            speaker_samples[sample["speaker_id"]].append(sample)

        train_samples = []
        val_samples = []
        test_samples = []

        # Split per speaker to maintain balance
        for speaker_id, speaker_data in speaker_samples.items():
            n = len(speaker_data)
            n_train = max(1, int(n * train_split))
            n_val = max(1, int(n * val_split))

            # Shuffle
            indices = np.random.permutation(n)

            train_idx = indices[:n_train]
            val_idx = indices[n_train:n_train + n_val]
            test_idx = indices[n_train + n_val:]

            train_samples.extend([speaker_data[i] for i in train_idx])
            val_samples.extend([speaker_data[i] for i in val_idx])
            test_samples.extend([speaker_data[i] for i in test_idx])

        return train_samples, val_samples, test_samples

    def _compute_statistics(self, samples: List[Dict[str, Any]]) -> None:
        """Compute dataset statistics."""
        # Mel spectrogram statistics
        mel_features = []

        logger.info("Computing mel spectrogram statistics...")
        for i, sample in enumerate(samples):
            if i % 100 == 0:
                logger.info(f"  Processing {i}/{len(samples)}")

            try:
                audio_path = sample["audio_path"]
                waveform, sr = librosa.load(audio_path, sr=self.sample_rate)
                mel_spec = self.audio_processor.compute_mel_spectrogram(waveform)
                mel_features.append(mel_spec)
            except Exception as e:
                logger.warning(f"Error computing mel for {sample['filename']}: {e}")
                continue

        if mel_features:
            mel_array = np.stack(mel_features)  # (n_samples, n_mels, time)
            self.stats["mel_stats"] = {
                "mean": mel_array.mean(axis=(0, 2)).tolist(),  # Per-channel mean
                "std": mel_array.std(axis=(0, 2)).tolist(),    # Per-channel std
                "min": float(mel_array.min()),
                "max": float(mel_array.max()),
            }

        # Speaker statistics
        for sample in samples:
            speaker_id = sample["speaker_id"]
            if speaker_id not in self.stats["speaker_stats"]:
                self.stats["speaker_stats"][speaker_id] = {
                    "count": 0,
                    "total_duration": 0.0,
                    "genders": set(),
                }

            self.stats["speaker_stats"][speaker_id]["count"] += 1
            self.stats["speaker_stats"][speaker_id]["total_duration"] += sample["duration"]
            if sample["gender"]:
                self.stats["speaker_stats"][speaker_id]["genders"].add(sample["gender"])

        # Convert sets to lists for JSON
        for speaker_id in self.stats["speaker_stats"]:
            genders = self.stats["speaker_stats"][speaker_id]["genders"]
            self.stats["speaker_stats"][speaker_id]["genders"] = list(genders)

    def _save_manifests(
        self,
        train_samples: List[Dict],
        val_samples: List[Dict],
        test_samples: List[Dict],
    ) -> None:
        """Save manifest files."""
        splits = [
            ("train", train_samples),
            ("val", val_samples),
            ("test", test_samples),
        ]

        for split_name, samples in splits:
            manifest_path = self.output_dir / f"{split_name}_manifest.json"

            # Remove normalized_text from output (for space efficiency)
            output_samples = []
            for sample in samples:
                output_sample = {
                    "audio_path": sample["audio_path"],
                    "text": sample["text"],
                    "speaker_id": sample["speaker_id"],
                    "gender": sample["gender"],
                    "duration": float(sample["duration"]),
                }
                output_samples.append(output_sample)

            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(output_samples, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved {len(output_samples)} samples to {manifest_path}")

    def _generate_report(self) -> None:
        """Generate and save preparation report."""
        report_path = self.output_dir / "dataset_report.txt"

        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append("Hindi TTS Dataset Preparation Report")
        report_lines.append("=" * 70)
        report_lines.append(f"Generated: {datetime.now().isoformat()}")
        report_lines.append("")

        # Summary
        report_lines.append("SUMMARY")
        report_lines.append("-" * 70)
        report_lines.append(f"Total samples:        {self.stats['total_samples']}")
        report_lines.append(f"Valid samples:        {self.stats['valid_samples']}")
        report_lines.append(f"Invalid samples:      {self.stats['invalid_samples']}")
        report_lines.append(f"Total duration:       {self.stats['total_duration']:.2f}s ({self.stats['total_duration']/3600:.2f}h)")
        report_lines.append("")

        # Skip reasons
        if self.stats["skipped_reasons"]:
            report_lines.append("SKIP REASONS")
            report_lines.append("-" * 70)
            for reason, count in sorted(self.stats["skipped_reasons"].items(), key=lambda x: -x[1]):
                report_lines.append(f"  {reason:40s}: {count}")
            report_lines.append("")

        # Mel statistics
        if self.stats["mel_stats"]:
            report_lines.append("MEL SPECTROGRAM STATISTICS")
            report_lines.append("-" * 70)
            mel_mean = np.array(self.stats["mel_stats"]["mean"])
            mel_std = np.array(self.stats["mel_stats"]["std"])
            report_lines.append(f"  Mean (per channel): min={mel_mean.min():.4f}, max={mel_mean.max():.4f}, mean={mel_mean.mean():.4f}")
            report_lines.append(f"  Std (per channel):  min={mel_std.min():.4f}, max={mel_std.max():.4f}, mean={mel_std.mean():.4f}")
            report_lines.append(f"  Overall min:        {self.stats['mel_stats']['min']:.4f}")
            report_lines.append(f"  Overall max:        {self.stats['mel_stats']['max']:.4f}")
            report_lines.append("")

        # Speaker statistics
        if self.stats["speaker_stats"]:
            report_lines.append("SPEAKER STATISTICS")
            report_lines.append("-" * 70)
            for speaker_id, stats in sorted(self.stats["speaker_stats"].items()):
                genders = ", ".join(stats["genders"]) if stats["genders"] else "unknown"
                report_lines.append(
                    f"  {speaker_id:20s}: {stats['count']:4d} samples, "
                    f"{stats['total_duration']:7.2f}s, gender(s): {genders}"
                )
            report_lines.append("")

        # Invalid samples (first 20)
        if self.invalid_samples:
            report_lines.append("INVALID SAMPLES (first 20)")
            report_lines.append("-" * 70)
            for filename, reason in self.invalid_samples[:20]:
                report_lines.append(f"  {filename:40s} - {reason}")
            if len(self.invalid_samples) > 20:
                report_lines.append(f"  ... and {len(self.invalid_samples) - 20} more")
            report_lines.append("")

        report_lines.append("=" * 70)

        # Save report
        report_text = "\n".join(report_lines)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text)

        logger.info(f"Report saved to {report_path}")
        print(report_text)

        # Save stats as JSON
        stats_path = self.output_dir / "dataset_stats.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(self.stats, f, indent=2)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Prepare Hindi TTS dataset")
    parser.add_argument("audio_dir", help="Directory containing audio files")
    parser.add_argument("transcript_file", help="File with transcripts (audio_filename\\ttext\\tspeaker_id\\tgender)")
    parser.add_argument("--output_dir", default="./dataset_manifests", help="Output directory")
    parser.add_argument("--sample_rate", type=int, default=44100, help="Target sample rate")
    parser.add_argument("--min_duration", type=float, default=0.5, help="Minimum audio duration")
    parser.add_argument("--max_duration", type=float, default=30.0, help="Maximum audio duration")
    parser.add_argument("--max_silence", type=float, default=0.5, help="Maximum silence ratio")
    parser.add_argument("--train_split", type=float, default=0.9, help="Train split ratio")
    parser.add_argument("--val_split", type=float, default=0.05, help="Val split ratio")
    parser.add_argument("--test_split", type=float, default=0.05, help="Test split ratio")

    args = parser.parse_args()

    # Create preparer
    preparer = DatasetPreparer(
        audio_dir=args.audio_dir,
        transcript_file=args.transcript_file,
        output_dir=args.output_dir,
        sample_rate=args.sample_rate,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        max_silence_ratio=args.max_silence,
    )

    # Prepare dataset
    preparer.prepare_dataset(
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
    )

    logger.info("Dataset preparation complete!")


if __name__ == "__main__":
    main()
