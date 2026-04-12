"""
Comprehensive evaluation script for SupertonicTTS Hindi with W&B integration.

Evaluates synthesized speech against ground truth using:
- WER/CER: Word and character error rate
- UTMOS: Speech quality prediction
- PESQ/STOI: Objective quality metrics
- Speaker Similarity: Speaker consistency
- RTF: Real-time factor

Supports evaluation on subsets by duration, speaker, or gender.
Logs all metrics and samples to Weights & Biases for experiment tracking.
"""

import logging
import json
import csv
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import argparse
from datetime import datetime
import time

import numpy as np
import pandas as pd
import torch

from .inference import SupertonicTTSInference
from .metrics import MetricComputer
from supertonic_hindi_tts.utils.logging_utils import ExperimentTracker

logger = logging.getLogger(__name__)


class SupertonicEvaluator:
    """Complete evaluation pipeline for SupertonicTTS with W&B tracking."""

    def __init__(
        self,
        autoencoder_path: str,
        text_to_latent_path: str,
        duration_predictor_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize evaluator.

        Args:
            autoencoder_path: Path to speech autoencoder
            text_to_latent_path: Path to text-to-latent model
            duration_predictor_path: Path to duration predictor
            device: Computation device
        """
        self.device = device

        # Initialize inference
        logger.info("Initializing inference pipeline...")
        self.inference = SupertonicTTSInference(
            autoencoder_path=autoencoder_path,
            text_to_latent_path=text_to_latent_path,
            duration_predictor_path=duration_predictor_path,
            device=device,
        )

        # Initialize metrics
        logger.info("Initializing metrics computer...")
        self.metric_computer = MetricComputer(device=device)

        self.tracker = None

    def evaluate_manifest(
        self,
        manifest_path: str,
        output_dir: str,
        reference_speaker: Optional[str] = None,
        duration_range: Optional[Tuple[float, float]] = None,
        gender_filter: Optional[str] = None,
        max_samples: Optional[int] = None,
        cfg_scale: float = 3.0,
        duration_scale: float = 1.0,
        n_steps: int = 32,
        config: Optional[Dict] = None,
    ) -> Dict:
        """Evaluate on a test manifest with W&B logging.

        Manifest format (JSON Lines or CSV):
        - text: Ground truth text
        - reference_audio: Path to reference audio
        - audio: Path to ground truth audio
        - speaker: Speaker ID
        - gender: Speaker gender (optional)
        - duration: Audio duration in seconds (optional)

        Args:
            manifest_path: Path to test manifest
            output_dir: Directory to save results
            reference_speaker: Filter to specific speaker (optional)
            duration_range: Tuple of (min_sec, max_sec) to filter by duration
            gender_filter: Filter by speaker gender ("M"/"F")
            max_samples: Max samples to evaluate
            cfg_scale: Classifier-free guidance scale
            duration_scale: Duration scaling factor
            n_steps: Number of ODE solver steps
            config: Configuration dict for W&B tracking

        Returns:
            Dictionary with aggregated metrics
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize W&B tracker
        if config is None:
            config = {
                "cfg_scale": cfg_scale,
                "duration_scale": duration_scale,
                "n_steps": n_steps,
                "device": self.device,
            }

        self.tracker = ExperimentTracker(
            log_dir=str(output_dir / "logs"),
            stage="evaluation",
            config=config,
            use_tensorboard=False,
            use_wandb=True,
            wandb_tags=["evaluation", "hindi-tts"],
        )
        logger.info("W&B ExperimentTracker initialized")

        # Load manifest
        logger.info(f"Loading manifest from {manifest_path}")
        samples = self._load_manifest(manifest_path)
        logger.info(f"Loaded {len(samples)} samples")

        # Apply filters
        samples = self._filter_samples(
            samples,
            reference_speaker=reference_speaker,
            duration_range=duration_range,
            gender_filter=gender_filter,
            max_samples=max_samples,
        )
        logger.info(f"After filtering: {len(samples)} samples")

        # Evaluate each sample
        results = []
        per_sample_rows = []
        audio_samples = []
        mel_comparison_count = 0

        for i, sample in enumerate(samples):
            logger.info(f"\n[{i+1}/{len(samples)}] Evaluating: {sample.get('text', '')[:50]}...")

            try:
                sample_results = self._evaluate_sample(
                    sample=sample,
                    output_dir=output_dir,
                    cfg_scale=cfg_scale,
                    duration_scale=duration_scale,
                    n_steps=n_steps,
                    sample_idx=i,
                )
                results.append(sample_results)

                # Prepare row for W&B table
                per_sample_rows.append({
                    "text": sample.get("text", "")[:100],
                    "speaker_id": sample.get("speaker", ""),
                    "gender": sample.get("gender", ""),
                    "duration": sample.get("duration", 0.0),
                    "wer": sample_results.get("wer", np.nan),
                    "cer": sample_results.get("cer", np.nan),
                    "utmos": sample_results.get("utmos", np.nan),
                    "pesq": sample_results.get("pesq", np.nan),
                    "stoi": sample_results.get("stoi", np.nan),
                    "sim": sample_results.get("speaker_similarity", np.nan),
                    "rtf": sample_results.get("rtf", np.nan),
                })

                # Collect audio samples (up to 5)
                if len(audio_samples) < 5 and "generated_audio_path" in sample_results:
                    audio_samples.append((
                        sample_results["generated_audio_path"],
                        sample.get("text", "")[:100]
                    ))

                # Log mel comparison for first 3 samples
                if mel_comparison_count < 3 and "mel_gt" in sample_results and "mel_pred" in sample_results:
                    try:
                        self.tracker.log_mel_comparison(
                            "evaluation/mel_comparison",
                            mel_gt=sample_results["mel_gt"],
                            mel_pred=sample_results["mel_pred"],
                            step=mel_comparison_count,
                            caption=f"Sample {i}: {sample.get('text', '')[:50]}"
                        )
                        mel_comparison_count += 1
                    except Exception as e:
                        logger.warning(f"Could not log mel comparison: {e}")

            except Exception as e:
                logger.error(f"Error evaluating sample {i}: {e}", exc_info=True)
                results.append({
                    "text": sample.get("text", ""),
                    "speaker": sample.get("speaker", ""),
                    "error": str(e),
                })

        # Log per-sample metrics table
        if per_sample_rows:
            logger.info("Logging per-sample metrics table to W&B...")
            try:
                self.tracker.log_evaluation_table(
                    rows=per_sample_rows,
                    step=0,
                    tag="evaluation/per_sample"
                )
            except Exception as e:
                logger.warning(f"Could not log evaluation table: {e}")

        # Log synthesized audio samples
        if audio_samples:
            logger.info(f"Logging {len(audio_samples)} audio samples to W&B...")
            try:
                waveforms = []
                captions = []
                sample_rate = 44100  # Default sample rate

                for audio_path, caption in audio_samples:
                    audio_path = Path(audio_path)
                    if audio_path.exists():
                        # Load audio waveform
                        import torchaudio
                        waveform, sr = torchaudio.load(str(audio_path))
                        waveforms.append(waveform.numpy())
                        captions.append(caption)
                        sample_rate = sr

                if waveforms:
                    self.tracker.log_audio_batch(
                        "evaluation/synthesised_audio",
                        waveforms=waveforms,
                        step=0,
                        sr=sample_rate,
                        captions=captions,
                    )
            except Exception as e:
                logger.warning(f"Could not log audio batch: {e}")

        # Save detailed results
        results_csv = output_dir / "results_detailed.csv"
        self._save_detailed_results(results, results_csv)

        # Compute and save summary
        summary = self._compute_summary(results)

        # Log aggregate metrics to W&B
        if not ("error" in summary and len(summary) == 1):
            logger.info("Logging aggregate metrics to W&B...")
            try:
                aggregate_metrics = {
                    "eval/wer": summary.get("wer", {}).get("mean", 0),
                    "eval/cer": summary.get("cer", {}).get("mean", 0),
                    "eval/utmos": summary.get("utmos", {}).get("mean", 0),
                    "eval/pesq": summary.get("pesq", {}).get("mean", 0),
                    "eval/stoi": summary.get("stoi", {}).get("mean", 0),
                    "eval/sim": summary.get("speaker_similarity", {}).get("mean", 0),
                    "eval/rtf": summary.get("rtf", {}).get("mean", 0),
                }
                self.tracker.log_scalars(aggregate_metrics, step=0)
            except Exception as e:
                logger.warning(f"Could not log aggregate metrics: {e}")

            # Log per-speaker breakdown
            if "by_speaker" in summary and summary["by_speaker"]:
                logger.info("Logging per-speaker metrics to W&B...")
                try:
                    speaker_metrics_dict = {}
                    for speaker_id, metrics in summary["by_speaker"].items():
                        speaker_metrics_dict[speaker_id] = {
                            "wer": metrics.get("wer", {}).get("mean", 0),
                            "cer": metrics.get("cer", {}).get("mean", 0),
                            "utmos": metrics.get("utmos", {}).get("mean", 0),
                            "pesq": metrics.get("pesq", {}).get("mean", 0),
                            "stoi": metrics.get("stoi", {}).get("mean", 0),
                            "speaker_similarity": metrics.get("speaker_similarity", {}).get("mean", 0),
                            "rtf": metrics.get("rtf", {}).get("mean", 0),
                        }
                    self.tracker.log_per_speaker_table(speaker_metrics_dict, step=0)
                except Exception as e:
                    logger.warning(f"Could not log per-speaker table: {e}")

        summary_json = output_dir / "summary.json"
        self._save_summary(summary, summary_json)

        logger.info(f"\nEvaluation complete!")
        logger.info(f"Results saved to {output_dir}")

        # Close W&B tracker
        if self.tracker:
            self.tracker.close()
            logger.info("W&B tracker closed")

        return summary

    def _load_manifest(self, manifest_path: str) -> List[Dict]:
        """Load manifest from JSON Lines or CSV.

        Args:
            manifest_path: Path to manifest file

        Returns:
            List of sample dictionaries
        """
        manifest_path = Path(manifest_path)

        samples = []
        if manifest_path.suffix == ".jsonl":
            import json
            with open(manifest_path) as f:
                for line in f:
                    if line.strip():
                        samples.append(json.loads(line))

        elif manifest_path.suffix in [".csv", ".tsv"]:
            delimiter = "\t" if manifest_path.suffix == ".tsv" else ","
            df = pd.read_csv(manifest_path, delimiter=delimiter)
            samples = df.to_dict("records")

        elif manifest_path.suffix == ".json":
            import json
            with open(manifest_path) as f:
                data = json.load(f)
                samples = data if isinstance(data, list) else data.get("samples", [])

        return samples

    def _filter_samples(
        self,
        samples: List[Dict],
        reference_speaker: Optional[str] = None,
        duration_range: Optional[Tuple[float, float]] = None,
        gender_filter: Optional[str] = None,
        max_samples: Optional[int] = None,
    ) -> List[Dict]:
        """Filter samples based on criteria.

        Args:
            samples: List of sample dicts
            reference_speaker: Filter by speaker ID
            duration_range: Filter by duration (min, max) in seconds
            gender_filter: Filter by gender
            max_samples: Max samples to keep

        Returns:
            Filtered list of samples
        """
        filtered = samples

        # Filter by speaker
        if reference_speaker:
            filtered = [s for s in filtered if s.get("speaker") == reference_speaker]
            logger.info(f"Filtered by speaker '{reference_speaker}': {len(filtered)} samples")

        # Filter by duration
        if duration_range:
            min_dur, max_dur = duration_range
            filtered = [
                s for s in filtered
                if min_dur <= s.get("duration", float('inf')) <= max_dur
            ]
            logger.info(f"Filtered by duration {duration_range}: {len(filtered)} samples")

        # Filter by gender
        if gender_filter:
            filtered = [s for s in filtered if s.get("gender") == gender_filter]
            logger.info(f"Filtered by gender '{gender_filter}': {len(filtered)} samples")

        # Limit samples
        if max_samples:
            filtered = filtered[:max_samples]
            logger.info(f"Limited to {max_samples} samples")

        return filtered

    def _evaluate_sample(
        self,
        sample: Dict,
        output_dir: Path,
        cfg_scale: float = 3.0,
        duration_scale: float = 1.0,
        n_steps: int = 32,
        sample_idx: int = 0,
    ) -> Dict:
        """Evaluate a single sample.

        Args:
            sample: Sample dictionary
            output_dir: Directory to save generated audio
            cfg_scale: CFG scale for synthesis
            duration_scale: Duration scaling
            n_steps: ODE steps
            sample_idx: Index of sample for logging

        Returns:
            Dictionary with all computed metrics
        """
        text = sample.get("text", "")
        ref_audio = sample.get("reference_audio", "")
        ground_truth_audio = sample.get("audio", "")

        if not text or not ref_audio or not ground_truth_audio:
            raise ValueError(f"Missing required fields in sample: {sample}")

        # Synthesize
        logger.debug(f"Synthesizing: {text[:50]}...")
        gen_start = time.time()
        waveform, rtf = self.inference.synthesize(
            text=text,
            reference_audio_path=ref_audio,
            duration_scale=duration_scale,
            cfg_scale=cfg_scale,
            n_steps=n_steps,
            return_rtf=True,
        )
        gen_time = time.time() - gen_start

        # Save generated audio
        speaker = sample.get("speaker", "unknown")
        sample_id = sample.get("id", "")
        gen_audio_name = f"{speaker}_{sample_id}_gen.wav"
        gen_audio_path = output_dir / gen_audio_name
        self.inference.save_audio(waveform, str(gen_audio_path))

        # Compute metrics
        logger.debug("Computing metrics...")
        metrics = self.metric_computer.compute_all(
            gen_audio_path=str(gen_audio_path),
            ref_audio_path=ground_truth_audio,
            ground_truth_text=text,
            generation_time=gen_time,
        )

        # Add sample info
        metrics.update({
            "sample_id": sample_id,
            "speaker": speaker,
            "gender": sample.get("gender", ""),
            "duration": sample.get("duration", ""),
            "generated_audio_path": str(gen_audio_path),
        })

        # Attempt to extract mel spectrograms for logging
        try:
            mel_gt = self._compute_mel_spectrogram(ground_truth_audio)
            mel_pred = self._compute_mel_spectrogram(str(gen_audio_path))
            metrics["mel_gt"] = mel_gt
            metrics["mel_pred"] = mel_pred
        except Exception as e:
            logger.debug(f"Could not extract mel spectrograms: {e}")

        return metrics

    def _compute_mel_spectrogram(self, audio_path: str) -> Optional[np.ndarray]:
        """Compute mel spectrogram from audio.

        Args:
            audio_path: Path to audio file

        Returns:
            Mel spectrogram as numpy array or None
        """
        try:
            import torchaudio
            import torchaudio.transforms as T

            # Load audio
            waveform, sr = torchaudio.load(audio_path)

            # Resample if needed
            if sr != 22050:
                resampler = T.Resample(sr, 22050)
                waveform = resampler(waveform)

            # Compute mel spectrogram
            mel_transform = T.MelSpectrogram(
                sample_rate=22050,
                n_mels=80,
                n_fft=1024,
                hop_length=256,
            )
            mel_spec = mel_transform(waveform)
            mel_spec = T.AmplitudeToDB()(mel_spec)

            return mel_spec.squeeze(0).numpy()
        except Exception as e:
            logger.debug(f"Could not compute mel spectrogram: {e}")
            return None

    def _save_detailed_results(self, results: List[Dict], output_path: Path):
        """Save detailed per-sample results to CSV.

        Args:
            results: List of result dictionaries
            output_path: Path to save CSV
        """
        # Flatten and prepare for CSV
        csv_results = []
        for result in results:
            csv_row = {
                "sample_id": result.get("sample_id", ""),
                "speaker": result.get("speaker", ""),
                "gender": result.get("gender", ""),
                "ground_truth": result.get("ground_truth", "")[:100],
                "transcribed": result.get("transcribed_text", "")[:100],
                "wer": result.get("wer", ""),
                "cer": result.get("cer", ""),
                "utmos": result.get("utmos", ""),
                "pesq": result.get("pesq", ""),
                "stoi": result.get("stoi", ""),
                "speaker_similarity": result.get("speaker_similarity", ""),
                "rtf": result.get("rtf", ""),
                "generation_time": result.get("generation_time", ""),
                "error": result.get("error", ""),
            }
            csv_results.append(csv_row)

        # Write CSV
        if csv_results:
            df = pd.DataFrame(csv_results)
            df.to_csv(output_path, index=False)
            logger.info(f"Saved detailed results to {output_path}")

    def _compute_summary(self, results: List[Dict]) -> Dict:
        """Compute summary statistics.

        Args:
            results: List of result dictionaries

        Returns:
            Dictionary with aggregated metrics
        """
        # Filter out results with errors
        valid_results = [r for r in results if "error" not in r or not r["error"]]

        if not valid_results:
            return {"error": "No valid results to summarize"}

        # Extract metrics
        metrics_dict = {
            "wer": [r.get("wer", np.nan) for r in valid_results],
            "cer": [r.get("cer", np.nan) for r in valid_results],
            "utmos": [r.get("utmos", np.nan) for r in valid_results],
            "pesq": [r.get("pesq", np.nan) for r in valid_results],
            "stoi": [r.get("stoi", np.nan) for r in valid_results],
            "speaker_similarity": [r.get("speaker_similarity", np.nan) for r in valid_results],
            "rtf": [r.get("rtf", np.nan) for r in valid_results],
        }

        # Compute stats
        summary = {
            "timestamp": datetime.now().isoformat(),
            "num_samples": len(valid_results),
            "num_errors": len(results) - len(valid_results),
        }

        for metric_name, values in metrics_dict.items():
            valid_values = [v for v in values if not np.isnan(v)]
            if valid_values:
                summary[metric_name] = {
                    "mean": float(np.mean(valid_values)),
                    "std": float(np.std(valid_values)),
                    "min": float(np.min(valid_values)),
                    "max": float(np.max(valid_values)),
                    "median": float(np.median(valid_values)),
                }

        # Summary by speaker
        speakers = {}
        for result in valid_results:
            speaker = result.get("speaker", "unknown")
            if speaker not in speakers:
                speakers[speaker] = []
            speakers[speaker].append(result)

        summary["by_speaker"] = {}
        for speaker, speaker_results in speakers.items():
            speaker_metrics = {
                "wer": [r.get("wer", np.nan) for r in speaker_results],
                "cer": [r.get("cer", np.nan) for r in speaker_results],
                "utmos": [r.get("utmos", np.nan) for r in speaker_results],
                "pesq": [r.get("pesq", np.nan) for r in speaker_results],
                "stoi": [r.get("stoi", np.nan) for r in speaker_results],
                "speaker_similarity": [r.get("speaker_similarity", np.nan) for r in speaker_results],
                "rtf": [r.get("rtf", np.nan) for r in speaker_results],
            }
            speaker_summary = {}
            for metric_name, values in speaker_metrics.items():
                valid_values = [v for v in values if not np.isnan(v)]
                if valid_values:
                    speaker_summary[metric_name] = {
                        "mean": float(np.mean(valid_values)),
                        "std": float(np.std(valid_values)),
                    }
            summary["by_speaker"][speaker] = speaker_summary

        return summary

    def _save_summary(self, summary: Dict, output_path: Path):
        """Save summary as JSON.

        Args:
            summary: Summary dictionary
            output_path: Path to save JSON
        """
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved summary to {output_path}")

        # Print summary to console
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Timestamp: {summary.get('timestamp', '')}")
        print(f"Samples: {summary.get('num_samples', 0)} valid, {summary.get('num_errors', 0)} errors")
        print("\nAggregate Metrics:")
        for metric in ["wer", "cer", "utmos", "pesq", "stoi", "speaker_similarity", "rtf"]:
            if metric in summary:
                stats = summary[metric]
                print(f"  {metric.upper()}:")
                print(f"    Mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
                print(f"    Range: [{stats['min']:.4f}, {stats['max']:.4f}]")

        if "by_speaker" in summary:
            print("\nBy Speaker:")
            for speaker, metrics in summary["by_speaker"].items():
                print(f"  {speaker}:")
                for metric_name, stats in metrics.items():
                    print(f"    {metric_name}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        print("="*60 + "\n")


def main():
    """CLI interface for evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate SupertonicTTS Hindi"
    )

    parser.add_argument(
        "--autoencoder",
        required=True,
        help="Path to speech autoencoder checkpoint"
    )
    parser.add_argument(
        "--text-to-latent",
        required=True,
        help="Path to text-to-latent checkpoint"
    )
    parser.add_argument(
        "--duration-predictor",
        required=True,
        help="Path to duration predictor checkpoint"
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to test manifest (JSONL/CSV/JSON)"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for results"
    )
    parser.add_argument(
        "--speaker",
        help="Filter to specific speaker"
    )
    parser.add_argument(
        "--duration-min",
        type=float,
        help="Minimum audio duration in seconds"
    )
    parser.add_argument(
        "--duration-max",
        type=float,
        help="Maximum audio duration in seconds"
    )
    parser.add_argument(
        "--gender",
        choices=["M", "F"],
        help="Filter by speaker gender"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum samples to evaluate"
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=3.0,
        help="Classifier-free guidance scale"
    )
    parser.add_argument(
        "--duration-scale",
        type=float,
        default=1.0,
        help="Duration scaling factor"
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=32,
        help="Number of ODE solver steps"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Initialize evaluator
    evaluator = SupertonicEvaluator(
        autoencoder_path=args.autoencoder,
        text_to_latent_path=args.text_to_latent,
        duration_predictor_path=args.duration_predictor,
        device=args.device,
    )

    # Prepare filters
    duration_range = None
    if args.duration_min is not None or args.duration_max is not None:
        duration_range = (
            args.duration_min or 0.0,
            args.duration_max or float('inf')
        )

    # Build config for W&B
    config = {
        "cfg_scale": args.cfg_scale,
        "duration_scale": args.duration_scale,
        "n_steps": args.n_steps,
        "device": args.device,
        "manifest": manifest_path,
        "reference_speaker": args.speaker,
        "gender_filter": args.gender,
        "max_samples": args.max_samples,
    }

    # Run evaluation
    evaluator.evaluate_manifest(
        manifest_path=args.manifest,
        output_dir=args.output,
        reference_speaker=args.speaker,
        duration_range=duration_range,
        gender_filter=args.gender,
        max_samples=args.max_samples,
        cfg_scale=args.cfg_scale,
        duration_scale=args.duration_scale,
        n_steps=args.n_steps,
        config=config,
    )


if __name__ == "__main__":
    main()
