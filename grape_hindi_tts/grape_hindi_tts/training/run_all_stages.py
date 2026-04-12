"""
Master training orchestration script for SupertonicTTS.

Runs all 3 training stages sequentially with proper checkpointing and logging.

Usage:
    python run_all_stages.py \
        --output_dir ./outputs \
        --stage1_config config_autoencoder.yaml \
        --stage2_config config_text_to_latent.yaml \
        --stage3_config config_duration.yaml \
        --resume_stage 2  # Optional: resume from specific stage
"""
import os
import sys
import argparse
import logging
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime


class TrainingOrchestrator:
    """Orchestrates training of all 3 stages."""

    def __init__(
        self,
        output_dir: str,
        stage1_config: str,
        stage2_config: str,
        stage3_config: str,
        resume_stage: Optional[int] = None
    ):
        self.output_dir = output_dir
        self.stage1_config = stage1_config
        self.stage2_config = stage2_config
        self.stage3_config = stage3_config
        self.resume_stage = resume_stage

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Setup logging
        self.setup_logging()

        # Training metadata
        self.metadata = {
            "start_time": datetime.now().isoformat(),
            "stages": {}
        }

    def setup_logging(self):
        """Setup logging for orchestration."""
        log_dir = os.path.join(self.output_dir, "orchestration_logs")
        os.makedirs(log_dir, exist_ok=True)

        log_file = os.path.join(log_dir, "training_orchestration.log")

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )

        logging.info(f"Logging initialized. Log file: {log_file}")

    def get_latest_checkpoint(self, stage_dir: str, prefix: str) -> Optional[str]:
        """Get latest checkpoint from stage directory."""
        ckpt_dir = os.path.join(stage_dir, "checkpoints")
        if not os.path.exists(ckpt_dir):
            return None

        checkpoints = [f for f in os.listdir(ckpt_dir) if f.startswith(prefix) and not "_best" in f]
        if not checkpoints:
            return None

        # Sort by step number
        checkpoints.sort(
            key=lambda x: int(x.split("_")[-1].split(".")[0]),
            reverse=True
        )

        return os.path.join(ckpt_dir, checkpoints[0])

    def run_stage(
        self,
        stage: int,
        script_name: str,
        config_file: str,
        stage_dir: str,
        resume_checkpoint: Optional[str] = None
    ) -> bool:
        """
        Run a single training stage.

        Args:
            stage: Stage number (1, 2, or 3)
            script_name: Name of training script
            config_file: Config file path
            stage_dir: Output directory for this stage
            resume_checkpoint: Checkpoint to resume from (optional)

        Returns:
            True if successful, False otherwise
        """
        logging.info(f"{'='*80}")
        logging.info(f"Starting Stage {stage}: {script_name}")
        logging.info(f"{'='*80}")

        stage_start_time = datetime.now()

        # Build command
        cmd = [
            "python",
            script_name,
            "--config", config_file,
            "--output_dir", stage_dir
        ]

        if resume_checkpoint:
            logging.info(f"Resuming from checkpoint: {resume_checkpoint}")
            cmd.extend(["--resume", resume_checkpoint])

        # Run training
        try:
            result = subprocess.run(
                cmd,
                cwd=os.path.dirname(os.path.abspath(__file__)),
                capture_output=False
            )

            stage_end_time = datetime.now()
            duration = stage_end_time - stage_start_time

            if result.returncode != 0:
                logging.error(f"Stage {stage} failed with return code {result.returncode}")
                return False

            logging.info(f"Stage {stage} completed successfully in {duration}")

            # Record metadata
            self.metadata["stages"][f"stage{stage}"] = {
                "status": "completed",
                "start_time": stage_start_time.isoformat(),
                "end_time": stage_end_time.isoformat(),
                "duration_seconds": duration.total_seconds()
            }

            return True

        except KeyboardInterrupt:
            logging.warning(f"Stage {stage} interrupted by user")
            self.metadata["stages"][f"stage{stage}"] = {
                "status": "interrupted",
                "start_time": stage_start_time.isoformat(),
                "end_time": datetime.now().isoformat()
            }
            return False

        except Exception as e:
            logging.error(f"Stage {stage} failed with error: {e}")
            self.metadata["stages"][f"stage{stage}"] = {
                "status": "failed",
                "error": str(e),
                "start_time": stage_start_time.isoformat(),
                "end_time": datetime.now().isoformat()
            }
            return False

    def run_all_stages(self) -> bool:
        """Run all training stages sequentially."""
        logging.info(f"Starting SupertonicTTS training pipeline")
        logging.info(f"Output directory: {self.output_dir}")

        stages = [
            (1, "train_autoencoder.py", self.stage1_config, "autoencoder"),
            (2, "train_text_to_latent.py", self.stage2_config, "text_to_latent"),
            (3, "train_duration.py", self.stage3_config, "duration"),
        ]

        for stage, script, config, stage_name in stages:
            # Check if should resume this stage
            if self.resume_stage is not None and stage < self.resume_stage:
                logging.info(f"Skipping Stage {stage} (resume_stage={self.resume_stage})")
                continue

            stage_dir = os.path.join(self.output_dir, f"stage{stage}_{stage_name}")

            # Check for resume checkpoint
            resume_checkpoint = None
            if self.resume_stage == stage:
                prefix = f"{stage_name}_step_"
                resume_checkpoint = self.get_latest_checkpoint(stage_dir, prefix)

            # Run stage
            success = self.run_stage(
                stage,
                script,
                config,
                stage_dir,
                resume_checkpoint=resume_checkpoint
            )

            if not success:
                if self.resume_stage == stage:
                    # If explicitly resuming, continue anyway
                    logging.warning(f"Stage {stage} did not complete, but continuing...")
                else:
                    logging.error(f"Stage {stage} failed, stopping pipeline")
                    return False

        return True

    def save_metadata(self):
        """Save training metadata."""
        self.metadata["end_time"] = datetime.now().isoformat()

        metadata_path = os.path.join(self.output_dir, "training_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)

        logging.info(f"Training metadata saved to {metadata_path}")

    def print_summary(self):
        """Print training summary."""
        logging.info(f"{'='*80}")
        logging.info("Training Summary")
        logging.info(f"{'='*80}")

        for stage_key, stage_info in self.metadata.get("stages", {}).items():
            status = stage_info.get("status", "unknown")
            logging.info(f"{stage_key}: {status}")

            if "duration_seconds" in stage_info:
                hours = stage_info["duration_seconds"] / 3600
                logging.info(f"  Duration: {hours:.2f} hours")

        if "end_time" in self.metadata and "start_time" in self.metadata:
            start = datetime.fromisoformat(self.metadata["start_time"])
            end = datetime.fromisoformat(self.metadata["end_time"])
            total_duration = end - start
            logging.info(f"Total training time: {total_duration}")


def main():
    parser = argparse.ArgumentParser(
        description="Run SupertonicTTS training pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all stages
  python run_all_stages.py \\
      --output_dir ./outputs \\
      --stage1_config config_autoencoder.yaml \\
      --stage2_config config_text_to_latent.yaml \\
      --stage3_config config_duration.yaml

  # Resume from stage 2
  python run_all_stages.py \\
      --output_dir ./outputs \\
      --stage1_config config_autoencoder.yaml \\
      --stage2_config config_text_to_latent.yaml \\
      --stage3_config config_duration.yaml \\
      --resume_stage 2
        """
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Base output directory"
    )
    parser.add_argument(
        "--stage1_config",
        type=str,
        required=True,
        help="Config file for Stage 1 (Autoencoder)"
    )
    parser.add_argument(
        "--stage2_config",
        type=str,
        required=True,
        help="Config file for Stage 2 (Text-to-Latent)"
    )
    parser.add_argument(
        "--stage3_config",
        type=str,
        required=True,
        help="Config file for Stage 3 (Duration)"
    )
    parser.add_argument(
        "--resume_stage",
        type=int,
        choices=[1, 2, 3],
        help="Resume from this stage (1-3)"
    )

    args = parser.parse_args()

    # Create orchestrator
    orchestrator = TrainingOrchestrator(
        output_dir=args.output_dir,
        stage1_config=args.stage1_config,
        stage2_config=args.stage2_config,
        stage3_config=args.stage3_config,
        resume_stage=args.resume_stage
    )

    # Run all stages
    try:
        success = orchestrator.run_all_stages()

        # Save metadata and print summary
        orchestrator.save_metadata()
        orchestrator.print_summary()

        if success:
            logging.info("All training stages completed successfully!")
            sys.exit(0)
        else:
            logging.error("Training pipeline failed!")
            sys.exit(1)

    except KeyboardInterrupt:
        logging.warning("Training interrupted by user")
        orchestrator.save_metadata()
        orchestrator.print_summary()
        sys.exit(130)


if __name__ == "__main__":
    main()
