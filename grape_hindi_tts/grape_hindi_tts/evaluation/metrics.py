"""
Metric implementations for SupertonicTTS evaluation.

Includes:
- WER/CER: Word/Character Error Rate using Whisper ASR
- Speaker similarity: Cosine similarity using speaker embeddings
- UTMOS: Universal TMOS model for MOS prediction
- PESQ: Perceptual Evaluation of Speech Quality
- STOI: Short-Time Objective Intelligibility
- RTF: Real-Time Factor
"""

import logging
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import librosa
import torch
import torchaudio
from jiwer import wer, cer
import warnings

logger = logging.getLogger(__name__)


class MetricComputer:
    """Compute various quality metrics for TTS evaluation."""

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """Initialize metric computer.

        Args:
            device: Device for model inference
        """
        self.device = device
        self._whisper_model = None
        self._utmos_model = None
        self._speaker_encoder = None

    @property
    def whisper_model(self):
        """Lazy load Whisper model."""
        if self._whisper_model is None:
            logger.info("Loading Whisper large-v2 model...")
            try:
                import whisper
                self._whisper_model = whisper.load_model("large-v2", device=self.device)
            except ImportError:
                logger.error("whisper not installed. Install with: pip install openai-whisper")
                raise

        return self._whisper_model

    @property
    def utmos_model(self):
        """Lazy load UTMOS model."""
        if self._utmos_model is None:
            logger.info("Loading UTMOS model...")
            try:
                # UTMOS implementation
                from google_universal_tts_mos import GoogLEUniversalTTSMOS
                self._utmos_model = GoogLEUniversalTTSMOS(device=self.device)
            except ImportError:
                logger.warning("UTMOS model not available. Install: pip install google-universal-tts-mos")
                return None

        return self._utmos_model

    @property
    def speaker_encoder(self):
        """Lazy load speaker encoder (ResNet TDNN based)."""
        if self._speaker_encoder is None:
            logger.info("Loading speaker encoder...")
            try:
                import speechbrain as sb
                self._speaker_encoder = sb.resource.HuggingFaceHub.load_or_create(
                    model_id="speechbrain/spkrec-resnet-voxceleb",
                    savedir="speechbrain_checkpoints"
                )
            except ImportError:
                logger.warning("SpeechBrain not installed. Install: pip install speechbrain")
                # Fallback to resemblyzer
                try:
                    from resemblyzer import VoiceEncoder
                    self._speaker_encoder = VoiceEncoder()
                except ImportError:
                    logger.warning("Resemblyzer not installed. Install: pip install resemblyzer")
                    return None

        return self._speaker_encoder

    def transcribe_audio(
        self,
        audio_path: str,
        language: str = "hi",
    ) -> str:
        """Transcribe audio using Whisper ASR.

        Args:
            audio_path: Path to audio file
            language: Language code ("hi" for Hindi)

        Returns:
            Transcribed text
        """
        try:
            result = self.whisper_model.transcribe(
                audio_path,
                language=language,
                fp16=False
            )
            return result["text"].strip()
        except Exception as e:
            logger.error(f"Error transcribing {audio_path}: {e}")
            return ""

    def compute_wer(
        self,
        reference_text: str,
        hypothesis_text: str
    ) -> float:
        """Compute Word Error Rate.

        Args:
            reference_text: Ground truth text
            hypothesis_text: Recognized text

        Returns:
            WER as fraction (0-1)
        """
        if not reference_text or not hypothesis_text:
            return 1.0

        try:
            error_rate = wer(reference_text, hypothesis_text)
            return min(error_rate, 1.0)  # Cap at 1.0
        except Exception as e:
            logger.error(f"Error computing WER: {e}")
            return 1.0

    def compute_cer(
        self,
        reference_text: str,
        hypothesis_text: str
    ) -> float:
        """Compute Character Error Rate.

        Args:
            reference_text: Ground truth text
            hypothesis_text: Recognized text

        Returns:
            CER as fraction (0-1)
        """
        if not reference_text or not hypothesis_text:
            return 1.0

        try:
            error_rate = cer(reference_text, hypothesis_text)
            return min(error_rate, 1.0)  # Cap at 1.0
        except Exception as e:
            logger.error(f"Error computing CER: {e}")
            return 1.0

    def compute_speaker_similarity(
        self,
        ref_audio_path: str,
        gen_audio_path: str,
        sr: int = 16000
    ) -> float:
        """Compute speaker similarity using speaker embeddings.

        Uses SpeechBrain ResNet TDNN or Resemblyzer.

        Args:
            ref_audio_path: Path to reference audio
            gen_audio_path: Path to generated audio
            sr: Sample rate for audio loading

        Returns:
            Cosine similarity (0-1)
        """
        try:
            encoder = self.speaker_encoder
            if encoder is None:
                logger.warning("Speaker encoder unavailable")
                return 0.0

            # Load audios
            ref_audio, _ = librosa.load(ref_audio_path, sr=sr, mono=True)
            gen_audio, _ = librosa.load(gen_audio_path, sr=sr, mono=True)

            # Extract embeddings
            if hasattr(encoder, 'encode_batch'):  # SpeechBrain
                ref_tensor = torch.from_numpy(ref_audio).float().unsqueeze(0).to(self.device)
                gen_tensor = torch.from_numpy(gen_audio).float().unsqueeze(0).to(self.device)

                with torch.no_grad():
                    ref_embed = encoder.encode_batch(ref_tensor)
                    gen_embed = encoder.encode_batch(gen_tensor)

                # Cosine similarity
                similarity = torch.nn.functional.cosine_similarity(
                    ref_embed,
                    gen_embed,
                    dim=-1
                ).item()
            else:  # Resemblyzer
                ref_embed = encoder.embed_utterance(ref_audio)
                gen_embed = encoder.embed_utterance(gen_audio)
                similarity = np.dot(ref_embed, gen_embed) / (
                    np.linalg.norm(ref_embed) * np.linalg.norm(gen_embed)
                )

            return float(max(0.0, min(1.0, similarity)))

        except Exception as e:
            logger.error(f"Error computing speaker similarity: {e}")
            return 0.0

    def compute_utmos(
        self,
        audio_path: str,
    ) -> float:
        """Compute UTMOS score (Mean Opinion Score prediction).

        Args:
            audio_path: Path to audio file

        Returns:
            UTMOS score (typically 1-5)
        """
        model = self.utmos_model
        if model is None:
            logger.warning("UTMOS model not available")
            return 0.0

        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
            audio = torch.from_numpy(audio).float().to(self.device)

            # Compute score
            with torch.no_grad():
                score = model.predict(audio)

            return float(score)

        except Exception as e:
            logger.error(f"Error computing UTMOS: {e}")
            return 0.0

    def compute_pesq(
        self,
        ref_audio_path: str,
        gen_audio_path: str,
        sr: int = 16000
    ) -> float:
        """Compute PESQ (Perceptual Evaluation of Speech Quality).

        Args:
            ref_audio_path: Path to reference audio
            gen_audio_path: Path to generated audio
            sr: Sample rate

        Returns:
            PESQ score (typically -0.5 to 4.5)
        """
        try:
            from pesq import pesq
        except ImportError:
            logger.warning("pesq library not installed. Install: pip install pesq")
            return 0.0

        try:
            # Load audios at 16kHz for PESQ
            ref_audio, _ = librosa.load(ref_audio_path, sr=16000, mono=True)
            gen_audio, _ = librosa.load(gen_audio_path, sr=16000, mono=True)

            # Ensure same length
            min_len = min(len(ref_audio), len(gen_audio))
            ref_audio = ref_audio[:min_len]
            gen_audio = gen_audio[:min_len]

            # Compute PESQ
            score = pesq(16000, ref_audio, gen_audio)
            return float(score)

        except Exception as e:
            logger.error(f"Error computing PESQ: {e}")
            return 0.0

    def compute_stoi(
        self,
        ref_audio_path: str,
        gen_audio_path: str,
        sr: int = 16000
    ) -> float:
        """Compute STOI (Short-Time Objective Intelligibility).

        Args:
            ref_audio_path: Path to reference audio
            gen_audio_path: Path to generated audio
            sr: Sample rate

        Returns:
            STOI score (0-1)
        """
        try:
            from pystoi import stoi
        except ImportError:
            logger.warning("pystoi library not installed. Install: pip install pystoi")
            return 0.0

        try:
            # Load audios
            ref_audio, _ = librosa.load(ref_audio_path, sr=sr, mono=True)
            gen_audio, _ = librosa.load(gen_audio_path, sr=sr, mono=True)

            # Ensure same length
            min_len = min(len(ref_audio), len(gen_audio))
            ref_audio = ref_audio[:min_len]
            gen_audio = gen_audio[:min_len]

            # Compute STOI
            score = stoi(ref_audio, gen_audio, sr)
            return float(max(0.0, min(1.0, score)))

        except Exception as e:
            logger.error(f"Error computing STOI: {e}")
            return 0.0

    def compute_rtf(
        self,
        audio_duration: float,
        generation_time: float
    ) -> float:
        """Compute Real-Time Factor (RTF).

        RTF = generation_time / audio_duration
        RTF < 1.0 = faster than real-time
        RTF = 1.0 = real-time
        RTF > 1.0 = slower than real-time

        Args:
            audio_duration: Duration of generated audio in seconds
            generation_time: Time taken for generation in seconds

        Returns:
            RTF as float
        """
        if audio_duration <= 0:
            return float('inf')

        rtf = generation_time / audio_duration
        return float(rtf)

    def compute_all(
        self,
        gen_audio_path: str,
        ref_audio_path: str,
        ground_truth_text: str,
        generation_time: float,
    ) -> dict:
        """Compute all metrics for a single sample.

        Args:
            gen_audio_path: Path to generated audio
            ref_audio_path: Path to reference audio
            ground_truth_text: Ground truth text
            generation_time: Time taken for generation

        Returns:
            Dictionary of computed metrics
        """
        results = {
            "gen_audio": gen_audio_path,
            "ref_audio": ref_audio_path,
            "ground_truth": ground_truth_text,
            "generation_time": generation_time,
        }

        try:
            # Load generated audio to compute duration
            gen_audio, sr = librosa.load(gen_audio_path, sr=None, mono=True)
            audio_duration = len(gen_audio) / sr

            # Transcribe generated audio
            logger.info(f"Transcribing {gen_audio_path}...")
            transcribed_text = self.transcribe_audio(gen_audio_path, language="hi")
            results["transcribed_text"] = transcribed_text

            # WER and CER
            logger.info("Computing WER and CER...")
            results["wer"] = self.compute_wer(ground_truth_text, transcribed_text)
            results["cer"] = self.compute_cer(ground_truth_text, transcribed_text)

            # Speaker similarity
            logger.info("Computing speaker similarity...")
            results["speaker_similarity"] = self.compute_speaker_similarity(
                ref_audio_path, gen_audio_path
            )

            # Quality metrics
            logger.info("Computing UTMOS...")
            results["utmos"] = self.compute_utmos(gen_audio_path)

            logger.info("Computing PESQ...")
            results["pesq"] = self.compute_pesq(ref_audio_path, gen_audio_path)

            logger.info("Computing STOI...")
            results["stoi"] = self.compute_stoi(ref_audio_path, gen_audio_path)

            # RTF
            logger.info("Computing RTF...")
            results["rtf"] = self.compute_rtf(audio_duration, generation_time)

        except Exception as e:
            logger.error(f"Error computing metrics: {e}", exc_info=True)

        return results


def main():
    """Test metrics computation."""
    import argparse

    parser = argparse.ArgumentParser(description="Compute TTS evaluation metrics")
    parser.add_argument("--gen-audio", required=True, help="Path to generated audio")
    parser.add_argument("--ref-audio", required=True, help="Path to reference audio")
    parser.add_argument("--text", required=True, help="Ground truth text")
    parser.add_argument("--time", type=float, default=1.0, help="Generation time in seconds")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    computer = MetricComputer(device=args.device)
    results = computer.compute_all(
        gen_audio_path=args.gen_audio,
        ref_audio_path=args.ref_audio,
        ground_truth_text=args.text,
        generation_time=args.time,
    )

    print("\nMetric Results:")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
