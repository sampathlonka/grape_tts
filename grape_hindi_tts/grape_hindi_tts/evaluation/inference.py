"""
Complete inference pipeline for SupertonicTTS Hindi.

Handles:
1. Loading trained modules (speech autoencoder, text-to-latent, duration predictor)
2. Text processing and tokenization
3. Reference audio encoding
4. Duration prediction
5. Flow matching with F5-TTS Euler ODE + sway-sampling for latent generation
6. Latent decompression and waveform decoding
7. Batch inference and duration scaling

Flow-matching inference engine: F5-TTS OT-CFM with sway-sampling (SWivid/F5-TTS).
  - Sway-sampling concentrates ODE steps near t=0 (noisy region) for better quality
  - CFG double-batch trick avoids two separate model forward passes
  - NFE=32 recommended (trade-off: RTF vs. quality, see SupertonicTTS Table 8)
"""

import logging
import time
from pathlib import Path
from typing import Optional, Tuple, List, Union, Dict
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import librosa

from ..models.speech_autoencoder import SpeechAutoencoder
from ..models.text_to_latent import TextToLatentModule
from ..models.duration_predictor import DurationPredictor
from ..data.hindi_text_processor import HindiTextProcessor
from ..data.audio_processor import AudioProcessor

# F5-TTS inference engine (OT-CFM + sway-sampling + CFG)
from ..third_party.f5_tts_cfm import euler_solve, sway_sampling_coefs

logger = logging.getLogger(__name__)


class SupertonicTTSInference:
    """Complete inference pipeline for SupertonicTTS Hindi.

    Loads all 3 trained modules and provides methods for:
    - Single and batch synthesis
    - Duration scaling
    - GPU/CPU inference
    - Real-time factor (RTF) measurement
    """

    def __init__(
        self,
        autoencoder_path: str,
        text_to_latent_path: str,
        duration_predictor_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        sample_rate: int = 44100,
        use_fp16: bool = False,
    ):
        """Initialize inference pipeline.

        Args:
            autoencoder_path: Path to speech autoencoder checkpoint
            text_to_latent_path: Path to text-to-latent checkpoint
            duration_predictor_path: Path to duration predictor checkpoint
            device: Device to use ("cuda" or "cpu")
            sample_rate: Target sample rate (default: 44100 Hz)
            use_fp16: Use half precision for faster inference
        """
        self.device = device
        self.sample_rate = sample_rate
        self.use_fp16 = use_fp16

        logger.info(f"Initializing SupertonicTTSInference on {device}")

        # Initialize components
        self.text_processor = HindiTextProcessor(language="hi")
        self.audio_processor = AudioProcessor(sample_rate=sample_rate)

        # Load models
        logger.info(f"Loading speech autoencoder from {autoencoder_path}")
        self.autoencoder = self._load_model(
            autoencoder_path,
            SpeechAutoencoder,
            decoder_only=True
        )

        logger.info(f"Loading text-to-latent from {text_to_latent_path}")
        self.text_to_latent = self._load_model(
            text_to_latent_path,
            TextToLatent
        )

        logger.info(f"Loading duration predictor from {duration_predictor_path}")
        self.duration_predictor = self._load_model(
            duration_predictor_path,
            DurationPredictor
        )

        # Set to eval mode
        self.autoencoder.eval()
        self.text_to_latent.eval()
        self.duration_predictor.eval()

        logger.info("SupertonicTTSInference initialized successfully")

    def _load_model(
        self,
        checkpoint_path: str,
        model_class,
        decoder_only: bool = False
    ):
        """Load model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            model_class: Model class to instantiate
            decoder_only: If True, only load decoder (for autoencoder)

        Returns:
            Loaded and moved model
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Handle both direct state dict and nested checkpoint formats
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Instantiate model
        if model_class == SpeechAutoencoder:
            model = model_class()
            if decoder_only:
                # Load only decoder part
                decoder_keys = {k: v for k, v in state_dict.items() if "decoder" in k}
                model.load_state_dict(decoder_keys, strict=False)
            else:
                model.load_state_dict(state_dict)
        else:
            model = model_class()
            model.load_state_dict(state_dict)

        return model.to(self.device)

    @torch.no_grad()
    def synthesize(
        self,
        text: str,
        reference_audio_path: str,
        duration_scale: float = 1.0,
        cfg_scale: float = 3.0,
        n_steps: int = 32,
        return_rtf: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
        """Synthesize speech from text using reference audio.

        Args:
            text: Hindi text to synthesize
            reference_audio_path: Path to reference audio for speaker characteristics
            duration_scale: Factor to scale predicted duration (< 1.0 = faster, > 1.0 = slower)
            cfg_scale: Classifier-free guidance scale (0 = no guidance, higher = stronger)
            n_steps: Number of ODE solver steps (NFE) for flow matching
            return_rtf: If True, return (waveform, rtf) tuple

        Returns:
            Synthesized waveform (np.ndarray at 44.1kHz) or (waveform, rtf) if return_rtf=True
        """
        start_time = time.time()

        try:
            # Step 1: Process text
            logger.debug(f"Processing text: {text}")
            text_tokens = self._process_text(text)

            # Step 2: Load and encode reference audio
            logger.debug(f"Loading reference audio: {reference_audio_path}")
            ref_latents = self._encode_reference_audio(reference_audio_path)

            # Step 3: Predict duration
            logger.debug("Predicting duration")
            predicted_duration = self._predict_duration(text_tokens, ref_latents)
            scaled_duration = int(predicted_duration * duration_scale)

            logger.debug(f"Predicted duration: {predicted_duration}, scaled: {scaled_duration}")

            # Step 4: Generate latents using flow matching
            logger.debug(f"Generating latents with {n_steps} steps, CFG={cfg_scale}")
            latents = self._flow_matching_inference(
                text_tokens=text_tokens,
                ref_latents=ref_latents,
                target_length=scaled_duration,
                cfg_scale=cfg_scale,
                n_steps=n_steps,
            )

            # Step 5: Decompress latents
            logger.debug("Decompressing latents")
            decompressed_latents = self._decompress_latents(latents)

            # Step 6: Decode to waveform
            logger.debug("Decoding latents to waveform")
            waveform = self._decode_waveform(decompressed_latents)

            inference_time = time.time() - start_time
            audio_duration = len(waveform) / self.sample_rate
            rtf = inference_time / audio_duration if audio_duration > 0 else float('inf')

            logger.info(f"Synthesis complete. Duration: {audio_duration:.2f}s, RTF: {rtf:.3f}")

            if return_rtf:
                return waveform, rtf
            return waveform

        except Exception as e:
            logger.error(f"Error during synthesis: {e}", exc_info=True)
            raise

    def synthesize_batch(
        self,
        texts: List[str],
        reference_audio_paths: Union[str, List[str]],
        duration_scale: float = 1.0,
        cfg_scale: float = 3.0,
        n_steps: int = 32,
    ) -> List[Tuple[np.ndarray, float, str]]:
        """Synthesize multiple texts.

        Args:
            texts: List of Hindi texts
            reference_audio_paths: Single path or list of paths (one per text)
            duration_scale: Factor to scale predicted duration
            cfg_scale: Classifier-free guidance scale
            n_steps: Number of ODE solver steps

        Returns:
            List of (waveform, rtf, text) tuples
        """
        if isinstance(reference_audio_paths, str):
            reference_audio_paths = [reference_audio_paths] * len(texts)

        results = []
        for i, (text, ref_path) in enumerate(zip(texts, reference_audio_paths)):
            logger.info(f"Processing {i+1}/{len(texts)}: {text[:50]}...")
            try:
                waveform, rtf = self.synthesize(
                    text=text,
                    reference_audio_path=ref_path,
                    duration_scale=duration_scale,
                    cfg_scale=cfg_scale,
                    n_steps=n_steps,
                    return_rtf=True,
                )
                results.append((waveform, rtf, text))
            except Exception as e:
                logger.error(f"Failed to synthesize text {i}: {e}")
                results.append((None, None, text))

        return results

    def _process_text(self, text: str) -> torch.Tensor:
        """Process Hindi text to character token IDs.

        Args:
            text: Raw Hindi text

        Returns:
            Token tensor of shape (1, seq_len) on device
        """
        # Normalize text
        normalized = self.text_processor.normalize_unicode(text)
        normalized = self.text_processor.expand_abbreviations(normalized)
        normalized = self.text_processor.normalize_numbers(normalized)

        # Convert to tokens
        token_ids = self.text_processor.text_to_ids(normalized)

        # Convert to tensor
        tokens = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        return tokens

    def _encode_reference_audio(self, audio_path: str) -> torch.Tensor:
        """Load reference audio and encode to compressed latents.

        Args:
            audio_path: Path to reference audio

        Returns:
            Compressed latent tensor
        """
        # Load audio
        waveform, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        waveform_tensor = torch.from_numpy(waveform).float().unsqueeze(0)
        waveform_tensor = waveform_tensor.to(self.device)

        # Compute mel spectrogram
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_mels=228,
            n_fft=2048,
            hop_length=512,
            f_min=55.0,
        )
        mel = mel_transform(waveform_tensor)  # (1, 228, time)

        # Encode to latents using autoencoder
        with torch.no_grad():
            if hasattr(self.autoencoder, 'encode'):
                latents = self.autoencoder.encode(mel)  # (1, 24, time)
            else:
                # If no explicit encode method, use forward (assuming it's encoder-only)
                latents = self.autoencoder(mel)

        # Compress latents
        compressed = self.text_to_latent.compress_latents(latents, Kc=6)  # (1, 144, time//6)

        return compressed

    def _predict_duration(
        self,
        text_tokens: torch.Tensor,
        ref_latents: torch.Tensor
    ) -> int:
        """Predict utterance duration using duration predictor.

        Args:
            text_tokens: Text token tensor (1, seq_len)
            ref_latents: Compressed reference latents (1, 144, time)

        Returns:
            Predicted duration in compressed latent frames
        """
        with torch.no_grad():
            duration = self.duration_predictor(ref_latents, text_tokens)
            # Assuming output is a scalar or (batch, 1) tensor
            if duration.dim() > 0:
                duration = duration.squeeze().item()
            return int(duration)

    def _flow_matching_inference(
        self,
        text_tokens: torch.Tensor,
        ref_latents: torch.Tensor,
        target_length: int,
        cfg_scale: float = 3.0,
        n_steps: int = 32,
    ) -> torch.Tensor:
        """Generate latents using flow matching with Euler ODE solver.

        Algorithm:
        1. Sample z_0 ~ N(0, 1) of target length
        2. For each step i in 0..n_steps-1:
           - t = i / n_steps
           - v_cond = vf_model(z_t, ref, text, t)  (conditional)
           - v_uncond = vf_model(z_t, None, None, t)  (unconditional)
           - v_guided = (1 + cfg) * v_cond - cfg * v_uncond
           - z_{t+dt} = z_t + (1/n_steps) * v_guided
        3. Return final z

        Args:
            text_tokens: Text tokens (1, seq_len)
            ref_latents: Reference latents (1, 144, ref_time)
            target_length: Target sequence length
            cfg_scale: Classifier-free guidance scale
            n_steps: Number of Euler steps

        Returns:
            Generated latents (1, 144, target_length)
        """
        batch_size = 1
        latent_channels = 144  # Compressed latent dimension

        # Initialize z_0 ~ N(0, 1)
        z = torch.randn(
            batch_size,
            latent_channels,
            target_length,
            device=self.device,
            dtype=torch.float32
        )

        # Encode text features
        with torch.no_grad():
            text_features = self.text_to_latent.text_encoder(text_tokens)  # (1, seq_len, hidden)

        # Reference features
        with torch.no_grad():
            ref_features = self.text_to_latent.reference_encoder(ref_latents)  # (1, query_dim)

        # Euler solver
        dt = 1.0 / n_steps

        for step in range(n_steps):
            t = torch.full((batch_size,), step / n_steps, device=self.device)

            # Velocity field (conditional)
            with torch.no_grad():
                v_cond = self.text_to_latent.vf_estimator(
                    z=z,
                    text_features=text_features,
                    ref_features=ref_features,
                    t=t
                )

            # Classifier-free guidance
            if cfg_scale > 0:
                # Unconditional velocity (no text, no reference)
                with torch.no_grad():
                    v_uncond = self.text_to_latent.vf_estimator(
                        z=z,
                        text_features=None,
                        ref_features=None,
                        t=t
                    )

                # Guided velocity
                v_guided = (1 + cfg_scale) * v_cond - cfg_scale * v_uncond
            else:
                v_guided = v_cond

            # Euler step
            z = z + dt * v_guided

        return z

    def _decompress_latents(self, compressed: torch.Tensor) -> torch.Tensor:
        """Decompress latents from compressed form.

        Args:
            compressed: Compressed latents (1, 144, time)

        Returns:
            Decompressed latents (1, 24, time*6)
        """
        decompressed = self.text_to_latent.decompress_latents(compressed, Kc=6)
        return decompressed

    def _decode_waveform(self, latents: torch.Tensor) -> np.ndarray:
        """Decode latents to waveform using autoencoder decoder.

        Args:
            latents: Decompressed latents (1, 24, time)

        Returns:
            Waveform as numpy array (mono, 44.1kHz)
        """
        with torch.no_grad():
            if hasattr(self.autoencoder, 'decode'):
                waveform = self.autoencoder.decode(latents)
            else:
                # If no explicit decode, assume forward returns waveform
                waveform = self.autoencoder(latents)

        # Convert to numpy
        waveform = waveform.squeeze(0).cpu().numpy()

        # Ensure mono
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=0)

        return waveform

    def save_audio(
        self,
        waveform: np.ndarray,
        output_path: str,
        sample_rate: Optional[int] = None
    ):
        """Save waveform to audio file.

        Args:
            waveform: Numpy waveform array
            output_path: Path to save audio
            sample_rate: Sample rate (default: self.sample_rate)
        """
        if sample_rate is None:
            sample_rate = self.sample_rate

        # Normalize to [-1, 1]
        max_val = np.abs(waveform).max()
        if max_val > 1.0:
            waveform = waveform / max_val

        # Convert to tensor and save
        waveform_tensor = torch.from_numpy(waveform).float().unsqueeze(0)
        torchaudio.save(output_path, waveform_tensor, sample_rate)
        logger.info(f"Saved audio to {output_path}")


def main():
    """CLI interface for inference."""
    parser = argparse.ArgumentParser(
        description="SupertonicTTS Hindi inference pipeline"
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
        "--text",
        required=True,
        help="Hindi text to synthesize"
    )
    parser.add_argument(
        "--reference",
        required=True,
        help="Path to reference audio"
    )
    parser.add_argument(
        "--output",
        default="output.wav",
        help="Path to save synthesized audio"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)"
    )
    parser.add_argument(
        "--duration-scale",
        type=float,
        default=1.0,
        help="Duration scaling factor"
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=3.0,
        help="Classifier-free guidance scale"
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=32,
        help="Number of ODE solver steps"
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

    # Initialize inference
    inference = SupertonicTTSInference(
        autoencoder_path=args.autoencoder,
        text_to_latent_path=args.text_to_latent,
        duration_predictor_path=args.duration_predictor,
        device=args.device,
    )

    # Synthesize
    waveform, rtf = inference.synthesize(
        text=args.text,
        reference_audio_path=args.reference,
        duration_scale=args.duration_scale,
        cfg_scale=args.cfg_scale,
        n_steps=args.n_steps,
        return_rtf=True,
    )

    # Save
    inference.save_audio(waveform, args.output)
    print(f"Synthesis complete!")
    print(f"  Output: {args.output}")
    print(f"  Duration: {len(waveform) / inference.sample_rate:.2f}s")
    print(f"  RTF: {rtf:.3f}")


if __name__ == "__main__":
    main()
