"""
Audio Processor for SupertonicTTS

Handles audio preprocessing including:
- Loading and resampling audio
- Mel spectrogram computation
- Voice Activity Detection (VAD) for silence trimming
- Audio normalization
- Long audio segmentation
"""

import numpy as np
import librosa
import torch
import torchaudio
import torchaudio.transforms as T
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


class AudioProcessor:
    """Audio processor for TTS training."""

    def __init__(
        self,
        sample_rate: int = 44100,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 228,
        f_min: float = 55.0,
        f_max: Optional[float] = None,
        ref_db: float = 20.0,
        min_level_db: float = -100.0,
    ):
        """
        Initialize audio processor.

        Args:
            sample_rate: Target sample rate in Hz (default: 44100)
            n_fft: FFT window size (default: 2048)
            hop_length: Number of samples between frames (default: 512)
            n_mels: Number of mel frequency bins (default: 228)
            f_min: Minimum frequency (default: 55.0 Hz)
            f_max: Maximum frequency (default: sample_rate // 2)
            ref_db: Decibel reference for normalization (default: 20.0)
            min_level_db: Minimum level in dB (default: -100.0)
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max if f_max is not None else sample_rate // 2
        self.ref_db = ref_db
        self.min_level_db = min_level_db

        # Create mel spectrogram transform
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            f_min=f_min,
            f_max=self.f_max,
            window_fn=torch.hann_window,
        )

    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file and return waveform and sample rate.

        Args:
            audio_path: Path to audio file

        Returns:
            Tuple of (waveform, sample_rate)
        """
        try:
            waveform, sr = librosa.load(audio_path, sr=None, mono=True)
            return waveform, sr
        except Exception as e:
            logger.error(f"Error loading audio {audio_path}: {e}")
            raise

    def resample_audio(
        self,
        waveform: np.ndarray,
        orig_sr: int,
        target_sr: Optional[int] = None
    ) -> np.ndarray:
        """
        Resample audio to target sample rate.

        Args:
            waveform: Input waveform
            orig_sr: Original sample rate
            target_sr: Target sample rate (default: self.sample_rate)

        Returns:
            Resampled waveform
        """
        if target_sr is None:
            target_sr = self.sample_rate

        if orig_sr == target_sr:
            return waveform

        return librosa.resample(waveform, orig_sr=orig_sr, res_type="kaiser_best")

    def normalize_audio(self, waveform: np.ndarray, peak_norm: bool = True) -> np.ndarray:
        """
        Normalize audio by peak value.

        Args:
            waveform: Input waveform
            peak_norm: Whether to use peak normalization (default: True)

        Returns:
            Normalized waveform
        """
        if peak_norm:
            # Peak normalization to [-1, 1]
            max_val = np.abs(waveform).max()
            if max_val > 0:
                return waveform / max_val
            return waveform
        else:
            # RMS normalization
            rms = np.sqrt(np.mean(waveform ** 2))
            if rms > 0:
                return waveform / rms
            return waveform

    def compute_vad_energy(
        self,
        waveform: np.ndarray,
        frame_length: int = 512,
        energy_threshold: float = 0.02,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Voice Activity Detection based on frame energy.

        Args:
            waveform: Input waveform
            frame_length: Length of each frame
            energy_threshold: Energy threshold for speech detection

        Returns:
            Tuple of (vad_mask, energy_values)
        """
        # Compute frame energy
        energy = []
        for i in range(0, len(waveform) - frame_length, frame_length):
            frame = waveform[i:i + frame_length]
            frame_energy = np.sqrt(np.mean(frame ** 2))
            energy.append(frame_energy)

        energy = np.array(energy)

        # Compute threshold (mean energy)
        if len(energy) > 0:
            threshold = np.mean(energy) * energy_threshold
        else:
            threshold = energy_threshold

        # Create VAD mask
        vad_mask = energy > threshold

        return vad_mask, energy

    def trim_silence(
        self,
        waveform: np.ndarray,
        top_db: float = 40.0,
        ref: float = np.max,
    ) -> np.ndarray:
        """
        Trim silence from beginning and end of audio.

        Args:
            waveform: Input waveform
            top_db: Threshold in dB below reference to consider as silence
            ref: Reference value (default: max)

        Returns:
            Trimmed waveform
        """
        trimmed, _ = librosa.effects.trim(waveform, top_db=top_db, ref=ref)
        return trimmed

    def compute_mel_spectrogram(self, waveform: np.ndarray) -> np.ndarray:
        """
        Compute log-scaled mel spectrogram from waveform.

        Args:
            waveform: Input waveform (numpy array)

        Returns:
            Log mel spectrogram (n_mels, time_steps)
        """
        # Convert to torch tensor
        if isinstance(waveform, np.ndarray):
            waveform_torch = torch.from_numpy(waveform).float()
        else:
            waveform_torch = waveform.float()

        # Ensure 1D
        if waveform_torch.dim() == 1:
            waveform_torch = waveform_torch.unsqueeze(0)

        # Compute mel spectrogram
        mel_spec = self.mel_transform(waveform_torch)

        # Convert to dB scale with clipping
        mel_db = torch.log(torch.clamp(mel_spec, min=1e-9)) * self.ref_db
        mel_db = torch.clamp(mel_db, min=self.min_level_db)

        # Normalize to [0, 1]
        mel_db_normalized = (mel_db - self.min_level_db) / (-self.min_level_db)
        mel_db_normalized = torch.clamp(mel_db_normalized, 0, 1)

        return mel_db_normalized.squeeze(0).cpu().numpy()

    def griffin_lim(
        self,
        mel_spec: np.ndarray,
        n_iter: int = 60,
    ) -> np.ndarray:
        """
        Reconstruct waveform from mel spectrogram using Griffin-Lim algorithm.
        (For debugging purposes - not used in main pipeline)

        Args:
            mel_spec: Mel spectrogram (n_mels, time_steps)
            n_iter: Number of iterations (default: 60)

        Returns:
            Reconstructed waveform
        """
        # Denormalize mel spectrogram
        mel_db = mel_spec * (-self.min_level_db) + self.min_level_db
        mel_linear = 10 ** (mel_db / self.ref_db)

        # Convert mel back to linear spectrogram
        inverse_mel_matrix = librosa.feature.inverse.mel_to_audio(
            mel_linear,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )

        # Apply Griffin-Lim
        try:
            # Use spectrogram inversion for better results
            S = librosa.feature.melspectrogram(
                y=inverse_mel_matrix,
                sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
                f_min=self.f_min,
                f_max=self.f_max,
            )
            return librosa.griffinlim(np.abs(S), n_iter=n_iter)
        except Exception as e:
            logger.warning(f"Griffin-Lim reconstruction failed: {e}")
            return inverse_mel_linear

    def segment_long_audio(
        self,
        waveform: np.ndarray,
        segment_length: int = 32000,
        overlap: int = 4000,
    ) -> List[np.ndarray]:
        """
        Split long audio into overlapping segments.

        Args:
            waveform: Input waveform
            segment_length: Length of each segment in samples
            overlap: Overlap between segments in samples

        Returns:
            List of audio segments
        """
        segments = []
        stride = segment_length - overlap

        for start in range(0, len(waveform) - segment_length + 1, stride):
            segment = waveform[start:start + segment_length]
            if len(segment) == segment_length:
                segments.append(segment)

        # Handle last segment
        if len(waveform) > start + segment_length:
            last_segment = waveform[-segment_length:]
            if len(last_segment) == segment_length:
                segments.append(last_segment)

        return segments

    def process_audio_file(
        self,
        audio_path: str,
        normalize: bool = True,
        trim_silence: bool = True,
    ) -> Tuple[np.ndarray, int]:
        """
        Complete audio processing pipeline.

        Args:
            audio_path: Path to audio file
            normalize: Whether to normalize audio
            trim_silence: Whether to trim silence

        Returns:
            Tuple of (processed waveform, sample_rate)
        """
        # Load audio
        waveform, sr = self.load_audio(audio_path)

        # Resample if needed
        if sr != self.sample_rate:
            waveform = self.resample_audio(waveform, sr, self.sample_rate)

        # Trim silence
        if trim_silence:
            waveform = self.trim_silence(waveform, top_db=40.0)

        # Normalize
        if normalize:
            waveform = self.normalize_audio(waveform, peak_norm=True)

        return waveform, self.sample_rate

    def get_audio_duration(self, waveform: np.ndarray) -> float:
        """
        Get duration of audio in seconds.

        Args:
            waveform: Input waveform

        Returns:
            Duration in seconds
        """
        return len(waveform) / self.sample_rate

    def compute_rms_energy(self, waveform: np.ndarray) -> float:
        """
        Compute RMS energy of waveform.

        Args:
            waveform: Input waveform

        Returns:
            RMS energy value
        """
        return float(np.sqrt(np.mean(waveform ** 2)))


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    processor = AudioProcessor()

    print(f"Sample rate: {processor.sample_rate}")
    print(f"Mel bins: {processor.n_mels}")
    print(f"FFT size: {processor.n_fft}")
    print(f"Hop length: {processor.hop_length}")

    # Example: Create synthetic audio and process it
    print("\n--- Synthetic Audio Test ---")
    duration = 3.0  # seconds
    t = np.linspace(0, duration, int(processor.sample_rate * duration))
    # 440 Hz sine wave
    waveform = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    # Process
    normalized = processor.normalize_audio(waveform)
    mel_spec = processor.compute_mel_spectrogram(normalized)

    print(f"Waveform shape: {waveform.shape}")
    print(f"Normalized min/max: {normalized.min():.4f} / {normalized.max():.4f}")
    print(f"Mel spectrogram shape: {mel_spec.shape}")
    print(f"Mel spectrogram value range: {mel_spec.min():.4f} / {mel_spec.max():.4f}")
    print(f"Audio duration: {processor.get_audio_duration(normalized):.2f}s")
