"""Audio utility functions for loading, saving, and processing audio signals."""

import numpy as np
import torch
import torchaudio
import librosa
import soundfile as sf
from scipy.signal import get_window
from typing import Tuple, Union, Optional


def load_audio(
    path: str,
    sr: int = 44100,
) -> torch.Tensor:
    """
    Load audio from file and convert to mono.

    Args:
        path: Path to audio file
        sr: Target sample rate

    Returns:
        Waveform tensor of shape (1, num_samples) or (num_samples,)
    """
    waveform, sample_rate = torchaudio.load(path)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if sample_rate != sr:
        resampler = torchaudio.transforms.Resample(sample_rate, sr)
        waveform = resampler(waveform)

    return waveform.squeeze(0)  # Return 1D tensor


def save_audio(
    waveform: torch.Tensor,
    path: str,
    sr: int = 44100,
) -> None:
    """
    Save waveform to audio file.

    Args:
        waveform: Audio tensor of shape (num_samples,) or (1, num_samples)
        path: Output path
        sr: Sample rate
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    waveform = waveform.cpu().detach()
    torchaudio.save(path, waveform, sr)


def compute_mel_spectrogram(
    waveform: torch.Tensor,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 128,
    sr: int = 44100,
) -> torch.Tensor:
    """
    Compute log mel-scale spectrogram from waveform.

    Args:
        waveform: Audio tensor of shape (num_samples,)
        n_fft: FFT size
        hop_length: Number of samples between successive frames
        n_mels: Number of mel bands
        sr: Sample rate

    Returns:
        Log mel-spectrogram of shape (n_mels, time_steps)
    """
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        center=True,
    )

    mel_spec = mel_transform(waveform)
    # Convert to log scale
    log_mel_spec = torch.log(torch.clamp(mel_spec, min=1e-9))

    return log_mel_spec


def mel_to_audio_griffin_lim(
    mel_spectrogram: torch.Tensor,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 128,
    sr: int = 44100,
    n_iter: int = 32,
) -> torch.Tensor:
    """
    Convert log mel-spectrogram to waveform using Griffin-Lim algorithm.

    Args:
        mel_spectrogram: Log mel-spectrogram of shape (n_mels, time_steps)
        n_fft: FFT size
        hop_length: Number of samples between successive frames
        n_mels: Number of mel bands
        sr: Sample rate
        n_iter: Number of Griffin-Lim iterations

    Returns:
        Waveform tensor
    """
    # Convert log mel-spectrogram back to linear mel
    mel_spec_linear = torch.exp(mel_spectrogram)

    # Create mel-scale to Hz conversion matrix
    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    mel_basis = torch.from_numpy(mel_basis).float()

    # Pseudo-inverse to convert mel back to linear magnitude spectrogram
    inv_mel_basis = torch.linalg.pinv(mel_basis)
    magnitude = torch.matmul(inv_mel_basis, mel_spec_linear.cpu().numpy() if isinstance(mel_spec_linear, torch.Tensor) else mel_spec_linear)
    magnitude = torch.from_numpy(magnitude).float()

    # Apply Griffin-Lim algorithm
    phase = np.random.randn(*magnitude.shape)
    for _ in range(n_iter):
        complex_spec = magnitude.numpy() * np.exp(1j * phase)
        waveform = librosa.istft(complex_spec, hop_length=hop_length, win_length=n_fft)
        phase = np.angle(librosa.stft(waveform, n_fft=n_fft, hop_length=hop_length))

    waveform_final = librosa.istft(magnitude.numpy() * np.exp(1j * phase), hop_length=hop_length)

    return torch.from_numpy(waveform_final).float()


def get_audio_duration(path: str) -> float:
    """
    Get audio duration in seconds.

    Args:
        path: Path to audio file

    Returns:
        Duration in seconds
    """
    metadata = torchaudio.info(path)
    return metadata.num_frames / metadata.sample_rate


def normalize_audio(
    waveform: torch.Tensor,
    target_db: float = -20.0,
) -> torch.Tensor:
    """
    Normalize audio to target loudness using simple peak normalization.

    Args:
        waveform: Audio tensor
        target_db: Target loudness in dB (for peak normalization, db is relative to max)

    Returns:
        Normalized waveform
    """
    # Peak normalization first
    peak = waveform.abs().max()
    if peak > 0:
        waveform = waveform / peak

    # Scale to target level
    target_linear = 10.0 ** (target_db / 20.0)
    waveform = waveform * target_linear

    # Clip to prevent clipping
    waveform = torch.clamp(waveform, -1.0, 1.0)

    return waveform


def trim_silence(
    waveform: torch.Tensor,
    sr: int = 44100,
    threshold_db: float = -40.0,
) -> torch.Tensor:
    """
    Trim silence from beginning and end of audio.

    Args:
        waveform: Audio tensor
        sr: Sample rate
        threshold_db: Threshold below which to consider silence

    Returns:
        Trimmed waveform
    """
    # Compute magnitude in dB
    S = librosa.feature.melspectrogram(y=waveform.numpy(), sr=sr)
    S_db = librosa.power_to_db(S, ref=np.max)

    # Find frames above threshold
    threshold_linear = 10.0 ** (threshold_db / 10.0)
    S_energy = S.mean(axis=0)
    above_thresh = (S_energy > threshold_linear).astype(int)

    # Find start and end indices
    nonzero_frames = np.nonzero(above_thresh)[0]
    if len(nonzero_frames) == 0:
        return waveform

    start_frame = nonzero_frames[0]
    end_frame = nonzero_frames[-1]

    # Convert frames to samples
    hop_length = 512  # Default for melspectrogram
    start_sample = max(0, start_frame * hop_length)
    end_sample = min(len(waveform), (end_frame + 1) * hop_length)

    return waveform[start_sample:end_sample]


def resample_audio(
    waveform: torch.Tensor,
    orig_sr: int,
    target_sr: int,
) -> torch.Tensor:
    """
    Resample audio to target sample rate.

    Args:
        waveform: Audio tensor
        orig_sr: Original sample rate
        target_sr: Target sample rate

    Returns:
        Resampled waveform
    """
    if orig_sr == target_sr:
        return waveform

    resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
    return resampler(waveform)


def power_to_db(
    magnitude: torch.Tensor,
    ref: float = 1.0,
    amin: float = 1e-10,
) -> torch.Tensor:
    """
    Convert power/magnitude to decibel scale.

    Args:
        magnitude: Magnitude spectrogram
        ref: Reference magnitude for 0 dB
        amin: Minimum value to prevent log(0)

    Returns:
        dB spectrogram
    """
    magnitude = torch.clamp(magnitude, min=amin)
    db = 20 * torch.log10(magnitude / ref)
    return db


def db_to_power(
    db: torch.Tensor,
    ref: float = 1.0,
) -> torch.Tensor:
    """
    Convert dB to power/magnitude scale.

    Args:
        db: dB spectrogram
        ref: Reference magnitude for 0 dB

    Returns:
        Magnitude spectrogram
    """
    return ref * torch.pow(10.0, db / 20.0)
