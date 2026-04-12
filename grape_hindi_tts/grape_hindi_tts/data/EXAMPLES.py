"""
SupertonicTTS Data Pipeline - Usage Examples

Demonstrates common usage patterns for all pipeline components.
"""

# ==============================================================================
# EXAMPLE 1: Text Normalization
# ==============================================================================

def example_text_normalization():
    """Demonstrate Hindi text normalization."""
    from hindi_text_processor import HindiTextProcessor

    processor = HindiTextProcessor()

    # Various text samples
    texts = [
        "नमस्ते, यह 123 रुपये का है।",
        "डॉ. शर्मा ने 5 दवाई दी।",
        "भारत (इ.आ.) में रहते हैं।",
        "245 किलोमीटर दूर है।",
    ]

    print("=" * 70)
    print("TEXT NORMALIZATION EXAMPLES")
    print("=" * 70)

    for text in texts:
        normalized = processor.normalize_text(text)
        token_ids = processor.text_to_token_ids(text)
        reconstructed = processor.token_ids_to_text(token_ids)

        print(f"\nOriginal:      {text}")
        print(f"Normalized:    {normalized}")
        print(f"Tokens:        {len(token_ids)} tokens")
        print(f"Reconstructed: {reconstructed}")

    # Number conversion examples
    print("\n" + "=" * 70)
    print("NUMBER TO WORDS CONVERSION")
    print("=" * 70)
    for num in [0, 5, 10, 15, 100, 123, 1000, 10000, 100000, 1000000]:
        words = processor.number_to_words_hindi(num)
        print(f"{num:10d} → {words}")

    # Vocabulary info
    print("\n" + "=" * 70)
    print("VOCABULARY INFORMATION")
    print("=" * 70)
    print(f"Vocabulary size: {processor.get_vocab_size()}")
    print(f"Special tokens: PAD={processor.PAD_IDX}, UNK={processor.UNK_IDX}, "
          f"BOS={processor.BOS_IDX}, EOS={processor.EOS_IDX}")


# ==============================================================================
# EXAMPLE 2: Audio Processing
# ==============================================================================

def example_audio_processing():
    """Demonstrate audio processing pipeline."""
    from audio_processor import AudioProcessor
    import numpy as np

    processor = AudioProcessor()

    print("\n" + "=" * 70)
    print("AUDIO PROCESSING EXAMPLES")
    print("=" * 70)

    # Create synthetic audio for demonstration
    print("\nGenerating synthetic test audio...")
    duration = 2.0  # seconds
    sample_rate = processor.sample_rate
    t = np.linspace(0, duration, int(sample_rate * duration))

    # 440 Hz sine wave (A note)
    waveform = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    print(f"Sample rate: {processor.sample_rate} Hz")
    print(f"Duration: {processor.get_audio_duration(waveform):.2f}s")
    print(f"RMS energy: {processor.compute_rms_energy(waveform):.4f}")

    # Normalize
    normalized = processor.normalize_audio(waveform)
    print(f"\nAfter normalization:")
    print(f"  Min: {normalized.min():.4f}, Max: {normalized.max():.4f}")

    # Compute mel spectrogram
    mel_spec = processor.compute_mel_spectrogram(waveform)
    print(f"\nMel spectrogram:")
    print(f"  Shape: {mel_spec.shape}")
    print(f"  Range: [{mel_spec.min():.4f}, {mel_spec.max():.4f}]")
    print(f"  Mel bins: {mel_spec.shape[0]}")
    print(f"  Time steps: {mel_spec.shape[1]}")

    # Segment audio
    segments = processor.segment_long_audio(
        waveform,
        segment_length=int(sample_rate * 0.5),  # 0.5s segments
        overlap=int(sample_rate * 0.1)  # 0.1s overlap
    )
    print(f"\nAudio segmentation (0.5s segments, 0.1s overlap):")
    print(f"  Number of segments: {len(segments)}")


# ==============================================================================
# EXAMPLE 3: Dataset Creation
# ==============================================================================

def example_dataset_creation():
    """Demonstrate dataset creation and usage."""
    import json
    import tempfile
    from dataset import HindiTTSDataset, collate_tts_batch
    from torch.utils.data import DataLoader

    print("\n" + "=" * 70)
    print("DATASET CREATION EXAMPLES")
    print("=" * 70)

    # Create a dummy manifest
    manifest = [
        {
            "audio_path": "/path/to/audio1.wav",
            "text": "नमस्ते दुनिया",
            "speaker_id": "speaker_001",
            "duration": 2.5,
            "gender": "M"
        },
        {
            "audio_path": "/path/to/audio2.wav",
            "text": "यह एक परीक्षण है",
            "speaker_id": "speaker_002",
            "duration": 1.8,
            "gender": "F"
        },
    ]

    # Save manifest to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(manifest, f, ensure_ascii=False)
        manifest_path = f.name

    print(f"\nCreated test manifest with {len(manifest)} samples")
    print(f"Manifest path: {manifest_path}")

    # Create dataset (without loading audio since files don't exist)
    try:
        dataset = HindiTTSDataset(
            manifest_path=manifest_path,
            max_text_length=256,
            normalize_text=True,
            load_audio=False  # Don't try to load actual audio files
        )

        print(f"Dataset size: {len(dataset)}")

        # Show sample
        print("\nSample from dataset (load_audio=False):")
        sample = dataset[0]
        print(f"  Text: {sample['text']}")
        print(f"  Text tokens: {len(sample['text_token_ids'])} tokens")
        print(f"  Speaker: {sample['speaker_id']}")
        print(f"  Duration: {sample['duration']}s")
        print(f"  Mel spectrogram: {sample['mel_spectrogram']}")

    except Exception as e:
        print(f"Note: Error loading sample (expected without actual audio files)")
        print(f"  Error: {e}")

    print(f"\nFor real usage, collate_tts_batch is used in DataLoader:")
    print(f"  loader = DataLoader(dataset, batch_size=32, collate_fn=collate_tts_batch)")


# ==============================================================================
# EXAMPLE 4: Dataset Preparation
# ==============================================================================

def example_dataset_preparation():
    """Demonstrate dataset preparation workflow."""
    import tempfile
    from pathlib import Path

    print("\n" + "=" * 70)
    print("DATASET PREPARATION WORKFLOW")
    print("=" * 70)

    print("""
Dataset preparation follows these steps:

1. SETUP:
   Create audio directory and transcripts file:

   transcripts.txt format:
   audio_001.wav\\tनमस्ते\\tspeaker_001\\tM
   audio_002.wav\\tदुनिया\\tspeaker_002\\tF

2. VALIDATION:
   - Check audio format and duration (0.5-30s default)
   - Check silence ratio (max 50% default)
   - Normalize all text
   - Track skip reasons

3. SPLITTING:
   - Train/Val/Test (90/5/5 by default)
   - Speaker-balanced splitting
   - Reproducible with seed

4. STATISTICS:
   - Compute mel spectrogram stats (mean, std per channel)
   - Per-speaker statistics (duration, gender)
   - Count and duration tracking

5. OUTPUT:
   - train_manifest.json
   - val_manifest.json
   - test_manifest.json
   - dataset_report.txt (human readable)
   - dataset_stats.json (machine readable)

COMMAND LINE USAGE:

python prepare_dataset.py \\
    /path/to/audio \\
    /path/to/transcripts.txt \\
    --output_dir ./manifests \\
    --sample_rate 44100 \\
    --min_duration 0.5 \\
    --max_duration 30.0 \\
    --max_silence 0.5 \\
    --train_split 0.9 \\
    --val_split 0.05 \\
    --test_split 0.05

PYTHON USAGE:

from prepare_dataset import DatasetPreparer

preparer = DatasetPreparer(
    audio_dir="/path/to/audio",
    transcript_file="/path/to/transcripts.txt",
    output_dir="./manifests",
    sample_rate=44100,
)

preparer.prepare_dataset(
    train_split=0.9,
    val_split=0.05,
    test_split=0.05,
)
    """)


# ==============================================================================
# EXAMPLE 5: Latent Precomputation
# ==============================================================================

def example_latent_precomputation():
    """Demonstrate latent precomputation workflow."""

    print("\n" + "=" * 70)
    print("LATENT PRECOMPUTATION WORKFLOW")
    print("=" * 70)

    print("""
Latent precomputation is used after training a speech autoencoder:

1. TRAINING AUTOENCODER:
   - Train speech autoencoder on audio
   - Autoencoder has encode() method
   - Save checkpoint with model weights

2. PRECOMPUTE LATENTS:
   - Load trained autoencoder
   - Encode all audio through encoder
   - Save latent vectors as .npy files
   - Compute normalization statistics

3. USE IN TRAINING:
   - Use TTLDataset with precomputed latents
   - Train text-to-latent model
   - Much faster training (no encoding needed)

COMMAND LINE:

python precompute_latents.py \\
    ./checkpoints/autoencoder.pt \\
    ./manifests/train_manifest.json \\
    --output_dir ./precomputed_latents \\
    --device cuda \\
    --batch_size 32

PYTHON USAGE:

from precompute_latents import LatentPrecomputer, LatentNormalizer
from audio_processor import AudioProcessor
from your_model import SpeechAutoencoder

# Initialize
precomputer = LatentPrecomputer(
    model_path="checkpoints/autoencoder.pt",
    output_dir="precomputed_latents/",
    device="cuda"
)

# Load model
model = SpeechAutoencoder(...)
model = precomputer.load_model(model)

# Precompute
audio_processor = AudioProcessor()
precomputer.precompute_latents_from_manifest(
    model,
    "manifests/train_manifest.json",
    audio_processor
)

# Use statistics for normalization
normalizer = LatentNormalizer("precomputed_latents/latent_statistics.json")
normalized_latent = normalizer.normalize(latent)
original_latent = normalizer.denormalize(normalized_latent)
    """)


# ==============================================================================
# EXAMPLE 6: Complete Training Pipeline
# ==============================================================================

def example_complete_pipeline():
    """Demonstrate complete training pipeline integration."""

    print("\n" + "=" * 70)
    print("COMPLETE TRAINING PIPELINE")
    print("=" * 70)

    print("""
from hindi_text_processor import HindiTextProcessor
from audio_processor import AudioProcessor
from dataset import HindiTTSDataset, collate_tts_batch
from torch.utils.data import DataLoader
import torch
import torch.nn as nn

# 1. INITIALIZE PROCESSORS
text_processor = HindiTextProcessor()
audio_processor = AudioProcessor()

# 2. CREATE TRAIN/VAL DATASETS
train_dataset = HindiTTSDataset(
    "manifests/train_manifest.json",
    audio_processor=audio_processor,
    text_processor=text_processor,
    max_text_length=256,
    normalize_text=True
)

val_dataset = HindiTTSDataset(
    "manifests/val_manifest.json",
    audio_processor=audio_processor,
    text_processor=text_processor,
    max_text_length=256,
    normalize_text=True
)

# 3. CREATE DATA LOADERS
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_tts_batch
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_tts_batch
)

# 4. TRAINING LOOP
model = YourTTSModel(...)
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(num_epochs):
    for batch in train_loader:
        text_ids = batch["text_token_ids"]        # (B, seq_len)
        mel_spec = batch["mel_spectrogram"]       # (B, 228, T)
        speaker_ids = batch["speaker_id"]         # List[str]
        durations = batch["duration"]             # (B,)

        # Forward pass
        output = model(
            text_ids,
            mel_spec,
            speaker_ids=speaker_ids
        )

        # Compute loss and backward
        loss = criterion(output, mel_spec)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# 5. VALIDATION
with torch.no_grad():
    for batch in val_loader:
        text_ids = batch["text_token_ids"]
        mel_spec = batch["mel_spectrogram"]
        output = model(text_ids, mel_spec)
        val_loss = criterion(output, mel_spec)
    """)


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 20 + "SUPERTONIC TTS DATA PIPELINE" + " " * 20 + "║")
    print("║" + " " * 22 + "USAGE EXAMPLES AND PATTERNS" + " " * 20 + "║")
    print("╚" + "=" * 68 + "╝")

    # Run examples
    example_text_normalization()
    example_audio_processing()
    example_dataset_creation()
    example_dataset_preparation()
    example_latent_precomputation()
    example_complete_pipeline()

    print("\n" + "=" * 70)
    print("For more information, see:")
    print("  - QUICK_START.md for quick setup")
    print("  - PIPELINE_DOCUMENTATION.md for detailed API")
    print("=" * 70 + "\n")
