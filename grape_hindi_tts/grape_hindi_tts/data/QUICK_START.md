# Quick Start Guide - SupertonicTTS Data Pipeline

## 5-Minute Setup

### 1. Install Dependencies
```bash
pip install torch torchaudio librosa numpy tqdm
```

### 2. Prepare Your Data

Create a transcript file (`transcripts.txt`):
```
audio/sample1.wav	नमस्ते दुनिया	speaker_001	M
audio/sample2.wav	यह एक परीक्षण है	speaker_002	F
audio/sample3.wav	सुप्रभात मित्र	speaker_001	M
```

### 3. Generate Manifests
```bash
python prepare_dataset.py \
  ./audio \
  ./transcripts.txt \
  --output_dir ./manifests
```

### 4. Create Training Dataset
```python
from dataset import HindiTTSDataset, collate_tts_batch
from torch.utils.data import DataLoader

dataset = HindiTTSDataset("manifests/train_manifest.json")
loader = DataLoader(dataset, batch_size=32, collate_fn=collate_tts_batch)

for batch in loader:
    print(f"Text: {batch['text_token_ids'].shape}")
    print(f"Audio: {batch['mel_spectrogram'].shape}")
    break
```

## Common Tasks

### Text Normalization Only
```python
from hindi_text_processor import HindiTextProcessor

processor = HindiTextProcessor()
normalized = processor.normalize_text("नमस्ते, 123 रुपये!")
# Output: "नमस्ते , एक सौ तेईस रुपये !"
```

### Audio Processing Only
```python
from audio_processor import AudioProcessor

processor = AudioProcessor()
waveform, sr = processor.process_audio_file("audio.wav")
mel = processor.compute_mel_spectrogram(waveform)
print(mel.shape)  # (228, time_steps)
```

### Tokenize Text
```python
processor = HindiTextProcessor()

# Text to IDs
token_ids = processor.text_to_token_ids("नमस्ते")
print(token_ids)  # [2, 45, 67, 89, 3]  (BOS...EOS)

# IDs back to text
text = processor.token_ids_to_text(token_ids)
print(text)  # "नमस्ते"
```

### Precompute Latents
```bash
python precompute_latents.py \
  ./checkpoints/autoencoder.pt \
  ./manifests/train_manifest.json \
  --output_dir ./latents
```

Then use in TTL dataset:
```python
from dataset import TTLDataset

dataset = TTLDataset(
    "manifests/train_manifest.json",
    "latents/"
)
```

## Troubleshooting

### Audio not loading
- Check file exists and is valid format
- Ensure sample rate supported (most formats OK)
- Check file permissions

### Text encoding issues
- Text must be UTF-8 encoded
- Ensure Devanagari characters are correct
- Check for hidden Unicode characters

### Dataset too small
- Check `dataset_report.txt` for skip reasons
- Increase `--max_duration` or decrease `--min_duration`
- Reduce `--max_silence` threshold

### Out of memory
- Reduce `batch_size` in DataLoader
- Use `load_audio=False` to load on-demand
- Process audio in separate worker processes

## File Size Reference

For 1000 audio samples (~1 hour total):
- Raw WAV: ~300-500 MB
- Manifests: ~100 KB
- Precomputed latents: ~100-200 MB

## Next Steps

1. **Explore data statistics**: Check `dataset_report.txt`
2. **Verify tokenization**: Test HindiTextProcessor examples
3. **Check audio quality**: Listen to processed samples
4. **Start training**: Use HindiTTSDataset in your model
5. **Monitor metrics**: Track loss, attention alignment, etc.

## Documentation

For detailed API documentation, see `PIPELINE_DOCUMENTATION.md`
