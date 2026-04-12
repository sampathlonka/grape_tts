# SupertonicTTS Hindi Evaluation - Quick Start Guide

Get up and running in 5 minutes with the evaluation pipeline.

## Installation (2 minutes)

```bash
# Install core dependencies
pip install torch torchaudio librosa

# Install evaluation dependencies
pip install openai-whisper speechbrain pesq pystoi jiwer pandas

# Optional: Download Whisper model in advance
python -c "import whisper; whisper.load_model('large-v2')"
```

## Quick Example 1: Single Synthesis (1 minute)

```python
from supertonic_hindi_tts.evaluation import SupertonicTTSInference

# Initialize
inference = SupertonicTTSInference(
    autoencoder_path="models/autoencoder.pt",
    text_to_latent_path="models/text_to_latent.pt",
    duration_predictor_path="models/duration_predictor.pt",
)

# Synthesize
waveform = inference.synthesize(
    text="नमस्ते, यह एक परीक्षण है।",
    reference_audio_path="speaker_sample.wav",
)

# Save
inference.save_audio(waveform, "output.wav")

print("✓ Generated output.wav")
```

## Quick Example 2: Generate Demo Samples (1 minute)

```python
from supertonic_hindi_tts.evaluation import SampleGenerator

generator = SampleGenerator(
    autoencoder_path="models/autoencoder.pt",
    text_to_latent_path="models/text_to_latent.pt",
    duration_predictor_path="models/duration_predictor.pt",
)

# Generate samples
generator.generate_samples(
    reference_audio_path="speaker_sample.wav",
    output_dir="demo_samples/",
)

print("✓ Generated demo_samples/index.html")
print("  Open in browser to listen")
```

## Quick Example 3: Full Evaluation (2 minutes)

**Step 1: Create test manifest** (test_manifest.jsonl)
```jsonl
{"text": "पहला वाक्य", "reference_audio": "ref.wav", "audio": "gt1.wav", "speaker": "speaker1", "id": "1"}
{"text": "दूसरा वाक्य", "reference_audio": "ref.wav", "audio": "gt2.wav", "speaker": "speaker1", "id": "2"}
```

**Step 2: Run evaluation**
```python
from supertonic_hindi_tts.evaluation import SupertonicEvaluator

evaluator = SupertonicEvaluator(
    autoencoder_path="models/autoencoder.pt",
    text_to_latent_path="models/text_to_latent.pt",
    duration_predictor_path="models/duration_predictor.pt",
)

summary = evaluator.evaluate_manifest(
    manifest_path="test_manifest.jsonl",
    output_dir="eval_results/",
)
```

**Step 3: Check results**
```
eval_results/
├── results_detailed.csv    # Per-sample metrics
├── summary.json           # Aggregate stats
└── speaker1_1_gen.wav     # Generated audio
└── speaker1_2_gen.wav
```

## Command Line Quick Start

### Synthesis
```bash
python -m supertonic_hindi_tts.evaluation.inference \
    --autoencoder models/ae.pt \
    --text-to-latent models/t2l.pt \
    --duration-predictor models/dp.pt \
    --text "नमस्ते" \
    --reference ref.wav \
    --output output.wav
```

### Generate Samples
```bash
python -m supertonic_hindi_tts.evaluation.generate_samples \
    --autoencoder models/ae.pt \
    --text-to-latent models/t2l.pt \
    --duration-predictor models/dp.pt \
    --reference ref.wav \
    --output demo/
```

### Evaluate
```bash
python -m supertonic_hindi_tts.evaluation.evaluate \
    --autoencoder models/ae.pt \
    --text-to-latent models/t2l.pt \
    --duration-predictor models/dp.pt \
    --manifest test_manifest.jsonl \
    --output results/
```

## Common Parameters

### Duration Control
- **1.0**: Natural speed (default)
- **0.8**: 20% faster
- **1.2**: 20% slower

### Quality vs Speed
```
Fast (RTF 0.3-0.5):
  --n-steps 16 --cfg-scale 0.0

Balanced (RTF 0.5-1.5):
  --n-steps 32 --cfg-scale 3.0

High Quality (RTF 1.5-3.0):
  --n-steps 64 --cfg-scale 3.0
```

## Output Interpretation

### CSV Results (results_detailed.csv)
```
sample_id | speaker | wer   | cer   | utmos | pesq  | stoi  | rtf
----------|---------|-------|-------|-------|-------|-------|-------
1         | s1      | 0.05  | 0.02  | 4.2   | 3.8   | 0.92  | 0.85
```

- **WER**: 0.0-0.4 (lower is better)
- **CER**: 0.0-0.4 (lower is better)
- **UTMOS**: 1-5 (higher is better)
- **PESQ**: -0.5 to 4.5 (higher is better)
- **STOI**: 0-1 (higher is better)
- **RTF**: < 1.0 is faster than real-time

### JSON Summary (summary.json)
```json
{
  "num_samples": 10,
  "num_errors": 0,
  "wer": {
    "mean": 0.08,
    "std": 0.05,
    "min": 0.02,
    "max": 0.15
  },
  "utmos": {
    "mean": 4.1,
    "std": 0.3
  },
  "by_speaker": {
    "speaker1": {
      "wer": {"mean": 0.07, "std": 0.04},
      "utmos": {"mean": 4.2, "std": 0.2}
    }
  }
}
```

## Troubleshooting

### "CUDA out of memory"
```python
# Use CPU instead
inference = SupertonicTTSInference(device="cpu")
```

### "Whisper model not found"
```bash
# Pre-download it
python -c "import whisper; whisper.load_model('large-v2')"
```

### "Speaker encoder not available"
```bash
# Install SpeechBrain
pip install speechbrain
```

### Wrong audio format
```python
# Make sure reference audio is readable
import librosa
audio, sr = librosa.load("reference.wav")
print(f"Loaded: {sr} Hz, {len(audio)/sr:.2f}s duration")
```

## Tips & Tricks

### Faster Evaluation
```python
# Evaluate only 10 samples for testing
evaluator.evaluate_manifest(
    manifest_path="test_manifest.jsonl",
    output_dir="results/",
    max_samples=10,
)
```

### Different Speakers
```python
# Evaluate per speaker
evaluator.evaluate_manifest(
    manifest_path="test_manifest.jsonl",
    output_dir="speaker1_results/",
    reference_speaker="speaker1",
)
```

### Batch Synthesis
```python
texts = ["पहला", "दूसरा", "तीसरा"]
results = inference.synthesize_batch(
    texts=texts,
    reference_audio_paths="ref.wav",
)

for waveform, rtf, text in results:
    inference.save_audio(waveform, f"{text}.wav")
```

## Expected Results

For a well-trained Hindi TTS model:

| Metric | Expected Range |
|--------|-----------------|
| WER    | 0.05 - 0.15 (92-97% accuracy) |
| CER    | 0.02 - 0.08 (92-98% accuracy) |
| UTMOS  | 3.8 - 4.5 (good to excellent) |
| PESQ   | 2.5 - 4.0 (good to excellent) |
| STOI   | 0.85 - 0.95 (good intelligibility) |
| Speaker Sim | 0.75 - 0.95 (good to high) |
| RTF    | 0.3 - 1.5 (fast to real-time) |

## Next Steps

1. **Prepare your test data**
   - Create JSONL manifest with text, reference audio, ground truth
   - Check file paths are correct

2. **Run evaluation**
   - Start with `max_samples=5` for testing
   - Check `results_detailed.csv` for metrics

3. **Analyze results**
   - Review per-sample WER, CER
   - Check speaker similarity scores
   - Inspect summary statistics

4. **Optimize**
   - Adjust `cfg_scale` for quality vs naturalness
   - Use `n_steps=16` for faster inference
   - Use `duration_scale` to adjust speech pace

## File Locations

All evaluation code is in:
```
supertonic_hindi_tts/evaluation/
├── inference.py           # Main synthesis
├── metrics.py             # Metric computation
├── evaluate.py            # Full evaluation
├── generate_samples.py    # Demo generation
├── __init__.py            # Exports
├── EVALUATION_GUIDE.md    # Full documentation
└── QUICK_START.md         # This file
```

## Getting Help

- Check `EVALUATION_GUIDE.md` for detailed documentation
- Review code docstrings: `help(SupertonicTTSInference.synthesize)`
- Look at error messages - they usually indicate the issue
- Check if required files exist: `--autoencoder`, `--reference`, etc.

## Summary

You now have a complete TTS inference and evaluation pipeline:

✓ Synthesis with duration control and CFG  
✓ 7 quality metrics (WER, CER, UTMOS, PESQ, STOI, Speaker Sim, RTF)  
✓ Batch evaluation on test manifests  
✓ Interactive HTML demo page  
✓ CSV and JSON result export  
✓ CLI and Python API  

Happy synthesizing! 🎙️
