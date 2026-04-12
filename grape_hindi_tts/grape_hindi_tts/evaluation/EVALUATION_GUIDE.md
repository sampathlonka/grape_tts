# SupertonicTTS Hindi - Evaluation Pipeline Guide

Complete inference and evaluation pipeline for SupertonicTTS Hindi TTS system.

## Overview

The evaluation module provides:

1. **inference.py** - Complete inference pipeline with:
   - Text processing and tokenization
   - Reference audio encoding
   - Duration prediction
   - Flow matching with Euler ODE solver (CFG support)
   - Latent decompression and waveform decoding
   - Batch inference and RTF measurement

2. **metrics.py** - Comprehensive metrics computation:
   - **WER/CER**: Word/Character Error Rate via Whisper ASR
   - **UTMOS**: Universal TMOS model for speech quality
   - **PESQ**: Perceptual Evaluation of Speech Quality
   - **STOI**: Short-Time Objective Intelligibility
   - **Speaker Similarity**: Cosine similarity of speaker embeddings
   - **RTF**: Real-Time Factor measurement

3. **evaluate.py** - Full evaluation on test manifests:
   - Batch evaluation with filtering
   - Per-sample and aggregated metrics
   - CSV and JSON report generation
   - Speaker-wise breakdown

4. **generate_samples.py** - Demo sample generation:
   - 25+ default Hindi test sentences
   - HTML page for easy listening
   - Covers multiple sentence types

## Installation

### Required Dependencies

```bash
# Core dependencies
pip install torch torchaudio librosa

# ASR (Whisper)
pip install openai-whisper

# Speaker similarity
pip install speechbrain
# OR
pip install resemblyzer

# Quality metrics
pip install pesq pystoi

# Text processing
pip install jiwer

# Data handling
pip install pandas
```

### Optional: Specialized Model Installations

```bash
# UTMOS (if available)
pip install google-universal-tts-mos

# For enhanced speaker embeddings
pip install speechbrain
```

## Usage

### 1. Basic Synthesis (inference.py)

#### Command Line

```bash
python -m supertonic_hindi_tts.evaluation.inference \
    --autoencoder /path/to/autoencoder.pt \
    --text-to-latent /path/to/text_to_latent.pt \
    --duration-predictor /path/to/duration_predictor.pt \
    --text "मुझे बहुत खुशी है आपसे मिलकर।" \
    --reference /path/to/reference.wav \
    --output output.wav \
    --cfg-scale 3.0 \
    --duration-scale 1.0 \
    --n-steps 32
```

#### Python API

```python
from supertonic_hindi_tts.evaluation import SupertonicTTSInference

# Initialize
inference = SupertonicTTSInference(
    autoencoder_path="models/autoencoder.pt",
    text_to_latent_path="models/text_to_latent.pt",
    duration_predictor_path="models/duration_predictor.pt",
    device="cuda"
)

# Single synthesis
waveform = inference.synthesize(
    text="यह एक परीक्षण है।",
    reference_audio_path="reference.wav",
    duration_scale=1.0,
    cfg_scale=3.0,
    n_steps=32
)

# Save
inference.save_audio(waveform, "output.wav")

# Batch synthesis
texts = ["पहला वाक्य।", "दूसरा वाक्य।"]
results = inference.synthesize_batch(
    texts=texts,
    reference_audio_paths="reference.wav",
    cfg_scale=3.0,
    n_steps=32
)

for waveform, rtf, text in results:
    print(f"RTF: {rtf:.3f} for: {text}")
```

#### Parameters

- **duration_scale**: Factor to scale predicted duration
  - < 1.0: Faster speech (e.g., 0.8x)
  - = 1.0: Natural speed
  - > 1.0: Slower speech (e.g., 1.2x)

- **cfg_scale**: Classifier-free guidance strength
  - 0.0: No guidance (pure unconditional)
  - 3.0: Default balanced guidance
  - > 5.0: Strong guidance, less natural variation

- **n_steps**: ODE solver steps (NFE)
  - 16-32: Fast inference
  - 32-64: Higher quality
  - 64+: Highest quality (slower)

### 2. Compute Metrics (metrics.py)

#### Command Line

```bash
python -m supertonic_hindi_tts.evaluation.metrics \
    --gen-audio generated.wav \
    --ref-audio reference.wav \
    --text "मूल पाठ" \
    --time 2.5
```

#### Python API

```python
from supertonic_hindi_tts.evaluation import MetricComputer

computer = MetricComputer(device="cuda")

# Transcribe audio
text = computer.transcribe_audio("audio.wav", language="hi")

# Individual metrics
wer = computer.compute_wer("reference text", text)
cer = computer.compute_cer("reference text", text)
similarity = computer.compute_speaker_similarity("ref.wav", "gen.wav")
utmos = computer.compute_utmos("audio.wav")
pesq = computer.compute_pesq("ref.wav", "gen.wav")
stoi = computer.compute_stoi("ref.wav", "gen.wav")
rtf = computer.compute_rtf(audio_duration=10.5, generation_time=3.2)

# All metrics at once
results = computer.compute_all(
    gen_audio_path="generated.wav",
    ref_audio_path="reference.wav",
    ground_truth_text="मूल पाठ",
    generation_time=2.5
)

print(f"WER: {results['wer']:.4f}")
print(f"UTMOS: {results['utmos']:.2f}")
print(f"RTF: {results['rtf']:.3f}")
```

### 3. Full Evaluation (evaluate.py)

#### Test Manifest Format

**JSON Lines (.jsonl)**
```jsonl
{"text": "पहला वाक्य", "reference_audio": "ref1.wav", "audio": "gt1.wav", "speaker": "speaker1", "id": "s1_1"}
{"text": "दूसरा वाक्य", "reference_audio": "ref1.wav", "audio": "gt2.wav", "speaker": "speaker1", "id": "s1_2"}
```

**CSV (.csv)**
```csv
text,reference_audio,audio,speaker,gender,duration,id
"पहला वाक्य",ref1.wav,gt1.wav,speaker1,M,5.2,s1_1
"दूसरा वाक्य",ref1.wav,gt2.wav,speaker1,M,4.8,s1_2
```

**JSON (.json)**
```json
{
  "samples": [
    {"text": "...", "reference_audio": "...", "audio": "...", "speaker": "..."},
    ...
  ]
}
```

#### Command Line

```bash
# Full evaluation
python -m supertonic_hindi_tts.evaluation.evaluate \
    --autoencoder models/autoencoder.pt \
    --text-to-latent models/text_to_latent.pt \
    --duration-predictor models/duration_predictor.pt \
    --manifest test_manifest.jsonl \
    --output results/ \
    --cfg-scale 3.0 \
    --n-steps 32

# With filters
python -m supertonic_hindi_tts.evaluation.evaluate \
    --manifest test_manifest.jsonl \
    --output results/ \
    --speaker speaker1 \
    --gender M \
    --duration-min 3.0 \
    --duration-max 10.0 \
    --max-samples 20 \
    [other options...]
```

#### Python API

```python
from supertonic_hindi_tts.evaluation import SupertonicEvaluator

evaluator = SupertonicEvaluator(
    autoencoder_path="models/autoencoder.pt",
    text_to_latent_path="models/text_to_latent.pt",
    duration_predictor_path="models/duration_predictor.pt",
    device="cuda"
)

summary = evaluator.evaluate_manifest(
    manifest_path="test_manifest.jsonl",
    output_dir="results/",
    reference_speaker="speaker1",
    duration_range=(3.0, 10.0),
    gender_filter="M",
    max_samples=50,
    cfg_scale=3.0,
    duration_scale=1.0,
    n_steps=32
)

print(f"WER: {summary['wer']['mean']:.4f} ± {summary['wer']['std']:.4f}")
print(f"UTMOS: {summary['utmos']['mean']:.2f} ± {summary['utmos']['std']:.2f}")
```

#### Output Files

- **results_detailed.csv**: Per-sample metrics
  - Columns: sample_id, speaker, gender, ground_truth, transcribed, wer, cer, utmos, pesq, stoi, speaker_similarity, rtf, generation_time, error

- **summary.json**: Aggregated statistics
  - Overall metrics (mean, std, min, max, median)
  - Per-speaker breakdown
  - Timestamp and sample count

### 4. Generate Demo Samples (generate_samples.py)

#### Command Line

```bash
python -m supertonic_hindi_tts.evaluation.generate_samples \
    --autoencoder models/autoencoder.pt \
    --text-to-latent models/text_to_latent.pt \
    --duration-predictor models/duration_predictor.pt \
    --reference reference.wav \
    --output demo_samples/ \
    --cfg-scale 3.0 \
    --n-steps 32
```

#### Python API

```python
from supertonic_hindi_tts.evaluation import SampleGenerator, DEFAULT_TEST_SENTENCES

generator = SampleGenerator(
    autoencoder_path="models/autoencoder.pt",
    text_to_latent_path="models/text_to_latent.pt",
    duration_predictor_path="models/duration_predictor.pt",
    device="cuda"
)

# Default test sentences
samples_data = generator.generate_samples(
    reference_audio_path="reference.wav",
    output_dir="demo_samples/",
    sentences=None,  # Uses DEFAULT_TEST_SENTENCES
    cfg_scale=3.0,
    n_steps=32
)

# Custom sentences
custom_sentences = {
    "my_category": [
        "पहला वाक्य",
        "दूसरा वाक्य",
    ]
}

samples_data = generator.generate_samples(
    reference_audio_path="reference.wav",
    output_dir="demo_samples/",
    sentences=custom_sentences,
)
```

#### Output

- Audio files: `{category}_{index}.wav`
- Metadata: `metadata.json` with audio file info
- HTML page: `index.html` for easy listening comparison

Open `index.html` in a web browser to listen to samples with metadata.

## Default Test Sentences

The `DEFAULT_TEST_SENTENCES` dictionary includes 25+ sentences covering:

1. **Simple Declarative Sentences** (5 samples)
   - "मुझे बहुत खुशी है आपसे मिलकर।"
   - "यह एक सुंदर दिन है।"
   - "भारत एक महान देश है।"

2. **Questions** (5 samples)
   - "आप कैसे हैं?"
   - "क्या आप कल आ सकते हैं?"

3. **Exclamations** (5 samples)
   - "वाह, कितना शानदार है!"
   - "यह तो अद्भुत है!"

4. **Long Paragraph** (1 sample, ~30 seconds)
   - Tests extended generation ability

5. **Numbers and Dates** (5 samples)
   - "पचास प्रतिशत की छूट पाएं।"
   - "आज की तारीख पन्द्रह अप्रैल है।"

6. **Code-Switching** (5 samples)
   - "Artificial Intelligence को Hindi में कृत्रिम बुद्धिमत्ता कहते हैं।"
   - "मैं हर दिन software development करता हूँ।"

## Architecture Details

### Flow Matching Inference

The inference uses a rectified flow matching approach with Euler ODE solver:

```
1. Sample z_0 ~ N(0, I) of target length
2. For t = 0 to T:
   - Encode text features
   - Encode reference features
   - Compute velocity field:
     v_cond = VF_model(z_t, text, ref, t)
     v_uncond = VF_model(z_t, None, None, t)
     v_guided = (1 + cfg) * v_cond - cfg * v_uncond
   - Euler step: z_{t+dt} = z_t + (1/N_steps) * v_guided
3. Return final z, decompress, decode to waveform
```

### Latent Compression/Decompression

- **Compression**: (B, 24, T) → (B, 144, T//6)
  - Groups 6 consecutive frames and flattens channels
- **Decompression**: (B, 144, T) → (B, 24, T*6)
  - Reverses the compression process

### Key Model Dimensions

- Text vocab: 256 characters
- Latent dimension: 24
- Compressed latent: 144 (24 * 6)
- Mel spectrogram: 228 dimensions
- Sample rate: 44.1 kHz

## Performance Tips

### Speed vs Quality Trade-off

```
Fastest (Real-time or faster):
- n_steps=16, cfg_scale=0.0 (no guidance)
- RTF: 0.3-0.5 (3-5x faster than real-time)

Balanced:
- n_steps=32, cfg_scale=3.0 (default)
- RTF: 0.5-1.5 (near real-time)

Highest Quality:
- n_steps=64, cfg_scale=3.0
- RTF: 1.5-3.0 (slower than real-time)
```

### GPU Memory Requirements

- Batch size 1: ~6GB VRAM
- Batch size 4: ~12GB VRAM

### CPU Inference

- Slower but possible: set `device="cpu"`
- Expect 10-30x slower than GPU

## Evaluation Results Interpretation

### WER/CER Scores
- 0.0: Perfect transcription match
- 0.1-0.2: Excellent quality (92-98% accuracy)
- 0.2-0.4: Good quality (80-92% accuracy)
- > 0.4: Poor transcription quality

### UTMOS Score (MOS Prediction)
- 1-2: Poor quality
- 2-3: Fair quality
- 3-4: Good quality
- 4-5: Excellent quality

### PESQ Score
- -0.5 to 0.5: Poor
- 0.5 to 1.5: Fair
- 1.5 to 2.5: Good
- 2.5 to 4.5: Excellent

### STOI Score
- 0.0-0.3: Poor intelligibility
- 0.3-0.6: Fair intelligibility
- 0.6-0.9: Good intelligibility
- 0.9-1.0: Excellent intelligibility

### Speaker Similarity
- 0.5-0.6: Low similarity
- 0.6-0.7: Medium similarity
- 0.7-0.8: Good similarity
- 0.8-1.0: High similarity

### RTF (Real-Time Factor)
- < 0.1: Very fast (10x+ real-time)
- 0.1-1.0: Real-time or faster
- 1.0-10: Slower than real-time
- > 10: Very slow

## Troubleshooting

### CUDA Out of Memory
```python
# Reduce batch size
results = inference.synthesize_batch(texts, ..., max_batch=1)

# Or use CPU
inference = SupertonicTTSInference(..., device="cpu")
```

### Whisper ASR Errors
```bash
# Ensure Whisper is installed
pip install openai-whisper

# Pre-download model
python -c "import whisper; whisper.load_model('large-v2')"
```

### Missing Speaker Encoder
```bash
# Install SpeechBrain
pip install speechbrain

# Or Resemblyzer
pip install resemblyzer
```

### Manifest Loading Errors
```python
# Check format
import json
with open("manifest.jsonl") as f:
    sample = json.loads(f.readline())
    print(sample.keys())  # Must have: text, reference_audio, audio
```

## Citation

If you use this evaluation pipeline, please cite:

```
SupertonicTTS: A Lightweight and Efficient Text-to-Speech System
for Hindi Language Synthesis
```

## License

Same as SupertonicTTS main project

## Support

For issues or questions, please open an issue on the project repository.
