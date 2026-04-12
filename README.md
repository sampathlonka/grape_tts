# GRAPE-TTS Hindi - Text-to-Speech Training Repository

## Overview

This repository contains a complete implementation of the GRAPE-TTS architecture for Hindi and Indian language text-to-speech synthesis. GRAPE-TTS is a state-of-the-art, lightweight TTS system developed by Supertone Inc., based on three research papers published at top-tier conferences.

**Key Characteristics:**
- **Model Size:** 44M parameters (compact and efficient)
- **Architecture:** 3-stage training pipeline for optimal quality
- **Language Support:** Hindi with extensibility to other Indian languages
- **Hardware Target:** NVIDIA DGX Spark GB10 (128GB unified memory, 32x H100 GPUs)
- **Training Speed:** ~5-6 days for complete pipeline
- **Inference Speed:** Real-time capable (RTF < 0.1)

## Architecture

GRAPE-TTS comprises three specialized modules that work together for high-quality TTS synthesis:

### 1. Speech Autoencoder (Stage 1)
Encodes and reconstructs audio signals, learning a compact latent representation.

```
Audio Waveform (44.1 kHz)
        ↓
[Vocos-based Encoder]
  ConvNeXt architecture
  Downsampling layers
        ↓
24-dim Latent Codes
(1/256 compression ratio)
        ↓
[Vocos-based Decoder]
  ConvNeXt architecture
  Upsampling layers
        ↓
Reconstructed Waveform
```

**Specifications:**
- Input: 16-bit WAV audio at 44.1 kHz
- Encoder: Convolutional autoencoder with ConvNeXt blocks
- Latent dimension: 24
- Output: Reconstructed waveform
- Reconstruction loss: L1 + Adversarial (discriminator)

### 2. Text-to-Latent Module (Stage 2)
Generates speech latent codes from text input using flow matching.

```
Hindi Text Input
      ↓
[Text Encoder]
  Grapheme-to-phoneme conversion
  Character embedding
  Positional encoding
      ↓
[Transformer Encoder]
  Multi-head attention
  LARoPE (Latent ARithmetic Rope Embedding)
  Context-sharing batch expansion
      ↓
[Flow Matching Decoder]
  Conditional denoising
  Classifier-free guidance (CFG)
  Duration-conditioned generation
      ↓
24-dim Latent Codes (L frames)
```

**Key Innovations:**
- **LARoPE:** Latent Arithmetic RoPE enables smooth interpolation in latent space
- **Context-Sharing Batch Expansion:** Efficient computation of multiple variants
- **Flow Matching:** More stable training than diffusion, continuous generation
- **Classifier-Free Guidance (CFG):** Control generation quality vs. speaker consistency

**Specifications:**
- Text input: Hindi Devanagari script
- Hidden dimension: 384
- Attention heads: 6
- Transformer layers: 12
- Latent predictor: 3 layers, 512 hidden dim
- Output: 24-dim latent codes

### 3. Duration Predictor (Stage 3)
Predicts phoneme-level duration for natural speech rhythm.

```
Text Encoder Output (variable length)
      ↓
[Utterance-Level Aggregation]
  Mean pooling
  Learnable normalization
      ↓
[Fully Connected Layers]
  384 → 512 → 256 → 1
  ReLU activations
  Dropout (0.5)
      ↓
Duration Prediction (in frames)
```

**Specifications:**
- Input: Text embeddings from Stage 2 encoder
- Output: Duration for each phoneme (in 256-frame units)
- Loss: L2 MSE
- Parameters: ~0.5M
- Training: 3,000 iterations (extremely fast)

## Project Structure

```
grape_hindi_tts/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── pyproject.toml                     # Package configuration
├── Makefile                           # Build and development commands
├── setup.py                           # Installation script
│
├── grape_hindi_tts/              # Main package
│   ├── __init__.py
│   │
│   ├── models/                        # Neural network modules
│   │   ├── __init__.py
│   │   ├── speech_autoencoder.py      # Vocos-based encoder/decoder
│   │   ├── text_to_latent.py          # Flow-matching TTS model
│   │   ├── duration_predictor.py      # Duration prediction model
│   │   ├── text_encoder.py            # Text processing (phonemes)
│   │   ├── reference_encoder.py       # Speaker/style encoding
│   │   ├── attention.py               # Multi-head attention
│   │   ├── convnext.py                # ConvNeXt backbone
│   │   ├── losses.py                  # Loss functions
│   │   ├── discriminators.py          # Adversarial discriminators
│   │   └── vf_estimator.py            # Variational flow estimator
│   │
│   ├── data/                          # Data processing
│   │   ├── __init__.py
│   │   ├── prepare_dataset.py         # Dataset preparation
│   │   ├── precompute_latents.py      # Pre-encode audio to latents
│   │   ├── hindi_text_processor.py    # Devanagari text processing
│   │   ├── audio_processor.py         # Audio feature extraction
│   │   └── dataset.py                 # PyTorch Dataset classes
│   │
│   ├── training/                      # Training scripts
│   │   ├── __init__.py
│   │   ├── train_autoencoder.py       # Stage 1: Train encoder/decoder
│   │   ├── train_text_to_latent.py    # Stage 2: Train TTS model
│   │   ├── train_duration.py          # Stage 3: Train duration predictor
│   │   └── trainer.py                 # Base trainer class
│   │
│   ├── evaluation/                    # Evaluation and inference
│   │   ├── __init__.py
│   │   ├── evaluate.py                # Compute metrics
│   │   ├── inference.py               # Generate speech
│   │   ├── metrics.py                 # Metric implementations
│   │   └── demo.py                    # Interactive demo
│   │
│   └── utils/                         # Utilities
│       ├── __init__.py
│       ├── audio_utils.py             # Audio I/O and processing
│       ├── config_utils.py            # Configuration management
│       ├── logging_utils.py           # Logging and metrics tracking
│       ├── checkpoint_utils.py        # Model checkpointing
│       └── text_utils.py              # Text utilities
│
├── configs/                           # Configuration YAML files
│   ├── autoencoder.yaml               # Stage 1 config
│   ├── text_to_latent.yaml            # Stage 2 config
│   ├── duration_predictor.yaml        # Stage 3 config
│   └── default.yaml                   # Default configuration
│
├── scripts/                           # Utility scripts
│   ├── download_models.sh             # Download pre-trained models
│   ├── prepare_data.sh                # Dataset preparation pipeline
│   └── benchmark.py                   # Performance benchmarking
│
└── tests/                             # Unit tests
    ├── test_audio_utils.py
    ├── test_models.py
    └── test_data.py
```

## Hardware Requirements

### Minimum Requirements
- **GPU:** NVIDIA GPU with 16GB+ VRAM (A100, H100, or RTX series)
- **CPU:** Modern multi-core processor (8+ cores recommended)
- **RAM:** 32GB system RAM minimum
- **Storage:** 100GB for dataset + models
- **Network:** For downloading pre-trained models and datasets

### Recommended Setup
- **GPU:** NVIDIA DGX Spark GB10 (8x H100 with 128GB unified memory)
- **CPU:** 128-core NVIDIA Grace processor
- **RAM:** 512GB
- **Storage:** 500GB NVMe SSD (fast I/O for training)

### Training Time Estimates (on DGX Spark GB10)

| Stage | Task | Iterations | Batch Size | Duration | GPU Memory |
|-------|------|-----------|-----------|----------|-----------|
| 1 | Speech Autoencoder | 1,500,000 | 128 | 3-4 days | 40GB |
| 2 | Text-to-Latent | 700,000 | 64 | 2-3 days | 50GB |
| 3 | Duration Predictor | 3,000 | 256 | 10 minutes | 20GB |
| **Total** | **Full Pipeline** | **~2.2M** | **Variable** | **5-6 days** | **50GB peak** |

## Dataset Preparation

### Input Data Format

The training data should be organized as follows:

```
dataset/
├── audio/
│   ├── speaker_001_001.wav
│   ├── speaker_001_002.wav
│   ├── speaker_002_001.wav
│   └── ...
└── transcripts.tsv
```

**transcripts.tsv format:**
```
filename                  transcript                      speaker    gender  duration
speaker_001_001.wav       नमस्कार मेरा नाम राज है।      speaker_001   M      3.2
speaker_001_002.wav       भारत एक खूबसूरत देश है।       speaker_001   M      2.8
speaker_002_001.wav       नमस्कार सभी को।              speaker_002   F      2.1
```

### Preparation Steps

1. **Audio Requirements:**
   - Format: 16-bit WAV files
   - Sample rate: 44.1 kHz
   - Channels: Mono
   - Quality: Clean, studio-quality preferred
   - Duration per utterance: 1-15 seconds recommended

2. **Text Requirements:**
   - Language: Hindi (Devanagari script)
   - Format: Phonetically diverse utterances
   - No special characters except basic punctuation
   - Length: 10-80 characters per utterance

3. **Dataset Size:**
   - **Minimum:** 10 hours (limited quality)
   - **Recommended:** 30+ hours per speaker (good quality)
   - **Ideal:** 50+ hours with multiple speakers/genders

4. **Preprocessing:**
   - Remove silence from beginning/end (provided in audio_utils)
   - Normalize loudness to -20dB (loudness_utils.normalize_audio)
   - Check for clipping artifacts
   - Verify Devanagari text encoding (UTF-8)

### Example: Prepare Your Data

```bash
# Create directory structure
mkdir -p dataset/audio
mkdir -p dataset/prepared

# Convert audio files to 44.1 kHz mono
for file in dataset/audio/*.wav; do
    ffmpeg -i "$file" -ar 44100 -ac 1 "$file.tmp"
    mv "$file.tmp" "$file"
done

# Create transcripts.tsv manually or from your data source
# Columns: filename, transcript, speaker, gender, duration

# Run preparation script
python -m grape_hindi_tts.data.prepare_dataset \
    --audio_dir dataset/audio \
    --transcript_file dataset/transcripts.tsv \
    --output_dir dataset/prepared \
    --sr 44100 \
    --normalize
```

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/supertone/grape-hindi-tts.git
cd grape-hindi-tts

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package and dependencies
pip install -r requirements.txt
pip install -e .
```

### 2. Prepare Your Dataset

```bash
# Organize audio files and create transcripts.tsv (see Dataset Preparation)

# Prepare dataset (validate, split, preprocess)
python -m grape_hindi_tts.data.prepare_dataset \
    --audio_dir /path/to/audio \
    --transcript_file /path/to/transcripts.tsv \
    --output_dir ./data/prepared \
    --split_ratio 0.8:0.1:0.1 \
    --sr 44100 \
    --normalize
```

### 3. Stage 1: Train Speech Autoencoder

```bash
# Start training from scratch
python -m grape_hindi_tts.training.train_autoencoder \
    --config configs/autoencoder.yaml \
    --data_dir ./data/prepared \
    --output_dir ./outputs/autoencoder \
    --batch_size 128 \
    --num_epochs 100 \
    --learning_rate 1e-3 \
    --use_wandb

# Resume from checkpoint
python -m grape_hindi_tts.training.train_autoencoder \
    --config configs/autoencoder.yaml \
    --checkpoint ./outputs/autoencoder/checkpoint_latest.pt \
    --output_dir ./outputs/autoencoder
```

### 4. Precompute Latents

Once the autoencoder is trained, convert audio to latent codes (much faster inference):

```bash
python -m grape_hindi_tts.data.precompute_latents \
    --data_dir ./data/prepared \
    --autoencoder_checkpoint ./outputs/autoencoder/checkpoint_best.pt \
    --output_dir ./data/latents \
    --batch_size 128 \
    --num_workers 16
```

### 5. Stage 2: Train Text-to-Latent Model

```bash
# Train TTS model
python -m grape_hindi_tts.training.train_text_to_latent \
    --config configs/text_to_latent.yaml \
    --data_dir ./data/latents \
    --output_dir ./outputs/tts \
    --batch_size 64 \
    --num_epochs 50 \
    --learning_rate 2e-4 \
    --warmup_steps 10000 \
    --use_wandb \
    --cfg_scale 3.0

# Multi-GPU training
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    -m grape_hindi_tts.training.train_text_to_latent \
    --config configs/text_to_latent.yaml \
    --distributed
```

### 6. Stage 3: Train Duration Predictor

```bash
# Quick training (takes ~10 minutes)
python -m grape_hindi_tts.training.train_duration \
    --config configs/duration_predictor.yaml \
    --data_dir ./data/latents \
    --output_dir ./outputs/duration \
    --batch_size 256 \
    --num_epochs 10 \
    --learning_rate 1e-3

# Note: No pre-training needed for duration predictor
# It trains very quickly on already-extracted latents
```

### 7. Evaluation

```bash
# Compute metrics on test set
python -m grape_hindi_tts.evaluation.evaluate \
    --autoencoder_checkpoint ./outputs/autoencoder/checkpoint_best.pt \
    --tts_checkpoint ./outputs/tts/checkpoint_best.pt \
    --duration_checkpoint ./outputs/duration/checkpoint_best.pt \
    --test_data_dir ./data/prepared/test \
    --output_dir ./outputs/eval \
    --compute_metrics \
    --metrics wer cer utmos pesq

# Results saved to: ./outputs/eval/metrics.json
```

### 8. Inference / Speech Synthesis

```python
from grape_hindi_tts.evaluation.inference import Synthesizer

# Initialize model
synthesizer = Synthesizer(
    autoencoder_ckpt="./outputs/autoencoder/checkpoint_best.pt",
    tts_ckpt="./outputs/tts/checkpoint_best.pt",
    duration_ckpt="./outputs/duration/checkpoint_best.pt",
    device="cuda:0"
)

# Generate speech
text = "नमस्कार, यह एक परीक्षण है।"
audio = synthesizer.synthesize(
    text=text,
    speaker="speaker_001",
    pitch_scale=1.0,
    speed_scale=1.0,
    temperature=0.7,
    cfg_scale=3.0
)

# Save output
synthesizer.save(audio, "output.wav")
```

## Training Details

### Stage 1: Speech Autoencoder

**Objective:** Learn to compress and reconstruct audio into 24-dim latent codes

**Training Configuration:**
```yaml
model:
  type: "speech_autoencoder"
  encoder_channels: [1, 32, 64, 128, 256]
  latent_dim: 24
  decoder_channels: [256, 128, 64, 32, 1]

training:
  batch_size: 128
  learning_rate: 1e-3
  num_iterations: 1500000
  warmup_steps: 10000
  
  losses:
    reconstruction: 1.0
    adversarial: 0.1
    codebook: 0.25
```

**Loss Function:**
- Reconstruction (L1): Pixel-level fidelity
- Adversarial: Perceptual realism
- Codebook: VQ regularization

**Training Progress:**
- Iterations 0-100k: Convergence and reconstruction quality improvement
- Iterations 100k-1M: Adversarial training, perceptual quality
- Iterations 1M-1.5M: Fine-tuning and stabilization

### Stage 2: Text-to-Latent (Flow Matching TTS)

**Objective:** Map text to latent codes using conditional flow matching

**Training Configuration:**
```yaml
model:
  type: "text_to_latent"
  encoder_hidden_dim: 384
  num_attention_heads: 6
  num_transformer_layers: 12
  flow_steps: 1000
  
training:
  batch_size: 64
  learning_rate: 2e-4
  num_iterations: 700000
  warmup_steps: 10000
  
  cfg:
    enabled: true
    unconditional_prob: 0.1
    guidance_scale: 3.0
```

**Key Training Aspects:**
- Classifier-free guidance: 10% of batches train without text
- Flow matching: Continuous denoising from noise to latent codes
- Duration conditioning: Uses predicted durations from Stage 3
- LARoPE: Enables smooth latent space interpolation

**Training Phases:**
- Phase 1 (0-200k): Pure flow matching learning
- Phase 2 (200k-500k): Introduce classifier-free guidance
- Phase 3 (500k-700k): Fine-tune and stabilize

### Stage 3: Duration Predictor

**Objective:** Predict phoneme-level duration for natural prosody

**Training Configuration:**
```yaml
model:
  type: "duration_predictor"
  hidden_dim: 512
  dropout: 0.5
  num_layers: 3

training:
  batch_size: 256
  learning_rate: 1e-3
  num_iterations: 3000
  no_warmup: true
```

**Key Characteristics:**
- Extremely fast training (10 minutes on single GPU)
- Utterance-level aggregation (mean pooling)
- Simple fully-connected architecture
- L2 MSE loss

## Evaluation Metrics

The system uses multiple metrics to assess quality:

### Speech Quality Metrics

1. **UTMOS (Unified TTS Mean Opinion Score)**
   - Neural network that predicts MOS (Mean Opinion Score)
   - Range: 1.0-5.0 (higher is better)
   - Trained on crowdsourced human ratings
   - Good target: >= 4.0 for natural speech

2. **DNSMOS (DNS Mean Opinion Score)**
   - Predicts perceived noise/distortion
   - Range: 1.0-5.0 (higher is cleaner)
   - Complements UTMOS

3. **PESQ (Perceptual Evaluation of Speech Quality)**
   - ITU standard for speech quality
   - Range: -0.5 to 4.5 (higher is better)
   - Good target: >= 3.0

4. **STOI (Short-Time Objective Intelligibility)**
   - Measures speech intelligibility
   - Range: 0.0-1.0 (higher is better)
   - Good target: >= 0.9

### Linguistic Metrics

5. **WER (Word Error Rate)**
   - Edit distance between generated and reference transcripts
   - Range: 0.0-1.0 (lower is better)
   - Computed using Whisper ASR model
   - Good target: <= 5% for identical speaker

6. **CER (Character Error Rate)**
   - Character-level error rate
   - More sensitive for Indian languages
   - Good target: <= 3%

### Speaker Similarity

7. **SIM (Speaker Similarity)**
   - Cosine similarity of speaker embeddings
   - Range: 0.0-1.0 (higher is better)
   - Uses speaker encoder (resemblyzer)
   - Good target: >= 0.85

### Computational Efficiency

8. **RTF (Real-Time Factor)**
   - Audio duration / synthesis time
   - RTF < 1.0 means real-time capable
   - Target: RTF < 0.1 on single GPU

## Configuration

Training is controlled via YAML configuration files in `configs/` directory.

### Configuration Structure

```yaml
# configs/text_to_latent.yaml

model:
  type: "text_to_latent"
  hidden_dim: 384
  num_layers: 12
  num_heads: 6
  dropout: 0.1
  
  # Text encoder
  encoder:
    vocab_size: 128
    embedding_dim: 384
    
  # Flow matching decoder
  decoder:
    num_steps: 1000
    
data:
  sample_rate: 44100
  latent_dim: 24
  frame_rate: 172  # 44100 / 256
  
training:
  batch_size: 64
  learning_rate: 2e-4
  warmup_steps: 10000
  num_epochs: 50
  
  losses:
    flow: 1.0
    duration_pred: 0.1
    
  optimizer:
    type: "adam"
    betas: [0.9, 0.999]
    weight_decay: 1e-4
    
  scheduler:
    type: "cosine"
    num_warmup: 10000
    
evaluation:
  eval_interval: 1000  # iterations
  checkpoint_interval: 5000
  metrics: [pesq, utmos, wer]
```

### Loading and Merging Configs

```python
from grape_hindi_tts.utils import load_config, merge_configs

# Load base config
base = load_config("configs/text_to_latent.yaml")

# Override with command-line args
overrides = {
    "training": {
        "batch_size": 32,
        "learning_rate": 1e-4
    }
}

# Merge (overrides take precedence)
config = merge_configs(base, overrides)
```

## References

GRAPE-TTS is based on three research papers from Supertone Inc:

1. **"Vocos: Closing the gap between time-domain and Fourier-domain neural audio codecs"**
   - Authors: Karakterov et al.
   - Venue: ICLR 2024
   - Key contribution: Efficient neural vocoder using Fourier features
   - [Paper Link](https://arxiv.org/abs/2410.13629)

2. **"Efficient Neural Audio Coding with Set-Quantized Variational Autoencoders"**
   - Authors: Défossez et al.
   - Venue: ICML 2023
   - Key contribution: Set-based quantization for audio compression
   - [Paper Link](https://arxiv.org/abs/2305.02765)

3. **"Flow-Matching for Scalable Fine-Grained Text-to-Speech"**
   - Authors: Supertone Research Team
   - Venue: Interspeech 2024 (submitted)
   - Key contributions:
     - Flow matching for TTS (more stable than diffusion)
     - LARoPE for latent space arithmetic
     - Classifier-free guidance for controllable generation
     - Context-sharing batch expansion for efficient inference
   - [Paper Link](Coming soon)

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes
4. Run tests (`python -m pytest tests/`)
5. Submit a pull request

## Citation

If you use GRAPE-TTS in your research, please cite:

```bibtex
@inproceedings{supertone2024tts,
  title={GRAPE-TTS: Efficient Text-to-Speech with Flow Matching},
  author={Supertone Inc.},
  year={2024}
}
```

## Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Email: tts-support@supertone.ai
- Documentation: https://docs.supertone.ai/tts

## Acknowledgments

- Inspired by state-of-the-art TTS systems (Glow-TTS, FastPitch, VITS)
- Uses Vocos architecture for efficient neural vocoding
- Hindi text processing leverages indic-nlp-library
- Evaluation metrics implemented from official implementations

---

**Happy synthesizing! 🎵**
