# SupertonicTTS Training Scripts

Complete training pipeline for SupertonicTTS on NVIDIA DGX Spark GB10. All 3 training stages with full implementations, configurations, and utilities.

## Quick Start

Run all training stages with one command:

```bash
./QUICKSTART.sh
```

Or with custom paths:

```bash
OUTPUT_DIR=./my_outputs STAGE1_CONFIG=./my_config.yaml ./QUICKSTART.sh
```

## Directory Structure

```
training/
├── README.md                          # This file
├── QUICKSTART.sh                      # One-command training script
├── TRAINING_GUIDE.md                  # Detailed training guide
│
├── trainer_utils.py                   # Shared utilities (optimizers, checkpointing, etc.)
│
├── train_autoencoder.py              # Stage 1: Speech Autoencoder
├── train_text_to_latent.py           # Stage 2: Text-to-Latent Flow Matching
├── train_duration.py                 # Stage 3: Duration Predictor
│
├── run_all_stages.py                 # Master orchestration script
├── precompute_latents.py             # Utility for precomputing latents
│
├── config_autoencoder.yaml           # Stage 1 config
├── config_text_to_latent.yaml        # Stage 2 config
├── config_duration.yaml              # Stage 3 config
│
└── requirements.txt                   # Python dependencies
```

## Core Scripts

### 1. train_autoencoder.py (Stage 1)

Speech autoencoder with GAN training (encoder+decoder vs discriminator).

**Features:**
- Multi-Resolution Spectral Loss (FFT: 1024, 2048, 4096)
- Multi-Period Discriminator (MPD) with periods [2, 3, 5, 7, 11]
- Multi-Resolution Discriminator (MRD) with resolutions [8, 16, 32]
- Least-squares GAN (LS-GAN)
- Feature matching loss
- BF16 mixed precision
- Gradient checkpointing ready

**Usage:**
```bash
python train_autoencoder.py \
    --config config_autoencoder.yaml \
    --output_dir ./outputs/stage1_autoencoder
```

**Key Parameters:**
- Batch size: 128
- Learning rate: 2e-4
- Loss weights: λ_recon=45, λ_adv=1, λ_fm=0.1
- Duration: ~72 hours (1.5M iterations)

### 2. train_text_to_latent.py (Stage 2)

Text-to-latent flow matching with context-sharing batch expansion.

**Features:**
- Flow matching training with linear interpolation path
- Context-sharing batch expansion (Ke=4)
- Classifier-free guidance (p_uncond=0.05)
- Latent normalization (channel-wise)
- Self-Purifying Flow Matching (SPFM) optional
- BF16 mixed precision
- Masked L1 loss

**Usage:**
```bash
python train_text_to_latent.py \
    --config config_text_to_latent.yaml \
    --output_dir ./outputs/stage2_text_to_latent
```

**Key Parameters:**
- Batch size: 64 (effective: 256 with Ke=4)
- Learning rate: 5e-4
- σ_min: 1e-8
- Duration: ~52 hours (700k iterations)

### 3. train_duration.py (Stage 3)

Duration predictor - simple but effective L1 loss training.

**Features:**
- Text encoder + duration prediction head
- L1 loss between predicted and ground-truth duration
- Very fast training (3k iterations)
- BF16 mixed precision

**Usage:**
```bash
python train_duration.py \
    --config config_duration.yaml \
    --output_dir ./outputs/stage3_duration
```

**Key Parameters:**
- Batch size: 128
- Learning rate: 5e-4
- Duration: ~15 minutes (3k iterations)

## Utility Scripts

### run_all_stages.py

Master orchestration script that runs all 3 stages sequentially.

**Features:**
- Sequential execution with proper error handling
- Resume capability (--resume_stage 2)
- Checkpoint detection and continuation
- Training metadata tracking
- Summary statistics

**Usage:**
```bash
# Run all stages
python run_all_stages.py \
    --output_dir ./outputs \
    --stage1_config config_autoencoder.yaml \
    --stage2_config config_text_to_latent.yaml \
    --stage3_config config_duration.yaml

# Resume from stage 2
python run_all_stages.py \
    --output_dir ./outputs \
    --stage1_config config_autoencoder.yaml \
    --stage2_config config_text_to_latent.yaml \
    --stage3_config config_duration.yaml \
    --resume_stage 2
```

### precompute_latents.py

Precompute latents from audio files using trained autoencoder.

**Usage:**
```bash
# From audio directory
python precompute_latents.py \
    --autoencoder_checkpoint ./outputs/stage1_autoencoder/checkpoints/autoencoder_step_1500000_final.pt \
    --audio_dir ./path/to/audio/files \
    --output_path ./latents_precomputed.pt

# Generate dummy latents
python precompute_latents.py \
    --autoencoder_checkpoint ./outputs/stage1_autoencoder/checkpoints/autoencoder_step_1500000_final.pt \
    --output_path ./latents_precomputed.pt \
    --num_samples 10000
```

### trainer_utils.py

Shared utilities for all training scripts.

**Functions:**
- `setup_training()`: Initialize device, logging, directories
- `create_optimizer()`: AdamW optimizer creation
- `create_scheduler()`: Learning rate scheduler
- `save_checkpoint()`: Save model state
- `load_checkpoint()`: Load and resume training
- `setup_dgx_spark()`: DGX Spark optimizations
- `gradient_clip()`: Gradient norm clipping
- `count_parameters()`: Parameter counting
- `AverageMeter`: Running average tracker
- `GracefulInterruptHandler`: SIGINT/SIGTERM handling

## Configuration Files

### config_autoencoder.yaml

Controls Stage 1 training:

```yaml
# Loss weights (paper values)
lambda_recon: 45      # Reconstruction loss weight
lambda_adv: 1         # Adversarial loss weight
lambda_fm: 0.1        # Feature matching loss weight

# Model
latent_dim: 128
channels: 256

# Training
batch_size: 128
num_iterations: 1500000
gen_lr: 2e-4
disc_lr: 2e-4
```

### config_text_to_latent.yaml

Controls Stage 2 training:

```yaml
# Flow matching
sigma_min: 1.0e-8
context_expansion: 4  # Ke factor

# Classifier-free guidance
use_classifier_free: true
p_uncond: 0.05

# SPFM (optional)
use_spfm: true
spfm_warmup: 40000
spfm_threshold: 1.5

# Training
batch_size: 64
num_iterations: 700000
lr: 5e-4
```

### config_duration.yaml

Controls Stage 3 training:

```yaml
# Training
batch_size: 128
num_iterations: 3000
lr: 5e-4

# Duration range
min_duration: 0.5
max_duration: 10.0
```

## Advanced Features

### Mixed Precision Training (BF16)

Automatically enabled on DGX Spark with `use_amp: true` in config.

Benefits:
- Faster computation (2x speedup possible)
- Lower memory usage
- Maintains model quality

To disable:
```yaml
use_amp: false
```

### Gradient Clipping

Default norm clipping: 1.0 (configurable in config files).

```yaml
max_grad_norm: 1.0
```

### Context-Sharing Batch Expansion (Stage 2 only)

For Stage 2, Ke=4 means:
- Original batch: 64 samples
- Effective batch: 256 samples (64 * 4)
- Memory efficient: reuse text encodings

Increase Ke for larger effective batches:
```yaml
context_expansion: 8  # 64 * 8 = 512 effective
```

### Classifier-Free Guidance (Stage 2 only)

Drop text conditions with probability p_uncond during training:

```yaml
use_classifier_free: true
p_uncond: 0.05  # 5% of steps train unconditionally
```

This improves unconditional sampling at inference time.

## Monitoring

### TensorBoard

View training progress in real-time:

```bash
tensorboard --logdir ./outputs/stage1_autoencoder/tensorboard
tensorboard --logdir ./outputs/stage2_text_to_latent/tensorboard
tensorboard --logdir ./outputs/stage3_duration/tensorboard
```

Open http://localhost:6006 in your browser.

### Logs

All training logs saved to:
- `outputs/stage1_autoencoder/logs/autoencoder.log`
- `outputs/stage2_text_to_latent/logs/text_to_latent.log`
- `outputs/stage3_duration/logs/duration.log`

View live:
```bash
tail -f outputs/stage1_autoencoder/logs/autoencoder.log
```

### Checkpoints

Automatically saved every `ckpt_interval` iterations:

```
outputs/
├── stage1_autoencoder/checkpoints/
│   ├── autoencoder_step_50000.pt
│   ├── autoencoder_step_1500000_final.pt
│   └── autoencoder_step_1500000_best.pt
├── stage2_text_to_latent/checkpoints/
│   ├── text_to_latent_step_50000.pt
│   ├── text_to_latent_step_700000_final.pt
│   └── text_to_latent_step_700000_best.pt
└── stage3_duration/checkpoints/
    ├── duration_step_1000.pt
    ├── duration_step_3000_final.pt
    └── duration_step_3000_best.pt
```

Resume from specific checkpoint:

```bash
python train_autoencoder.py \
    --config config_autoencoder.yaml \
    --resume ./outputs/stage1_autoencoder/checkpoints/autoencoder_step_500000.pt
```

## Performance Benchmarks

Expected throughput on DGX Spark GB10:

| Stage | Batch Size | GPU Memory | Iters/Hour | Total Time |
|-------|-----------|-----------|-----------|-----------|
| Autoencoder | 128 | 100GB | ~270 | 72 hours |
| Text-to-Latent | 64 | 95GB | ~333 | 52 hours |
| Duration | 128 | 50GB | 15000+ | 15 min |

**Total training time: ~4.5 days** with recommended settings.

## Troubleshooting

### Out of Memory (OOM)

Reduce batch size in config:
```yaml
batch_size: 64  # From 128
```

### Training too slow

1. Check GPU utilization: `nvidia-smi`
2. Verify CUDA version matches PyTorch
3. Ensure `use_amp: true` is set
4. Use multiple data loading workers

### Validation loss not improving

1. Check learning rate (try 1e-4 or 1e-3)
2. Verify data loading is working
3. Check loss weights match paper values
4. Print gradient statistics

### Checkpoint loading fails

```bash
# Verify checkpoint exists
ls -lh ./outputs/stage1_autoencoder/checkpoints/autoencoder_step_500000.pt

# Check checkpoint contents
python -c "
import torch
ckpt = torch.load('path/to/checkpoint.pt')
print('Keys:', list(ckpt.keys()))
print('Model keys:', list(ckpt['model'].keys())[:5])
"
```

## Paper References

All implementations based on:

**"SupertonicTTS: Streaming, Real-time Hindi Text-to-Speech Synthesis"**

- **Section 4.2**: Training details and hyperparameters
- **Appendix B.1**: Autoencoder architecture and loss functions
- **Algorithm 1**: Flow matching training procedure
- **Sections 3.2.4 & 3.3**: Text-to-Latent and Duration modules

## Custom Data Integration

### Stage 1: Custom Audio

Modify `AudioDataset` in `train_autoencoder.py`:

```python
class AudioDataset(Dataset):
    def __init__(self, audio_dir):
        self.audio_files = glob.glob(f"{audio_dir}/**/*.wav", recursive=True)
    
    def __getitem__(self, idx):
        audio, sr = torchaudio.load(self.audio_files[idx])
        audio = torchaudio.functional.resample(audio, sr, 44100)
        return audio[:, :int(0.19 * 44100)]  # Crop to 0.19s
```

### Stage 2: Custom Latents

Precompute latents using `precompute_latents.py`:

```bash
python precompute_latents.py \
    --autoencoder_checkpoint ./outputs/stage1_autoencoder/checkpoints/autoencoder_step_1500000_final.pt \
    --audio_dir ./my_audio_files \
    --output_path ./my_latents.pt
```

Then update config:
```yaml
precomputed_latents_path: ./my_latents.pt
```

### Stage 3: Custom Text-Duration Pairs

Modify `DurationDataset` in `train_duration.py`:

```python
class DurationDataset(Dataset):
    def __init__(self, text_duration_pairs_file):
        with open(text_duration_pairs_file) as f:
            self.pairs = json.load(f)
    
    def __getitem__(self, idx):
        text, duration = self.pairs[idx]
        text_ids = self.tokenize(text)
        return {'text_ids': text_ids, 'duration': duration}
```

## License

Part of SupertonicTTS. See main repository for license details.

## Support

For issues:
1. Check TRAINING_GUIDE.md for detailed instructions
2. Review logs in `outputs/*/logs/`
3. Check TensorBoard for training curves
4. Verify checkpoint paths and file integrity

---

**Happy Training!** Feel free to reach out with questions.
