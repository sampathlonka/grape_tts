# SupertonicTTS Training Guide

This guide explains how to train all 3 stages of SupertonicTTS on NVIDIA DGX Spark GB10.

## System Requirements

- **Hardware**: NVIDIA DGX Spark GB10 (single GPU, 128GB unified memory)
- **CUDA**: 12.0+
- **PyTorch**: 2.1.0+ with CUDA support
- **Memory**: 128GB unified memory (supports BF16)

## Installation

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install librosa tensorboard wandb pyyaml tqdm numpy scipy
```

## Training Pipeline Overview

The training consists of 3 sequential stages:

### Stage 1: Speech Autoencoder (train_autoencoder.py)
- **Duration**: ~72 hours (1.5M iterations at ~270 iters/hour on DGX Spark)
- **Output**: Pretrained autoencoder checkpoint
- **Key Parameters**:
  - Batch size: 128
  - Learning rate: 2e-4
  - Loss weights: λ_recon=45, λ_adv=1, λ_fm=0.1

### Stage 2: Text-to-Latent (train_text_to_latent.py)
- **Duration**: ~52 hours (700k iterations at ~333 iters/hour on DGX Spark)
- **Dependencies**: Requires Stage 1 checkpoint
- **Key Parameters**:
  - Batch size: 64 (expanded to 256 with context-sharing Ke=4)
  - Learning rate: 5e-4
  - Flow matching with σ_min=1e-8

### Stage 3: Duration Predictor (train_duration.py)
- **Duration**: ~15 minutes (3k iterations on DGX Spark)
- **Dependencies**: Requires Stage 1 checkpoint
- **Key Parameters**:
  - Batch size: 128
  - Learning rate: 5e-4
  - L1 loss

## Quick Start

### Stage 1: Autoencoder Training

```bash
python train_autoencoder.py \
    --config config_autoencoder.yaml \
    --output_dir ./outputs/stage1_autoencoder
```

**Resume from checkpoint:**
```bash
python train_autoencoder.py \
    --config config_autoencoder.yaml \
    --output_dir ./outputs/stage1_autoencoder \
    --resume ./outputs/stage1_autoencoder/checkpoints/autoencoder_step_500000.pt
```

### Stage 2: Text-to-Latent Training

```bash
# First, precompute latents from Stage 1 checkpoint (optional)
python precompute_latents.py \
    --autoencoder_checkpoint ./outputs/stage1_autoencoder/checkpoints/autoencoder_step_1500000_final.pt \
    --output_path ./latents_precomputed.pt

# Then train
python train_text_to_latent.py \
    --config config_text_to_latent.yaml \
    --output_dir ./outputs/stage2_text_to_latent \
    --autoencoder_checkpoint ./outputs/stage1_autoencoder/checkpoints/autoencoder_step_1500000_final.pt
```

### Stage 3: Duration Predictor Training

```bash
python train_duration.py \
    --config config_duration.yaml \
    --output_dir ./outputs/stage3_duration
```

## Configuration Files

### config_autoencoder.yaml
Controls Stage 1 (Autoencoder) training:
- `lambda_recon`: 45 (reconstruction loss weight)
- `lambda_adv`: 1 (adversarial loss weight)
- `lambda_fm`: 0.1 (feature matching loss weight)
- `batch_size`: 128
- `num_iterations`: 1500000

### config_text_to_latent.yaml
Controls Stage 2 (Text-to-Latent) training:
- `context_expansion`: 4 (Ke factor for context-sharing batch expansion)
- `p_uncond`: 0.05 (classifier-free guidance dropout)
- `sigma_min`: 1e-8 (flow matching noise schedule min)
- `spfm_warmup`: 40000 (Self-Purifying Flow Matching warmup)
- `batch_size`: 64 (base batch size, effectively 256 with Ke=4)
- `num_iterations`: 700000

### config_duration.yaml
Controls Stage 3 (Duration Predictor) training:
- `batch_size`: 128
- `num_iterations`: 3000
- `min_duration`: 0.5 seconds
- `max_duration`: 10.0 seconds

## Monitoring Training

### TensorBoard

View training progress with TensorBoard:

```bash
tensorboard --logdir ./outputs/stage1_autoencoder/tensorboard
tensorboard --logdir ./outputs/stage2_text_to_latent/tensorboard
tensorboard --logdir ./outputs/stage3_duration/tensorboard
```

Then open http://localhost:6006 in your browser.

### Logs

Training logs are saved to:
- `./outputs/stage1_autoencoder/logs/autoencoder.log`
- `./outputs/stage2_text_to_latent/logs/text_to_latent.log`
- `./outputs/stage3_duration/logs/duration.log`

View live logs:
```bash
tail -f ./outputs/stage1_autoencoder/logs/autoencoder.log
```

## Checkpoints

Checkpoints are saved every `ckpt_interval` iterations:

```
outputs/
├── stage1_autoencoder/
│   └── checkpoints/
│       ├── autoencoder_step_50000.pt
│       ├── autoencoder_step_100000.pt
│       ├── autoencoder_step_1500000_final.pt
│       └── autoencoder_step_1500000_best.pt
├── stage2_text_to_latent/
│   └── checkpoints/
│       ├── text_to_latent_step_50000.pt
│       ├── text_to_latent_step_700000_final.pt
│       └── text_to_latent_step_700000_best.pt
└── stage3_duration/
    └── checkpoints/
        ├── duration_step_1000.pt
        ├── duration_step_3000_final.pt
        └── duration_step_3000_best.pt
```

## Advanced Configuration

### Mixed Precision Training (BF16)

The scripts automatically enable BF16 mixed precision on DGX Spark. To disable:

```yaml
use_amp: false
```

This is NOT recommended on DGX Spark due to memory constraints.

### Gradient Clipping

Default gradient norm clipping is 1.0. To change:

```yaml
max_grad_norm: 1.0  # In config file
```

### Batch Size Tuning

For Stage 1 (Autoencoder):
- Recommended: 128 on DGX Spark
- Min: 32 (slower convergence)
- Max: 256 (may exceed memory)

For Stage 2 (Text-to-Latent):
- Base batch size: 64
- Effective batch size: 256 (with Ke=4 context-sharing)
- Can increase Ke to 8 for larger effective batches

For Stage 3 (Duration):
- Recommended: 128
- Can be increased to 256+ safely (very fast stage)

### Custom Data

To use custom data:

1. **Stage 1**: Modify `AudioDataset` in `train_autoencoder.py` to load your audio files
2. **Stage 2**: Precompute latents from your audio, modify `LatentDataset`
3. **Stage 3**: Modify `DurationDataset` to load your text-duration pairs

## Optimization Tips for DGX Spark

1. **Use gradient accumulation** (not yet implemented - add if needed):
   ```python
   for i, batch in enumerate(loader):
       loss = model(batch)
       loss.backward()
       if (i + 1) % accumulation_steps == 0:
           optimizer.step()
           optimizer.zero_grad()
   ```

2. **Enable torch.compile** for additional speedup:
   ```python
   model = torch.compile(model)
   ```

3. **Monitor memory usage**:
   ```python
   print(torch.cuda.memory_allocated() / 1e9)  # GB
   ```

4. **Use persistent workers** for DataLoader:
   ```python
   loader = DataLoader(dataset, num_workers=4, persistent_workers=True)
   ```

## Troubleshooting

### Out of Memory (OOM)

If you get OOM errors:

1. **Reduce batch size** in config file
2. **Enable gradient checkpointing** (can be added to models)
3. **Reduce number of workers** in DataLoader
4. **Clear cache** before training: `torch.cuda.empty_cache()`

### Training is too slow

1. Check GPU utilization: `nvidia-smi`
2. Ensure you're using correct CUDA version
3. Enable BF16 mixed precision (should be default)
4. Use multiple workers for data loading
5. Pin memory in DataLoader: `pin_memory=True`

### Validation loss not decreasing

1. Check learning rate (try 1e-4 or 1e-3)
2. Verify data is loaded correctly
3. Check loss weights in config
4. Ensure gradients are flowing (print grad norms)

### Checkpoint loading fails

```bash
# Verify checkpoint exists and is readable
ls -lh ./outputs/stage1_autoencoder/checkpoints/autoencoder_step_500000.pt

# Check checkpoint contents
python -c "import torch; ckpt = torch.load('path/to/checkpoint.pt'); print(ckpt.keys())"
```

## Distributed Training (Optional)

For multi-GPU training (modify scripts):

```python
from torch.nn.parallel import DataParallel, DistributedDataParallel

model = DataParallel(model)  # For single machine, multiple GPUs
# OR
model = DistributedDataParallel(model)  # For multi-machine
```

## Performance Benchmarks (DGX Spark GB10)

Expected throughput with recommended settings:

| Stage | Batch Size | Iters/Hour | Total Time |
|-------|-----------|-----------|-----------|
| Autoencoder | 128 | ~270 | ~72 hours |
| Text-to-Latent | 64 (256 eff) | ~333 | ~52 hours |
| Duration | 128 | ~15000 | ~15 min |

**Total training time: ~4.5 days** (with optimal settings)

## References

- Paper: "SupertonicTTS: Streaming, Real-time Hindi Text-to-Speech Synthesis"
- Section 4.2: Training details and hyperparameters
- Appendix B.1: Autoencoder architecture and loss functions
- Algorithm 1: Flow matching training procedure
- Sections 3.2.4 & 3.3: Text-to-Latent and Duration prediction modules

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review training logs in `./outputs/*/logs/`
3. Check TensorBoard visualizations
4. Verify data format and checkpoint paths
