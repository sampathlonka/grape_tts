# SupertonicTTS Text-to-Latent: Quick Start Guide

## Installation

No additional dependencies beyond PyTorch required.

```bash
# Already part of the project
from supertonic_hindi_tts.models import TextToLatent
```

## Basic Usage (Training)

```python
import torch
from supertonic_hindi_tts.models import TextToLatent

# Create model
model = TextToLatent(
    vocab_size=256,              # Character vocabulary
    text_transformer_hidden=512,
    vf_channel_dim=256,
    latent_channels=24,
    compression_ratio=6,
    use_cfg=True,                # Classifier-free guidance
)

# Move to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Create dummy batch
batch_size = 4
text_len = 50
seq_len = 600                    # Uncompressed
seq_len_compressed = seq_len // 6  # 100

text_tokens = torch.randint(0, 256, (batch_size, text_len), device=device)
noisy_latents = torch.randn(batch_size, 24, seq_len, device=device)
compressed_ref = torch.randn(batch_size, 144, seq_len_compressed, device=device)
timesteps = model.sample_training_timesteps(batch_size, device, torch.float32)

# Forward pass
velocity = model(
    noisy_latents=noisy_latents,
    compressed_latents_ref=compressed_ref,
    text_tokens=text_tokens,
    timestep=timesteps,
    cfg_scale=1.0,  # Set to 1.0 during training
)

print(f"Velocity shape: {velocity.shape}")  # (4, 144, 100)

# Compute loss (flow matching objective)
loss = torch.mean(velocity ** 2)  # Simplified
loss.backward()
```

## Basic Usage (Inference)

```python
import torch
from supertonic_hindi_tts.models import TextToLatent

model = TextToLatent(...)
model.eval()

text_tokens = torch.randint(0, 256, (1, 50))
compressed_ref = torch.randn(1, 144, 100)

# Generate latents using ODE solver
with torch.no_grad():
    latents = model.inference(
        text_tokens=text_tokens,
        compressed_latents_ref=compressed_ref,
        num_inference_steps=50,      # ODE steps
        cfg_scale=7.5,               # Guidance strength
    )

print(f"Generated latents: {latents.shape}")  # (1, 144, 100)

# Optionally decompress to original space
uncompressed = model.decompress_latents(latents, Kc=6)
print(f"Uncompressed: {uncompressed.shape}")  # (1, 24, 600)
```

## Common Configurations

### Minimal (Development)
```python
TextToLatent(
    vocab_size=128,
    text_transformer_hidden=256,
    vf_channel_dim=128,
    latent_channels=24,
    use_cfg=False,
)
```

### Standard (Production)
```python
TextToLatent(
    vocab_size=256,
    text_transformer_hidden=512,
    vf_channel_dim=256,
    latent_channels=24,
    use_cfg=True,
)
```

### Large (High Quality)
```python
TextToLatent(
    vocab_size=512,
    text_transformer_hidden=768,
    vf_channel_dim=384,
    latent_channels=24,
    vf_num_main_blocks=6,
    use_cfg=True,
)
```

## Key Components

### TextEncoder
- Input: text tokens (B, text_len)
- Output: text embeddings (B, text_len, transformer_hidden)
- Processing: Char embedding → ConvNeXt → Self-Attention → Cross-Attention

### ReferenceEncoder
- Input: compressed speech latents (B, 144, T/6)
- Output: reference keys (B, 50, 128)
- Processing: Conv → ConvNeXt → Cross-Attention with learnable queries

### VFEstimator
- Input: noisy latents + conditioning
- Output: velocity field (B, 144, T/6)
- Processing: ConvNeXt + time/text/ref conditioning (4 main blocks)

## Attention Mechanisms

### Standard RoPE (Self-Attention)
```
angle = position * θ_j
Used for absolute position awareness in text
```

### LARoPE (Cross-Attention)
```
angle = gamma * (position / sequence_length) * θ_j
Used for relative position awareness
```

## Latent Compression

```python
# Compress: (B, 24, 600) → (B, 144, 100)
compressed = model.compress_latents(latents, Kc=6)

# Decompress: (B, 144, 100) → (B, 24, 600)
uncompressed = model.decompress_latents(compressed, Kc=6)
```

## Training Tips

1. **Batch Size**: 32+ recommended for stable training
2. **Learning Rate**: Start with 1e-4, use warmup
3. **Timesteps**: Sample uniformly from [0, 1]
4. **CFG Training**: Set cfg_uncond_prob=0.1 to randomly use learnable params
5. **Loss**: Use flow matching objective (MSE of velocity)

## Inference Tips

1. **ODE Steps**: 50 typically sufficient (balance quality/speed)
2. **CFG Scale**: 
   - 1.0 = no guidance
   - 7.5 = strong guidance
   - Higher = more adherence to text/speaker
3. **Sampling**: Use model.inference() for automatic ODE solving
4. **Batch Inference**: Process multiple samples simultaneously

## Classifier-Free Guidance

```python
# Training
if random.random() < 0.1:
    # Replace text/ref with learnable unconditional embeddings
    use_uncond = True
    
# Inference
velocity = model(
    ...,
    cfg_scale=7.5  # High scale for strong guidance
)
# Internally: v = v_uncond + 7.5 * (v_cond - v_uncond)
```

## Device Management

```python
# CPU
device = torch.device("cpu")
model = model.to(device)

# Single GPU
device = torch.device("cuda:0")
model = model.to(device)

# Multi-GPU (DataParallel)
model = torch.nn.DataParallel(model)

# Distributed
model = torch.nn.parallel.DistributedDataParallel(model)
```

## Saving/Loading

```python
# Save checkpoint
torch.save(model.state_dict(), "checkpoint.pt")

# Load checkpoint
model.load_state_dict(torch.load("checkpoint.pt"))

# Save full model
torch.save(model, "model.pt")

# Load full model
model = torch.load("model.pt")
```

## Debugging

```python
# Check parameter count
total_params = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total: {total_params}, Trainable: {trainable}")

# Check memory usage
import torch
print(f"Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# Enable profiling
from torch.profiler import profile
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    velocity = model(...)
print(prof.key_averages().table())
```

## Common Issues

### Out of Memory
- Reduce batch size
- Reduce sequence length
- Use gradient checkpointing (if implemented)
- Use mixed precision training

### Poor Quality Output
- Increase CFG scale
- Increase num_inference_steps
- Check training loss is decreasing
- Verify correct text/reference conditioning

### Slow Inference
- Reduce num_inference_steps
- Use batch processing
- Enable flash attention (if available)
- Profile to identify bottleneck

## Files Reference

| File | Purpose |
|------|---------|
| attention.py | RoPE, LARoPE, attention modules |
| convnext.py | ConvNeXt building blocks |
| text_encoder.py | Text encoding with speaker adaptation |
| reference_encoder.py | Speaker characteristic extraction |
| vf_estimator.py | Velocity field prediction with conditioning |
| text_to_latent.py | Complete integrated module |
| example_usage.py | Usage examples (training/inference) |

## Next Steps

1. Integrate with speech autoencoder for actual latent extraction
2. Implement flow matching loss function
3. Create training script with logging
4. Add duration predictor if needed
5. Evaluate on downstream TTS tasks
