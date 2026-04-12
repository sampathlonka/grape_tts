# SupertonicTTS Speech Autoencoder Implementation Guide

## Overview

This document describes the complete Speech Autoencoder implementation for SupertonicTTS based on the Vocos architecture with ConvNeXt blocks. The autoencoder encodes 228-dimensional mel spectrograms into 24-dimensional continuous latents and reconstructs waveforms from these latents.

## Architecture Summary

### 1. ConvNeXt Blocks (`convnext.py`)
- **Lines of Code**: ~160
- **Key Classes**:
  - `ConvNeXtBlock`: Standard ConvNeXt block with depthwise convolution
  - `DilatedConvNeXtBlock`: ConvNeXt with configurable dilation rates
  - `CausalConvNeXtBlock`: Strict causal padding for streaming
  - `CausalConv1d`: Causal 1D convolution layer

#### ConvNeXtBlock Architecture
```
Input (B, hidden_dim, T)
  ↓
Depthwise Conv1d (kernel=7, padding=3, groups=hidden_dim)
  ↓
BatchNorm1d
  ↓
Transpose to (B, T, hidden_dim)
  ↓
Linear (hidden_dim → intermediate_dim)
  ↓
GELU activation
  ↓
Linear (intermediate_dim → hidden_dim)
  ↓
Transpose back to (B, hidden_dim, T)
  ↓
Residual connection with input
  ↓
Output (B, hidden_dim, T)
```

#### Key Features
- **Depthwise separable convolution**: Groups=hidden_dim makes convolution operate independently per channel
- **Inverted bottleneck**: Feed-forward expands to intermediate_dim (2048) then contracts
- **BatchNorm placement**: After convolution, normalizes across time and batch
- **Causal variants**: Support for left-only padding to maintain causality in streaming mode

### 2. Speech Autoencoder (`speech_autoencoder.py`)
- **Lines of Code**: ~387
- **Key Classes**:
  - `LatentEncoder`: 228-dim mel → 24-dim latents
  - `LatentDecoder`: 24-dim latents → waveform
  - `SpeechAutoencoder`: Complete encoder-decoder system

#### LatentEncoder Architecture
```
Mel Spectrogram Input (B, T, 228)
  ↓
Transpose to (B, 228, T)
  ↓
Conv1d (228 → 512, kernel=1)
  ↓
BatchNorm1d
  ↓
[10x ConvNeXt Blocks]
  (hidden=512, intermediate=2048, kernel=7)
  ↓
Transpose to (B, T, 512)
  ↓
Linear (512 → 24)
  ↓
LayerNorm
  ↓
Latent Output (B, T, 24)
```

**Input/Output Shapes**:
- Input: (B, T, 228) or (B, 228, T)
- Output: (B, T, 24) latent representation

#### LatentDecoder Architecture
```
Latent Input (B, T, 24)
  ↓
Transpose to (B, 24, T)
  ↓
CausalConv1d (24 → 512, kernel=1)
  ↓
BatchNorm1d
  ↓
[10x Dilated CausalConvNeXt Blocks]
  dilation_rates = [1, 2, 4, 1, 2, 4, 1, 1, 1, 1]
  (hidden=512, intermediate=2048, kernel=7)
  ↓
BatchNorm1d
  ↓
CausalConv1d (512 → 512, kernel=3)
  ↓
Transpose to (B, T, 512)
  ↓
Linear (512 → 512)
  ↓
PReLU activation
  ↓
Linear (512 → hop_length=256)
  ↓
Reshape: (B, T, 256) → (B, T*256)
  ↓
Waveform Output (B, T*hop_length)
```

**Input/Output Shapes**:
- Input: (B, T, 24) or (B, 24, T)
- Output: (B, T*256) waveform

**Dilation Rates**: [1, 2, 4, 1, 2, 4, 1, 1, 1, 1]
- Early blocks use exponential dilation to expand receptive field
- Later blocks use dilation=1 for detail synthesis

#### SpeechAutoencoder API
```python
autoencoder = SpeechAutoencoder()

# Full forward pass
waveform, latents = autoencoder(mel)  # (B, T*256), (B, T, 24)

# Separate encode/decode
latents = autoencoder.encode(mel)      # (B, T, 24)
waveform = autoencoder.decode(latents) # (B, T*256)
```

### 3. Discriminators (`discriminators.py`)
- **Lines of Code**: ~350
- **Key Classes**:
  - `MultiPeriodDiscriminator`: Period-based waveform discrimination
  - `MultiResolutionDiscriminator`: Resolution-based spectrogram discrimination
  - `CompositeDiscriminator`: Combined MPD+MRD

#### MultiPeriodDiscriminator (MPD)
**Concept**: Discriminator applied to waveforms reshaped at different periods.
- Periods: [2, 3, 5, 7, 11]
- Each period sub-discriminator:
  1. Reshapes input (B, 1, T) → (B, 1, period, T//period)
  2. Applies 6 Conv2D layers:
     - Channels: 1 → 16 → 64 → 256 → 512 → 512 → 1
     - Kernel: (5, 5)
     - Stride: (1, 2) for temporal downsampling
     - LeakyReLU (slope=0.1) between layers
  3. Returns discriminator score

**Why periods?**: Different periods capture different temporal correlations. Period 2 catches high-frequency aliasing, period 7 catches mid-range patterns, etc.

#### MultiResolutionDiscriminator (MRD)
**Concept**: Discriminator applied to spectrograms at different resolutions.
- FFT sizes: [512, 1024, 2048]
- Corresponding hop sizes: [50, 100, 200]
- Window sizes: [240, 480, 960]

For each FFT size:
1. Compute STFT: (B, T) → (B, n_fft//2 + 1, n_frames, 2)
2. Extract magnitude: √(real² + imag²)
3. Convert to log-magnitude: log(magnitude)
4. Apply 6 Conv2D layers:
   - Channels: 1 → 16 → 16 → 16 → 16 → 16 → 1
   - Kernel: (5, 5)
   - Stride: (2, 2)
   - LeakyReLU between layers

**Why resolutions?**: Different FFT sizes capture different spectral details. Large FFT (2048) captures fine spectral detail, small FFT (512) captures transients.

#### CompositeDiscriminator API
```python
discriminator = CompositeDiscriminator(use_mpd=True, use_mrd=True)

# Forward pass
sub_discs, all_features = discriminator(waveform)
# sub_discs: List[(features, score)] from all sub-discriminators
# all_features: Flattened list of intermediate features for feature matching
```

### 4. Loss Functions (`losses.py`)
- **Lines of Code**: ~447
- **Key Functions**:

#### spectral_reconstruction_loss
**Purpose**: Multi-resolution spectral reconstruction loss
**Implementation**:
```
For each FFT size in [1024, 2048, 4096]:
  1. Compute STFT of predicted and target waveforms
  2. Extract log-magnitude spectrograms
  3. Compute L1 loss between:
     - Linear magnitude spectrograms
     - Log-magnitude spectrograms (perceptual loss)
  4. Average across all resolutions
```
**Use case**: Trains autoencoder to reconstruct spectrogram details

#### adversarial_loss_generator
**Formula**: L_gen = E[(D(G(x)) - 1)²]
**Purpose**: Encourages discriminator to output 1 for generated samples
**Implementation**: MSE loss between discriminator outputs and ones

#### adversarial_loss_discriminator
**Formula**: L_disc = E[(D(x_real) - 1)²] + E[D(x_fake)²]
**Purpose**: Trains discriminator to distinguish real (→1) from fake (→0)
**Implementation**: MSE loss for real and fake outputs separately

#### feature_matching_loss
**Purpose**: Match intermediate discriminator features
**Implementation**:
```
L_fm = E[||D^(l)(x_real) - D^(l)(x_fake)||₁]
```
Provides richer gradients to generator beyond just adversarial loss

#### flow_matching_loss
**Purpose**: Flow-based generative modeling loss
**Formula**: L_flow = ||v_θ(t, x_t) - (x₁ - x₀)||²
**Use case**: Optional, for flow-based continuous latent dynamics

#### duration_loss
**Purpose**: Phoneme duration prediction loss
**Formula**: L_dur = L1(predicted_duration, target_duration)
**Use case**: For TTS duration alignment

#### Combined Losses
```python
# Generator training
loss_gen, loss_dict = combined_generator_loss(
    reconstructed_waveform,
    target_waveform,
    disc_outputs,
    real_features,
    fake_features,
    lambda_adv=1.0,
    lambda_feat=10.0,
    lambda_spec=1.0,
)

# Discriminator training
loss_disc, loss_dict = combined_discriminator_loss(
    real_outputs,
    fake_outputs,
    lambda_disc=1.0,
)
```

## Training Pipeline

### 1. Forward Pass
```python
# Encode mel to latents
mel_spec = ...  # (B, T, 228)
latents = autoencoder.encode(mel_spec)  # (B, T, 24)

# Discriminate latents
sub_discs, features = discriminator(latents)
```

### 2. Generator Training
```python
# Reconstruct waveform from latents
waveform_reconstructed, latents = autoencoder(mel_spec)

# Discriminator evaluation
disc_outputs_real, real_features = discriminator(waveform_target)
disc_outputs_fake, fake_features = discriminator(waveform_reconstructed)

# Loss computation
spec_loss = spectral_reconstruction_loss(waveform_reconstructed, waveform_target)
adv_loss = adversarial_loss_generator(disc_outputs_fake)
feat_loss = feature_matching_loss(real_features, fake_features)

total_loss = spec_loss + λ_adv * adv_loss + λ_feat * feat_loss
```

### 3. Discriminator Training
```python
# Real discriminator output
disc_real_output, _ = discriminator(waveform_target)

# Fake discriminator output
disc_fake_output, _ = discriminator(waveform_reconstructed.detach())

# Discriminator loss
loss_disc = adversarial_loss_discriminator(disc_real_output, disc_fake_output)
```

## Implementation Details

### Causal Padding in Decoder
The decoder uses **causal (left-only) padding** to maintain the property that each output depends only on past and current inputs:
```python
# For CausalConv1d with kernel_size=7
# Standard padding: (3, 3) on both sides
# Causal padding: (6, 0) only on left side

F.pad(x, (causal_padding, 0))  # (left, right)
```

### Dilation Scheduling
Decoder dilation rates: [1, 2, 4, 1, 2, 4, 1, 1, 1, 1]
- **Purpose**: Exponential dilation expands receptive field early
- **Receptive field growth**: Block i has RF = ∏(dilation[j])
- **Rationale**: Large dilation for coarse structure, small for detail

### Batch Normalization vs LayerNorm
- **ConvNeXt blocks use BatchNorm1d**: Operates across batch and spatial dims
  - Input to BN: (B, hidden_dim, T)
  - Normalizes across B and T dimensions
- **Encoder output uses LayerNorm**: Normalizes per-sample
  - Input to LN: (B, T, latent_dim)
  - Normalizes across latent_dim dimension

## Hyperparameter Reference

### Model Dimensions
| Parameter | Value | Notes |
|-----------|-------|-------|
| mel_dim | 228 | Input spectrogram dimension |
| hidden_dim | 512 | ConvNeXt hidden dimension |
| latent_dim | 24 | Learned continuous latent dimension |
| intermediate_dim | 2048 | ConvNeXt feed-forward expansion |
| hop_length | 256 | Frame-to-sample conversion |
| kernel_size | 7 | ConvNeXt depthwise conv kernel |

### Training Parameters
| Parameter | Typical Value | Notes |
|-----------|---------------|-------|
| λ_spec | 1.0 | Spectral loss weight |
| λ_adv | 1.0 | Adversarial loss weight |
| λ_feat | 10.0 | Feature matching loss weight |
| MPD periods | [2,3,5,7,11] | Period-based discrimination |
| MRD FFT sizes | [512,1024,2048] | Resolution-based discrimination |

## File Structure

```
models/
├── __init__.py                  # Package initialization
├── convnext.py                  # ConvNeXt blocks (~160 lines)
├── speech_autoencoder.py        # Encoder/Decoder (~387 lines)
├── discriminators.py            # MPD/MRD discriminators (~350 lines)
├── losses.py                    # All loss functions (~447 lines)
├── IMPLEMENTATION_GUIDE.md      # This file
└── [Other modules...]
```

## Usage Examples

### Basic Autoencoding
```python
from models import SpeechAutoencoder
import torch

# Initialize
autoencoder = SpeechAutoencoder(
    mel_dim=228,
    latent_dim=24,
    hidden_dim=512,
    hop_length=256,
)

# Forward pass
mel = torch.randn(2, 100, 228)  # Batch of mels
waveform, latents = autoencoder(mel)

# Expected outputs
assert waveform.shape == (2, 25600)  # 100 * 256
assert latents.shape == (2, 100, 24)
```

### Discriminator Training
```python
from models import CompositeDiscriminator
import torch

# Initialize
disc = CompositeDiscriminator()

# Forward pass
waveform = torch.randn(2, 8192)
sub_discs, all_features = disc(waveform)

# sub_discs: List of (features, score) tuples
# all_features: List of all intermediate activations
```

### Loss Computation
```python
from models.losses import (
    spectral_reconstruction_loss,
    adversarial_loss_generator,
    feature_matching_loss,
)

# Compute losses
spec_loss = spectral_reconstruction_loss(waveform_pred, waveform_true)
adv_loss = adversarial_loss_generator(disc_outputs)
feat_loss = feature_matching_loss(real_features, fake_features)

# Total loss
total = spec_loss + adv_loss + 10 * feat_loss
```

## Tensor Shape Reference

### Mel Spectrogram
- Input shape: (B, T, 228) or (B, 228, T)
- B: Batch size (typically 4-16)
- T: Time steps (typically 100-500)
- 228: Mel band dimension (fixed)

### Latent Representation
- Shape: (B, T, 24)
- Temporal length matches input mel
- Continuous 24-dimensional representation

### Waveform
- Shape: (B, T * 256)
- Each mel frame expands to 256 samples (hop_length)
- Example: 100 mel frames → 25,600 samples

### Discriminator Inputs
- MPD: (B, 1, T) or (B, T) waveform
- MRD: (B, T) waveform (computes STFT internally)

## Performance Considerations

1. **Memory**: ConvNeXt blocks have moderate memory usage
   - 10 encoder + 10 decoder blocks = 20 layers total
   - Hidden dim 512, batch size 4-8 recommended

2. **Computation**: Causal padding adds minimal overhead
   - Causal padding = standard padding + F.pad operation
   - No additional convolutions or parameters

3. **Inference**: Streaming-friendly architecture
   - CausalConvNeXtBlock supports frame-by-frame inference
   - Can be optimized for real-time use with stateful mode

## References

- **Vocos paper**: "Vocos: Closing the Gap Between Time-Domain and Frequency-Domain Speech Synthesis Networks"
- **ConvNeXt paper**: "A ConvNet for the 2020s"
- **HiFi-GAN**: "HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis"
