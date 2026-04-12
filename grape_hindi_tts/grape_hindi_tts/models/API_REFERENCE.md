# Speech Autoencoder API Reference

Complete API documentation for the SupertonicTTS Speech Autoencoder implementation.

## Table of Contents
1. [ConvNeXt Blocks](#convnext-blocks)
2. [Speech Autoencoder](#speech-autoencoder)
3. [Discriminators](#discriminators)
4. [Loss Functions](#loss-functions)

---

## ConvNeXt Blocks

### `ConvNeXtBlock`
Standard ConvNeXt block for sequence modeling.

```python
from models.convnext import ConvNeXtBlock

block = ConvNeXtBlock(
    hidden_dim: int = 512,
    intermediate_dim: int = 2048,
    kernel_size: int = 7,
    dilation: int = 1,
    dropout: float = 0.0,
)
```

**Parameters:**
- `hidden_dim`: Feature dimension (channels)
- `intermediate_dim`: Feed-forward expansion dimension
- `kernel_size`: Depthwise convolution kernel size
- `dilation`: Dilation rate for receptive field
- `dropout`: Dropout probability

**Forward:**
```python
# Input: (B, hidden_dim, T)
# Output: (B, hidden_dim, T)
x = torch.randn(2, 512, 100)
y = block(x)
```

---

### `DilatedConvNeXtBlock`
ConvNeXt block with explicit dilation support.

```python
from models.convnext import DilatedConvNeXtBlock

block = DilatedConvNeXtBlock(
    hidden_dim: int = 512,
    intermediate_dim: int = 2048,
    kernel_size: int = 7,
    dilation: int = 2,
    causal: bool = False,
    dropout: float = 0.0,
)
```

**Parameters:**
- `causal`: If True, use left-only padding
- Other parameters same as `ConvNeXtBlock`

**Use Case:** Decoder blocks where you want explicit dilation control.

---

### `CausalConvNeXtBlock`
ConvNeXt with strict causal (left-only) padding for streaming.

```python
from models.convnext import CausalConvNeXtBlock

block = CausalConvNeXtBlock(
    hidden_dim: int = 512,
    intermediate_dim: int = 2048,
    kernel_size: int = 7,
    dilation: int = 1,
    dropout: float = 0.0,
)
```

**Key Property:** Each output depends only on current and past inputs.

**Use Case:** Streaming/real-time decoding.

---

### `CausalConv1d`
Simple causal convolution layer.

```python
from models.convnext import CausalConv1d

layer = CausalConv1d(
    in_channels: int = 24,
    out_channels: int = 512,
    kernel_size: int = 3,
    dilation: int = 1,
    bias: bool = True,
)
```

**Causal Padding:**
- Total padding: `dilation * (kernel_size - 1)`
- Direction: All padding on left (past), none on right (future)

---

## Speech Autoencoder

### `LatentEncoder`
Encodes mel spectrograms to latent representations.

```python
from models.speech_autoencoder import LatentEncoder

encoder = LatentEncoder(
    mel_dim: int = 228,
    hidden_dim: int = 512,
    latent_dim: int = 24,
    num_blocks: int = 10,
    intermediate_dim: int = 2048,
    kernel_size: int = 7,
    dropout: float = 0.0,
)
```

**Architecture:**
1. Conv1d: 228 → 512 (input projection)
2. BatchNorm1d
3. 10x ConvNeXtBlock (hidden=512)
4. Linear: 512 → 24 (latent projection)
5. LayerNorm

**Forward:**
```python
# Input: mel spectrogram
mel = torch.randn(4, 100, 228)  # (B, T, mel_dim)
# or: mel = torch.randn(4, 228, 100)  # (B, mel_dim, T)

# Output: latent codes
latents = encoder(mel)  # (B, T, 24)
```

---

### `LatentDecoder`
Decodes latent codes to waveform.

```python
from models.speech_autoencoder import LatentDecoder

decoder = LatentDecoder(
    latent_dim: int = 24,
    hidden_dim: int = 512,
    hop_length: int = 256,
    num_blocks: int = 10,
    intermediate_dim: int = 2048,
    kernel_size: int = 7,
    dilation_rates: list = [1, 2, 4, 1, 2, 4, 1, 1, 1, 1],
    dropout: float = 0.0,
)
```

**Architecture:**
1. CausalConv1d: 24 → 512 (input projection)
2. BatchNorm1d
3. 10x Dilated CausalConvNeXtBlock (with dilation_rates)
4. BatchNorm1d
5. CausalConv1d: 512 → 512 (hidden representation)
6. Linear (MLP): 512 → 512 → hop_length
7. Reshape: (B, T, 256) → (B, T*256)

**Dilation Rates:** [1, 2, 4, 1, 2, 4, 1, 1, 1, 1]
- Early blocks: Exponential dilation for receptive field expansion
- Later blocks: Small dilation for fine detail

**Forward:**
```python
# Input: latent codes
latents = torch.randn(4, 100, 24)  # (B, T, latent_dim)
# or: latents = torch.randn(4, 24, 100)  # (B, latent_dim, T)

# Output: waveform
waveform = decoder(latents)  # (B, T*256) = (B, 25600)
```

---

### `SpeechAutoencoder`
Complete autoencoder combining encoder and decoder.

```python
from models.speech_autoencoder import SpeechAutoencoder

autoencoder = SpeechAutoencoder(
    mel_dim: int = 228,
    latent_dim: int = 24,
    hidden_dim: int = 512,
    hop_length: int = 256,
    num_encoder_blocks: int = 10,
    num_decoder_blocks: int = 10,
    intermediate_dim: int = 2048,
    kernel_size: int = 7,
    dropout: float = 0.0,
)
```

**Methods:**

#### `encode(mel) → latents`
```python
mel = torch.randn(4, 100, 228)
latents = autoencoder.encode(mel)  # (4, 100, 24)
```

#### `decode(latents) → waveform`
```python
latents = torch.randn(4, 100, 24)
waveform = autoencoder.decode(latents)  # (4, 25600)
```

#### `forward(mel) → (waveform, latents)`
```python
mel = torch.randn(4, 100, 228)
waveform, latents = autoencoder(mel)
# waveform: (4, 25600)
# latents: (4, 100, 24)
```

---

## Discriminators

### `MultiPeriodDiscriminator`
Discriminator applied at multiple periods of the waveform.

```python
from models.discriminators import MultiPeriodDiscriminator

mpd = MultiPeriodDiscriminator(
    periods: list = [2, 3, 5, 7, 11],
    kernel_size: tuple = (5, 5),
    stride: tuple = (1, 2),
)
```

**Periods:** [2, 3, 5, 7, 11] (default)
- Each period captures different temporal patterns
- Coprime periods ensure diverse receptive fields

**Forward:**
```python
# Input: waveform
waveform = torch.randn(4, 8192)  # (B, T)
# or: waveform = torch.randn(4, 1, 8192)  # (B, 1, T)

# Output: list of scores and features
scores, features = mpd(waveform)
# scores: List[5] of (B, 1, h, w) tensors
# features: List of intermediate activations
```

---

### `MultiResolutionDiscriminator`
Discriminator applied to spectrograms of different resolutions.

```python
from models.discriminators import MultiResolutionDiscriminator

mrd = MultiResolutionDiscriminator(
    fft_sizes: list = [512, 1024, 2048],
    hop_sizes: list = [50, 100, 200],
    win_sizes: list = [240, 480, 960],
    mel_bands: list = [64, 128, 128],
)
```

**FFT Resolutions:**
- 512: Transient details
- 1024: Mid-range spectral content
- 2048: Fine spectral details

**Forward:**
```python
# Input: waveform
waveform = torch.randn(4, 8192)

# Internally computes STFT at each resolution
scores, features = mrd(waveform)
# scores: List[3] of discriminator outputs
# features: List of intermediate activations
```

---

### `CompositeDiscriminator`
Combined MPD + MRD discriminator.

```python
from models.discriminators import CompositeDiscriminator

discriminator = CompositeDiscriminator(
    use_mpd: bool = True,
    use_mrd: bool = True,
    periods: list = [2, 3, 5, 7, 11],
    fft_sizes: list = [512, 1024, 2048],
)
```

**Forward:**
```python
waveform = torch.randn(4, 8192)

sub_discriminators, all_features = discriminator(waveform)
# sub_discriminators: List of (features, score) tuples
# all_features: Flattened list of all features (for feature matching)
```

---

## Loss Functions

### `spectral_reconstruction_loss`
Multi-resolution spectral reconstruction loss.

```python
from models.losses import spectral_reconstruction_loss

loss = spectral_reconstruction_loss(
    y_hat: torch.Tensor,        # Predicted waveform (B, T)
    y: torch.Tensor,            # Target waveform (B, T)
    fft_sizes: list = [1024, 2048, 4096],
    mel_bands: list = [64, 128, 128],
    reduction: str = "mean",
)
```

**Details:**
- Computes STFT at multiple resolutions
- L1 loss on both linear and log-magnitude spectrograms
- Averages across all resolutions

**Use:** Trains autoencoder for perceptual quality

---

### `adversarial_loss_generator`
Generator adversarial loss (LS-GAN).

```python
from models.losses import adversarial_loss_generator

loss = adversarial_loss_generator(
    disc_outputs: list,      # List of discriminator outputs
    reduction: str = "mean",
)
```

**Formula:** L_gen = E[(D(G(x)) - 1)²]

**Use:** Encourages discriminator to believe generated samples are real

---

### `adversarial_loss_discriminator`
Discriminator adversarial loss (LS-GAN).

```python
from models.losses import adversarial_loss_discriminator

loss = adversarial_loss_discriminator(
    real_outputs: list,      # D(real) outputs
    fake_outputs: list,      # D(fake) outputs
    reduction: str = "mean",
)
```

**Formula:** L_disc = E[(D(real) - 1)²] + E[D(fake)²]

**Use:** Trains discriminator to distinguish real from fake

---

### `feature_matching_loss`
Matching intermediate discriminator features.

```python
from models.losses import feature_matching_loss

loss = feature_matching_loss(
    real_features: list,     # List of real feature tensors
    fake_features: list,     # List of fake feature tensors
    reduction: str = "mean",
)
```

**Formula:** L_fm = Σ ||D^(l)(real) - D^(l)(fake)||₁

**Use:** Provides richer gradients to generator

---

### `flow_matching_loss`
Flow-based continuous latent trajectory loss.

```python
from models.losses import flow_matching_loss

loss = flow_matching_loss(
    predicted_velocity: torch.Tensor,    # (B, T, latent_dim)
    target_velocity: torch.Tensor,       # (B, T, latent_dim)
    mask: torch.Tensor = None,           # (B, T) optional
    reduction: str = "mean",
)
```

**Formula:** L_flow = ||v_θ - (x₁ - x₀)||²

**Use:** Optional for flow-based generative models

---

### `duration_loss`
Phoneme duration prediction loss.

```python
from models.losses import duration_loss

loss = duration_loss(
    predicted_duration: torch.Tensor,    # (B, num_phonemes)
    target_duration: torch.Tensor,       # (B, num_phonemes)
    reduction: str = "mean",
)
```

**Formula:** L_dur = L1(predicted, target)

**Use:** TTS duration alignment training

---

### `combined_generator_loss`
Combined loss for generator training.

```python
from models.losses import combined_generator_loss

loss, loss_dict = combined_generator_loss(
    reconstructed_waveform: torch.Tensor,
    target_waveform: torch.Tensor,
    disc_outputs: list,
    real_features: list,
    fake_features: list,
    lambda_adv: float = 1.0,
    lambda_feat: float = 10.0,
    lambda_spec: float = 1.0,
)
```

**Returns:**
- `loss`: Scalar loss tensor
- `loss_dict`: Dictionary with individual loss components
  - `loss_generator`: Total loss
  - `loss_spec`: Spectral loss
  - `loss_adv_gen`: Adversarial loss
  - `loss_feat_matching`: Feature matching loss

---

### `combined_discriminator_loss`
Combined loss for discriminator training.

```python
from models.losses import combined_discriminator_loss

loss, loss_dict = combined_discriminator_loss(
    real_outputs: list,
    fake_outputs: list,
    lambda_disc: float = 1.0,
)
```

**Returns:**
- `loss`: Scalar loss tensor
- `loss_dict`: Dictionary with loss components
  - `loss_discriminator`: Total loss
  - `loss_adv_disc`: Adversarial loss

---

## Training Loop Example

```python
import torch
import torch.optim as optim
from models import SpeechAutoencoder, CompositeDiscriminator
from models.losses import combined_generator_loss, combined_discriminator_loss

# Initialize models
autoencoder = SpeechAutoencoder()
discriminator = CompositeDiscriminator()

# Initialize optimizers
opt_gen = optim.Adam(autoencoder.parameters(), lr=1e-4)
opt_disc = optim.Adam(discriminator.parameters(), lr=1e-4)

# Training loop
for epoch in range(num_epochs):
    for mel_batch, waveform_batch in dataloader:
        # Generator step
        opt_gen.zero_grad()
        
        # Forward pass
        waveform_recon, latents = autoencoder(mel_batch)
        
        # Discriminator evaluation
        real_subs, real_feats = discriminator(waveform_batch)
        fake_subs, fake_feats = discriminator(waveform_recon.detach())
        
        # Extract outputs
        real_outputs = [feat for feat, score in real_subs]
        fake_outputs = [score for feat, score in fake_subs]
        
        # Generator loss
        gen_loss, gen_dict = combined_generator_loss(
            waveform_recon, waveform_batch,
            fake_outputs, real_feats, fake_feats
        )
        
        gen_loss.backward()
        opt_gen.step()
        
        # Discriminator step
        opt_disc.zero_grad()
        
        real_subs, _ = discriminator(waveform_batch)
        fake_subs, _ = discriminator(waveform_recon.detach())
        
        real_outputs = [score for feat, score in real_subs]
        fake_outputs = [score for feat, score in fake_subs]
        
        disc_loss, disc_dict = combined_discriminator_loss(
            real_outputs, fake_outputs
        )
        
        disc_loss.backward()
        opt_disc.step()
        
        print(f"Gen: {gen_dict['loss_generator']:.4f}, "
              f"Disc: {disc_dict['loss_discriminator']:.4f}")
```

---

## Type Hints Reference

All classes and functions use complete type hints:

```python
# Import type hints
from typing import List, Tuple, Optional
import torch
import torch.nn as nn

# Function signature example
def spectral_reconstruction_loss(
    y_hat: torch.Tensor,                    # Predicted waveform
    y: torch.Tensor,                        # Target waveform
    fft_sizes: Optional[List[int]] = None,  # FFT sizes to use
    mel_bands: Optional[List[int]] = None,  # Mel bands per FFT
    reduction: str = "mean",                # Reduction method
) -> torch.Tensor:                          # Returns scalar tensor
    ...
```

---

## Common Pitfalls

1. **Input Format:** Both (B, T, D) and (B, D, T) formats supported
   - Autoencoder detects format automatically
   - Ensure consistency for custom code

2. **Causal Padding:** Decoder uses causal padding
   - Cannot use random cropping during training
   - Must preserve temporal ordering

3. **Feature List Length:** Different sub-discriminators have different feature counts
   - Always handle variable-length feature lists
   - Don't hardcode feature count

4. **STFT Window:** MRD uses Hann window internally
   - Reproducible on same GPU/CPU
   - Different across platforms (float precision)

---

## Performance Tips

1. **Memory:** Use smaller batch sizes for 24-bit model
   - Recommended: B=4-8 for 16GB GPU
   - gradient accumulation for larger effective batches

2. **Speed:** Causal padding is efficient
   - Single F.pad operation per layer
   - Minimal CPU-GPU overhead

3. **Quality:** Freeze encoder during discriminator warmup
   - Stabilizes adversarial training
   - Prevents mode collapse in early epochs

---
