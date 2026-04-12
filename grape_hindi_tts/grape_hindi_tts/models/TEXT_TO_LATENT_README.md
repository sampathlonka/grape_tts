# SupertonicTTS Text-to-Latent Module

Complete implementation of the Text-to-Latent module for SupertonicTTS flow-matching based latent prediction.

## Architecture Overview

The module combines three main components:

### 1. **TextEncoder** (`text_encoder.py`)
Processes text tokens with speaker conditioning to produce speaker-adaptive text embeddings.

**Architecture:**
- Character embedding: vocab_size → 128 dim
- 6 ConvNeXt blocks (hidden=128, intermediate=512, kernel=5)
- Projection to transformer hidden dimension (512)
- 4 Self-Attention blocks with RoPE (Rotary Position Embedding)
- 2 Cross-Attention layers with reference features using LARoPE
  - First layer: uses 50 learnable reference key vectors as K,V
  - Second layer: uses first layer output as K,V

**Key Features:**
- Pre-norm architecture (LayerNorm before attention/FFN)
- Residual connections
- Standard RoPE for self-attention, LARoPE for cross-attention

### 2. **ReferenceEncoder** (`reference_encoder.py`)
Extracts speaker characteristics from reference speech compressed latents.

**Architecture:**
- Input: compressed latents (B, 144, T/Kc)
- Linear projection: 144 → 128
- 6 ConvNeXt blocks (hidden=128, intermediate=512, kernel=5)
- 2 Cross-Attention layers with learnable query vectors:
  - First layer: 50 learnable queries as Q, ConvNeXt output as K,V
  - Second layer: same learnable queries as Q, first layer output as K,V
- Output: reference key vectors (50, 128)

**Key Features:**
- Efficient speaker characteristic extraction
- Learnable query vectors for stable conditioning

### 3. **VFEstimator** (`vf_estimator.py`)
Predicts velocity field for flow matching using text and speaker conditioning.

**Architecture (Main Block, repeated 4 times):**
- 4 Dilated ConvNeXt blocks (dilations=[1,2,4,8])
- 2 Standard ConvNeXt blocks
- TimeCondBlock: sinusoidal time embedding → MLP → global addition
- TextCondBlock: cross-attention with text embeddings (K,V)
- RefCondBlock: cross-attention with reference embeddings (K,V)

**Time Embedding:**
- Sinusoidal positional encoding
- 64-dimensional embedding
- Used for conditioning at each main block

**Key Features:**
- Dilated convolutions for large receptive field
- Multi-scale conditioning (time, text, speaker)
- Residual connections

## Attention Mechanisms (`attention.py`)

### RoPE (Rotary Position Embedding)
Standard RoPE implementation with rotation angle: `θ_j * p`

**Usage:** Self-attention where absolute position is important

### LARoPE (Length-Aware Rotary Position Embedding)
Modified RoPE with length-aware scaling: `γ * (p/L) * θ_j`

**Key Properties:**
- Normalizes positions by sequence length
- Creates diagonal bias in cross-attention
- γ=10 is the scaling hyperparameter
- Supports separate query_length and key_length

**Usage:** Cross-attention for relative position modeling

### MultiHeadSelfAttention
Standard multi-head self-attention with RoPE.

**Inputs:** x (batch, seq_len, dim)
**Output:** attended (batch, seq_len, dim)

### MultiHeadCrossAttention
Cross-attention with optional LARoPE support.

**Inputs:**
- query: (batch, query_len, dim)
- key: (batch, key_len, dim)
- value: (batch, key_len, dim)

**Output:** attended (batch, query_len, dim)

## Latent Compression

The module handles latent compression/decompression:

```
compress(B, 24, T) → (B, 144, T//6)    via reshape: (B, 24, T) → (B, 24, T//6, 6) → (B, 144, T//6)
decompress(B, 144, T//6) → (B, 24, T)  inverse operation
```

**Compression Ratio:** 6 (default)
**Compressed Channels:** 24 × 6 = 144

## Usage Examples

### Training

```python
from models import TextToLatent

# Create model
model = TextToLatent(
    vocab_size=256,
    text_transformer_hidden=512,
    vf_channel_dim=256,
    latent_channels=24,
    compression_ratio=6,
    use_cfg=True,  # Enable classifier-free guidance
)

# Prepare data
text_tokens = torch.randint(0, 256, (batch_size, text_len))
noisy_latents = torch.randn(batch_size, 24, seq_len)  # Uncompressed
compressed_ref = torch.randn(batch_size, 144, seq_len // 6)  # Reference
timesteps = torch.rand(batch_size)  # [0, 1]

# Forward pass
velocity = model(
    noisy_latents=noisy_latents,
    compressed_latents_ref=compressed_ref,
    text_tokens=text_tokens,
    timestep=timesteps,
    cfg_scale=1.0,  # No guidance
)

# velocity.shape = (batch_size, 144, seq_len // 6)
```

### Inference

```python
# Using ODE solver (Euler method)
with torch.no_grad():
    latents = model.inference(
        text_tokens=text_tokens,
        compressed_latents_ref=compressed_ref,
        num_inference_steps=50,
        cfg_scale=7.5,  # Strong guidance
    )

# Decompress if needed
uncompressed = model.decompress_latents(latents, Kc=6)
```

### Classifier-Free Guidance

When `use_cfg=True`, the model supports classifier-free guidance:

```python
# Training: randomly set cfg_scale = 1.0 for 10% of samples
if random.random() < 0.1:
    cfg_scale = 1.0
else:
    cfg_scale = 1.0  # Standard training

# Inference: use high guidance scale
velocity = model.forward(..., cfg_scale=7.5)
# Effectively: v = v_uncond + 7.5 * (v_cond - v_uncond)
```

## File Structure

```
models/
├── __init__.py                    # Package exports
├── attention.py                   # RoPE, LARoPE, attention mechanisms
├── convnext.py                    # ConvNeXt blocks
├── text_encoder.py               # TextEncoder implementation
├── reference_encoder.py           # ReferenceEncoder implementation
├── vf_estimator.py               # VFEstimator and conditioning blocks
├── text_to_latent.py             # Complete TextToLatent module
└── example_usage.py              # Usage examples
```

## Key Design Decisions

### 1. Pre-Norm Architecture
- LayerNorm before attention/FFN
- More stable training
- Better gradient flow

### 2. ConvNeXt Blocks
- Depthwise convolution for efficiency
- Inverted bottleneck structure
- Residual connections

### 3. Multi-Head Attention
- 4 attention heads by default
- Scales well with sequence length
- LARoPE for cross-attention provides relative position bias

### 4. Time Conditioning
- Sinusoidal embeddings (learnable frequency scaling possible)
- Global addition to feature maps
- Used in every main block of VF Estimator

### 5. Classifier-Free Guidance
- Learnable unconditional parameters
- Interpolation between conditional and unconditional predictions
- Guidance scale as hyperparameter

## Hyperparameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| vocab_size | 256 | Character vocabulary |
| text_char_embed_dim | 128 | Character embedding dimension |
| text_transformer_hidden | 512 | Transformer hidden dimension |
| vf_channel_dim | 256 | VF Estimator channel dimension |
| latent_channels | 24 | Uncompressed latent channels |
| compression_ratio | 6 | Time compression factor |
| num_convnext (text) | 6 | ConvNeXt blocks in text encoder |
| num_convnext (ref) | 6 | ConvNeXt blocks in reference encoder |
| num_main_blocks (vf) | 4 | Main blocks in VF Estimator |
| n_heads | 4 | Number of attention heads |
| num_reference_keys | 50 | Learnable reference vectors |
| gamma | 10.0 | LARoPE scaling factor |

## Shape Specifications

### Text Encoder Input/Output
```
Input:  text_tokens (batch, text_len)
        reference_features (batch, num_ref_keys, ref_dim)
Output: text_embeddings (batch, text_len, transformer_hidden)
```

### Reference Encoder Input/Output
```
Input:  compressed_latents (batch, 144, T_compressed)
Output: ref_keys (batch, num_queries, query_dim)
        ref_values (batch, num_queries, query_dim)
```

### VF Estimator Input/Output
```
Input:  noisy_latents (batch, 144, T_compressed)
        text_embeddings (batch, text_len, 512)
        ref_keys (batch, 50, 128)
        timestep (batch,)
Output: velocity (batch, 144, T_compressed)
```

### Full Pipeline
```
Uncompressed latents:  (batch, 24, T) → compress → (batch, 144, T//6)
                                                         ↓
Text tokens:          (batch, text_len) → TextEncoder → (batch, text_len, 512)
                                                         ↓
Reference latents:    (batch, 144, T_ref//6) → ReferenceEncoder → (batch, 50, 128)
                                                                      ↓
Timestep:             (batch,) ──────────────────────────────────→ VF Estimator
                                                                      ↓
Velocity output:      (batch, 144, T//6)
```

## Notes

- All code uses pure PyTorch with type annotations
- Supports both training (with masks) and inference (with ODE solver)
- Efficient implementation with grouped convolutions
- Compatible with distributed training (DataParallel/DistributedDataParallel)
- Flexible hyperparameter configuration for experimentation

## References

- RoPE: Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding"
- Flow Matching: Liphardt et al., "Flow Matching for Generative Modeling"
- ConvNeXt: Liu et al., "A ConvNet for the 2020s"
