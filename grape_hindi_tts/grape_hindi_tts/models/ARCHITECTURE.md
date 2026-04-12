# SupertonicTTS Text-to-Latent: Architecture Deep Dive

## Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TextToLatent Module                          │
└─────────────────────────────────────────────────────────────────────┘

Input: text_tokens, noisy_latents, compressed_ref_latents, timestep

                              ↓
                   ┌──────────┼──────────┐
                   ↓          ↓          ↓
            ┌────────────┐ ┌──────────┐ ┌─────────────┐
            │   Text     │ │Reference │ │   Compress  │
            │  Encoder   │ │ Encoder  │ │  Latents    │
            └────────────┘ └──────────┘ └─────────────┘
                   ↓          ↓          ↓
          text_emb (B,T,512)  ref_keys   noisy_compressed
                    (B,50,128)  (B,144,T/6)
                   ↓          ↓          ↓
                   └──────────┼──────────┘
                              ↓
                    ┌──────────────────────┐
                    │   VF Estimator       │
                    │  + time embedding    │
                    │  + text conditioning │
                    │  + ref conditioning  │
                    └──────────────────────┘
                              ↓
                    Output: velocity (B,144,T/6)
```

## TextEncoder Architecture

```
text_tokens (B, T_text)
         ↓
    char embedding (128)
         ↓
    ConvNeXt×6 (128→128)
         ↓
    Linear (128→512)
         ↓
  ┌─────────────────────┐
  │  Self-Attention×4   │  ← RoPE
  │  (512→512)          │
  │  Pre-norm, FFN      │
  └─────────────────────┘
         ↓
  ┌─────────────────────┐
  │ Cross-Attention×2   │  ← LARoPE
  │ (512→512)           │
  │ Q: text features    │
  │ K,V: reference keys │
  │ Pre-norm, FFN       │
  └─────────────────────┘
         ↓
text_embeddings (B, T_text, 512)
```

**Key Components:**
- **Char Embedding:** Converts discrete tokens to dense vectors
- **ConvNeXt Blocks:** Local feature extraction via depthwise convolutions
- **Self-Attention:** Global text relationships using RoPE
- **Cross-Attention:** Speaker adaptation via reference features using LARoPE

## ReferenceEncoder Architecture

```
compressed_latents_ref (B, 144, T_ref/6)
         ↓
    Linear (144→128)
         ↓
    ConvNeXt×6 (128→128)
         ↓
  ┌──────────────────────────┐
  │  Cross-Attention Layer 1 │  ← LARoPE
  │  Q: learnable (50×128)   │
  │  K,V: ConvNeXt output    │
  └──────────────────────────┘
         ↓
  ┌──────────────────────────┐
  │  Cross-Attention Layer 2 │  ← LARoPE
  │  Q: learnable (50×128)   │
  │  K,V: Layer1 output      │
  └──────────────────────────┘
         ↓
ref_keys (B, 50, 128)
```

**Key Components:**
- **Input Projection:** Prepares compressed latents
- **ConvNeXt Blocks:** Temporal feature extraction
- **Learnable Query Vectors:** Fixed 50 queries attending to speech features
- **Cascaded Cross-Attention:** Progressive refinement of speaker features

## VFEstimator Architecture

```
noisy_latents (B, 144, T/6) + timestep (B,)
         ↓
    Linear (144→256)
         ↓
    ╔════════════════════════════════════════════════════════╗
    ║           Main Block (repeated 4 times)                ║
    ║  ┌──────────────────────────────────────────────────┐ ║
    ║  │ ConvNeXt×4 with dilations [1,2,4,8]             │ ║
    ║  │ (Dilated receptive field expansion)              │ ║
    ║  └──────────────────────────────────────────────────┘ ║
    ║  ┌──────────────────────────────────────────────────┐ ║
    ║  │ ConvNeXt×2 (standard)                            │ ║
    ║  └──────────────────────────────────────────────────┘ ║
    ║  ┌──────────────────────────────────────────────────┐ ║
    ║  │ TimeCondBlock                                    │ ║
    ║  │ time_emb → SiLU → project → global add           │ ║
    ║  └──────────────────────────────────────────────────┘ ║
    ║  ┌──────────────────────────────────────────────────┐ ║
    ║  │ TextCondBlock (Cross-Attention)                  │ ║
    ║  │ Q: latent features, K,V: text_embeddings         │ ║
    ║  │ LARoPE enables relative position modeling        │ ║
    ║  └──────────────────────────────────────────────────┘ ║
    ║  ┌──────────────────────────────────────────────────┐ ║
    ║  │ RefCondBlock (Cross-Attention)                   │ ║
    ║  │ Q: latent features, K,V: reference keys          │ ║
    ║  │ LARoPE enables relative position modeling        │ ║
    ║  └──────────────────────────────────────────────────┘ ║
    ╚════════════════════════════════════════════════════════╝
         ↓
    ConvNeXt×4 (final)
         ↓
    Linear (256→144)
         ↓
velocity (B, 144, T/6)
```

**Key Features:**
- **Dilated ConvNeXt:** Exponential receptive field growth (1,2,4,8)
- **TimeCondBlock:** Projects time embedding to channel space, adds globally
- **TextCondBlock:** Cross-attention between latents and text features
- **RefCondBlock:** Cross-attention between latents and reference features
- **Residual Connections:** Throughout all blocks

## Attention Mechanism Details

### MultiHeadSelfAttention
```python
# Standard dot-product attention with RoPE
Q = Linear_Q(x)  # (B, n_heads, seq_len, head_dim)
K = Linear_K(x)  # (B, n_heads, seq_len, head_dim)
V = Linear_V(x)  # (B, n_heads, seq_len, head_dim)

# Apply RoPE to Q and K
Q_rot = apply_rope(Q, cos_pos, sin_pos)
K_rot = apply_rope(K, cos_pos, sin_pos)

# Attention
scores = Q_rot @ K_rot.T / sqrt(head_dim)
attn = softmax(scores)
output = attn @ V
```

### MultiHeadCrossAttention with LARoPE
```python
# Cross-attention with length-aware position encoding
Q = Linear_Q(query)      # (B, n_heads, q_len, head_dim)
K = Linear_K(key)        # (B, n_heads, k_len, head_dim)
V = Linear_V(value)      # (B, n_heads, k_len, head_dim)

# Apply LARoPE: angle = gamma * (pos / length) * theta_j
Q_rot = apply_rope(Q, cos_q, sin_q)  # Normalized by q_len
K_rot = apply_rope(K, cos_k, sin_k)  # Normalized by k_len

# Attention with relative position bias
scores = Q_rot @ K_rot.T / sqrt(head_dim)
attn = softmax(scores)
output = attn @ V
```

## Latent Compression/Decompression

### Compression (B, C, T) → (B, C×K, T//K)
```python
# Group K consecutive timesteps and flatten with channels
x: (B, 24, 600)
   ↓ reshape to (B, 24, 100, 6)
   ↓ permute to (B, 24, 6, 100)
   ↓ reshape to (B, 144, 100)

# Conceptually: fold 6 timesteps into channel dimension
```

### Decompression (B, C×K, T//K) → (B, C, T)
```python
# Unfold channel dimension back to timestep dimension
x: (B, 144, 100)
   ↓ reshape to (B, 24, 6, 100)
   ↓ permute to (B, 24, 100, 6)
   ↓ reshape to (B, 24, 600)
```

## Time Embedding Process

```
timestep ∈ [0, 1]
   ↓
Sinusoidal encoding:
  freq_j = 10000^(-j/(d/2))
  emb[2j] = sin(timestep * freq_j)
  emb[2j+1] = cos(timestep * freq_j)
   ↓
(B, 64) time embedding
   ↓
Linear (64 → 128)
   ↓
SiLU activation
   ↓
Linear (128 → channel_dim)
   ↓
(B, channel_dim) → broadcast to (B, channel_dim, 1)
   ↓
Add globally to feature map
```

## Classifier-Free Guidance

During training:
```python
if random.random() < cfg_uncond_prob:
    # Unconditional: set text/ref to learnable parameters
    text_emb = uncond_text_embed
    ref_keys = uncond_ref_keys
```

During inference:
```python
# Get conditional prediction
v_cond = model(noisy, text_emb, ref_keys, t)

# Get unconditional prediction
v_uncond = model(noisy, uncond_text_emb, uncond_ref_keys, t)

# Guided prediction
v_guided = v_uncond + scale * (v_cond - v_uncond)
# High scale → stronger adherence to text/speaker
```

## Memory Efficiency

The architecture is designed for efficiency:

1. **Grouped Convolutions:** Depthwise convolutions reduce parameters
2. **Pre-norm:** Slightly more stable, doesn't require post-norm parameters
3. **Latent Compression:** 6× time compression reduces sequence length
4. **Selective Attention:** Cross-attention only at specific layers
5. **ConvNeXt vs Transformer:** More parameter-efficient local modeling

**Approximate Parameter Counts:**
- TextEncoder: ~20M
- ReferenceEncoder: ~10M
- VFEstimator: ~50M
- **Total: ~80M parameters**

## Training Considerations

1. **Timestep Distribution:** Sample uniformly from [0, 1]
2. **Masking:** Attention supports sequence masks for variable lengths
3. **Gradient Flow:** Pre-norm helps, but monitor for saturation
4. **Batch Size:** Recommend 32+ for stable training
5. **Learning Rate:** Start with 1e-4, use learning rate warmup
6. **CFG Training:** Set cfg_uncond_prob = 0.1 during training

## Inference Considerations

1. **ODE Solver:** Default is Euler (simple, first-order)
2. **Steps:** 50 steps typically sufficient for good quality
3. **CFG Scale:** 7.5 recommended for strong guidance
4. **Temperature:** Can be added to noise sampling if desired
5. **Decompression:** Optional if latents are needed in original space
