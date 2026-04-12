"""
Example usage of the TextToLatent module.

Demonstrates:
1. Creating the module
2. Forward pass (training)
3. Inference with ODE solver
4. Latent compression/decompression
"""

import torch
from text_to_latent import TextToLatent


def example_training():
    """Example training loop."""
    
    # Create module
    model = TextToLatent(
        vocab_size=256,
        text_char_embed_dim=128,
        text_transformer_hidden=512,
        vf_channel_dim=256,
        latent_channels=24,
        compression_ratio=6,
        use_cfg=True,
    )

    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Example batch
    batch_size = 4
    text_len = 50
    seq_len = 600  # Uncompressed speech length
    seq_len_compressed = seq_len // 6  # 100

    # Create dummy data
    text_tokens = torch.randint(0, 256, (batch_size, text_len), device=device)
    
    # Reference latents (compressed)
    compressed_latents_ref = torch.randn(
        batch_size, 144, seq_len_compressed, device=device
    )
    
    # Noisy latents (uncompressed)
    noisy_latents = torch.randn(batch_size, 24, seq_len, device=device)
    
    # Timesteps
    timesteps = model.sample_training_timesteps(batch_size, device, noisy_latents.dtype)

    # Forward pass
    velocity = model(
        noisy_latents=noisy_latents,
        compressed_latents_ref=compressed_latents_ref,
        text_tokens=text_tokens,
        timestep=timesteps,
    )

    print(f"Input shapes:")
    print(f"  noisy_latents: {noisy_latents.shape}")
    print(f"  compressed_latents_ref: {compressed_latents_ref.shape}")
    print(f"  text_tokens: {text_tokens.shape}")
    print(f"  timesteps: {timesteps.shape}")
    print(f"\nOutput shapes:")
    print(f"  velocity: {velocity.shape}")
    print(f"  Expected: {(batch_size, 144, seq_len_compressed)}")


def example_inference():
    """Example inference using ODE solver."""
    
    # Create module
    model = TextToLatent(
        vocab_size=256,
        text_char_embed_dim=128,
        text_transformer_hidden=512,
        vf_channel_dim=256,
        latent_channels=24,
        compression_ratio=6,
        use_cfg=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Example data
    batch_size = 2
    text_len = 50
    seq_len_compressed = 100

    text_tokens = torch.randint(0, 256, (batch_size, text_len), device=device)
    compressed_latents_ref = torch.randn(
        batch_size, 144, seq_len_compressed, device=device
    )

    # Inference with ODE solver
    with torch.no_grad():
        latents = model.inference(
            text_tokens=text_tokens,
            compressed_latents_ref=compressed_latents_ref,
            num_inference_steps=50,
            cfg_scale=7.5,  # Classifier-free guidance
        )

    print(f"Inference output shape: {latents.shape}")
    print(f"Expected: {(batch_size, 144, seq_len_compressed)}")

    # Decompress if needed
    uncompressed = model.decompress_latents(latents, Kc=6)
    print(f"Decompressed shape: {uncompressed.shape}")
    print(f"Expected: {(batch_size, 24, seq_len_compressed * 6)}")


def example_latent_compression():
    """Example of latent compression and decompression."""
    
    model = TextToLatent()
    
    # Original latents: (batch=2, channels=24, time=600)
    original = torch.randn(2, 24, 600)
    
    # Compress
    compressed = model.compress_latents(original, Kc=6)
    print(f"Original: {original.shape}")
    print(f"Compressed: {compressed.shape}")
    
    # Decompress
    reconstructed = model.decompress_latents(compressed, Kc=6)
    print(f"Reconstructed: {reconstructed.shape}")
    
    # Check reconstruction error
    error = torch.mean((original - reconstructed) ** 2)
    print(f"Reconstruction error: {error:.6f} (should be ~0)")


if __name__ == "__main__":
    print("=" * 60)
    print("Training Example")
    print("=" * 60)
    example_training()
    
    print("\n" + "=" * 60)
    print("Inference Example")
    print("=" * 60)
    example_inference()
    
    print("\n" + "=" * 60)
    print("Latent Compression Example")
    print("=" * 60)
    example_latent_compression()
