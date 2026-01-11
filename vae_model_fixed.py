"""
FINAL FIX: Fully convolutional VAE - NO fully connected layers
This avoids the billion-parameter bug that causes 48GB memory usage
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import config

class FullyConvVAE(nn.Module):
    """Fully convolutional VAE - no FC layers, pure convolutions"""
    def __init__(self):
        super().__init__()

        # Encoder: (1, 1, 128, 128) → (32, 1, 32, 32)
        # Stage 1: 128→64
        self.enc1 = nn.Conv3d(1, 32, kernel_size=3, stride=(1,2,2), padding=1)
        self.bn1 = nn.BatchNorm3d(32)

        # Stage 2: 64→32
        self.enc2 = nn.Conv3d(32, 64, kernel_size=3, stride=(1,2,2), padding=1)
        self.bn2 = nn.BatchNorm3d(64)

        # Stage 3: No spatial reduction (VAE_R: 4x instead of 8x)
        self.enc3 = nn.Conv3d(64, 128, kernel_size=3, stride=(1,1,1), padding=1)
        self.bn3 = nn.BatchNorm3d(128)

        # Latent projection (CONVOLUTIONAL - not FC!)
        # From 128 channels to 32 channels (LATENT_CHANNELS)
        self.conv_mu = nn.Conv3d(128, 32, kernel_size=1)  # 1x1 conv = channel reduction
        self.conv_logvar = nn.Conv3d(128, 32, kernel_size=1)

        # Decoder input projection
        self.conv_decode = nn.Conv3d(32, 128, kernel_size=1)  # 1x1 conv = channel expansion

        # Decoder: (128, 1, 32, 32) → (1, 1, 128, 128)
        # Stage 1: No spatial change
        self.dec1 = nn.Conv3d(128, 64, kernel_size=3, stride=1, padding=1)
        self.dbn1 = nn.BatchNorm3d(64)

        # Stage 2: 32→64
        self.dec2 = nn.ConvTranspose3d(64, 32, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1))
        self.dbn2 = nn.BatchNorm3d(32)

        # Stage 3: 64→128
        self.dec3 = nn.ConvTranspose3d(32, 1, kernel_size=(1,4,4), stride=(1,2,2), padding=(0,1,1))

    def encode_forward(self, x):
        """Encode to mu, logvar (spatial format)"""
        # Encoder pathway
        x = F.relu(self.bn1(self.enc1(x)))  # → (32, 1, 64, 64)
        x = F.relu(self.bn2(self.enc2(x)))  # → (64, 1, 32, 32)
        x = F.relu(self.bn3(self.enc3(x)))  # → (128, 1, 32, 32)

        # Latent projection (1x1 conv - NO FC layer!)
        mu = self.conv_mu(x)        # → (32, 1, 32, 32)
        logvar = self.conv_logvar(x)  # → (32, 1, 32, 32)

        # Flatten for compatibility with training code
        batch_size = mu.size(0)
        mu_flat = mu.view(batch_size, -1)        # → (B, 32*1*32*32=32768)
        logvar_flat = logvar.view(batch_size, -1)  # → (B, 32768)

        return mu_flat, logvar_flat

    def reparameterize(self, mu_flat, logvar_flat):
        """Reparameterization trick (flattened)"""
        std = torch.exp(0.5 * logvar_flat)
        eps = torch.randn_like(std)
        return mu_flat + eps * std

    def decode_forward(self, z_flat):
        """Decode latent to image"""
        batch_size = z_flat.size(0)

        # Reshape to spatial (32, 1, 32, 32)
        z = z_flat.view(batch_size, 32, 1, 32, 32)

        # Project back to decoder channels (1x1 conv)
        x = self.conv_decode(z)  # → (128, 1, 32, 32)

        # Decoder pathway
        x = F.relu(self.dbn1(self.dec1(x)))  # → (64, 1, 32, 32)
        x = F.relu(self.dbn2(self.dec2(x)))  # → (32, 1, 64, 64)
        x = torch.sigmoid(self.dec3(x))      # → (1, 1, 128, 128)

        return x

    def forward(self, x):
        """Full forward pass"""
        mu, logvar = self.encode_forward(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode_forward(z)
        return recon, mu, logvar

def get_vae_model():
    """Returns fully convolutional VAE"""
    return FullyConvVAE()

def compute_kl_loss(mu, logvar):
    """KL divergence loss"""
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return kl_loss

def encode_vae(vae, x):
    """
    Encode input through VAE and return latent sample in SPATIAL format.
    Compatible with AGLU which operates on spatial latents.

    Args:
        vae: VAE model
        x: Input tensor (B, C, D, H, W)

    Returns:
        latent: Spatial latent tensor (B, LATENT_CHANNELS, D', H', W')
                For this VAE: (B, 32, 1, 32, 32)
    """
    mu, logvar = vae.encode_forward(x)
    z_flat = vae.reparameterize(mu, logvar)

    # Reshape to spatial format for U-Net processing
    batch_size = z_flat.shape[0]
    # This VAE: latent is (32, 1, 32, 32) -> flattened to 32*1*32*32 = 32768
    z_spatial = z_flat.view(batch_size, 32, 1, 32, 32)

    return z_spatial

def decode_vae(vae, latent):
    """
    Decode latent through VAE.
    Handles both flattened and spatial latent formats.

    Args:
        vae: VAE model
        latent: Either flattened (B, latent_dim) or spatial (B, C, D, H, W) latent tensor

    Returns:
        reconstruction: Reconstructed image (B, C, D, H, W)
    """
    # If spatial (5D), flatten it
    if latent.dim() == 5:
        batch_size = latent.shape[0]
        latent = latent.view(batch_size, -1)  # Flatten to (B, latent_dim)

    return vae.decode_forward(latent)

def get_vae_downsample_factors():
    """
    Calculate per-dimension downsample factors for this VAE.

    This VAE downsamples as follows:
    - Depth: 1x (no downsampling, stays at 1)
    - Height: 4x (128 -> 64 -> 32)
    - Width: 4x (128 -> 64 -> 32)

    Returns:
        tuple: (depth_factor, height_factor, width_factor)
    """
    return (1, 4, 4)

def get_latent_shape(batch_size):
    """
    Calculate the expected latent shape for this VAE.

    Args:
        batch_size: Number of samples in batch

    Returns:
        tuple: (batch_size, LATENT_CHANNELS, latent_depth, latent_height, latent_width)
    """
    # This VAE: input (1, 1, 128, 128) -> latent (32, 1, 32, 32)
    return (batch_size, config.LATENT_CHANNELS, 1, 32, 32)

if __name__ == "__main__":
    model = FullyConvVAE()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Memory (FP32): {total_params * 4 / 1024**2:.2f} MB")

    # Test forward pass
    test_input = torch.randn(1, 1, 1, 128, 128)
    with torch.no_grad():
        mu, logvar = model.encode_forward(test_input)
        z = model.reparameterize(mu, logvar)
        recon = model.decode_forward(z)

    print(f"\nInput shape:  {test_input.shape}")
    print(f"Latent shape: {mu.shape}")
    print(f"Output shape: {recon.shape}")

    if recon.shape == test_input.shape:
        print("\nOK: Shape test PASSED!")
    else:
        print(f"\nERROR: Shape test FAILED! Expected {test_input.shape}, got {recon.shape}")

    # Test on CUDA
    if torch.cuda.is_available():
        print("\n[CUDA Memory Test]")
        torch.cuda.empty_cache()
        allocated_before = torch.cuda.memory_allocated(0) / 1024**3
        print(f"Before model load: {allocated_before:.3f} GB")

        model_cuda = FullyConvVAE().cuda()
        allocated_after = torch.cuda.memory_allocated(0) / 1024**3
        print(f"After model load:  {allocated_after:.3f} GB")
        print(f"Model memory:      {allocated_after - allocated_before:.3f} GB")

        if allocated_after < 0.5:
            print("OK: Memory usage is NORMAL (< 0.5 GB)")
        elif allocated_after < 2.0:
            print("WARNING: Memory usage is high but acceptable (< 2 GB)")
        else:
            print(f"ERROR: High memory usage ({allocated_after:.2f} GB)!")

        del model_cuda
        torch.cuda.empty_cache()
