import torch
import sys

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data Configuration
# IMPORTANT: Data currently has depth=1 (single slices, not true 3D volumes)
# VAE configured to work with depth=1 by not downsampling in depth dimension
ROI_SIZE = (1, 128, 128)   # (Depth, Height, Width) - depth=1 to match actual data
SAMPLES_PER_VOLUME = 2     # VAE_R: Reduced from 4 to reduce memory (4x larger latents)
NUM_WORKERS = 0            # Set to 0 for Windows compatibility (use 4+ on Linux with GPU)
BATCH_SIZE = 2             # VAE_R: Reduced from 4 (4x larger latents use more memory)
GRADIENT_ACCUMULATION_STEPS = 2  # Gradient accumulation for stability

# VAE Configuration
VAE_MODEL_PATH = "vae.pth"

# --- VAE_R: REDUCE SPATIAL COMPRESSION FOR SUB-NUCLEAR DETAILS ---
VAE_CHANNELS = (32, 64, 128)   # Larger capacity to capture texture
LATENT_CHANNELS = 32           # More bandwidth for high-frequency details

# FIX R: Reduce spatial compression from 8x to 4x
# OLD: ((1,2,2), (1,2,2), (1,2,2)) → 128→16 (8x compression) - destroys sub-nuclear structures
# NEW: ((1,2,2), (1,2,2), (1,1,1)) → 128→32 (4x compression) - preserves spatial details
VAE_STRIDES = ((1, 2, 2), (1, 2, 2), (1, 1, 1))
# -----------------------------------------------------------------------

VAE_EPOCHS = 100
VAE_LR = 2e-4                # Standard learning rate

# U-Net Translation Model Configuration (Phase 2)
# Direct latent-space translation: QPI latent → DAPI latent
# NO diffusion, NO timesteps, NO noise - deterministic mapping
UNET_CHANNELS = (32, 64, 128, 256)  # Encoder channel progression
UNET_USE_ATTENTION = True           # Spatial attention in bottleneck
UNET_DROPOUT = 0.3                  # Dropout rate for regularization (prevent overfitting)

# Training Configuration
EPOCHS = 300                        # Total training epochs (simpler model converges faster)
WARMUP_EPOCHS = 20                  # Stage 1: latent loss only
LEARNING_RATE = 1e-4                # Stage 1 learning rate (REDUCED for stability)
LEARNING_RATE_FINETUNE = 5e-5       # Stage 2 learning rate (fine-tuning)
EARLY_STOPPING_PATIENCE = None      # Disabled - let training complete (val variance is normal)

# Loss Weights (Stage 2: Fine-tuning)
LATENT_LOSS_WEIGHT = 1.0            # Primary: direct latent space supervision
PIXEL_LOSS_WEIGHT = 0.7             # Secondary: perceptual quality (INCREASED for accuracy)
GRADIENT_LOSS_WEIGHT = 0.3          # Tertiary: edge sharpness (INCREASED for sharp nuclei)
NUCLEUS_EMPHASIS = 15.0             # Weighted loss multiplier for bright nuclear regions (INCREASED)
BRIGHTNESS_PENALTY_WEIGHT = 0.15    # Penalty for predicting darkness in cell regions (prevents "safe zero" failure) 

# VAE Loss Weights - VAE_R Configuration
L1_WEIGHT = 1.0                # Pixel-wise reconstruction
KL_WEIGHT = 1e-6               # REDUCED from 1e-4 - allow latent to use full capacity for details
GRADIENT_WEIGHT = 10.0         # REPLACED LPIPS (memory leak) - Gradient loss preserves edges
KL_ANNEAL_EPOCHS = 20          # Gradually increase KL weight over first 20 epochs

def validate_config():
    """
    Validates configuration parameters for consistency and correctness.
    Raises ValueError if any validation fails.
    """
    errors = []
    
    # Check ROI_SIZE dimensions are divisible by downsample factors
    # Handle both tuple strides and integer strides
    if isinstance(VAE_STRIDES[0], (tuple, list)):
        # Tuple strides: [(1,2,2), (1,2,2), (1,2,2)]
        depth_factor = height_factor = width_factor = 1
        for stride in VAE_STRIDES:
            depth_factor *= stride[0]
            height_factor *= stride[1]
            width_factor *= stride[2]
        factors = [depth_factor, height_factor, width_factor]
    else:
        # Integer strides: [2, 2, 2]
        downsample_factor = 1
        for stride in VAE_STRIDES:
            downsample_factor *= stride
        factors = [downsample_factor] * 3

    dim_names = ['Depth', 'Height', 'Width']
    for i, (dim, factor) in enumerate(zip(ROI_SIZE, factors)):
        if dim % factor != 0:
            errors.append(
                f"ROI_SIZE[{i}] ({dim_names[i]}) = {dim} must be divisible by downsample_factor = {factor}"
            )

    # Check channel/strides consistency
    if len(VAE_CHANNELS) != len(VAE_STRIDES):
        errors.append(
            f"VAE_CHANNELS length ({len(VAE_CHANNELS)}) must match VAE_STRIDES length ({len(VAE_STRIDES)})"
        )

    # Check positive values
    if BATCH_SIZE <= 0:
        errors.append(f"BATCH_SIZE = {BATCH_SIZE} must be positive")
    if NUM_WORKERS < 0:
        errors.append(f"NUM_WORKERS = {NUM_WORKERS} must be non-negative")
    if SAMPLES_PER_VOLUME <= 0:
        errors.append(f"SAMPLES_PER_VOLUME = {SAMPLES_PER_VOLUME} must be positive")
    if VAE_EPOCHS <= 0:
        errors.append(f"VAE_EPOCHS = {VAE_EPOCHS} must be positive")
    if LATENT_CHANNELS <= 0:
        errors.append(f"LATENT_CHANNELS = {LATENT_CHANNELS} must be positive")

    # Check learning rates
    if VAE_LR <= 0:
        errors.append(f"VAE_LR = {VAE_LR} must be positive")
    
    # Check loss weights
    if L1_WEIGHT < 0:
        errors.append(f"L1_WEIGHT = {L1_WEIGHT} must be non-negative")
    if KL_WEIGHT < 0:
        errors.append(f"KL_WEIGHT = {KL_WEIGHT} must be non-negative")
    if GRADIENT_WEIGHT < 0:
        errors.append(f"GRADIENT_WEIGHT = {GRADIENT_WEIGHT} must be non-negative")
    
    if errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ValueError(error_msg)
    
    return True


# Auto-validate on import (can be disabled if needed)
if __name__ != "__main__":
    try:
        validate_config()
    except ValueError as e:
        print(f"Warning: Configuration validation failed: {e}")
        print("Continuing anyway, but errors may occur during execution.")