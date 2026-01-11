import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from monai.metrics import PSNRMetric, SSIMMetric
from monai.utils import set_determinism
import config
import data_pipeline
import vae_model_fixed as vae_model  # EMERGENCY FIX: Use fixed VAE (no 48GB bug)

def validate():
    # 1. Setup & Configuration
    device = config.DEVICE
    print(f"Running VAE validation on: {device}")

    # Use truly random samples (no deterministic seed for sample selection)
    # Note: Metrics are still deterministic based on the full validation set

    # 2. Data Loading
    print("Loading validation dataset...")
    try:
        val_loader = data_pipeline.get_dataloader(
            data_dir="3d_data", 
            for_vae=True, 
            split='val', 
            val_split=0.2,
            for_eval=True  # Uses CenterSpatialCropd (deterministic)
        )
    except ValueError as e:
        print(f"Error loading data: {e}")
        print("Ensure '3d_data/dapi' exists and contains .tif or .nii.gz files.")
        return

    # 3. Model Loading
    model = vae_model.get_vae_model().to(device)
    
    if os.path.exists(config.VAE_MODEL_PATH):
        print(f"Loading checkpoint from {config.VAE_MODEL_PATH}...")
        model.load_state_dict(torch.load(config.VAE_MODEL_PATH, map_location=device))
    else:
        print(f"CRITICAL ERROR: Checkpoint '{config.VAE_MODEL_PATH}' not found.")
        return

    model.eval()

    # 4. Metrics Setup
    # CRITICAL FIX: Use spatial_dims=2. Even though data is stored as (1, D, H, W),
    # the depth is 1. We will squeeze the depth dim before passing to these metrics.
    psnr_metric = PSNRMetric(max_val=1.0) 
    ssim_metric = SSIMMetric(data_range=1.0, spatial_dims=2)

    # Trackers for latent statistics
    latent_means = []
    latent_stds = []

    print("\nStarting evaluation loop...")
    import random

    with torch.no_grad():
        # Collect all validation samples and compute metrics in single pass
        all_val_samples = []

        for batch_idx, batch in enumerate(val_loader):
            # Input shape: (B, 1, D, H, W) -> D is typically 1
            inputs = batch["dapi"].to(device)

            # DEBUG: Print first batch statistics
            if batch_idx == 0:
                print(f"\n[DEBUG] First batch from loader:")
                print(f"  Shape: {inputs.shape}")
                print(f"  Min: {inputs.min().item():.6f}, Max: {inputs.max().item():.6f}")
                print(f"  Mean: {inputs.mean().item():.6f}, Std: {inputs.std().item():.6f}")

            # Forward pass
            latents = vae_model.encode_vae(model, inputs)
            recons = vae_model.decode_vae(model, latents)

            # DEBUG: Print reconstruction quality for first batch
            if batch_idx == 0:
                print(f"\n[DEBUG] After VAE reconstruction:")
                print(f"  Reconstruction - Shape: {recons.shape}")
                print(f"  Reconstruction - Min: {recons.min().item():.6f}, Max: {recons.max().item():.6f}")
                print(f"  Reconstruction - Mean: {recons.mean().item():.6f}, Std: {recons.std().item():.6f}")
                print(f"  Diff - Mean: {torch.mean(torch.abs(recons - inputs)).item():.6f}")

            # Calculate Latent Statistics (Keep dimensions for calculation)
            latent_means.append(latents.mean().item())
            latent_stds.append(latents.std().item())

            # Metric Calculation
            # Squeeze depth dimension (index 2) to make it (B, C, H, W) for 2D metrics
            # This fixes the "Kernel size can't be greater than actual input size" error
            inputs_2d = inputs.squeeze(2)
            recons_2d = recons.squeeze(2)

            psnr_metric(y_pred=recons_2d, y=inputs_2d)
            ssim_metric(y_pred=recons_2d, y=inputs_2d)

            # Collect samples for random visualization
            dapi_batch = batch["dapi"]
            for j in range(dapi_batch.shape[0]):
                all_val_samples.append(dapi_batch[j])

        # Randomly select 5 samples for visualization (truly random)
        num_samples_to_visualize = 5
        print(f"\nSelecting {num_samples_to_visualize} truly random validation samples from {len(all_val_samples)} total samples...")
        if len(all_val_samples) >= num_samples_to_visualize:
            val_samples = random.sample(all_val_samples, num_samples_to_visualize)
        else:
            val_samples = all_val_samples
        print(f"OK: Selected {len(val_samples)} random samples for visualization")

        # Visualization: Generate reconstructions for the selected samples
        print(f"\nGenerating visualizations for {len(val_samples)} random samples...")
        save_validation_grid_samples(model, val_samples, device, "vae_validation_sample.png")

    # 5. Aggregate Results
    avg_psnr = psnr_metric.aggregate().item()
    avg_ssim = ssim_metric.aggregate().item()
    global_latent_mean = np.mean(latent_means)
    global_latent_std = np.mean(latent_stds)

    # 6. Strict Quality Assessment
    print_quality_report(avg_psnr, avg_ssim, global_latent_mean, global_latent_std)

def print_quality_report(psnr, ssim, mean, std):
    """
    Prints a strict interpretation of the VAE quality based on project requirements.
    Benchmarks derived from PROJECT_STATUS_REPORT.md:
      - Ideal SSIM > 0.75
      - Ideal PSNR > 26 dB
      - [cite_start]Latent Std ~ 1.0 for Diffusion [cite: 139, 151]
    """
    print("\n" + "="*60)
    print("               VAE VALIDATION REPORT")
    print("="*60)
    
    # 1. Reconstruction Quality
    print(f"RECONSTRUCTION METRICS:")
    print(f"  PSNR: {psnr:.2f} dB")
    print(f"  SSIM: {ssim:.4f}")
    
    recon_status = "FAIL"
    if ssim >= 0.85 and psnr >= 28:
        recon_status = "EXCELLENT"
        print("  -> STATUS: EXCELLENT. High fidelity reconstruction.")
    elif ssim >= 0.75 and psnr >= 26:
        recon_status = "PASS"
        print("  -> STATUS: PASS. Meets ideal project targets.")
    elif ssim >= 0.70 and psnr >= 24:
        recon_status = "BORDERLINE"
        print("  -> STATUS: BORDERLINE. Meets minimum GAN baseline but may lack detail.")
    else:
        print("  -> STATUS: FAIL. Does not meet minimum requirements (>0.70 SSIM).")
        print("     ACTION: Do not proceed. Tune VAE (increase perceptual weight or capacity).")

    print("-" * 60)

    # 2. Latent Space Distribution
    print(f"LATENT SPACE STATISTICS:")
    print(f"  Mean: {mean:.4f} (Target: ~0.0)")
    print(f"  Std:  {std:.4f} (Target: ~1.0)")
    
    dist_status = "FAIL"
    if 0.9 <= std <= 1.1 and -0.1 <= mean <= 0.1:
        dist_status = "PERFECT"
        print("  -> STATUS: PERFECT. Ready for standard diffusion training.")
    elif 0.8 <= std <= 1.2:
        dist_status = "ACCEPTABLE"
        print("  -> STATUS: ACCEPTABLE. Slightly off-distribution.")
    else:
        print("  -> STATUS: WARNING/FAIL. Latent space is not Unit Gaussian.")
        print(f"     ACTION: You MUST normalize latents during AGLU training.")
        print(f"             Use scale factor: {1.0/std:.4f}")

    print("="*60)

    if recon_status in ["EXCELLENT", "PASS"] and dist_status in ["PERFECT", "ACCEPTABLE"]:
        print("\n[PASS] OVERALL DECISION: GO FOR AGLU TRAINING")
    else:
        print("\n[FAIL] OVERALL DECISION: STOP. FIX VAE BEFORE PROCEEDING.")
        if recon_status == "FAIL":
            print("   - Issue: Reconstruction quality is too low.")
        if dist_status == "FAIL":
            print("   - Issue: Latent distribution is malformed (requires scaling).")

def save_validation_grid_samples(model, val_samples, device, filename):
    """
    Visualize VAE reconstructions on random validation samples.
    Saves Nx3 grid: Ground Truth | Reconstruction | Abs Error for each sample.
    """
    num_samples = len(val_samples)
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))

    # Handle single sample case (axes won't be 2D)
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for idx in range(num_samples):
        dapi = val_samples[idx].unsqueeze(0).to(device)  # (1, 1, 1, 128, 128)

        # Encode and decode
        latents = vae_model.encode_vae(model, dapi)
        recon = vae_model.decode_vae(model, latents)

        # Move to CPU and numpy: (1, 1, D, H, W) -> (D, H, W)
        img_in = dapi[0, 0].cpu().numpy()
        img_out = recon[0, 0].cpu().numpy()

        # Extract center slice along depth
        d_idx = img_in.shape[0] // 2
        slice_in = img_in[d_idx]
        slice_out = img_out[d_idx]

        # Calculate Error Map
        diff = np.abs(slice_in - slice_out)

        # Plot Ground Truth
        axes[idx, 0].imshow(slice_in, cmap='magma', vmin=0, vmax=1)
        axes[idx, 0].set_title(f"Sample {idx+1}: Ground Truth")
        axes[idx, 0].axis('off')

        # Plot Reconstruction
        axes[idx, 1].imshow(slice_out, cmap='magma', vmin=0, vmax=1)
        axes[idx, 1].set_title(f"Sample {idx+1}: Reconstruction")
        axes[idx, 1].axis('off')

        # Plot Error
        im = axes[idx, 2].imshow(diff, cmap='inferno')
        axes[idx, 2].set_title(f"Abs Error (Max: {diff.max():.2f})")
        axes[idx, 2].axis('off')
        plt.colorbar(im, ax=axes[idx, 2])

    plt.suptitle(f"VAE Validation - {num_samples} Truly Random Samples", fontsize=16)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nVisual validation saved to: {os.path.abspath(filename)}")
    plt.close()

def save_validation_grid(inputs, recons, filename):
    """
    Saves a comparison image: Input | Reconstruction | Abs Error
    """
    # Move to CPU and numpy: (B, C, D, H, W) -> (C, D, H, W)
    img_in = inputs[0, 0].cpu().numpy()
    img_out = recons[0, 0].cpu().numpy()

    # Extract center slice along depth
    d_idx = img_in.shape[0] // 2
    slice_in = img_in[d_idx]
    slice_out = img_out[d_idx]

    # Calculate Error Map
    diff = np.abs(slice_in - slice_out)

    plt.figure(figsize=(12, 4))

    # Plot Original
    plt.subplot(1, 3, 1)
    plt.imshow(slice_in, cmap='magma', vmin=0, vmax=1)
    plt.title("Ground Truth (DAPI)")
    plt.axis('off')

    # Plot Reconstruction
    plt.subplot(1, 3, 2)
    plt.imshow(slice_out, cmap='magma', vmin=0, vmax=1)
    plt.title("VAE Reconstruction")
    plt.axis('off')

    # Plot Error
    plt.subplot(1, 3, 3)
    plt.imshow(diff, cmap='inferno')
    plt.title(f"Abs Error (Max: {diff.max():.2f})")
    plt.axis('off')
    plt.colorbar()

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"\nVisual validation saved to: {os.path.abspath(filename)}")
    plt.close()

if __name__ == "__main__":
    validate()