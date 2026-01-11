"""
Testing Script for QPI→DAPI Translation Model

This script validates the trained U-Net model by generating predictions on the validation set
and computing visual and quantitative metrics.

Key Features:
- Direct single-pass prediction (no noise, no timesteps)
- Visual comparison: QPI input | GT DAPI | Predicted DAPI | Error Map
- Quantitative metrics: MSE, MAE, PSNR, SSIM
- Saves results to aglu_test_results/
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import random
import config
from aglu_model import get_model
from vae_model_fixed import FullyConvVAE
from data_pipeline import get_dataloader


def normalize_for_display(img):
    """Normalize image to [0, 1] range for display."""
    img = img.copy() if isinstance(img, np.ndarray) else img.clone()
    img_min = img.min()
    img_max = img.max()

    if img_max - img_min > 0:
        img = (img - img_min) / (img_max - img_min)
    else:
        img = img * 0

    return img


def compute_error_map(pred, gt):
    """Compute absolute error map."""
    return np.abs(pred - gt)


def generate_comparison_figure(qpi, dapi_gt, dapi_pred, sample_idx, save_path):
    """
    Generate a side-by-side comparison figure:
    [Input QPI] | [Ground Truth DAPI] | [Predicted DAPI] | [Error Map]
    """
    # Normalize images for display
    qpi_norm = normalize_for_display(qpi)
    dapi_gt_norm = normalize_for_display(dapi_gt)
    dapi_pred_norm = normalize_for_display(dapi_pred)

    # Compute error map
    error_map = compute_error_map(dapi_pred_norm, dapi_gt_norm)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # 1. Input QPI
    axes[0].imshow(qpi_norm, cmap='viridis')
    axes[0].set_title('Input QPI', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # 2. Ground Truth DAPI
    axes[1].imshow(dapi_gt_norm, cmap='magma')
    axes[1].set_title('Ground Truth DAPI', fontsize=12, fontweight='bold')
    axes[1].axis('off')

    # 3. Predicted DAPI
    axes[2].imshow(dapi_pred_norm, cmap='magma')
    axes[2].set_title('Predicted DAPI', fontsize=12, fontweight='bold')
    axes[2].axis('off')

    # 4. Error Map
    im = axes[3].imshow(error_map, cmap='inferno', vmin=0, vmax=0.3)
    axes[3].set_title('Absolute Error Map', fontsize=12, fontweight='bold')
    axes[3].axis('off')

    cbar = plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)
    cbar.set_label('Error Magnitude', fontsize=10)

    # Metrics
    mse = np.mean((dapi_pred_norm - dapi_gt_norm) ** 2)
    mae = np.mean(error_map)
    fig.suptitle(f'Sample {sample_idx} | MSE: {mse:.6f} | MAE: {mae:.6f}',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return mse, mae


def test(num_samples=10, checkpoint_path=None):
    """
    Main testing function.

    Args:
        num_samples: Number of validation samples to test
        checkpoint_path: Path to model checkpoint (defaults to best model)
    """
    output_dir = "aglu_test_results"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Saving test results to: {output_dir}")
    print(f"Device: {config.DEVICE}")

    # ==================== 1. LOAD MODELS ====================
    print("\n[1/4] Loading models...")

    # Load VAE
    vae = FullyConvVAE().to(config.DEVICE)
    if not os.path.exists(config.VAE_MODEL_PATH):
        raise FileNotFoundError(f"VAE model not found at {config.VAE_MODEL_PATH}")
    vae.load_state_dict(torch.load(config.VAE_MODEL_PATH, map_location=config.DEVICE))
    vae.eval()
    print(f"✓ VAE loaded from {config.VAE_MODEL_PATH}")

    # Load translation model
    model = get_model(frozen_vae=vae).to(config.DEVICE)

    if checkpoint_path is None:
        # Prioritize best model
        paths_to_try = [
            "aglu_outputs/checkpoints/model_best.pth",
            "aglu_outputs/checkpoints/model_latest.pth",
        ]
        for p in paths_to_try:
            if os.path.exists(p):
                checkpoint_path = p
                break

    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Could not find model checkpoint. Checked: {paths_to_try}")

    model.load_state_dict(torch.load(checkpoint_path, map_location=config.DEVICE))
    model.eval()
    print(f"✓ Model loaded from {checkpoint_path}")

    # ==================== 2. LOAD VALIDATION DATA ====================
    print("\n[2/4] Loading validation data...")

    val_loader = get_dataloader(
        data_dir="3d_data",
        for_vae=False,
        split='val',
        val_split=0.2,
        for_eval=True
    )

    # ==================== 3. COLLECT SAMPLES ====================
    print(f"\n[3/4] Collecting {num_samples} random samples...")

    all_qpi = []
    all_dapi = []

    for batch_data in val_loader:
        if isinstance(batch_data, list):
            qpi_batch = torch.cat([s['qpi'] for s in batch_data], dim=0)
            dapi_batch = torch.cat([s['dapi'] for s in batch_data], dim=0)
        else:
            qpi_batch = batch_data['qpi']
            dapi_batch = batch_data['dapi']

        for i in range(qpi_batch.shape[0]):
            all_qpi.append(qpi_batch[i])
            all_dapi.append(dapi_batch[i])

    # Random selection
    if len(all_qpi) < num_samples:
        num_samples = len(all_qpi)
        selected_indices = list(range(num_samples))
    else:
        selected_indices = random.sample(range(len(all_qpi)), num_samples)

    print(f"✓ Selected {num_samples} random samples")

    # ==================== 4. INFERENCE ====================
    print(f"\n[4/4] Generating predictions...")

    metrics = {'sample_idx': [], 'mse': [], 'mae': []}

    with torch.no_grad():
        for i, idx in enumerate(tqdm(selected_indices, desc="Processing")):
            qpi_sample = all_qpi[idx].to(config.DEVICE)
            dapi_sample = all_dapi[idx].to(config.DEVICE)

            # Normalize if needed
            if qpi_sample.max() > 1.5:
                qpi_sample = qpi_sample / 255.0
            if dapi_sample.max() > 1.5:
                dapi_sample = dapi_sample / 255.0

            qpi_sample = torch.clamp(qpi_sample, 0.0, 1.0)
            dapi_sample = torch.clamp(dapi_sample, 0.0, 1.0)

            if qpi_sample.dim() == 4:
                qpi_sample = qpi_sample.unsqueeze(0)
            if dapi_sample.dim() == 4:
                dapi_sample = dapi_sample.unsqueeze(0)

            # Direct prediction (NO noise, NO timesteps)
            dapi_pred = model.predict_dapi(qpi_sample)

            # Visualization preparation
            qpi_np = qpi_sample[0, 0, 0].cpu().numpy()
            dapi_gt_np = dapi_sample[0, 0, 0].cpu().numpy()
            dapi_pred_np = dapi_pred[0, 0, 0].cpu().numpy()

            save_path = os.path.join(output_dir, f"sample_{i+1:02d}.png")
            mse, mae = generate_comparison_figure(qpi_np, dapi_gt_np, dapi_pred_np, i+1, save_path)

            metrics['sample_idx'].append(i + 1)
            metrics['mse'].append(mse)
            metrics['mae'].append(mae)

    # Save Stats
    df = pd.DataFrame(metrics)
    df.to_csv(os.path.join(output_dir, "test_metrics.csv"), index=False)

    print(f"\nResults saved to {output_dir}")
    print(f"Avg MSE: {np.mean(metrics['mse']):.6f}")
    print(f"Avg MAE: {np.mean(metrics['mae']):.6f}")


if __name__ == "__main__":
    test()
