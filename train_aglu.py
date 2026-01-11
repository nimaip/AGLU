"""
Training Script for QPI→DAPI Translation Model

This script trains a direct U-Net translator that operates in the frozen VAE's latent space.
No diffusion, no timesteps, no noise - just deterministic image-to-image translation.

Training Strategy:
1. Stage 1 (Warmup, epochs 1-20): Train with latent loss only
2. Stage 2 (Fine-tuning, epochs 21+): Add pixel and gradient losses

Key Features:
- Two-stage training for stable convergence
- Weighted loss emphasizing nuclear regions
- Multi-component loss: latent + pixel + gradient
- Gradient clipping for stability
- Checkpointing with CSV logging and plots
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import config
from aglu_model import get_model
from vae_model_fixed import FullyConvVAE, encode_vae, decode_vae
from data_pipeline import get_dataloader


# --- LOSS FUNCTIONS ---

class WeightedL1Loss(nn.Module):
    """
    Weighted L1 loss that emphasizes nuclear regions.

    Assigns higher weight to bright pixels (nuclei) to prevent the model
    from collapsing to predicting uniform background.

    Args:
        background_weight: Weight for background pixels (default: 1.0)
        nucleus_weight: Weight for bright nuclear pixels (default: 10.0)
    """
    def __init__(self, background_weight=1.0, nucleus_weight=10.0):
        super().__init__()
        self.bg_weight = background_weight
        self.nuc_weight = nucleus_weight

    def forward(self, pred, target):
        """
        Args:
            pred: Predicted DAPI image (B, 1, 1, H, W)
            target: Ground truth DAPI image (B, 1, 1, H, W)

        Returns:
            weighted_loss: Scalar loss value
        """
        # Compute L1 error
        l1_error = torch.abs(pred - target)

        # Create importance map based on target intensity
        # Brighter pixels (nuclei) get higher weight
        importance = torch.abs(target)
        importance = torch.clamp(importance, 0, 10) / 10.0  # Normalize to [0,1]

        # Linear interpolation between background and nucleus weights
        weights = self.bg_weight + (self.nuc_weight - self.bg_weight) * importance

        # Apply weights
        weighted_loss = l1_error * weights
        return weighted_loss.mean()


class GradientLoss(nn.Module):
    """
    Gradient loss to enforce sharp nuclear boundaries.

    Uses Sobel filters to compute image gradients and penalizes
    blurry predictions.
    """
    def __init__(self):
        super().__init__()

        # Sobel kernels for edge detection
        kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        self.register_buffer('kernel_x', kernel_x)
        self.register_buffer('kernel_y', kernel_y)

    def forward(self, pred, target):
        """
        Args:
            pred: Predicted DAPI image (B, C, D, H, W)
            target: Ground truth DAPI image (B, C, D, H, W)

        Returns:
            gradient_loss: Scalar loss value
        """
        b, c, d, h, w = pred.shape

        # Reshape to 2D for conv2d
        pred_2d = pred.view(b * c * d, 1, h, w)
        target_2d = target.view(b * c * d, 1, h, w)

        # Compute gradients
        pred_grad_x = F.conv2d(pred_2d, self.kernel_x, padding=1)
        pred_grad_y = F.conv2d(pred_2d, self.kernel_y, padding=1)
        target_grad_x = F.conv2d(target_2d, self.kernel_x, padding=1)
        target_grad_y = F.conv2d(target_2d, self.kernel_y, padding=1)

        # L1 loss on gradient magnitudes
        loss = F.l1_loss(
            torch.abs(pred_grad_x) + torch.abs(pred_grad_y),
            torch.abs(target_grad_x) + torch.abs(target_grad_y)
        )

        return loss


# --- MAIN TRAINING LOOP ---

def train():
    """
    Main training function.
    """
    output_dir = "aglu_outputs"
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    print(f"Device: {config.DEVICE}")
    print(f"Training Configuration:")
    print(f"  Total Epochs: {config.EPOCHS}")
    print(f"  Warmup Epochs: {config.WARMUP_EPOCHS}")
    print(f"  Stage 1 LR: {config.LEARNING_RATE}")
    print(f"  Stage 2 LR: {config.LEARNING_RATE_FINETUNE}")
    print(f"  Nucleus Emphasis: {config.NUCLEUS_EMPHASIS}×")

    # 1. Load frozen VAE
    print("\n[1/4] Loading frozen VAE...")
    vae = FullyConvVAE().to(config.DEVICE)
    vae.load_state_dict(torch.load(config.VAE_MODEL_PATH, map_location=config.DEVICE))
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False
    print(f"✓ VAE loaded and frozen")

    # 2. Load data
    print("\n[2/4] Loading data...")
    train_loader = get_dataloader("3d_data", split='train', val_split=0.2)
    val_loader = get_dataloader("3d_data", split='val', val_split=0.2)
    print(f"✓ Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # 3. Initialize model
    print("\n[3/4] Initializing model...")
    model = get_model(frozen_vae=vae).to(config.DEVICE)
    print(f"✓ Model initialized")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {total_params:,}")

    # 4. Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=1e-4
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.EPOCHS,
        eta_min=1e-6
    )

    scaler = GradScaler()

    # 5. Loss functions
    latent_loss_fn = nn.L1Loss()
    pixel_loss_fn = WeightedL1Loss(
        background_weight=1.0,
        nucleus_weight=config.NUCLEUS_EMPHASIS
    ).to(config.DEVICE)
    gradient_loss_fn = GradientLoss().to(config.DEVICE)

    # 6. Training metrics
    train_metrics = {'total': [], 'latent': [], 'pixel': [], 'gradient': []}
    val_metrics = {'loss': []}
    best_val_loss = float('inf')

    print(f"\n[4/4] Starting training...")
    print("="*70)

    for epoch in range(config.EPOCHS):
        model.train()

        # Determine training stage
        is_warmup = epoch < config.WARMUP_EPOCHS
        stage = "WARMUP" if is_warmup else "FINETUNE"

        # Adjust learning rate for stage 2
        if epoch == config.WARMUP_EPOCHS:
            print(f"\n{'='*70}")
            print(f"Switching to Stage 2: Fine-tuning")
            print(f"  Adding pixel and gradient losses")
            print(f"  Reducing LR: {config.LEARNING_RATE} → {config.LEARNING_RATE_FINETUNE}")
            print(f"{'='*70}\n")
            for param_group in optimizer.param_groups:
                param_group['lr'] = config.LEARNING_RATE_FINETUNE

        epoch_stats = {'total': 0.0, 'latent': 0.0, 'pixel': 0.0, 'gradient': 0.0}
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [{stage}]")

        for batch_data in pbar:
            qpi = batch_data['qpi'].to(config.DEVICE)
            dapi = batch_data['dapi'].to(config.DEVICE)

            # Normalize if needed
            if qpi.max() > 1.5:
                qpi = qpi / 255.0
            if dapi.max() > 1.5:
                dapi = dapi / 255.0

            with autocast():
                # Get ground truth DAPI latent
                with torch.no_grad():
                    dapi_latent_gt = encode_vae(vae, dapi)
                    dapi_latent_gt = torch.clamp(dapi_latent_gt, -20, 20)

                # Forward pass: QPI → DAPI latent
                dapi_latent_pred = model(qpi)

                # Safety check: Detect NaN
                if torch.isnan(dapi_latent_pred).any():
                    print(f"\n!!! NaN detected in predictions at batch {num_batches}")
                    print(f"QPI stats: min={qpi.min():.4f}, max={qpi.max():.4f}")
                    print(f"Skipping batch...")
                    continue

                # Stage 1: Latent loss only
                if is_warmup:
                    latent_loss = latent_loss_fn(dapi_latent_pred, dapi_latent_gt)
                    loss = latent_loss
                    pixel_loss = torch.tensor(0.0)
                    grad_loss = torch.tensor(0.0)

                # Stage 2: Combined losses
                else:
                    latent_loss = latent_loss_fn(dapi_latent_pred, dapi_latent_gt)

                    # Decode predictions for pixel-space losses
                    dapi_pred = decode_vae(vae, dapi_latent_pred)
                    pixel_loss = pixel_loss_fn(dapi_pred, dapi)
                    grad_loss = gradient_loss_fn(dapi_pred, dapi)

                    # Combined loss
                    loss = (
                        config.LATENT_LOSS_WEIGHT * latent_loss +
                        config.PIXEL_LOSS_WEIGHT * pixel_loss +
                        config.GRADIENT_LOSS_WEIGHT * grad_loss
                    )

            # Backward pass
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            # Gradient clipping with monitoring
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Detect gradient explosion
            if grad_norm > 10.0:
                print(f"\n!!! Large gradient norm: {grad_norm:.2f}")

            scaler.step(optimizer)
            scaler.update()

            # Track metrics
            epoch_stats['total'] += loss.item()
            epoch_stats['latent'] += latent_loss.item()
            epoch_stats['pixel'] += pixel_loss.item() if not is_warmup else 0.0
            epoch_stats['gradient'] += grad_loss.item() if not is_warmup else 0.0
            num_batches += 1

            # Update progress bar
            if is_warmup:
                pbar.set_postfix({'L_latent': f"{latent_loss.item():.4f}"})
            else:
                pbar.set_postfix({
                    'L_lat': f"{latent_loss.item():.3f}",
                    'L_pix': f"{pixel_loss.item():.3f}",
                    'L_grad': f"{grad_loss.item():.3f}"
                })

        # Compute average training metrics
        avg_train_loss = epoch_stats['total'] / num_batches
        avg_train_latent = epoch_stats['latent'] / num_batches
        avg_train_pixel = epoch_stats['pixel'] / num_batches if not is_warmup else 0.0
        avg_train_gradient = epoch_stats['gradient'] / num_batches if not is_warmup else 0.0

        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch_data in val_loader:
                qpi = batch_data['qpi'].to(config.DEVICE)
                dapi = batch_data['dapi'].to(config.DEVICE)

                if qpi.max() > 1.5:
                    qpi /= 255.0
                if dapi.max() > 1.5:
                    dapi /= 255.0

                # Ground truth latent
                dapi_latent_gt = encode_vae(vae, dapi)
                dapi_latent_gt = torch.clamp(dapi_latent_gt, -20, 20)

                # Prediction
                dapi_latent_pred = model(qpi)

                # Skip if NaN detected
                if torch.isnan(dapi_latent_pred).any():
                    print(f"!!! NaN in validation, skipping batch")
                    continue

                # Validation uses latent loss only for consistency
                loss = latent_loss_fn(dapi_latent_pred, dapi_latent_gt)

                # Skip if loss is NaN or extremely large
                if torch.isnan(loss) or loss.item() > 100:
                    print(f"!!! Abnormal val loss: {loss.item():.2f}, skipping")
                    continue

                val_loss += loss.item()
                val_batches += 1

        # Safety: Handle case where all batches were skipped
        if val_batches == 0:
            print("!!! All validation batches skipped due to NaN")
            avg_val_loss = float('inf')
        else:
            avg_val_loss = val_loss / val_batches

        # Update scheduler
        scheduler.step()

        # Save metrics
        train_metrics['total'].append(avg_train_loss)
        train_metrics['latent'].append(avg_train_latent)
        train_metrics['pixel'].append(avg_train_pixel)
        train_metrics['gradient'].append(avg_train_gradient)
        val_metrics['loss'].append(avg_val_loss)

        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{config.EPOCHS} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f} (Latent: {avg_train_latent:.4f}, Pixel: {avg_train_pixel:.4f}, Grad: {avg_train_gradient:.4f})")
        print(f"  Val Loss:   {avg_val_loss:.4f}")
        print(f"  Stage:      {stage}")

        # Checkpointing
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, "model_latest.pth"))

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "model_best.pth"))
            print(f"  ✓ Best model saved (val_loss: {best_val_loss:.4f})")

        # Save CSV
        epochs_list = list(range(1, len(train_metrics['total']) + 1))
        df_losses = pd.DataFrame({
            'epoch': epochs_list,
            'train_loss': train_metrics['total'],
            'val_loss': val_metrics['loss'],
            'train_latent': train_metrics['latent'],
            'train_pixel': train_metrics['pixel'],
            'train_gradient': train_metrics['gradient']
        })
        csv_path = os.path.join(output_dir, "training_losses.csv")
        df_losses.to_csv(csv_path, index=False)

        # Plot losses
        plt.figure(figsize=(12, 5))

        # Plot 1: Total loss
        plt.subplot(1, 2, 1)
        plt.plot(train_metrics['total'], label='Train Total', linewidth=2)
        plt.plot(val_metrics['loss'], label='Val Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 2: Component losses
        plt.subplot(1, 2, 2)
        plt.plot(train_metrics['latent'], label='Latent', linewidth=2)
        if not all(x == 0 for x in train_metrics['pixel']):
            plt.plot(train_metrics['pixel'], label='Pixel', linewidth=2)
            plt.plot(train_metrics['gradient'], label='Gradient', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Components')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "training_progress.png"), dpi=150)
        plt.close()

        print("="*70)

    print("\n" + "="*70)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}/")
    print(f"Metrics saved to: {csv_path}")
    print("="*70)


if __name__ == "__main__":
    config.validate_config()
    train()
