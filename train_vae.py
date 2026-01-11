import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler  # FIXED: Updated to newer unified API
import config
import data_pipeline
import vae_model_fixed as vae_model  # EMERGENCY FIX: Use fully convolutional VAE (no 48GB bug)
from tqdm import tqdm
import os
import sys
# VAE_R: Removed LPIPS (memory leak) - using gradient loss instead
# EMERGENCY: Using vae_model_fixed instead of vae_model to avoid 48GB MONAI bug

# --- EARLY STOPPING CLASS ---
class EarlyStopping:
    def __init__(self, patience=15, min_delta=1e-3, path='vae.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.last_save_loss = float('inf')

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
            if val_loss < self.last_save_loss - self.min_delta:
                self.save_checkpoint(val_loss, model)

    def save_checkpoint(self, val_loss, model):
        print(f'  [Saving Model] Loss improved to {val_loss:.6f}')
        torch.save(model.state_dict(), self.path)
        self.last_save_loss = val_loss

# --- UTILS ---
def sanitize_input(tensor):
    """
    Ensure input is in [0,1] and free of NaNs/Infs.
    FIXED: More robust handling of edge cases.
    """
    # Handle NaN and Inf values
    if torch.isnan(tensor).any():
        tensor = torch.nan_to_num(tensor, nan=0.0)
    if torch.isinf(tensor).any():
        tensor = torch.nan_to_num(tensor, posinf=1.0, neginf=0.0)

    # Normalize if values exceed expected range
    # Note: Data pipeline already normalizes to [0, 1], so this is a safety check
    max_val = tensor.max()
    min_val = tensor.min()

    if max_val > 1.01 or min_val < -0.01:  # Allow small numerical errors
        # Normalize to [0, 1] range
        if max_val > min_val:
            tensor = (tensor - min_val) / (max_val - min_val)
        else:
            tensor = torch.zeros_like(tensor)

    # Final clamp to ensure [0, 1] range
    return torch.clamp(tensor, 0.0, 1.0)

def weights_init(m):
    """Kaiming Init for stability."""
    if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, a=0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

def robust_forward_step(model, x):
    """
    Manual forward pass that CLAMPS latents.
    Handles variable return values from encoder.
    """
    encoded = None
    
    # 1. Attempt to Encode
    if hasattr(model, "encode_forward"):
        encoded = model.encode_forward(x)
    elif hasattr(model, "encode"):
        encoded = model.encode(x)
    else:
        # Fallback: Forward whole model? No, we need to clamp.
        # Try encoder -> linear layers if standard VAE
        raise RuntimeError("Model does not have 'encode' or 'encode_forward' methods.")

    # 2. Robust Unpacking (Handle 2, 3, or more return values)
    if isinstance(encoded, (tuple, list)):
        if len(encoded) >= 2:
            mu = encoded[0]
            logvar = encoded[1]
            # Ignore extra return values (like z or intermediates)
        else:
            raise ValueError(f"Encoder returned {len(encoded)} values, expected at least 2 (mu, logvar).")
    else:
        raise ValueError(f"Encoder returned unexpected type: {type(encoded)}")

    # 3. Clamp Latents (Prevent exploding gradients but allow more flexibility)
    # FIXED: Increased clamp range from [-10, 10] to [-20, 20] for better representation
    mu = torch.clamp(mu, -20, 20)
    logvar = torch.clamp(logvar, -10, 10)  # Keep logvar tighter to prevent exp(logvar) explosion

    # 4. Reparameterize
    z = model.reparameterize(mu, logvar)

    # 5. Decode
    if hasattr(model, "decode_forward"):
        recon = model.decode_forward(z)
    elif hasattr(model, "decode"):
        recon = model.decode(z)
    elif hasattr(model, "decoder"):
        recon = model.decoder(z)
    else:
        raise RuntimeError("Model does not have 'decode', 'decode_forward', or 'decoder'.")

    return recon, mu, logvar

# --- MAIN TRAINING LOOP ---
def train_vae():
    print(f"=" * 60)
    print(f"--- Phase 1: VAE Training (Robust Unpack Mode) ---")
    print(f"=" * 60)
    
    device = config.DEVICE
    data_dir = "3d_data"
    
    if not os.path.exists(data_dir):
        print(f"ERROR: Data directory '{data_dir}' not found!")
        sys.exit(1)

    print(f"Loading Data from: {data_dir}")
    # FIXED: Load both training and validation data
    train_loader = data_pipeline.get_dataloader(data_dir, for_vae=True, split='train')
    val_loader = data_pipeline.get_dataloader(data_dir, for_vae=True, split='val')
    if len(train_loader) == 0: raise ValueError("No training data found.")
    print(f"Validation samples: {len(val_loader.dataset) if hasattr(val_loader, 'dataset') else 'unknown'}")
    
    model = vae_model.get_vae_model().to(device)
    model.apply(weights_init)
    print("Model initialized.")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.VAE_LR,  # FIXED: Use config value instead of hardcoded
        betas=(0.9, 0.999),  # FIXED: Use standard Adam betas (was 0.5, 0.9 for GANs)
        eps=1e-8
    )

    # VAE_R: Use Gradient Loss instead of LPIPS (AlexNet was leaking memory)
    # Gradient loss preserves edges using Sobel filters - lightweight, no deep network
    class GradientLoss2D(nn.Module):
        """2D Gradient loss using Sobel filters to preserve edges."""
        def __init__(self):
            super().__init__()
            # Sobel kernels for edge detection
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)

            self.sobel_x = sobel_x.view(1, 1, 3, 3)
            self.sobel_y = sobel_y.view(1, 1, 3, 3)

        def forward(self, pred, target):
            # Ensure kernels are on same device
            sobel_x = self.sobel_x.to(pred.device)
            sobel_y = self.sobel_y.to(pred.device)

            # Compute gradients
            pred_grad_x = F.conv2d(pred, sobel_x, padding=1)
            pred_grad_y = F.conv2d(pred, sobel_y, padding=1)
            target_grad_x = F.conv2d(target, sobel_x, padding=1)
            target_grad_y = F.conv2d(target, sobel_y, padding=1)

            # L1 loss on gradients
            loss_x = F.l1_loss(pred_grad_x, target_grad_x)
            loss_y = F.l1_loss(pred_grad_y, target_grad_y)

            return (loss_x + loss_y) / 2.0

    gradient_loss_fn = GradientLoss2D()
    # FIXED: Increased patience from 15 to 20 epochs for better convergence
    # VAE training can be slow, especially with KL annealing
    early_stopping = EarlyStopping(patience=20, min_delta=0.001, path=config.VAE_MODEL_PATH)

    # FIXED: Add learning rate scheduler for better convergence
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,          # Reduce LR by half
        patience=10,         # Wait 10 epochs before reducing
        min_lr=1e-6          # Don't go below this
    )
    print("Learning rate scheduler initialized (ReduceLROnPlateau)")

    print(f"Starting training (Max Epochs: {config.VAE_EPOCHS})...")
    
    for epoch in range(config.VAE_EPOCHS):
        model.train()
        epoch_loss = 0
        epoch_kl = 0
        valid_batches = 0
        skipped_batches = 0  # FIXED: Track skipped batches

        # FIXED: Start with small non-zero KL weight to avoid posterior collapse
        # Original: epoch 0 → scale=0.0, which completely disables KL regularization
        # Fixed: Start at 0.01 and linearly increase to 1.0
        if epoch < config.KL_ANNEAL_EPOCHS:
            kl_scale = 0.01 + (0.99 * epoch / config.KL_ANNEAL_EPOCHS)
        else:
            kl_scale = 1.0
        current_kl_weight = config.KL_WEIGHT * kl_scale  # FIXED: Use config value

        # FIXED: Print epoch info including current hyperparameters
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\n[Epoch {epoch+1}/{config.VAE_EPOCHS}] LR={current_lr:.6f}, KL_weight={current_kl_weight:.6f}")

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}")

        for step, batch in progress_bar:
            if isinstance(batch, list):
                dapi_patches = torch.cat([b["dapi"] for b in batch], dim=0).to(device)
            else:
                dapi_patches = batch["dapi"].to(device)

            # FIXED: Track skipped batches for debugging
            if dapi_patches.shape[0] == 0:
                skipped_batches += 1
                continue
            if dapi_patches.max() == 0:  # Skip empty batches
                skipped_batches += 1
                continue
            
            dapi_patches = sanitize_input(dapi_patches)

            # EMERGENCY: Gradient accumulation to simulate larger batches with batch_size=1
            # Only zero gradients at start of accumulation cycle
            if step % config.GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.zero_grad()

            try:
                # VAE_R: DISABLE AUTOCAST - mixed precision causes memory leak with LPIPS
                # Run in full precision (slower but more stable)
                reconstructed, mu, logvar = robust_forward_step(model, dapi_patches)

                l1_loss = F.l1_loss(reconstructed, dapi_patches)

                # VAE_R: Gradient loss - preserves edges without memory leak
                # Remove depth dimension for 2D gradient computation
                # Input shape: (B, C, D, H, W) where D=1 for our data
                # Target shape: (B, C, H, W)
                grad_recon = reconstructed.squeeze(2)
                grad_target = dapi_patches.squeeze(2)

                # Ensure we have 4D tensors (B, C, H, W)
                if grad_recon.ndim == 3:
                    grad_recon = grad_recon.unsqueeze(0)
                    grad_target = grad_target.unsqueeze(0)
                elif grad_recon.ndim != 4:
                    raise ValueError(f"Unexpected gradient input shape: {grad_recon.shape}, expected 4D tensor (B, C, H, W)")

                gradient_loss = gradient_loss_fn(grad_recon, grad_target)
                kl_loss = vae_model.compute_kl_loss(mu, logvar).mean()

                # VAE_R: L1 + Gradient + reduced KL
                total_loss = (config.L1_WEIGHT * l1_loss) + \
                             (config.GRADIENT_WEIGHT * gradient_loss) + \
                             (current_kl_weight * kl_loss)

                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    print(f" [NaN] L1: {l1_loss.item():.4f}, KL: {kl_loss.item():.4f}")
                    if step % config.GRADIENT_ACCUMULATION_STEPS == 0:
                        optimizer.zero_grad()
                    continue

                # EMERGENCY: Scale loss for gradient accumulation
                scaled_loss = total_loss / config.GRADIENT_ACCUMULATION_STEPS

                # VAE_R: NO SCALER - causes memory leak, use regular backward
                scaled_loss.backward()

                # Only update weights after accumulating gradients
                if (step + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
                    # FIXED: Increased gradient clipping from 0.5 to 1.0 for better learning
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                # VAE_R: AGGRESSIVE MEMORY CLEANUP
                epoch_loss += total_loss.item()
                epoch_kl += kl_loss.item()
                valid_batches += 1

                # Explicitly delete tensors to free memory
                del reconstructed, mu, logvar, l1_loss, gradient_loss, kl_loss, total_loss
                del grad_recon, grad_target, dapi_patches

                # Clear cache every batch to prevent accumulation
                if step % 5 == 0:
                    torch.cuda.empty_cache()

                progress_bar.set_postfix({
                    "Loss": f"{epoch_loss/valid_batches:.4f}",
                    "KL": f"{epoch_kl/valid_batches:.1f}"
                })

            except Exception as e:
                # FIXED: Track failed batches
                skipped_batches += 1
                # VAE_R: Aggressive cleanup on ANY error
                optimizer.zero_grad()
                torch.cuda.empty_cache()
                import gc
                gc.collect()
                # Print once per epoch if really noisy, otherwise prints every fail
                if step % 10 == 0:
                    print(f"\n[Batch Error] {e}")
                continue

        if valid_batches > 0:
            avg_loss = epoch_loss / valid_batches
            avg_kl = epoch_kl / valid_batches
            # FIXED: Include batch statistics in summary
            total_batches = len(train_loader)
            print(f"Epoch {epoch+1}: Train Loss={avg_loss:.4f} | Avg KL={avg_kl:.1f} | "
                  f"Valid Batches={valid_batches}/{total_batches} | Skipped={skipped_batches}")

            # FIXED: Add validation at end of each epoch
            model.eval()
            val_loss = 0
            val_batches = 0
            with torch.no_grad():
                for batch in val_loader:
                    if isinstance(batch, list):
                        dapi_patches = torch.cat([b["dapi"] for b in batch], dim=0).to(device)
                    else:
                        dapi_patches = batch["dapi"].to(device)

                    if dapi_patches.shape[0] == 0 or dapi_patches.max() == 0:
                        continue

                    dapi_patches = sanitize_input(dapi_patches)

                    try:
                        # VAE_R: No autocast for consistency with training
                        reconstructed, mu, logvar = robust_forward_step(model, dapi_patches)
                        l1_loss = F.l1_loss(reconstructed, dapi_patches)

                        # VAE_R: Gradient loss (same as training)
                        grad_recon = reconstructed.squeeze(2)
                        grad_target = dapi_patches.squeeze(2)
                        if grad_recon.ndim == 3:
                            grad_recon = grad_recon.unsqueeze(0)
                            grad_target = grad_target.unsqueeze(0)
                        gradient_loss = gradient_loss_fn(grad_recon, grad_target)

                        kl_loss = vae_model.compute_kl_loss(mu, logvar).mean()
                        total_loss = (config.L1_WEIGHT * l1_loss) + \
                                     (config.GRADIENT_WEIGHT * gradient_loss) + \
                                     (current_kl_weight * kl_loss)

                        if not (torch.isnan(total_loss) or torch.isinf(total_loss)):
                            val_loss += total_loss.item()
                            val_batches += 1

                        # Cleanup
                        del reconstructed, mu, logvar, l1_loss, gradient_loss, kl_loss, total_loss
                        del grad_recon, grad_target, dapi_patches
                    except Exception:
                        continue

            if val_batches > 0:
                avg_val_loss = val_loss / val_batches
                print(f"         Val Loss={avg_val_loss:.4f}")
                # Use validation loss for early stopping
                early_stopping(avg_val_loss, model)
                # FIXED: Step scheduler based on validation loss
                scheduler.step(avg_val_loss)
            else:
                # Fallback to training loss if validation fails
                early_stopping(avg_loss, model)
                scheduler.step(avg_loss)

            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break
        else:
            print(f"Epoch {epoch+1}: All batches failed.")

if __name__ == "__main__":
    try:
        train_vae()
    except Exception as e:
        print(f"Fatal Error: {e}")
        import traceback
        traceback.print_exc()