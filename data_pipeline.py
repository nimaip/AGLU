import os
import glob
import torch
import numpy as np
import tifffile
from monai.data import DataLoader, CacheDataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
    RandSpatialCropSamplesd,
    CenterSpatialCropd,
    EnsureTyped,
    ToTensord,
    Transform,
    RandFlipd,
    RandRotate90d,
    RandGaussianNoised,
    RandShiftIntensityd,
)
import config


class LoadTiffd(Transform):
    """Custom transform to load TIFF files using tifffile."""
    def __init__(self, keys):
        self.keys = keys if isinstance(keys, list) else [keys]

    def __call__(self, data):
        for key in self.keys:
            if key in data:
                img = tifffile.imread(data[key])
                # Ensure 3D: if 2D, add depth dimension
                if img.ndim == 2:
                    img = img[np.newaxis, ...]  # Add depth dimension: (1, H, W)
                # Add channel dimension: (D, H, W) -> (1, D, H, W) for MONAI
                if img.ndim == 3:
                    img = img[np.newaxis, ...]  # (1, D, H, W)
                data[key] = img
        return data


def get_train_transforms():
    """
    Defines the MONAI transformation pipeline for training with data augmentation.

    Handles variable-depth 3D volumes by extracting fixed-size crops.
    RandSpatialCropSamplesd automatically handles volumes with different Z-depths
    by cropping random sub-volumes of the specified ROI_SIZE.

    Augmentation strategy (applied to both QPI and DAPI consistently):
    - Random flips (horizontal/vertical) - 50% probability
    - Random 90° rotations - 50% probability
    - Random intensity shifts - small perturbations for robustness
    - Random Gaussian noise - very small, helps prevent overfitting
    """
    return Compose([
        LoadTiffd(keys=["qpi", "dapi"]),  # Custom TIFF loader with padding (outputs: C, D, H, W)
        # Channel is already first from LoadTiffd, but EnsureChannelFirstd ensures it's correct
        EnsureChannelFirstd(keys=["qpi", "dapi"], channel_dim=0),
        ScaleIntensityRanged(
            keys=["qpi", "dapi"],
            a_min=0, a_max=255,  # Data is 8-bit (0-255)
            b_min=0.0, b_max=1.0,  # FIXED: [0,1] range for MONAI VAE (not [-1,1])
            clip=True,
        ),

        # --- DATA AUGMENTATION (improves generalization on small dataset) ---
        # Random flips along spatial dimensions (not depth)
        RandFlipd(
            keys=["qpi", "dapi"],
            prob=0.5,
            spatial_axis=[1, 2],  # Flip H and W, not D
        ),
        # Random 90-degree rotations in the H-W plane
        RandRotate90d(
            keys=["qpi", "dapi"],
            prob=0.5,
            spatial_axes=(1, 2),  # Rotate in H-W plane
        ),
        # Random intensity shift (±10% of intensity range)
        RandShiftIntensityd(
            keys=["qpi", "dapi"],
            offsets=0.1,
            prob=0.5,
        ),
        # Random Gaussian noise (very small - prevents overfitting)
        RandGaussianNoised(
            keys=["qpi", "dapi"],
            prob=0.3,
            mean=0.0,
            std=0.01,  # Small noise
        ),
        # --------------------------------------------------------------------

        # CRITICAL: Extract fixed-size 3D chunks from variable-depth stacks.
        # If a stack has Z-depth=10 and roi_size Z=8, it picks a random start point.
        # This enables training on volumes with different Z-depths.
        RandSpatialCropSamplesd(
            keys=["qpi", "dapi"],
            roi_size=config.ROI_SIZE,  # (Depth, Height, Width) = (8, 128, 128)
            num_samples=config.SAMPLES_PER_VOLUME,
            random_center=True,
            random_size=False,  # Fixed size crops
        ),
        EnsureTyped(keys=["qpi", "dapi"], dtype=torch.float32),
        ToTensord(keys=["qpi", "dapi"]),
    ])

def get_vae_transforms():
    """Defines the MONAI transformation pipeline for VAE training (DAPI only)."""
    return Compose([
        LoadTiffd(keys=["dapi"]),  # Custom TIFF loader with padding (outputs: C, D, H, W)
        EnsureChannelFirstd(keys=["dapi"], channel_dim=0),
        ScaleIntensityRanged(
            keys=["dapi"],
            a_min=0, a_max=255,  # Data is 8-bit (0-255)
            b_min=0.0, b_max=1.0,  # FIXED: [0,1] range for MONAI VAE (not [-1,1])
            clip=True,
        ),
        RandSpatialCropSamplesd(
            keys=["dapi"],
            roi_size=config.ROI_SIZE,
            num_samples=config.SAMPLES_PER_VOLUME,
            random_center=True,
            random_size=False,
        ),
        EnsureTyped(keys=["dapi"], dtype=torch.float32),
        ToTensord(keys=["dapi"]),
    ])

def get_eval_transforms():
    """
    Defines DETERMINISTIC transformation pipeline for evaluation.
    NO random cropping - uses center crop for consistent ground truth.
    """
    return Compose([
        LoadTiffd(keys=["qpi", "dapi"]),  # Custom TIFF loader with padding (outputs: C, D, H, W)
        EnsureChannelFirstd(keys=["qpi", "dapi"], channel_dim=0),
        ScaleIntensityRanged(
            keys=["qpi", "dapi"],
            a_min=0, a_max=255,  # Data is 8-bit (0-255)
            b_min=0.0, b_max=1.0,  # [0,1] range
            clip=True,
        ),
        # CRITICAL: Use CENTER crop (deterministic) instead of random crop for evaluation
        # This ensures ground truth is not degraded and is consistent across evaluations
        CenterSpatialCropd(
            keys=["qpi", "dapi"],
            roi_size=config.ROI_SIZE,  # (Depth, Height, Width) = (8, 128, 128)
        ),
        EnsureTyped(keys=["qpi", "dapi"], dtype=torch.float32),
        ToTensord(keys=["qpi", "dapi"]),
    ])

def get_dataloader(data_dir, for_vae=False, split='train', val_split=0.2, random_seed=42, for_eval=False):
    """
    Creates a MONAI DataLoader with train/validation split support.
    Assumes data_dir contains two subfolders: 'qpi' and 'dapi'
    with matching filenames (e.g., stack_*.tif or volume_*.tif).

    Supports variable-depth 3D volumes created by build_3d_dataset.py.
    The volumes are automatically cropped to fixed-size patches during training.

    Args:
        data_dir: Directory containing qpi/ and dapi/ subdirectories
        for_vae: If True, only load DAPI data for VAE training
        split: 'train', 'val', or 'all' - which split to return
        val_split: Fraction of data to use for validation (default: 0.2 = 20%)
        random_seed: Seed for reproducible train/val split
        for_eval: If True, use deterministic (center crop) transforms instead of random crops

    Returns:
        DataLoader for the specified split
    """
    # Support both .tif and .nii.gz files
    qpi_files = sorted(glob.glob(os.path.join(data_dir, "qpi", "*.tif")) +
                       glob.glob(os.path.join(data_dir, "qpi", "*.nii.gz")))
    dapi_files = sorted(glob.glob(os.path.join(data_dir, "dapi", "*.tif")) +
                        glob.glob(os.path.join(data_dir, "dapi", "*.nii.gz")))

    if not dapi_files:
        raise ValueError(f"No DAPI files found in {data_dir}/dapi")

    if not for_vae and not qpi_files:
        raise ValueError(f"No QPI files found in {data_dir}/qpi")

    if not for_vae and len(qpi_files) != len(dapi_files):
        raise ValueError("Mismatch between QPI and DAPI file counts. Ensure they are paired.")

    # Create data dictionaries
    if for_vae:
        data_dicts = [{"dapi": dapi_file} for dapi_file in dapi_files]
        transforms = get_vae_transforms()
    else:
        data_dicts = [
            {"qpi": qpi_file, "dapi": dapi_file}
            for qpi_file, dapi_file in zip(qpi_files, dapi_files)
        ]
        # CRITICAL: Use deterministic transforms for evaluation to preserve ground truth quality
        if for_eval:
            transforms = get_eval_transforms()
        else:
            transforms = get_train_transforms()

    # CRITICAL FIX: Implement train/validation split
    if split != 'all' and val_split > 0:
        # Set random seed for reproducibility
        np.random.seed(random_seed)

        # Shuffle indices
        num_samples = len(data_dicts)
        indices = np.arange(num_samples)
        np.random.shuffle(indices)

        # Split into train and validation
        num_val = int(num_samples * val_split)
        num_train = num_samples - num_val

        if split == 'train':
            selected_indices = indices[:num_train]
            print(f"Using {num_train}/{num_samples} samples for training ({(1-val_split)*100:.0f}%)")
        elif split == 'val':
            selected_indices = indices[num_train:]
            print(f"Using {num_val}/{num_samples} samples for validation ({val_split*100:.0f}%)")
        else:
            raise ValueError(f"Invalid split '{split}'. Must be 'train', 'val', or 'all'")

        # Select data for this split
        data_dicts = [data_dicts[i] for i in selected_indices]
    else:
        print(f"Using all {len(data_dicts)} samples (no train/val split)")

    dataset = CacheDataset(
        data=data_dicts,
        transform=transforms,
        cache_rate=1.0,  # 1.0 means cache all data
        num_workers=config.NUM_WORKERS,
    )

    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=(split == 'train'),  # Only shuffle training data
        num_workers=config.NUM_WORKERS,
    )
    return loader