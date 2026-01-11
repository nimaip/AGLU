"""
build_3d_dataset.py

Preprocessing script to convert 2D PNG files (with left/right QPI/DAPI pairs)
into 3D volumes for AGLU training.

Each PNG file contains:
- Left half: QPI image
- Right half: DAPI image

The script groups slices by (Date, Sample, CenterX, CenterY, Augmentation) 
and stacks them by Z-index to create 3D volumes.
"""

import os
import re
import glob
import numpy as np
from PIL import Image
import tifffile
from collections import defaultdict
from tqdm import tqdm
import argparse


def parse_filename(filename):
    """
    Parse filename to extract metadata for grouping.
    
    Example filename:
    20250206_60x_55deg_miceBrian_DAPI_z3_layerFluo7_layerPhase6_CenterX831Y854.mat_org.png
    
    Returns:
        stack_id: Unique identifier for the 3D stack (excluding Z-index)
        z_index: Z-index for sorting
        full_path: Full path to the file
    """
    # Extract base filename without path
    basename = os.path.basename(filename)
    
    # Pattern to match the filename structure
    # Example: 20250206_60x_55deg_miceBrian_DAPI_z3_layerFluo7_layerPhase6_CenterX831Y854.mat_org.png
    # Captures: date, sample info, z-index, center coordinates, augmentation
    pattern = re.compile(
        r"(?P<date>\d{8})_"           # Date: 20250206
        r"(?P<mag>\d+x)_"             # Magnification: 60x
        r"(?P<angle>\d+deg)_"        # Angle: 55deg
        r"(?P<sample>[^_]+)_"         # Sample: miceBrian (non-underscore characters)
        r"DAPI_"                      # Literal "DAPI_"
        r"z(?P<z>\d+)_"               # Z-index: z3 -> 3
        r".*?"                        # Skip layer info (layerFluo7_layerPhase6)
        r"_CenterX(?P<x>\d+)Y(?P<y>\d+)"  # Center coordinates: CenterX831Y854
        r"\.mat_"                     # Literal ".mat_"
        r"(?P<aug>.*?)"               # Augmentation: org, datarot90, _dataflip1, etc.
        r"\.png$"                     # End with .png
    )
    
    match = pattern.search(basename)
    if match:
        meta = match.groupdict()
        
        # Create unique stack ID (everything except Z-index)
        # Include date, sample, center coordinates, and augmentation
        stack_id = f"{meta['date']}_{meta['sample']}_X{meta['x']}Y{meta['y']}_{meta['aug']}"
        
        z_index = int(meta['z'])
        
        return stack_id, z_index, filename
    else:
        # Try a more flexible pattern if the first one fails
        # This handles variations in the filename structure
        flexible_pattern = re.compile(
            r"(?P<date>\d{8})_"
            r".*?"
            r"_z(?P<z>\d+)_"
            r".*?"
            r"_CenterX(?P<x>\d+)Y(?P<y>\d+)"
            r"\.mat_(?P<aug>.*?)\.png$"
        )
        match = flexible_pattern.search(basename)
        if match:
            meta = match.groupdict()
            # Extract sample name from the middle part (between date and DAPI)
            parts = basename.split('_')
            sample_part = '_'.join(parts[1:-1])  # Everything between date and last part
            # Try to find a meaningful sample identifier
            sample = parts[1] if len(parts) > 1 else "unknown"
            
            stack_id = f"{meta['date']}_{sample}_X{meta['x']}Y{meta['y']}_{meta['aug']}"
            z_index = int(meta['z'])
            return stack_id, z_index, filename
    
    return None, None, filename


def split_qpi_dapi(image_path):
    """
    Split a 2D PNG image into QPI (left) and DAPI (right) halves.
    
    Args:
        image_path: Path to the PNG file
        
    Returns:
        qpi: Left half of the image (numpy array)
        dapi: Right half of the image (numpy array)
    """
    img = Image.open(image_path)
    img_array = np.array(img, dtype=np.float32)
    
    # Image is expected to be (Height, Width) with Width = 2 * Height
    height, width = img_array.shape
    
    if width != 2 * height:
        raise ValueError(f"Expected image width to be 2x height, got {width}x{height}")
    
    # Split: left = QPI, right = DAPI
    mid = width // 2
    qpi = img_array[:, :mid]  # Left half
    dapi = img_array[:, mid:]  # Right half
    
    return qpi, dapi


def validate_z_sequence(z_indices):
    """
    Validate Z-index sequence for gaps.
    
    Args:
        z_indices: List of Z-indices
        
    Returns:
        is_valid: True if sequence is valid
        warnings: List of warning messages
    """
    warnings = []
    sorted_z = sorted(z_indices)
    
    # Check for minimum depth
    if len(sorted_z) < 4:
        warnings.append(f"Stack has only {len(sorted_z)} slices (minimum recommended: 4)")
    
    # Check for gaps
    gaps = []
    for i in range(len(sorted_z) - 1):
        if sorted_z[i+1] - sorted_z[i] > 1:
            gaps.append((sorted_z[i], sorted_z[i+1]))
    
    if gaps:
        warnings.append(f"Gaps detected in Z-sequence: {gaps}")
    
    return len(warnings) == 0, warnings


def build_3d_volumes(data_dir, output_dir, min_depth=4, validate=True):
    """
    Main function to build 3D volumes from 2D PNG files.
    
    Args:
        data_dir: Directory containing PNG files (can be recursive)
        output_dir: Output directory for 3D volumes
        min_depth: Minimum number of slices required for a valid stack
        validate: Whether to validate Z-sequences and report warnings
    """
    print(f"Scanning for PNG files in: {data_dir}")
    
    # Recursively find all PNG files
    png_files = glob.glob(os.path.join(data_dir, "**", "*.png"), recursive=True)
    print(f"Found {len(png_files)} PNG files")
    
    if len(png_files) == 0:
        raise ValueError(f"No PNG files found in {data_dir}")
    
    # Group files by stack_id
    stacks = defaultdict(list)
    failed_files = []
    
    print("Parsing filenames and grouping by stack...")
    for png_file in tqdm(png_files, desc="Parsing files"):
        stack_id, z_index, full_path = parse_filename(png_file)
        
        if stack_id is None or z_index is None:
            failed_files.append(png_file)
            continue
        
        stacks[stack_id].append((z_index, full_path))
    
    if failed_files:
        print(f"\nWarning: Failed to parse {len(failed_files)} files:")
        for f in failed_files[:10]:  # Show first 10
            print(f"  {os.path.basename(f)}")
        if len(failed_files) > 10:
            print(f"  ... and {len(failed_files) - 10} more")
    
    print(f"\nFound {len(stacks)} unique stacks")
    
    # Create output directories
    qpi_dir = os.path.join(output_dir, "qpi")
    dapi_dir = os.path.join(output_dir, "dapi")
    os.makedirs(qpi_dir, exist_ok=True)
    os.makedirs(dapi_dir, exist_ok=True)
    
    # Process each stack
    valid_stacks = 0
    rejected_stacks = 0
    
    print("\nBuilding 3D volumes...")
    for stack_id, files in tqdm(stacks.items(), desc="Processing stacks"):
        # Sort by Z-index
        files.sort(key=lambda x: x[0])
        z_indices = [z for z, _ in files]
        
        # Validate Z-sequence
        if validate:
            is_valid, warnings = validate_z_sequence(z_indices)
            if warnings and len(z_indices) < min_depth:
                print(f"\nWarning for stack {stack_id}:")
                for w in warnings:
                    print(f"  {w}")
        
        # Skip stacks with insufficient depth
        if len(z_indices) < min_depth:
            rejected_stacks += 1
            continue
        
        # Load and split all images in the stack
        qpi_slices = []
        dapi_slices = []
        
        try:
            for z_idx, file_path in files:
                qpi_slice, dapi_slice = split_qpi_dapi(file_path)
                qpi_slices.append(qpi_slice)
                dapi_slices.append(dapi_slice)
            
            # Stack slices into 3D volumes
            # Shape: (Depth, Height, Width)
            qpi_volume = np.stack(qpi_slices, axis=0)
            dapi_volume = np.stack(dapi_slices, axis=0)
            
            # Sanitize stack_id for filename (remove invalid characters)
            safe_stack_id = re.sub(r'[<>:"/\\|?*]', '_', stack_id)
            
            # Save as TIFF files
            qpi_path = os.path.join(qpi_dir, f"stack_{safe_stack_id}.tif")
            dapi_path = os.path.join(dapi_dir, f"stack_{safe_stack_id}.tif")
            
            tifffile.imwrite(qpi_path, qpi_volume.astype(np.float32))
            tifffile.imwrite(dapi_path, dapi_volume.astype(np.float32))
            
            valid_stacks += 1
            
        except Exception as e:
            print(f"\nError processing stack {stack_id}: {e}")
            rejected_stacks += 1
            continue
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"  Valid stacks created: {valid_stacks}")
    print(f"  Rejected stacks: {rejected_stacks}")
    print(f"  Output directory: {output_dir}")
    print(f"  QPI volumes: {qpi_dir}")
    print(f"  DAPI volumes: {dapi_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert 2D PNG files (QPI/DAPI pairs) into 3D volumes"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Directory containing PNG files (default: ./data)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./3d_data",
        help="Output directory for 3D volumes (default: ./3d_data)"
    )
    parser.add_argument(
        "--min_depth",
        type=int,
        default=4,
        help="Minimum number of slices required for a valid stack (default: 4)"
    )
    parser.add_argument(
        "--no_validate",
        action="store_true",
        help="Disable Z-sequence validation warnings"
    )
    
    args = parser.parse_args()
    
    build_3d_volumes(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        min_depth=args.min_depth,
        validate=not args.no_validate
    )

