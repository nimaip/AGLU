# AGLU

Virtual staining via attention-guided latent-space translation from label-free QPI to fluorescence DAPI.

## Overview

This project implements a novel approach to **virtual staining**: digitally generating fluorescence nuclear labels (DAPI) directly from label-free Quantitative Phase Imaging (QPI) inputs. The system translates cell morphology (shape/density from QPI) into molecular localization (DNA density from DAPI) without the need for physical chemical staining.

## Why Virtual Staining?

Physical chemical staining has significant limitations:
- **Toxic** to live cells
- **Time-consuming** to prepare
- **Destructive** (endpoints the experiment)

This AI-powered approach enables long-term, non-invasive monitoring of cell nuclei in live cultures without any chemical intervention.

## Technical Approach

The AGLU (Attention-Gated Latent U-Net) system uses a three-stage architecture:

### 1. Compression - The "Language" of the Cell
A Variational Autoencoder (VAE) compresses microscope images into a latent space, abstracting away pixel-level noise and representing images as high-level semantic features (shapes, textures, densities).

### 2. Translation - The Core Logic
A specialized U-Net architecture with attention mechanisms maps the "Latent QPI" representation directly to the "Latent DAPI" representation:
- Direct, deterministic mapping (Input A → Output B)
- Attention mechanisms specifically hunt for sparse signals (nuclei) in the phase data
- Ensures faint cells aren't ignored

### 3. Reconstruction - The Output
Predicted DAPI latents are decoded by the VAE to generate the final high-resolution fluorescence image.

## Key Innovations

### Latent Operation vs. Pixel Operation
Training on compressed features rather than raw pixels enables faster learning and reduces sensitivity to background sensor noise. The model learns the "concept" of a nucleus rather than memorizing bright pixels.

### Deterministic vs. Probabilistic
Unlike diffusion models, there is no random noise injection or iterative time-steps. The model yields consistent, repeatable results in a single forward pass, achieving millisecond-fast inference.

### Recall-Focused Training
A weighted importance strategy penalizes the model heavily (10x-50x) for missing nuclei, forcing aggressive detection of even faint signals. This prevents the common failure mode where models optimize for average pixel error and miss sparse features.

## Advantages Over Existing Solutions

| Approach | Issue | Failure Mode |
|----------|-------|--------------|
| **Pixel-Space GANs** (e.g., Pix2Pix) | Hallucination and stability | "Delete" faint nuclei because adversarial loss prioritizes realism over anatomical accuracy |
| **Latent Diffusion Models** (LDMs) | Computationally heavy and slow | Requires 50-100 iterative denoising steps, impractical for real-time microscope analysis |
| **AGLU (This Project)** | - | Single-pass deterministic inference with high recall for sparse features |

## Applications

- Long-term live cell monitoring
- High-throughput microscopy
- Real-time microscope analysis
- Non-invasive cell culture studies
