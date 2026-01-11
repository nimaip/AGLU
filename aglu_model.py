"""
Latent-Space U-Net for QPI→DAPI Translation

This module implements a direct deterministic translator that maps QPI images to DAPI
nuclear staining images by operating in the frozen VAE's latent space.

Architecture:
    QPI (B,1,1,128,128) → Frozen VAE → QPI Latent (B,32,1,32,32)
                                            ↓
                                    Latent U-Net (trainable)
                                            ↓
                                 DAPI Latent (B,32,1,32,32)
                                            ↓
                                Frozen VAE → DAPI (B,1,1,128,128)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import config
from vae_model_fixed import encode_vae, decode_vae


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Double convolution
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU()
        )

        self.downsample = nn.Conv3d(
            out_channels, out_channels,
            kernel_size=3,
            stride=(1, 2, 2),  # Only downsample H and W
            padding=1
        )

    def forward(self, x):
        x = self.conv(x)
        skip = x
        downsampled = self.downsample(x)
        return downsampled, skip


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()

        # Squeeze: Global spatial pooling
        self.avg_pool = nn.AdaptiveAvgPool3d(1)

        # Excitation: Channel importance learning
        self.fc = nn.Sequential(
            nn.Conv3d(channels, channels // reduction, kernel_size=1),
            nn.SiLU(),
            nn.Conv3d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        squeeze = self.avg_pool(x)
        excitation = self.fc(squeeze)
        return x * excitation


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, use_attention=True):
        super().__init__()
        self.upsample_conv = nn.Sequential(
            nn.Upsample(scale_factor=(1, 2, 2), mode='nearest'),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        )

        self.skip_fusion = nn.Conv3d(out_channels + skip_channels, out_channels, kernel_size=1)

        self.conv = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU()
        )
        
        self.use_attention = use_attention
        if use_attention:
            self.channel_attention = ChannelAttention(out_channels)

    def forward(self, x, skip):
        x = self.upsample_conv(x)
        x = torch.cat([x, skip], dim=1)
        x = self.skip_fusion(x) 
        x = self.conv(x)

        if self.use_attention:
            x = self.channel_attention(x)  

        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels)
        )

        self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        return F.silu(self.conv(x) + self.shortcut(x))


class SpatialAttention(nn.Module):
    def __init__(self, channels, lightweight=False):
        super().__init__()

        self.lightweight = lightweight

        if lightweight:
            # Lightweight version: Depthwise spatial attention
            self.spatial_attn = nn.Sequential(
                nn.Conv3d(channels, channels, kernel_size=7, padding=3, groups=channels),
                nn.Sigmoid()
            )
            # Learnable scaling factor
            self.gamma = nn.Parameter(torch.tensor(0.1))
        else:
            # Full self-attention 
            self.query = nn.Conv3d(channels, channels // 8, kernel_size=1)
            self.key = nn.Conv3d(channels, channels // 8, kernel_size=1)
            self.value = nn.Conv3d(channels, channels, kernel_size=1)
            self.gamma = nn.Parameter(torch.zeros(1))

            nn.init.normal_(self.query.weight, std=0.01)
            nn.init.normal_(self.key.weight, std=0.01)
            nn.init.normal_(self.value.weight, std=0.01)

    def forward(self, x):
        if self.lightweight:
            # Lightweight: Simple spatial gating
            attn_weights = self.spatial_attn(x)
            return x + self.gamma * (x * attn_weights)
        else:
            # Full self-attention
            B, C, D, H, W = x.shape

            # Compute query, key, value
            q = self.query(x).view(B, -1, D * H * W)  # (B, C//8, DHW)
            k = self.key(x).view(B, -1, D * H * W)     # (B, C//8, DHW)
            v = self.value(x).view(B, C, D * H * W)    # (B, C, DHW)

            # Compute attention map
            attention = F.softmax(torch.bmm(q.transpose(1, 2), k), dim=-1)  # (B, DHW, DHW)

            # Apply attention to values
            out = torch.bmm(v, attention.transpose(1, 2))  # (B, C, DHW)
            out = out.view(B, C, D, H, W)

            # Residual connection with learnable weight
            return x + self.gamma * out


class RefinementModule(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.refine = nn.Sequential(
            nn.Conv3d(channels, channels // 2, kernel_size=3, padding=1),
            nn.GroupNorm(4, channels // 2),
            nn.SiLU(),
            nn.Conv3d(channels // 2, channels, kernel_size=3, padding=1),
            nn.GroupNorm(4, channels)
        )

        # Initialize to near-zero for stable training
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)

        # Learnable residual weight (starts small)
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        residual = self.refine(x)
        return x + self.alpha * residual


class LatentUNet(nn.Module):
    def __init__(self, in_channels=32, out_channels=32, use_attention=True):
        super().__init__()

        # Get channel progression from config
        channels = config.UNET_CHANNELS  # (32, 64, 128, 256)

        # Encoder path (downsampling)
        self.enc1 = DownBlock(in_channels, channels[1])  # 32→64
        self.enc2 = DownBlock(channels[1], channels[2])  # 64→128

        # Bottleneck with full spatial attention
        bottleneck_layers = [
            ResidualBlock(channels[2], channels[3]),  # 128→256
            ResidualBlock(channels[3], channels[3])   # 256→256
        ]

        if use_attention:
            bottleneck_layers.append(SpatialAttention(channels[3], lightweight=False))

        self.bottleneck = nn.Sequential(*bottleneck_layers)

        # Decoder path with channel attention and lightweight spatial attention
        self.dec2 = UpBlock(channels[3], channels[2], channels[2], use_attention=True)  # 256+128→128
        self.dec1 = UpBlock(channels[2], channels[1], channels[1], use_attention=True)  # 128+64→64

        # Add lightweight spatial attention in decoder for nuclear focus
        self.use_attention = use_attention
        if use_attention:
            self.dec2_spatial_attn = SpatialAttention(channels[2], lightweight=True)
            self.dec1_spatial_attn = SpatialAttention(channels[1], lightweight=True)

        # Output projection
        self.out_conv = nn.Conv3d(channels[1], out_channels, kernel_size=1)

        # Refinement module for final polishing
        self.refinement = RefinementModule(out_channels)

    def forward(self, x):
        # Encoder
        x, skip1 = self.enc1(x)     # (B,64,1,16,16), skip1:(B,64,1,32,32)
        x, skip2 = self.enc2(x)     # (B,128,1,8,8), skip2:(B,128,1,16,16)

        # Bottleneck with full spatial attention
        x = self.bottleneck(x)      # (B,256,1,8,8)

        # Decoder with multi-scale attention
        x = self.dec2(x, skip2)     # (B,128,1,16,16) - includes channel attention
        if self.use_attention:
            x = self.dec2_spatial_attn(x)  # Add lightweight spatial attention

        x = self.dec1(x, skip1)     # (B,64,1,32,32) - includes channel attention
        if self.use_attention:
            x = self.dec1_spatial_attn(x)  # Add lightweight spatial attention

        # Output projection
        x = self.out_conv(x)        # (B,32,1,32,32)

        # Refinement for improved quality
        x = self.refinement(x)      # (B,32,1,32,32)

        return x


class QPI2DAPI(nn.Module):
    def __init__(self, frozen_vae, use_attention=True):
        super().__init__()

        # Freeze VAE
        self.vae = frozen_vae
        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False

        # Trainable latent translator
        self.unet = LatentUNet(
            in_channels=config.LATENT_CHANNELS,
            out_channels=config.LATENT_CHANNELS,
            use_attention=use_attention
        )

    def forward(self, qpi):
        # Encode QPI to latent space (frozen)
        with torch.no_grad():
            qpi_latent = encode_vae(self.vae, qpi)

        # Translate to DAPI latent (trainable)
        dapi_latent = self.unet(qpi_latent)

        # CRITICAL: Clamp outputs to prevent gradient explosion
        dapi_latent = torch.clamp(dapi_latent, -20, 20)

        return dapi_latent

    def predict_dapi(self, qpi):
        # Get predicted DAPI latent
        dapi_latent = self.forward(qpi)

        # Decode to image space (frozen)
        with torch.no_grad():
            dapi = decode_vae(self.vae, dapi_latent)

        return dapi


def get_model(frozen_vae):
    model = QPI2DAPI(
        frozen_vae=frozen_vae,
        use_attention=config.UNET_USE_ATTENTION
    )
    return model


