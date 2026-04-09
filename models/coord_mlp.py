"""
models/coord_mlp.py

Coordinate MLP with NeRF-style positional encoding.

Inspired by Neural Radiance Fields (Mildenhall et al., 2020):
  - Map (x, y, z) voxel coordinates through sinusoidal positional encoding
  - Pass through a small pointwise MLP (implemented as 1×1×1 convolutions)
  - Output a per-voxel spatial feature map

Used in UNet3D to inject a continuous coordinate-based prior into the
decoder, making the spatial representation explicitly field-like.
"""

import math
import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────────────────────

def positional_encoding(coords: torch.Tensor, num_freqs: int) -> torch.Tensor:
    """
    NeRF sinusoidal positional encoding.

    For each coordinate channel c and frequency band k:
        [sin(2^k · π · c),  cos(2^k · π · c)]

    Args:
        coords:    (B, 3, D, H, W)  normalized coordinates in [-1, 1]
        num_freqs: number of frequency bands L (NeRF paper uses L=10 for position)

    Returns:
        (B, 3 · 2 · num_freqs, D, H, W)
    """
    freqs = (2.0 ** torch.arange(num_freqs, device=coords.device, dtype=coords.dtype)
             * math.pi)                           # (num_freqs,)

    # coords: (B, 3, D, H, W) → (B, 3, 1, D, H, W)
    x = coords.unsqueeze(2)
    # freqs: (num_freqs,) → (1, 1, num_freqs, 1, 1, 1)
    freqs = freqs.view(1, 1, num_freqs, 1, 1, 1)

    angles = x * freqs                            # (B, 3, num_freqs, D, H, W)
    enc = torch.cat([angles.sin(), angles.cos()], dim=2)  # (B, 3, 2·num_freqs, D, H, W)

    B, C, F, D, H, W = enc.shape
    return enc.view(B, C * F, D, H, W)           # (B, 3·2·num_freqs, D, H, W)


def make_coord_grid(spatial_shape: tuple, device: torch.device) -> torch.Tensor:
    """
    Build a normalized coordinate grid in [-1, 1]^3.

    Args:
        spatial_shape: (D, H, W)

    Returns:
        (1, 3, D, H, W)  — coords along (depth, height, width)
    """
    D, H, W = spatial_shape
    d = torch.linspace(-1.0, 1.0, D, device=device)
    h = torch.linspace(-1.0, 1.0, H, device=device)
    w = torch.linspace(-1.0, 1.0, W, device=device)
    gd, gh, gw = torch.meshgrid(d, h, w, indexing="ij")
    return torch.stack([gd, gh, gw], dim=0).unsqueeze(0)  # (1, 3, D, H, W)


# ──────────────────────────────────────────────────────────────────────────────

class CoordMLP(nn.Module):
    """
    NeRF-inspired coordinate field network.

    Takes a (B, 3, D, H, W) coordinate grid, applies sinusoidal positional
    encoding, then processes it through a small pointwise MLP (1×1×1 convs)
    to produce a per-voxel spatial feature map of shape (B, out_features, D, H, W).

    This gives the U-Net an explicit continuous spatial prior: every voxel
    carries information about its absolute position in the volume, encoded
    at multiple spatial frequencies — analogous to NeRF's scene representation.

    Args:
        num_freqs:    frequency bands in positional encoding (default 6)
        hidden_dim:   MLP hidden layer width (default 64)
        out_features: output feature channels fused into U-Net decoder (default 16)
    """

    def __init__(self, num_freqs: int = 6, hidden_dim: int = 64, out_features: int = 16):
        super().__init__()
        self.num_freqs = num_freqs
        in_ch = 3 * 2 * num_freqs    # 3 coords × (sin + cos) × num_freqs

        self.mlp = nn.Sequential(
            nn.Conv3d(in_ch,     hidden_dim,   kernel_size=1, bias=False),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_dim, hidden_dim,  kernel_size=1, bias=False),
            nn.BatchNorm3d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_dim, out_features, kernel_size=1),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (B, 3, D, H, W)  normalized to [-1, 1]
        Returns:
            (B, out_features, D, H, W)
        """
        pe = positional_encoding(coords, self.num_freqs)
        return self.mlp(pe)
