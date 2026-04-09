import torch
import torch.nn as nn

from .coord_mlp import CoordMLP, make_coord_grid


# ──────────────────────────────────────────────────────────────────────────────
# Building blocks
# ──────────────────────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Two consecutive Conv3d → BN → ReLU layers."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """MaxPool3d then ConvBlock."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool3d(2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    """Transposed conv upsample then ConvBlock (with skip connection cat)."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        return self.conv(torch.cat([x, skip], dim=1))


# ──────────────────────────────────────────────────────────────────────────────
# 3D U-Net  (optionally with NeRF-inspired CoordMLP)
# ──────────────────────────────────────────────────────────────────────────────

class UNet3D(nn.Module):
    """
    3D U-Net with optional NeRF-inspired Coordinate MLP fusion.

    Base architecture (base_channels=32):
        Encoder:     1 → 32 → 64 → 128 → 256
        Bottleneck:  256 → 512
        Decoder:     512+256 → 256 → ... → 32
        Head:        32 (+coord_features) → num_classes

    With use_coord_mlp=True:
        A CoordMLP encodes (x,y,z) voxel coordinates via NeRF positional
        encoding and outputs a spatial feature map that is concatenated
        with the final decoder features before the segmentation head.
        This injects a continuous coordinate-based prior — analogous to
        NeRF's implicit scene representation — without replacing the
        U-Net's strong spatial context.

    Parameters: ~19 M (base_channels=32, no CoordMLP)
                ~19.2 M (with CoordMLP, coord_features=16)

    Recommended VRAM: >= 8 GB for patch (64, 128, 128), batch_size=2
    """

    def __init__(
        self,
        in_channels:    int  = 1,
        num_classes:    int  = 3,
        base_channels:  int  = 32,
        use_coord_mlp:  bool = False,
        coord_features: int  = 16,
        coord_freq_bands: int = 6,
    ):
        super().__init__()
        c = base_channels
        self.use_coord_mlp = use_coord_mlp

        # Encoder
        self.enc1 = ConvBlock(in_channels, c)
        self.enc2 = Down(c,     c * 2)
        self.enc3 = Down(c * 2, c * 4)
        self.enc4 = Down(c * 4, c * 8)

        # Bottleneck
        self.bottleneck = Down(c * 8, c * 16)

        # Decoder
        self.dec4 = Up(c * 16, c * 8, c * 8)
        self.dec3 = Up(c * 8,  c * 4, c * 4)
        self.dec2 = Up(c * 4,  c * 2, c * 2)
        self.dec1 = Up(c * 2,  c,     c)

        # Optional CoordMLP — fused into the final decoder features
        if use_coord_mlp:
            self.coord_mlp = CoordMLP(
                num_freqs=coord_freq_bands,
                hidden_dim=64,
                out_features=coord_features,
            )
            head_in = c + coord_features
        else:
            head_in = c

        # Segmentation head
        self.head = nn.Conv3d(head_in, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, D, H, W)  CT patch, float32, values in [0, 1]
        Returns:
            logits: (B, num_classes, D, H, W)
        """
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # Bottleneck
        b = self.bottleneck(e4)

        # Decoder
        d4 = self.dec4(b,  e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        # Fuse CoordMLP spatial features (NeRF-inspired continuous field)
        if self.use_coord_mlp:
            coords = make_coord_grid(d1.shape[2:], device=x.device)
            coords = coords.expand(x.shape[0], -1, -1, -1, -1)  # (B, 3, D, H, W)
            d1 = torch.cat([d1, self.coord_mlp(coords)], dim=1)

        return self.head(d1)
