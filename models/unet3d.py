import torch
import torch.nn as nn


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
# 3D U-Net
# ──────────────────────────────────────────────────────────────────────────────

class UNet3D(nn.Module):
    """
    Standard 3D U-Net with 4 encoder levels + bottleneck.

    Architecture (base_channels=32):
        Encoder:     1 → 32 → 64 → 128 → 256
        Bottleneck:  256 → 512
        Decoder:     512+256 → 256 → 128+128 → 128 → 64+64 → 64 → 32+32 → 32
        Head:        32 → num_classes (1×1×1 conv)

    Parameters: ~19 M  (base_channels=32)
    Recommended VRAM: ≥ 8 GB for patch (64, 128, 128), batch_size=2
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 3, base_channels: int = 32):
        super().__init__()
        c = base_channels

        # Encoder
        self.enc1 = ConvBlock(in_channels, c)
        self.enc2 = Down(c,     c * 2)
        self.enc3 = Down(c * 2, c * 4)
        self.enc4 = Down(c * 4, c * 8)

        # Bottleneck
        self.bottleneck = Down(c * 8, c * 16)

        # Decoder
        self.dec4 = Up(c * 16, c * 8,  c * 8)
        self.dec3 = Up(c * 8,  c * 4,  c * 4)
        self.dec2 = Up(c * 4,  c * 2,  c * 2)
        self.dec1 = Up(c * 2,  c,      c)

        # Segmentation head
        self.head = nn.Conv3d(c, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, D, H, W) CT patch, float32, values in [0, 1]
        Returns:
            logits: (B, num_classes, D, H, W)
        """
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        b = self.bottleneck(e4)

        d4 = self.dec4(b,  e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        return self.head(d1)
