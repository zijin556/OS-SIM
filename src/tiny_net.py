from __future__ import annotations


def require_torch():
    try:
        import torch  # type: ignore
        import torch.nn as nn  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional local install
        raise RuntimeError(
            "PyTorch is not installed. The tiny-net stage is optional and should only run "
            "after P0-P7 are validated and torch is available."
        ) from exc
    return torch, nn


def build_tiny_unet(in_channels: int, out_channels: int = 1, base_channels: int = 16):
    """Build a small U-Net-like model when PyTorch is available."""
    torch, nn = require_torch()

    class ConvBlock(nn.Module):
        def __init__(self, c_in: int, c_out: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(c_out, c_out, 3, padding=1),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.net(x)

    class TinyUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc1 = ConvBlock(in_channels, base_channels)
            self.pool = nn.MaxPool2d(2)
            self.enc2 = ConvBlock(base_channels, base_channels * 2)
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            self.dec = ConvBlock(base_channels * 3, base_channels)
            self.out = nn.Conv2d(base_channels, out_channels, 1)
            self.softplus = nn.Softplus()

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            u = self.up(e2)
            if u.shape[-2:] != e1.shape[-2:]:
                u = torch.nn.functional.interpolate(u, size=e1.shape[-2:], mode="bilinear", align_corners=False)
            return self.softplus(self.out(self.dec(torch.cat([u, e1], dim=1))))

    return TinyUNet()

