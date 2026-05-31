from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import torch
    from torch import nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
    TORCH_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - exercised only in missing-torch envs
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False
    TORCH_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


@dataclass(frozen=True)
class InverseNetConfig:
    in_channels: int = 6
    out_channels: int = 2
    base_channels: int = 24
    softplus_beta: float = 1.0
    architecture: str = "tiny_unet"
    freq_bins: int = 8
    transformer_heads: int = 4
    transformer_window: int = 8


def require_torch() -> None:
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch is required for inverse light-section network training. "
            f"Import error: {TORCH_IMPORT_ERROR}"
        )


if TORCH_AVAILABLE:

    class ConvBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.GroupNorm(num_groups=max(1, min(8, out_channels // 4)), num_channels=out_channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.GroupNorm(num_groups=max(1, min(8, out_channels // 4)), num_channels=out_channels),
                nn.SiLU(inplace=True),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.net(x)


    class ResidualBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, dilation: int = 1):
            super().__init__()
            padding = dilation
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding, dilation=dilation)
            self.norm1 = nn.GroupNorm(num_groups=max(1, min(8, out_channels // 4)), num_channels=out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=padding, dilation=dilation)
            self.norm2 = nn.GroupNorm(num_groups=max(1, min(8, out_channels // 4)), num_channels=out_channels)
            self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            y = F.silu(self.norm1(self.conv1(x)), inplace=True)
            y = self.norm2(self.conv2(y))
            return F.silu(y + self.skip(x), inplace=True)


    class FrequencyFiLM(nn.Module):
        """Global FFT descriptors that softly modulate bottleneck features.

        The descriptors are radial frequency-band energies plus horizontal and
        vertical high-frequency energies. They guide de-grid behavior without
        giving the network a free per-pixel stripe generator.
        """

        def __init__(self, channels: int, bins: int = 8):
            super().__init__()
            self.bins = int(bins)
            descriptor_dim = self.bins + 2
            hidden = max(16, channels // 2)
            self.mlp = nn.Sequential(
                nn.Linear(descriptor_dim, hidden),
                nn.SiLU(inplace=True),
                nn.Linear(hidden, 2 * channels),
            )

        def _descriptor(self, x: "torch.Tensor") -> "torch.Tensor":
            b, _c, h, w = x.shape
            freq = torch.fft.fftshift(torch.fft.fft2(x.float(), dim=(-2, -1), norm="ortho"), dim=(-2, -1))
            mag = torch.log1p(torch.abs(freq)).mean(dim=1)
            yy, xx = torch.meshgrid(
                torch.linspace(-1.0, 1.0, h, device=x.device, dtype=mag.dtype),
                torch.linspace(-1.0, 1.0, w, device=x.device, dtype=mag.dtype),
                indexing="ij",
            )
            radius = torch.sqrt(xx * xx + yy * yy)
            rows = []
            for idx in range(self.bins):
                lo = idx / float(self.bins)
                hi = (idx + 1) / float(self.bins)
                mask = (radius >= lo) & (radius < hi)
                rows.append(mag[:, mask].mean(dim=1) if bool(mask.any()) else mag.new_zeros((b,)))
            high = radius > 0.25
            horizontal = (torch.abs(yy) < 0.08) & high
            vertical = (torch.abs(xx) < 0.08) & high
            rows.append(mag[:, horizontal].mean(dim=1) if bool(horizontal.any()) else mag.new_zeros((b,)))
            rows.append(mag[:, vertical].mean(dim=1) if bool(vertical.any()) else mag.new_zeros((b,)))
            desc = torch.stack(rows, dim=1)
            return (desc - desc.mean(dim=1, keepdim=True)) / (desc.std(dim=1, keepdim=True) + 1e-6)

        def forward(self, bottleneck: "torch.Tensor", raw_input: "torch.Tensor") -> "torch.Tensor":
            gamma_beta = self.mlp(self._descriptor(raw_input)).to(dtype=bottleneck.dtype)
            gamma, beta = gamma_beta.chunk(2, dim=1)
            scale = 1.0 + 0.1 * torch.tanh(gamma)[:, :, None, None]
            bias = 0.1 * torch.tanh(beta)[:, :, None, None]
            return bottleneck * scale + bias


    class WindowTransformerBlock(nn.Module):
        def __init__(self, channels: int, heads: int = 4, window_size: int = 8):
            super().__init__()
            self.channels = int(channels)
            self.window_size = int(window_size)
            self.norm1 = nn.LayerNorm(channels)
            self.attn = nn.MultiheadAttention(channels, num_heads=heads, batch_first=True)
            self.norm2 = nn.LayerNorm(channels)
            self.mlp = nn.Sequential(
                nn.Linear(channels, 2 * channels),
                nn.GELU(),
                nn.Linear(2 * channels, channels),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            b, c, h, w = x.shape
            ws = self.window_size
            pad_h = (ws - h % ws) % ws
            pad_w = (ws - w % ws) % ws
            xp = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect") if pad_h or pad_w else x
            hp, wp = xp.shape[-2:]
            windows = xp.unfold(2, ws, ws).unfold(3, ws, ws)
            windows = windows.permute(0, 2, 3, 4, 5, 1).reshape(-1, ws * ws, c)
            y = self.norm1(windows)
            attn, _ = self.attn(y, y, y, need_weights=False)
            windows = windows + attn
            windows = windows + self.mlp(self.norm2(windows))
            y = windows.reshape(b, hp // ws, wp // ws, ws, ws, c).permute(0, 5, 1, 3, 2, 4)
            y = y.reshape(b, c, hp, wp)
            return y[..., :h, :w]


    class InverseLightSectionNet(nn.Module):
        """Small shared 2D U-Net that predicts ideal H/V light-section maps.

        Input channels are the six raw phase frames for one z layer:
        H100/H101/H102/V103/V104/V105. The output is constrained with
        softplus so the predicted modulation maps stay non-negative.
        """

        def __init__(self, config: InverseNetConfig | None = None, **kwargs: Any):
            super().__init__()
            if config is None:
                config = InverseNetConfig(**kwargs)
            self.config = config
            c = config.base_channels
            self.enc1 = ConvBlock(config.in_channels, c)
            self.enc2 = ConvBlock(c, 2 * c)
            self.bottleneck = ConvBlock(2 * c, 4 * c)
            self.up2 = ConvBlock(4 * c + 2 * c, 2 * c)
            self.up1 = ConvBlock(2 * c + c, c)
            self.out = nn.Conv2d(c, config.out_channels, kernel_size=1)
            self.softplus_beta = float(config.softplus_beta)

        def forward(self, x: "torch.Tensor") -> dict[str, "torch.Tensor"]:
            e1 = self.enc1(x)
            e2 = self.enc2(F.avg_pool2d(e1, kernel_size=2))
            b = self.bottleneck(F.avg_pool2d(e2, kernel_size=2))
            u2 = F.interpolate(b, size=e2.shape[-2:], mode="bilinear", align_corners=False)
            u2 = self.up2(torch.cat([u2, e2], dim=1))
            u1 = F.interpolate(u2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
            u1 = self.up1(torch.cat([u1, e1], dim=1))
            raw = self.out(u1)
            A = F.softplus(raw, beta=self.softplus_beta)
            return {
                "A": A,
                "A_h": A[:, 0],
                "A_v": A[:, 1] if A.shape[1] > 1 else A[:, 0],
                "raw_A": raw,
            }


    class FreqResInverseLightSectionNet(nn.Module):
        """Residual U-Net with frequency-conditioned bottleneck and optional windows."""

        def __init__(self, config: InverseNetConfig | None = None, use_transformer: bool = False, **kwargs: Any):
            super().__init__()
            if config is None:
                config = InverseNetConfig(**kwargs)
            self.config = config
            c = config.base_channels
            self.enc1 = ResidualBlock(config.in_channels, c)
            self.enc2 = ResidualBlock(c, 2 * c)
            self.bottleneck1 = ResidualBlock(2 * c, 4 * c)
            self.bottleneck2 = ResidualBlock(4 * c, 4 * c, dilation=2)
            self.freq_film = FrequencyFiLM(4 * c, bins=config.freq_bins)
            self.transformer = (
                WindowTransformerBlock(4 * c, heads=config.transformer_heads, window_size=config.transformer_window)
                if use_transformer
                else nn.Identity()
            )
            self.up2 = ResidualBlock(4 * c + 2 * c, 2 * c)
            self.up1 = ResidualBlock(2 * c + c, c)
            self.out = nn.Conv2d(c, config.out_channels, kernel_size=1)
            self.softplus_beta = float(config.softplus_beta)

        def forward(self, x: "torch.Tensor") -> dict[str, "torch.Tensor"]:
            e1 = self.enc1(x)
            e2 = self.enc2(F.avg_pool2d(e1, kernel_size=2))
            b = self.bottleneck1(F.avg_pool2d(e2, kernel_size=2))
            b = self.bottleneck2(b)
            b = self.freq_film(b, x)
            b = self.transformer(b)
            u2 = F.interpolate(b, size=e2.shape[-2:], mode="bilinear", align_corners=False)
            u2 = self.up2(torch.cat([u2, e2], dim=1))
            u1 = F.interpolate(u2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
            u1 = self.up1(torch.cat([u1, e1], dim=1))
            raw = self.out(u1)
            A = F.softplus(raw, beta=self.softplus_beta)
            return {
                "A": A,
                "A_h": A[:, 0],
                "A_v": A[:, 1] if A.shape[1] > 1 else A[:, 0],
                "raw_A": raw,
            }


    def build_inverse_lightsection_net(
        in_channels: int = 6,
        out_channels: int = 2,
        base_channels: int = 24,
        softplus_beta: float = 1.0,
        architecture: str = "tiny_unet",
        freq_bins: int = 8,
        transformer_heads: int = 4,
        transformer_window: int = 8,
    ) -> nn.Module:
        config = InverseNetConfig(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            softplus_beta=softplus_beta,
            architecture=architecture,
            freq_bins=freq_bins,
            transformer_heads=transformer_heads,
            transformer_window=transformer_window,
        )
        if architecture == "tiny_unet":
            return InverseLightSectionNet(config)
        if architecture == "freq_res_unet":
            return FreqResInverseLightSectionNet(config, use_transformer=False)
        if architecture == "freq_res_unet_tx":
            return FreqResInverseLightSectionNet(config, use_transformer=True)
        raise ValueError(f"Unknown inverse net architecture: {architecture}")


    def count_parameters(model: nn.Module) -> int:
        return int(sum(p.numel() for p in model.parameters() if p.requires_grad))

else:

    class InverseLightSectionNet:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any):
            require_torch()


    def build_inverse_lightsection_net(*args: Any, **kwargs: Any) -> Any:
        require_torch()


    def count_parameters(model: Any) -> int:
        require_torch()
        return 0
