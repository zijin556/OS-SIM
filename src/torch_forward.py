from __future__ import annotations

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


def require_torch() -> None:
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch is required for the differentiable OS-SIM forward model. "
            f"Import error: {TORCH_IMPORT_ERROR}"
        )


if TORCH_AVAILABLE:

    def charbonnier_loss(x: "torch.Tensor", eps: float = 1e-3) -> "torch.Tensor":
        return torch.sqrt(x * x + eps * eps).mean()


    def tv_l1_torch(x: "torch.Tensor") -> "torch.Tensor":
        dy = torch.abs(x[..., 1:, :] - x[..., :-1, :]).mean()
        dx = torch.abs(x[..., :, 1:] - x[..., :, :-1]).mean()
        return dx + dy


    def normalize_phase_patterns(patterns: "torch.Tensor", eps: float = 1e-6) -> "torch.Tensor":
        centered = patterns - patterns.mean(dim=1, keepdim=True)
        rms = torch.sqrt(torch.mean(centered * centered, dim=1, keepdim=True) + eps * eps)
        return centered / rms


    def lowpass_avg(x: "torch.Tensor", kernel_size: int = 9) -> "torch.Tensor":
        pad = kernel_size // 2
        channels = x.shape[1]
        weight = torch.ones((channels, 1, kernel_size, kernel_size), device=x.device, dtype=x.dtype)
        weight = weight / float(kernel_size * kernel_size)
        return F.conv2d(F.pad(x, (pad, pad, pad, pad), mode="reflect"), weight, groups=channels)


    def fft_grid_penalty(A: "torch.Tensor", masks: "torch.Tensor") -> "torch.Tensor":
        """Return mean masked FFT magnitude for A[B,G,H,W]."""
        if masks.numel() == 0:
            return A.new_tensor(0.0)
        Freq = torch.fft.fftshift(torch.fft.fft2(A, dim=(-2, -1)), dim=(-2, -1))
        mag = torch.abs(Freq)
        mask = masks[None].to(device=A.device, dtype=A.dtype)
        denom = torch.linalg.vector_norm(mag.flatten(start_dim=-2), dim=-1).clamp_min(1e-8)
        numer = torch.linalg.vector_norm((mag * mask).flatten(start_dim=-2), dim=-1)
        return (numer / denom).mean()


    def _phase_shift(patterns: "torch.Tensor", phase_offsets: "torch.Tensor", sharpness: float = 18.0) -> "torch.Tensor":
        g_count, m_count = patterns.shape[:2]
        src = torch.linspace(0.0, 2.0 * torch.pi, m_count + 1, device=patterns.device, dtype=patterns.dtype)[:-1]
        target = src[None, :, None] + phase_offsets[:, :, None]
        diff = torch.atan2(torch.sin(target - src[None, None, :]), torch.cos(target - src[None, None, :]))
        weights = torch.softmax(-sharpness * diff * diff, dim=-1)
        return torch.einsum("gms,gshw->gmhw", weights, patterns)


    def _apply_affine_residual(patterns: "torch.Tensor", residual: "torch.Tensor", residual_scale: float) -> "torch.Tensor":
        g_count, m_count, h, w = patterns.shape
        rows = []
        eye = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=patterns.device, dtype=patterns.dtype)
        for g in range(g_count):
            theta = eye + residual_scale * residual[g]
            theta_batch = theta[None].expand(m_count, 2, 3)
            images = patterns[g, :, None, :, :]
            grid = F.affine_grid(theta_batch, size=images.shape, align_corners=False)
            rows.append(F.grid_sample(images, grid, mode="bilinear", padding_mode="reflection", align_corners=False)[:, 0])
        return torch.stack(rows, dim=0)


    def _gaussian_blur_group(patterns: "torch.Tensor", sigma_raw: "torch.Tensor", radius: int = 4) -> "torch.Tensor":
        g_count, m_count, h, w = patterns.shape
        coords = torch.arange(-radius, radius + 1, device=patterns.device, dtype=patterns.dtype)
        rows = []
        for g in range(g_count):
            sigma = F.softplus(sigma_raw[g]) + 1e-4
            k1 = torch.exp(-0.5 * (coords / sigma) ** 2)
            k1 = k1 / k1.sum()
            k2 = (k1[:, None] * k1[None, :]).to(dtype=patterns.dtype)
            kernel = k2[None, None].expand(m_count, 1, -1, -1)
            images = patterns[g][None]
            rows.append(F.conv2d(F.pad(images, (radius, radius, radius, radius), mode="reflect"), kernel, groups=m_count)[0])
        return torch.stack(rows, dim=0)


    def _delta_q_residual(
        patterns: "torch.Tensor",
        delta_q: "torch.Tensor",
        delta_q_scale: float,
        delta_q_pattern_scale: float,
    ) -> "torch.Tensor":
        g_count, m_count, h, w = patterns.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-0.5, 0.5, h, device=patterns.device, dtype=patterns.dtype),
            torch.linspace(-0.5, 0.5, w, device=patterns.device, dtype=patterns.dtype),
            indexing="ij",
        )
        phase_steps = torch.arange(m_count, device=patterns.device, dtype=patterns.dtype) * (2.0 * torch.pi / float(m_count))
        spatial = delta_q_scale * (delta_q[:, 0, None, None] * xx[None] + delta_q[:, 1, None, None] * yy[None])
        shifted = torch.cos(phase_steps[None, :, None, None] + 2.0 * torch.pi * spatial[:, None])
        at_zero = torch.cos(phase_steps)[None, :, None, None]
        return patterns + delta_q_pattern_scale * (shifted - at_zero)


    class RestrictedStripeForward(nn.Module):
        """Differentiable low-dimensional OS-SIM degradation model.

        The model starts from measured/fixed camera-side phase patterns M[g,m,y,x].
        It can only alter them through a small delta_q frequency residual, global
        phase mixing, a small affine residual, a small blur parameter, and fixed
        grid/Moire basis coefficients. It does not allow a network to freely
        synthesize per-pixel stripe patterns.
        """

        def __init__(
            self,
            base_patterns: "torch.Tensor",
            grid_basis: "torch.Tensor | None" = None,
            learn_delta_q: bool = True,
            learn_phase: bool = True,
            learn_affine: bool = True,
            learn_blur: bool = True,
            learn_grid: bool = True,
            delta_q_scale: float = 0.15,
            delta_q_pattern_scale: float = 0.05,
            affine_residual_scale: float = 0.02,
            grid_coeff_scale: float = 0.05,
        ):
            super().__init__()
            arr = base_patterns.float()
            if arr.ndim != 4:
                raise ValueError(f"Expected base_patterns [G,M,H,W], got {tuple(arr.shape)}")
            self.register_buffer("base_patterns", normalize_phase_patterns(arr))
            if grid_basis is None:
                grid_basis = arr.new_zeros((arr.shape[0], 0, arr.shape[-2], arr.shape[-1]))
            if grid_basis.ndim == 4:
                if grid_basis.shape[0] != arr.shape[0]:
                    raise ValueError("grid_basis must have shape [G,B,H,W] or [G,M,B,H,W]")
                grid_basis_count = grid_basis.shape[1]
                self.grid_basis_phase_dependent = False
            elif grid_basis.ndim == 5:
                if grid_basis.shape[0] != arr.shape[0] or grid_basis.shape[1] != arr.shape[1]:
                    raise ValueError("phase-dependent grid_basis must have shape [G,M,B,H,W]")
                grid_basis_count = grid_basis.shape[2]
                self.grid_basis_phase_dependent = True
            else:
                raise ValueError("grid_basis must have shape [G,B,H,W] or [G,M,B,H,W]")
            self.register_buffer("grid_basis", grid_basis.float())
            g_count, m_count = arr.shape[:2]
            self.delta_q = nn.Parameter(torch.zeros(g_count, 2), requires_grad=learn_delta_q)
            self.phase_offsets = nn.Parameter(torch.zeros(g_count, m_count), requires_grad=learn_phase)
            self.affine_residual = nn.Parameter(torch.zeros(g_count, 2, 3), requires_grad=learn_affine)
            self.blur_sigma_raw = nn.Parameter(torch.full((g_count,), -5.0), requires_grad=learn_blur)
            self.grid_coeff = nn.Parameter(torch.zeros(g_count, grid_basis_count), requires_grad=learn_grid)
            self.delta_q_scale = float(delta_q_scale)
            self.delta_q_pattern_scale = float(delta_q_pattern_scale)
            self.affine_residual_scale = float(affine_residual_scale)
            self.grid_coeff_scale = float(grid_coeff_scale)

        def patterns(self) -> "torch.Tensor":
            out = self.base_patterns
            if self.phase_offsets.requires_grad:
                out = _phase_shift(out, self.phase_offsets)
            if self.delta_q.requires_grad:
                out = _delta_q_residual(out, self.delta_q, self.delta_q_scale, self.delta_q_pattern_scale)
            if self.affine_residual.requires_grad:
                out = _apply_affine_residual(out, self.affine_residual, self.affine_residual_scale)
            if self.grid_coeff.numel():
                if self.grid_basis_phase_dependent:
                    grid = torch.einsum("gb,gmbhw->gmhw", self.grid_coeff * self.grid_coeff_scale, self.grid_basis)
                    out = out + grid
                else:
                    grid = torch.einsum("gb,gbhw->ghw", self.grid_coeff * self.grid_coeff_scale, self.grid_basis)
                    out = out + grid[:, None]
            if self.blur_sigma_raw.requires_grad:
                out = _gaussian_blur_group(out, self.blur_sigma_raw)
            return normalize_phase_patterns(out)

        def predict_modulation(self, A: "torch.Tensor") -> "torch.Tensor":
            M = self.patterns()
            return A[:, :, None, :, :] * M[None, :, :, :, :]

        def forward(self, A: "torch.Tensor", D0: "torch.Tensor") -> "torch.Tensor":
            return D0[:, :, None, :, :] + self.predict_modulation(A)

        def theta_regularization(self) -> "torch.Tensor":
            reg = self.delta_q.square().mean()
            reg = reg + self.phase_offsets.square().mean()
            reg = reg + self.affine_residual.square().mean()
            reg = reg + F.softplus(self.blur_sigma_raw).square().mean()
            if self.grid_coeff.numel():
                reg = reg + self.grid_coeff.square().mean()
            return reg

        def theta_summary(self) -> dict[str, Any]:
            return {
                "delta_q": self.delta_q.detach().cpu().tolist(),
                "phase_offsets": self.phase_offsets.detach().cpu().tolist(),
                "affine_residual": self.affine_residual.detach().cpu().tolist(),
                "blur_sigma": F.softplus(self.blur_sigma_raw).detach().cpu().tolist(),
                "grid_coeff": self.grid_coeff.detach().cpu().tolist(),
                "delta_q_scale": self.delta_q_scale,
                "delta_q_pattern_scale": self.delta_q_pattern_scale,
                "affine_residual_scale": self.affine_residual_scale,
                "grid_coeff_scale": self.grid_coeff_scale,
            }


else:

    def charbonnier_loss(*args: Any, **kwargs: Any) -> Any:
        require_torch()


    def tv_l1_torch(*args: Any, **kwargs: Any) -> Any:
        require_torch()


    def normalize_phase_patterns(*args: Any, **kwargs: Any) -> Any:
        require_torch()


    def lowpass_avg(*args: Any, **kwargs: Any) -> Any:
        require_torch()


    def fft_grid_penalty(*args: Any, **kwargs: Any) -> Any:
        require_torch()


    class RestrictedStripeForward:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any):
            require_torch()
