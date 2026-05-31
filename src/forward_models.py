from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .fft_analysis import detect_frequency_peaks


def normalize_m(M: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    arr = np.asarray(M, dtype=np.float32)
    centered = arr - arr.mean(axis=-3, keepdims=True)
    rms = np.sqrt(np.mean(centered * centered, axis=-3, keepdims=True))
    return centered / (rms + eps)


def phases_for_m(m: int) -> np.ndarray:
    return np.linspace(0.0, 2.0 * np.pi, m, endpoint=False, dtype=np.float32)


def period_from_group(group: dict) -> float:
    period = str(group.get("period", "T6")).upper().replace("T", "")
    try:
        return float(period)
    except ValueError:
        return 6.0


def q_from_group(group: dict) -> tuple[float, float]:
    q = 2.0 * np.pi / period_from_group(group)
    direction = group.get("direction", "horizontal")
    if direction == "horizontal":
        return 0.0, q
    return q, 0.0


def cos_patterns(
    shape: tuple[int, int],
    group: dict,
    phases: np.ndarray | None = None,
    qxy: tuple[float, float] | None = None,
) -> np.ndarray:
    h, w = shape
    m = int(group.get("phase_steps", 3))
    phases = phases_for_m(m) if phases is None else np.asarray(phases, dtype=np.float32)
    qx, qy = q_from_group(group) if qxy is None else qxy
    y, x = np.indices((h, w), dtype=np.float32)
    phi = qx * x + qy * y
    M = np.stack([np.cos(phi + p) for p in phases], axis=0)
    return normalize_m(M)


def binary_grating_patterns(
    shape: tuple[int, int],
    group: dict,
    phases: np.ndarray | None = None,
    qxy: tuple[float, float] | None = None,
    duty: float = 0.5,
) -> np.ndarray:
    """Generate known binary DMD stripe patterns P_m for a frame group.

    The pattern is defined in the same coordinate system as the supplied shape.
    A following warp model can reinterpret these coordinates as DMD coordinates
    and sample them into the camera grid. Values are binary before the required
    phase-wise normalization.
    """
    h, w = shape
    m = int(group.get("phase_steps", 3))
    phases = phases_for_m(m) if phases is None else np.asarray(phases, dtype=np.float32)
    qx, qy = q_from_group(group) if qxy is None else qxy
    y, x = np.indices((h, w), dtype=np.float32)
    phi = qx * x + qy * y
    threshold = np.cos(np.pi * np.clip(duty, 1e-3, 1.0 - 1e-3))
    frames = [(np.cos(phi + p) >= threshold).astype(np.float32) for p in phases]
    return normalize_m(np.stack(frames, axis=0))


def estimate_q_from_frames(frames: np.ndarray, group: dict) -> tuple[float, float, dict]:
    y_tilde = frames - frames.mean(axis=0, keepdims=True)
    probe = y_tilde[0]
    peaks = detect_frequency_peaks(probe, top_n=8, min_radius=4)
    if not peaks:
        qx, qy = q_from_group(group)
        return qx, qy, {"source": "group_period", "peaks": []}
    direction = group.get("direction", "horizontal")
    target = 1.0 / period_from_group(group)
    if direction == "horizontal":
        candidates = [p for p in peaks if abs(p["freq_y_cycles_per_px"]) >= abs(p["freq_x_cycles_per_px"])]
        key = lambda p: abs(abs(p["freq_y_cycles_per_px"]) - target) + 0.25 * abs(p["freq_x_cycles_per_px"])
    else:
        candidates = [p for p in peaks if abs(p["freq_x_cycles_per_px"]) >= abs(p["freq_y_cycles_per_px"])]
        key = lambda p: abs(abs(p["freq_x_cycles_per_px"]) - target) + 0.25 * abs(p["freq_y_cycles_per_px"])
    peak = min(candidates or peaks, key=key)
    qx = 2.0 * np.pi * peak["freq_x_cycles_per_px"]
    qy = 2.0 * np.pi * peak["freq_y_cycles_per_px"]
    return float(qx), float(qy), {"source": "fft_nominal_period", "target_cycles_per_px": target, "peak": peak, "peaks": peaks}


@dataclass
class AffineWarp:
    matrix: np.ndarray
    offset: np.ndarray

    @classmethod
    def identity(cls) -> "AffineWarp":
        return cls(matrix=np.eye(2, dtype=np.float32), offset=np.zeros(2, dtype=np.float32))

    def sample(self, image: np.ndarray) -> np.ndarray:
        return ndimage.affine_transform(
            image,
            self.matrix,
            offset=self.offset,
            order=1,
            mode="reflect",
            prefilter=False,
        ).astype(np.float32)


@dataclass
class PolynomialWarp:
    """Second-order residual warp in image coordinates.

    Coefficients are ordered as [1, x, y, x^2, xy, y^2] for each output axis.
    The default zero coefficients produce an identity map.
    """

    coeff_y: np.ndarray
    coeff_x: np.ndarray

    @classmethod
    def identity(cls) -> "PolynomialWarp":
        return cls(coeff_y=np.zeros(6, dtype=np.float32), coeff_x=np.zeros(6, dtype=np.float32))

    def displacement(self, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        h, w = shape
        yy, xx = np.indices(shape, dtype=np.float32)
        xn = (xx - (w - 1) / 2.0) / max(w - 1, 1)
        yn = (yy - (h - 1) / 2.0) / max(h - 1, 1)
        basis = np.stack([np.ones_like(xn), xn, yn, xn * xn, xn * yn, yn * yn], axis=0)
        dy = np.tensordot(self.coeff_y.astype(np.float32), basis, axes=(0, 0))
        dx = np.tensordot(self.coeff_x.astype(np.float32), basis, axes=(0, 0))
        return dy.astype(np.float32), dx.astype(np.float32)

    def sample(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape
        yy, xx = np.indices((h, w), dtype=np.float32)
        dy, dx = self.displacement((h, w))
        return ndimage.map_coordinates(image, [yy + dy, xx + dx], order=1, mode="reflect").astype(np.float32)

    def regularization(self, shape: tuple[int, int]) -> dict[str, float]:
        dy, dx = self.displacement(shape)
        lap = ndimage.laplace(dy, mode="reflect") ** 2 + ndimage.laplace(dx, mode="reflect") ** 2
        return {
            "warp_mag": float(np.mean(dy * dy + dx * dx)),
            "warp_smooth": float(np.mean(lap)),
        }


@dataclass
class BSplineWarp:
    """Coarse-grid residual warp implemented as smooth upsampled control points."""

    ctrl_y: np.ndarray
    ctrl_x: np.ndarray

    @classmethod
    def zeros(cls, grid_shape: tuple[int, int] = (8, 8)) -> "BSplineWarp":
        return cls(np.zeros(grid_shape, dtype=np.float32), np.zeros(grid_shape, dtype=np.float32))

    def displacement(self, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        zoom_y = shape[0] / self.ctrl_y.shape[0]
        zoom_x = shape[1] / self.ctrl_y.shape[1]
        dy = ndimage.zoom(self.ctrl_y.astype(np.float32), (zoom_y, zoom_x), order=3)
        dx = ndimage.zoom(self.ctrl_x.astype(np.float32), (zoom_y, zoom_x), order=3)
        dy = dy[: shape[0], : shape[1]]
        dx = dx[: shape[0], : shape[1]]
        if dy.shape != shape:
            dy = np.pad(dy, ((0, shape[0] - dy.shape[0]), (0, shape[1] - dy.shape[1])), mode="edge")
            dx = np.pad(dx, ((0, shape[0] - dx.shape[0]), (0, shape[1] - dx.shape[1])), mode="edge")
        return dy.astype(np.float32), dx.astype(np.float32)

    def sample(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape
        yy, xx = np.indices((h, w), dtype=np.float32)
        dy, dx = self.displacement((h, w))
        return ndimage.map_coordinates(image, [yy + dy, xx + dx], order=1, mode="reflect").astype(np.float32)

    def regularization(self, shape: tuple[int, int]) -> dict[str, float]:
        dy, dx = self.displacement(shape)
        lap = ndimage.laplace(dy, mode="reflect") ** 2 + ndimage.laplace(dx, mode="reflect") ** 2
        return {
            "warp_mag": float(np.mean(dy * dy + dx * dx)),
            "warp_smooth": float(np.mean(lap)),
        }


class OptionalBlur:
    def __init__(self, sigma: float = 0.0):
        self.sigma = float(max(sigma, 0.0))

    def apply(self, patterns: np.ndarray) -> np.ndarray:
        if self.sigma <= 0.0:
            return np.asarray(patterns, dtype=np.float32)
        return np.stack(
            [ndimage.gaussian_filter(frame, sigma=self.sigma, mode="reflect") for frame in np.asarray(patterns, dtype=np.float32)],
            axis=0,
        ).astype(np.float32)


class FixedPatternForward:
    def __init__(self, patterns: np.ndarray, blur: OptionalBlur | None = None):
        arr = np.asarray(patterns, dtype=np.float32)
        if blur is not None:
            arr = blur.apply(arr)
        self.fixed_patterns = normalize_m(arr)

    def patterns(self, shape: tuple[int, int]) -> np.ndarray:
        if tuple(shape) != tuple(self.fixed_patterns.shape[-2:]):
            raise ValueError(f"Fixed patterns have shape {self.fixed_patterns.shape[-2:]}, requested {shape}")
        return self.fixed_patterns


def fixed_grid_basis(shape: tuple[int, int], peaks: list[dict[str, float]], max_basis: int = 4) -> np.ndarray:
    h, w = shape
    yy, xx = np.indices(shape, dtype=np.float32)
    basis = []
    for peak in peaks[:max_basis]:
        fx = float(peak["freq_x_cycles_per_px"])
        fy = float(peak["freq_y_cycles_per_px"])
        phase = 2.0 * np.pi * (fx * xx + fy * yy)
        basis.extend([np.cos(phase), np.sin(phase)])
    if not basis:
        return np.zeros((0, h, w), dtype=np.float32)
    return np.stack(basis, axis=0).astype(np.float32)


def phase_dependent_grid_basis(
    shape: tuple[int, int],
    peaks: list[dict[str, float]],
    m_count: int,
    max_basis: int = 4,
) -> np.ndarray:
    """Return grid/Moire basis with explicit phase dependence [M,B,H,W].

    A phase-invariant additive grid term is removed by phase-wise centering in
    normalize_m. This basis gives every phase frame a shifted nuisance pattern,
    so fitted coefficients remain identifiable after normalization.
    """
    h, w = shape
    yy, xx = np.indices(shape, dtype=np.float32)
    phase_steps = phases_for_m(int(m_count))
    basis = []
    for peak in peaks[:max_basis]:
        fx = float(peak["freq_x_cycles_per_px"])
        fy = float(peak["freq_y_cycles_per_px"])
        spatial = 2.0 * np.pi * (fx * xx + fy * yy)
        basis.append(np.stack([np.cos(spatial + phase) for phase in phase_steps], axis=0))
        basis.append(np.stack([np.sin(spatial + phase) for phase in phase_steps], axis=0))
    if not basis:
        return np.zeros((int(m_count), 0, h, w), dtype=np.float32)
    return np.stack(basis, axis=1).astype(np.float32)


class IdealCosForward:
    def __init__(self, group: dict, qxy: tuple[float, float] | None = None):
        self.group = group
        self.qxy = qxy

    def patterns(self, shape: tuple[int, int]) -> np.ndarray:
        return cos_patterns(shape, self.group, qxy=self.qxy)

    def predict(self, D0: np.ndarray, A: np.ndarray) -> np.ndarray:
        M = self.patterns(A.shape[-2:])
        return D0[:, None, :, :] + A[:, None, :, :] * M[None, :, :, :]


class ModulationDomainForward(IdealCosForward):
    def predict_modulation(self, A: np.ndarray) -> np.ndarray:
        M = self.patterns(A.shape[-2:])
        return A[:, None, :, :] * M[None, :, :, :]


class DMDPatternWarpForward(IdealCosForward):
    def __init__(
        self,
        group: dict,
        warp: AffineWarp | PolynomialWarp | BSplineWarp | None = None,
        qxy: tuple[float, float] | None = None,
        pattern_kind: str = "binary",
        duty: float = 0.5,
    ):
        super().__init__(group, qxy=qxy)
        self.warp = warp or AffineWarp.identity()
        self.pattern_kind = pattern_kind
        self.duty = float(duty)

    def patterns(self, shape: tuple[int, int]) -> np.ndarray:
        if self.pattern_kind == "binary":
            base = binary_grating_patterns(shape, self.group, qxy=self.qxy, duty=self.duty)
        elif self.pattern_kind == "cos":
            base = cos_patterns(shape, self.group, qxy=self.qxy)
        else:
            raise ValueError(f"Unknown pattern_kind: {self.pattern_kind}")
        warped = np.stack([self.warp.sample(frame) for frame in base], axis=0)
        return normalize_m(warped)


class HarmonicGridForward(IdealCosForward):
    def __init__(
        self,
        group: dict,
        qxy: tuple[float, float] | None = None,
        a3: float = 0.0,
        a5: float = 0.0,
        grid_basis: np.ndarray | None = None,
        grid_coeff: np.ndarray | None = None,
        blur: OptionalBlur | None = None,
    ):
        super().__init__(group, qxy=qxy)
        self.a3 = float(a3)
        self.a5 = float(a5)
        self.grid_basis = None if grid_basis is None else np.asarray(grid_basis, dtype=np.float32)
        self.grid_coeff = None if grid_coeff is None else np.asarray(grid_coeff, dtype=np.float32)
        self.blur = blur or OptionalBlur(0.0)

    def patterns(self, shape: tuple[int, int]) -> np.ndarray:
        h, w = shape
        m = int(self.group.get("phase_steps", 3))
        phases = phases_for_m(m)
        qx, qy = q_from_group(self.group) if self.qxy is None else self.qxy
        yy, xx = np.indices((h, w), dtype=np.float32)
        phi0 = qx * xx + qy * yy
        frames = []
        grid_invariant = 0.0
        grid_by_phase = None
        if self.grid_basis is not None and self.grid_basis.size:
            if self.grid_basis.ndim == 4:
                if self.grid_basis.shape[0] != m:
                    raise ValueError(f"Phase-dependent grid basis expects first axis M={m}, got {self.grid_basis.shape[0]}")
                basis_count = self.grid_basis.shape[1]
                coeff = np.zeros(basis_count, dtype=np.float32) if self.grid_coeff is None else self.grid_coeff
                grid_by_phase = np.tensordot(coeff[:basis_count], self.grid_basis, axes=(0, 1))
            else:
                coeff = np.zeros(self.grid_basis.shape[0], dtype=np.float32) if self.grid_coeff is None else self.grid_coeff
                grid_invariant = np.tensordot(coeff[: self.grid_basis.shape[0]], self.grid_basis, axes=(0, 0))
        for m_idx, phase in enumerate(phases):
            phi = phi0 + phase
            grid = grid_by_phase[m_idx] if grid_by_phase is not None else grid_invariant
            frame = np.cos(phi) + self.a3 * np.cos(3.0 * phi) + self.a5 * np.cos(5.0 * phi) + grid
            frames.append(frame)
        return normalize_m(self.blur.apply(np.stack(frames, axis=0)))


def _charbonnier_mean(values: np.ndarray, eps: float = 1e-3) -> float:
    return float(np.mean(np.sqrt(values * values + eps * eps)))


def model_losses(Y: np.ndarray, D0: np.ndarray, A: np.ndarray, M: np.ndarray) -> dict[str, float]:
    Y_hat = D0[:, None, :, :] + A[:, None, :, :] * M[None, :, :, :]
    Y_tilde = Y - Y.mean(axis=1, keepdims=True)
    Y_tilde_hat = A[:, None, :, :] * M[None, :, :, :]
    raw = Y - Y_hat
    mod = Y_tilde - Y_tilde_hat
    return {
        "raw_rmse": float(np.sqrt(np.mean(raw * raw))),
        "mod_rmse": float(np.sqrt(np.mean(mod * mod))),
        "raw_mae": float(np.mean(np.abs(raw))),
        "mod_mae": float(np.mean(np.abs(mod))),
        "raw_charbonnier": _charbonnier_mean(raw),
        "mod_charbonnier": _charbonnier_mean(mod),
    }
