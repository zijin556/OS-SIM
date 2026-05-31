from __future__ import annotations

import numpy as np
from scipy import ndimage


def charbonnier(x: np.ndarray, eps: float = 1e-3) -> float:
    arr = np.asarray(x, dtype=np.float32)
    return float(np.mean(np.sqrt(arr * arr + eps * eps)))


def tv_l1(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=np.float32)
    dy = np.abs(np.diff(arr, axis=-2)).mean()
    dx = np.abs(np.diff(arr, axis=-1)).mean()
    return float(dx + dy)


def laplacian_smooth_grad(x: np.ndarray) -> np.ndarray:
    return -ndimage.laplace(np.asarray(x, dtype=np.float32), mode="reflect")


def lowpass(x: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    return ndimage.gaussian_filter(np.asarray(x, dtype=np.float32), sigma=(0, sigma, sigma), mode="reflect")


def normalize_stack(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    lo = np.percentile(arr, 1.0)
    hi = np.percentile(arr, 99.0)
    return np.clip((arr - lo) / (hi - lo + 1e-8), 0.0, 1.0)

