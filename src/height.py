from __future__ import annotations

import numpy as np
from scipy import ndimage


def _as_z(z_values: np.ndarray | None, k: int) -> np.ndarray:
    if z_values is None:
        return np.arange(k, dtype=np.float32)
    z = np.asarray(z_values, dtype=np.float32)
    if z.shape[0] != k:
        raise ValueError(f"Expected {k} z values, got {z.shape[0]}")
    return z


def height_argmax(A_stack: np.ndarray, z_values: np.ndarray | None = None) -> np.ndarray:
    A = np.asarray(A_stack, dtype=np.float32)
    z = _as_z(z_values, A.shape[0])
    idx = np.argmax(A, axis=0)
    return z[idx].astype(np.float32)


def height_parabolic(A_stack: np.ndarray, z_values: np.ndarray | None = None) -> np.ndarray:
    A = np.asarray(A_stack, dtype=np.float32)
    z = _as_z(z_values, A.shape[0])
    idx = np.argmax(A, axis=0)
    h, w = idx.shape
    yy, xx = np.indices((h, w))
    idx0 = np.clip(idx, 1, A.shape[0] - 2)
    ym = A[idx0 - 1, yy, xx]
    y0 = A[idx0, yy, xx]
    yp = A[idx0 + 1, yy, xx]
    denom = ym - 2.0 * y0 + yp
    offset = np.zeros_like(y0, dtype=np.float32)
    valid = np.abs(denom) > 1e-8
    offset[valid] = 0.5 * (ym[valid] - yp[valid]) / denom[valid]
    offset = np.clip(offset, -1.0, 1.0)
    dz = np.median(np.diff(z)) if z.shape[0] > 1 else 1.0
    height = z[idx0] + offset * dz
    height[idx == 0] = z[0]
    height[idx == A.shape[0] - 1] = z[-1]
    return height.astype(np.float32)


def height_gaussian(A_stack: np.ndarray, z_values: np.ndarray | None = None, eps: float = 1e-8) -> np.ndarray:
    A = np.maximum(np.asarray(A_stack, dtype=np.float32), eps)
    return height_parabolic(np.log(A), z_values=z_values)


def height_softargmax(
    A_stack: np.ndarray,
    z_values: np.ndarray | None = None,
    beta: float = 20.0,
    normalize: bool = True,
) -> np.ndarray:
    A = np.asarray(A_stack, dtype=np.float32)
    z = _as_z(z_values, A.shape[0])
    B = A
    if normalize:
        amin = B.min(axis=0, keepdims=True)
        amax = B.max(axis=0, keepdims=True)
        B = (B - amin) / (amax - amin + 1e-8)
    logits = beta * (B - B.max(axis=0, keepdims=True))
    P = np.exp(logits)
    P /= P.sum(axis=0, keepdims=True) + 1e-8
    return np.sum(P * z[:, None, None], axis=0).astype(np.float32)


def confidence_metrics(A_stack: np.ndarray) -> dict[str, np.ndarray]:
    A = np.asarray(A_stack, dtype=np.float32)
    sorted_A = np.sort(A, axis=0)
    peak = sorted_A[-1]
    second = sorted_A[-2] if A.shape[0] > 1 else np.zeros_like(peak)
    background = np.mean(A, axis=0)
    sharpness = (peak - second) / (peak + 1e-8)
    peak_to_background = peak / (background + 1e-8)
    amin = A.min(axis=0, keepdims=True)
    amax = A.max(axis=0, keepdims=True)
    norm = (A - amin) / (amax - amin + 1e-8)
    P = np.exp(20.0 * (norm - norm.max(axis=0, keepdims=True)))
    P /= P.sum(axis=0, keepdims=True) + 1e-8
    entropy = -np.sum(P * np.log(P + 1e-8), axis=0) / np.log(A.shape[0])
    half = 0.5 * peak[None, :, :]
    width = np.sum(A >= half, axis=0).astype(np.float32)
    invalid = (peak <= 1e-6) | (peak_to_background < 1.05)
    return {
        "peak": peak.astype(np.float32),
        "sharpness": sharpness.astype(np.float32),
        "peak_to_background": peak_to_background.astype(np.float32),
        "softmax_entropy": entropy.astype(np.float32),
        "fwhm_layers": width,
        "invalid_mask": invalid,
    }


def summarize_height(height_map: np.ndarray, confidence: dict[str, np.ndarray] | None = None) -> dict[str, float]:
    h = np.asarray(height_map, dtype=np.float32)
    out = {
        "height_min": float(np.nanmin(h)),
        "height_max": float(np.nanmax(h)),
        "height_mean": float(np.nanmean(h)),
        "height_std": float(np.nanstd(h)),
    }
    if confidence is not None:
        out.update(
            {
                "confidence_sharpness_mean": float(np.nanmean(confidence["sharpness"])),
                "confidence_peak_to_background_mean": float(np.nanmean(confidence["peak_to_background"])),
                "invalid_fraction": float(np.mean(confidence["invalid_mask"])),
            }
        )
    return out


def auto_two_platform_metrics(
    height_map: np.ndarray,
    invalid_mask: np.ndarray | None = None,
    erode_px: int = 5,
    min_fraction: float = 0.02,
) -> dict[str, object]:
    """Estimate two plateau ROIs from a height map.

    This is a diagnostic helper for step-like samples. It does not know the
    calibrated ground-truth step height; it only reports the measured separation
    in the same z coordinate as the input height map.
    """
    h = np.asarray(height_map, dtype=np.float32)
    finite = np.isfinite(h)
    if invalid_mask is not None:
        finite &= ~np.asarray(invalid_mask, dtype=bool)
    values = h[finite]
    if values.size < 16:
        return {"status": "failed", "reason": "not_enough_valid_pixels"}
    p20, p80 = np.percentile(values, [20.0, 80.0])
    c0, c1 = float(p20), float(p80)
    if abs(c1 - c0) < 1e-6:
        return {"status": "failed", "reason": "height_distribution_has_no_separation"}
    for _ in range(30):
        d0 = np.abs(values - c0)
        d1 = np.abs(values - c1)
        low_vals = values[d0 <= d1]
        high_vals = values[d1 < d0]
        if low_vals.size == 0 or high_vals.size == 0:
            return {"status": "failed", "reason": "one_cluster_empty"}
        nc0, nc1 = float(np.mean(low_vals)), float(np.mean(high_vals))
        if abs(nc0 - c0) + abs(nc1 - c1) < 1e-5:
            c0, c1 = nc0, nc1
            break
        c0, c1 = nc0, nc1
    if c0 > c1:
        c0, c1 = c1, c0
    threshold = 0.5 * (c0 + c1)
    low_mask = finite & (h <= threshold)
    high_mask = finite & (h > threshold)
    if erode_px > 0:
        structure = np.ones((2 * erode_px + 1, 2 * erode_px + 1), dtype=bool)
        low_mask = ndimage.binary_erosion(low_mask, structure=structure, border_value=0)
        high_mask = ndimage.binary_erosion(high_mask, structure=structure, border_value=0)
    total = h.size
    min_count = max(16, int(total * min_fraction))
    if int(low_mask.sum()) < min_count or int(high_mask.sum()) < min_count:
        return {
            "status": "failed",
            "reason": "platform_roi_too_small_after_erosion",
            "low_count": int(low_mask.sum()),
            "high_count": int(high_mask.sum()),
            "min_count": int(min_count),
            "threshold": float(threshold),
        }

    def roi_stats(mask: np.ndarray) -> dict[str, float | int]:
        vals = h[mask]
        mean = float(np.mean(vals))
        std = float(np.std(vals))
        rms = float(np.sqrt(np.mean((vals - mean) ** 2)))
        return {
            "count": int(vals.size),
            "fraction": float(vals.size / total),
            "mean": mean,
            "std": std,
            "rms": rms,
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }

    low = roi_stats(low_mask)
    high = roi_stats(high_mask)
    return {
        "status": "ok",
        "method": "two_cluster_height_threshold_with_binary_erosion",
        "threshold": float(threshold),
        "erode_px": int(erode_px),
        "low_platform": low,
        "high_platform": high,
        "step_height": float(high["mean"] - low["mean"]),
        "pooled_platform_rms": float(np.sqrt((low["rms"] ** 2 + high["rms"] ** 2) / 2.0)),
        "low_mask": low_mask,
        "high_mask": high_mask,
    }


def platform_metrics_from_masks(
    height_map: np.ndarray,
    low_mask: np.ndarray,
    high_mask: np.ndarray,
) -> dict[str, object]:
    h = np.asarray(height_map, dtype=np.float32)
    low_mask = np.asarray(low_mask, dtype=bool)
    high_mask = np.asarray(high_mask, dtype=bool)
    if low_mask.shape != h.shape or high_mask.shape != h.shape:
        return {
            "status": "failed",
            "reason": "mask_shape_mismatch",
            "height_shape": list(h.shape),
            "low_mask_shape": list(low_mask.shape),
            "high_mask_shape": list(high_mask.shape),
        }
    if int(low_mask.sum()) < 1 or int(high_mask.sum()) < 1:
        return {
            "status": "failed",
            "reason": "empty_platform_mask",
            "low_count": int(low_mask.sum()),
            "high_count": int(high_mask.sum()),
        }

    def roi_stats(mask: np.ndarray) -> dict[str, float | int]:
        vals = h[mask]
        mean = float(np.mean(vals))
        std = float(np.std(vals))
        rms = float(np.sqrt(np.mean((vals - mean) ** 2)))
        return {
            "count": int(vals.size),
            "fraction": float(vals.size / h.size),
            "mean": mean,
            "std": std,
            "rms": rms,
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }

    low = roi_stats(low_mask)
    high = roi_stats(high_mask)
    if low["mean"] > high["mean"]:
        low, high = high, low
    return {
        "status": "ok",
        "method": "provided_platform_masks",
        "low_platform": low,
        "high_platform": high,
        "step_height": float(high["mean"] - low["mean"]),
        "pooled_platform_rms": float(np.sqrt((low["rms"] ** 2 + high["rms"] ** 2) / 2.0)),
    }
