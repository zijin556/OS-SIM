from __future__ import annotations

from typing import Any

import numpy as np

from .fft_analysis import circular_peak_mask, detect_frequency_peaks, fft_magnitude, masked_fft_energy
from .height import confidence_metrics, height_parabolic, platform_metrics_from_masks, summarize_height
from .losses import charbonnier, tv_l1
from .os_sim import fuse_hv


EPS = 1e-8


def phase_center(Y: np.ndarray) -> np.ndarray:
    return np.asarray(Y, dtype=np.float32) - np.asarray(Y, dtype=np.float32).mean(axis=-3, keepdims=True)


def project_a_signed(Y: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Closed-form signed projection for Y[K,G,M,H,W] and M[G,M,H,W]."""
    y_tilde = phase_center(Y)
    numer = np.sum(y_tilde * M[None, :, :, :, :], axis=2)
    denom = np.sum(M[None, :, :, :, :] ** 2, axis=2) + EPS
    return (numer / denom).astype(np.float32)


def project_a_nonnegative(Y: np.ndarray, M: np.ndarray) -> np.ndarray:
    return np.maximum(project_a_signed(Y, M), 0.0).astype(np.float32)


def predict_modulation(A: np.ndarray, M: np.ndarray) -> np.ndarray:
    return np.asarray(A, dtype=np.float32)[:, :, None, :, :] * np.asarray(M, dtype=np.float32)[None, :, :, :, :]


def modulation_residual(Y: np.ndarray, A: np.ndarray, M: np.ndarray) -> np.ndarray:
    return phase_center(Y) - predict_modulation(A, M)


def modulation_loss(Y: np.ndarray, A: np.ndarray, M: np.ndarray) -> float:
    return charbonnier(modulation_residual(Y, A, M))


def mask_from_peaks(image: np.ndarray, top_n: int = 4, radius: int = 4, min_radius: int = 4) -> tuple[np.ndarray, list[dict[str, float]]]:
    peaks = detect_frequency_peaks(np.asarray(image, dtype=np.float32), top_n=top_n, min_radius=min_radius)
    mask = circular_peak_mask(image.shape, peaks, radius=radius) if peaks else np.zeros(image.shape, dtype=bool)
    return mask, peaks


def a_grid_energy(A_stack: np.ndarray, mask: np.ndarray) -> float:
    A = np.asarray(A_stack, dtype=np.float32)
    if A.ndim == 2:
        return masked_fft_energy(A, mask)
    values = [masked_fft_energy(A[k], mask) for k in range(A.shape[0])]
    return float(np.mean(values)) if values else 0.0


def residual_energy(residual: np.ndarray, mask: np.ndarray) -> float:
    R = np.asarray(residual, dtype=np.float32)
    flat = R.reshape((-1,) + R.shape[-2:])
    values = [masked_fft_energy(frame, mask) for frame in flat]
    return float(np.mean(values)) if values else 0.0


def residual_fft_l2(residual: np.ndarray) -> float:
    R = np.asarray(residual, dtype=np.float32).reshape((-1,) + residual.shape[-2:])
    vals = []
    for frame in R:
        centered = frame - frame.mean()
        F = np.fft.fft2(centered)
        vals.append(float(np.linalg.norm(F.ravel()) / np.sqrt(centered.size)))
    return float(np.mean(vals)) if vals else 0.0


def stripe_mask_from_raw(Y_group: np.ndarray, top_n: int = 1, radius: int = 4) -> tuple[np.ndarray, list[dict[str, float]]]:
    probe = phase_center(Y_group[Y_group.shape[0] // 2])[0]
    return mask_from_peaks(probe, top_n=top_n, radius=radius, min_radius=4)


def grid_mask_from_a(A_stack: np.ndarray, top_n: int = 4, radius: int = 4) -> tuple[np.ndarray, list[dict[str, float]]]:
    A = np.asarray(A_stack, dtype=np.float32)
    probe = A[A.shape[0] // 2] if A.ndim == 3 else A
    return mask_from_peaks(probe, top_n=top_n, radius=radius, min_radius=4)


def height_summary(A_stack: np.ndarray, z_values: np.ndarray, low_mask: np.ndarray | None = None, high_mask: np.ndarray | None = None) -> dict[str, Any]:
    height = height_parabolic(A_stack, z_values)
    conf = confidence_metrics(A_stack)
    out: dict[str, Any] = summarize_height(height, conf)
    if low_mask is not None and high_mask is not None:
        platform = platform_metrics_from_masks(height, low_mask, high_mask)
        out["platform"] = {k: v for k, v in platform.items() if k not in {"low_mask", "high_mask"}}
    return out


def hv_consistency(A_h: np.ndarray, A_v: np.ndarray, z_values: np.ndarray) -> dict[str, float]:
    h_h = height_parabolic(A_h, z_values)
    h_v = height_parabolic(A_v, z_values)
    diff = h_h - h_v
    a_corr = float(np.corrcoef(np.asarray(A_h).ravel(), np.asarray(A_v).ravel())[0, 1])
    return {
        "height_diff_median_abs": float(np.median(np.abs(diff))),
        "height_diff_mean_abs": float(np.mean(np.abs(diff))),
        "height_diff_p95_abs": float(np.percentile(np.abs(diff), 95.0)),
        "A_corr_HV": a_corr,
    }


def method_metrics(
    name: str,
    Y: np.ndarray,
    A: np.ndarray,
    M: np.ndarray,
    z_values: np.ndarray,
    grid_masks: np.ndarray,
    stripe_masks: np.ndarray,
) -> dict[str, Any]:
    residual = modulation_residual(Y, A, M)
    A_h = A[:, 0]
    A_v = A[:, 1] if A.shape[1] > 1 else A[:, 0]
    A_fused = fuse_hv(A_h, A_v)
    fused_mask, fused_peaks = grid_mask_from_a(A_fused)
    group_rows = []
    for g in range(A.shape[1]):
        group_rows.append(
            {
                "A_grid_energy": a_grid_energy(A[:, g], grid_masks[g]),
                "residual_grid_energy": residual_energy(residual[:, g], grid_masks[g]),
                "residual_stripe_leakage": residual_energy(residual[:, g], stripe_masks[g]),
            }
        )
    return {
        "method": name,
        "modulation_loss": modulation_loss(Y, A, M),
        "A_grid_energy_mean": float(np.mean([row["A_grid_energy"] for row in group_rows])),
        "residual_grid_energy_mean": float(np.mean([row["residual_grid_energy"] for row in group_rows])),
        "residual_stripe_leakage_mean": float(np.mean([row["residual_stripe_leakage"] for row in group_rows])),
        "residual_fft_l2": residual_fft_l2(residual),
        "A_fused_grid_energy": a_grid_energy(A_fused, fused_mask),
        "A_fused_grid_peaks": fused_peaks,
        "A_fused_tv": tv_l1(A_fused),
        "height_fused": height_summary(A_fused, z_values),
        "hv_consistency": hv_consistency(A_h, A_v, z_values),
        "groups": group_rows,
    }

