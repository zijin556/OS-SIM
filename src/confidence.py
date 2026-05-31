from __future__ import annotations

from typing import Any

import numpy as np

from .height import height_parabolic


REASON_LABELS = {
    0: "valid_single_peak",
    1: "low_return",
    2: "broad_peak",
    3: "double_peak",
    4: "hv_inconsistent",
    5: "high_residual",
    6: "saturated_or_invalid_raw",
}


def count_peaks(A_stack: np.ndarray, relative_threshold: float = 0.25) -> np.ndarray:
    A = np.asarray(A_stack, dtype=np.float32)
    peak = A.max(axis=0)
    background = np.percentile(A, 10.0, axis=0)
    threshold = background + relative_threshold * (peak - background)
    count = np.zeros(A.shape[1:], dtype=np.int16)
    for k in range(1, A.shape[0] - 1):
        is_peak = (A[k] >= A[k - 1]) & (A[k] >= A[k + 1]) & (A[k] >= threshold)
        count += is_peak.astype(np.int16)
    return count


def axial_confidence_maps(
    A_fused: np.ndarray,
    z_values: np.ndarray,
    A_h: np.ndarray | None = None,
    A_v: np.ndarray | None = None,
    residual_energy_map: np.ndarray | None = None,
    raw_stack: np.ndarray | None = None,
) -> dict[str, Any]:
    A = np.asarray(A_fused, dtype=np.float32)
    sorted_A = np.sort(A, axis=0)
    peak = sorted_A[-1]
    second = sorted_A[-2] if A.shape[0] > 1 else np.zeros_like(peak)
    background = np.mean(A, axis=0)
    peak_to_background = peak / (background + 1e-8)
    sharpness = (peak - second) / (peak + 1e-8)
    fwhm_layers = np.sum(A >= (0.5 * peak[None, :, :]), axis=0).astype(np.float32)
    norm = (A - A.min(axis=0, keepdims=True)) / (np.ptp(A, axis=0, keepdims=True) + 1e-8)
    logits = 20.0 * (norm - norm.max(axis=0, keepdims=True))
    prob = np.exp(logits)
    prob /= prob.sum(axis=0, keepdims=True) + 1e-8
    entropy = -np.sum(prob * np.log(prob + 1e-8), axis=0) / np.log(A.shape[0])
    n_peaks = count_peaks(A)

    low_return = peak <= max(float(np.percentile(peak, 8.0)), 1e-6)
    low_return |= peak_to_background < 1.25
    broad_peak = fwhm_layers >= max(8.0, float(np.percentile(fwhm_layers, 80.0)))
    double_peak = n_peaks >= 2

    hv_diff = np.zeros_like(peak, dtype=np.float32)
    hv_inconsistent = np.zeros_like(peak, dtype=bool)
    if A_h is not None and A_v is not None:
        h_h = height_parabolic(A_h, z_values)
        h_v = height_parabolic(A_v, z_values)
        hv_diff = np.abs(h_h - h_v).astype(np.float32)
        hv_inconsistent = hv_diff > max(float(np.median(np.diff(z_values))) * 2.0, float(np.percentile(hv_diff, 85.0)))

    high_residual = np.zeros_like(peak, dtype=bool)
    if residual_energy_map is not None:
        residual_energy_map = np.asarray(residual_energy_map, dtype=np.float32)
        high_residual = residual_energy_map > float(np.percentile(residual_energy_map, 90.0))

    saturated = np.zeros_like(peak, dtype=bool)
    if raw_stack is not None:
        raw = np.asarray(raw_stack, dtype=np.float32)
        saturated = np.any((raw <= 1e-6) | (raw >= 0.999), axis=tuple(range(raw.ndim - 2)))

    reason = np.zeros_like(peak, dtype=np.uint8)
    reason[low_return] = 1
    reason[broad_peak] = 2
    reason[double_peak] = 3
    reason[hv_inconsistent] = 4
    reason[high_residual] = 5
    reason[saturated] = 6
    invalid = reason != 0

    confidence = (1.0 - entropy) * np.clip(sharpness, 0.0, None) * np.minimum(peak_to_background / 3.0, 1.0)
    confidence[invalid] *= 0.35

    return {
        "peak": peak.astype(np.float32),
        "peak_to_background": peak_to_background.astype(np.float32),
        "sharpness": sharpness.astype(np.float32),
        "fwhm_layers": fwhm_layers.astype(np.float32),
        "softmax_entropy": entropy.astype(np.float32),
        "number_of_peaks": n_peaks.astype(np.int16),
        "hv_peak_z_difference": hv_diff.astype(np.float32),
        "confidence": confidence.astype(np.float32),
        "invalid_mask": invalid,
        "reason_map": reason,
        "reason_labels": REASON_LABELS,
    }

