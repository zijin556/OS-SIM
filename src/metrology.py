from __future__ import annotations

from typing import Any

import numpy as np

from .height import confidence_metrics, height_parabolic, platform_metrics_from_masks, summarize_height


def height_from_a_stack(A_stack: np.ndarray, z_values: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    height = height_parabolic(A_stack, z_values)
    conf = confidence_metrics(A_stack)
    return height, summarize_height(height, conf)


def repeatability_metrics(height_maps: dict[str, np.ndarray]) -> dict[str, Any]:
    if not height_maps:
        return {"status": "not_run", "reason": "no_height_maps"}
    shapes = {tuple(v.shape) for v in height_maps.values()}
    if len(shapes) != 1:
        return {"status": "failed", "reason": "shape_mismatch", "shapes": {k: list(v.shape) for k, v in height_maps.items()}}
    stack = np.stack([np.asarray(v, dtype=np.float32) for v in height_maps.values()], axis=0)
    return {
        "status": "ok",
        "method_count": int(stack.shape[0]),
        "pixelwise_std_mean": float(np.mean(np.std(stack, axis=0))),
        "pixelwise_std_p95": float(np.percentile(np.std(stack, axis=0), 95.0)),
        "method_mean_range": float(np.max(np.mean(stack, axis=(1, 2))) - np.min(np.mean(stack, axis=(1, 2)))),
    }


def standard_step_metrics(
    height_map: np.ndarray,
    low_mask: np.ndarray,
    high_mask: np.ndarray,
    nominal_step: float | None = None,
) -> dict[str, Any]:
    platform = platform_metrics_from_masks(height_map, low_mask, high_mask)
    out = {k: v for k, v in platform.items() if k not in {"low_mask", "high_mask"}}
    if platform.get("status") == "ok" and nominal_step is not None:
        out["nominal_step"] = float(nominal_step)
        out["step_error"] = float(platform["step_height"] - nominal_step)
        out["abs_step_error"] = float(abs(platform["step_height"] - nominal_step))
    else:
        out["absolute_error_status"] = "not_reported_without_nominal_step"
    return out
