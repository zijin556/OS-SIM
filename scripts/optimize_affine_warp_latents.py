from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import data_io
from src.forward_models import AffineWarp, DMDPatternWarpForward, estimate_q_from_frames, q_from_group
from src.height import confidence_metrics, height_parabolic, platform_metrics_from_masks, summarize_height
from src.losses import charbonnier, tv_l1
from src.os_sim import demodulate_multigroup, fuse_hv
from src.visualization import save_loss_curve, save_montage, save_scalar_image


PARAMS_PER_GROUP = 6


def affine_from_params(params: np.ndarray, group_index: int) -> AffineWarp:
    base = group_index * PARAMS_PER_GROUP
    dyy, dyx, dxy, dxx, ty, tx = params[base : base + PARAMS_PER_GROUP]
    matrix = np.array([[1.0 + dyy, dyx], [dxy, 1.0 + dxx]], dtype=np.float32)
    offset = np.array([ty, tx], dtype=np.float32)
    return AffineWarp(matrix=matrix, offset=offset)


def q_initializers(Y: np.ndarray, groups: list[dict], q_source: str) -> tuple[list[tuple[float, float]], list[dict]]:
    q_init: list[tuple[float, float]] = []
    q_info: list[dict] = []
    for g, group in enumerate(groups):
        if q_source == "group":
            qx, qy = q_from_group(group)
            q_init.append((qx, qy))
            q_info.append({"source": "group_period"})
        else:
            qx, qy, info = estimate_q_from_frames(Y[Y.shape[0] // 2, g], group)
            q_init.append((qx, qy))
            q_info.append(info)
    return q_init, q_info


def build_patterns(
    shape: tuple[int, int],
    groups: list[dict],
    q_init: list[tuple[float, float]],
    params: np.ndarray,
    pattern_kind: str,
    duty: float,
) -> np.ndarray:
    patterns = []
    for g, group in enumerate(groups):
        model = DMDPatternWarpForward(
            group,
            warp=affine_from_params(params, g),
            qxy=q_init[g],
            pattern_kind=pattern_kind,
            duty=duty,
        )
        patterns.append(model.patterns(shape))
    return np.stack(patterns, axis=0).astype(np.float32)


def solve_A_closed_form(Y: np.ndarray, M: np.ndarray) -> np.ndarray:
    Y_tilde = Y - Y.mean(axis=2, keepdims=True)
    numerator = np.sum(Y_tilde * M[None, :, :, :, :], axis=2)
    denominator = np.sum(M[None, :, :, :, :] ** 2, axis=2) + 1e-6
    return np.maximum(numerator / denominator, 0.0).astype(np.float32)


def affine_regularization(params: np.ndarray, shape: tuple[int, int]) -> float:
    scale = float(max(shape))
    value = 0.0
    for g in range(params.size // PARAMS_PER_GROUP):
        base = g * PARAMS_PER_GROUP
        dyy, dyx, dxy, dxx, ty, tx = params[base : base + PARAMS_PER_GROUP]
        value += float(dyy * dyy + dyx * dyx + dxy * dxy + dxx * dxx)
        value += float((ty / scale) ** 2 + (tx / scale) ** 2)
    return value


def objective(
    params: np.ndarray,
    Y: np.ndarray,
    groups: list[dict],
    q_init: list[tuple[float, float]],
    A_ref: np.ndarray,
    pattern_kind: str,
    duty: float,
    lambda_affine: float,
    lambda_sec: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    M = build_patterns(Y.shape[-2:], groups, q_init, params, pattern_kind, duty)
    A = solve_A_closed_form(Y, M)
    Y_tilde = Y - Y.mean(axis=2, keepdims=True)
    residual = Y_tilde - A[:, :, None, :, :] * M[None, :, :, :, :]
    loss = charbonnier(residual)
    if lambda_sec:
        loss += lambda_sec * float(np.mean(np.abs(A - A_ref)))
    loss += lambda_affine * affine_regularization(params, Y.shape[-2:])
    return float(loss), A, M


def coordinate_search(
    Y: np.ndarray,
    groups: list[dict],
    q_init: list[tuple[float, float]],
    A_ref: np.ndarray,
    passes: int,
    pattern_kind: str,
    duty: float,
    lambda_affine: float,
    lambda_sec: float,
    max_matrix_delta: float,
    max_offset_px: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
    params = np.zeros(len(groups) * PARAMS_PER_GROUP, dtype=np.float32)
    steps_one = np.array([0.01, 0.006, 0.006, 0.01, 1.0, 1.0], dtype=np.float32)
    limits_one = np.array(
        [max_matrix_delta, max_matrix_delta, max_matrix_delta, max_matrix_delta, max_offset_px, max_offset_px],
        dtype=np.float32,
    )
    steps = np.tile(steps_one, len(groups))
    limits = np.tile(limits_one, len(groups))
    best_loss, best_A, best_M = objective(params, Y, groups, q_init, A_ref, pattern_kind, duty, lambda_affine, lambda_sec)
    curve = [best_loss]
    for _ in range(passes):
        improved = False
        for i in range(params.size):
            for sign in (1.0, -1.0):
                trial = params.copy()
                trial[i] = np.clip(trial[i] + sign * steps[i], -limits[i], limits[i])
                loss, A, M = objective(trial, Y, groups, q_init, A_ref, pattern_kind, duty, lambda_affine, lambda_sec)
                if loss < best_loss:
                    params, best_loss, best_A, best_M = trial, loss, A, M
                    improved = True
            curve.append(best_loss)
        if not improved:
            steps *= 0.5
    return params, best_A, best_M, curve


def params_to_report(params: np.ndarray, groups: list[dict]) -> list[dict]:
    rows = []
    for g, group in enumerate(groups):
        warp = affine_from_params(params, g)
        rows.append(
            {
                "group": group["name"],
                "matrix": warp.matrix.astype(float).tolist(),
                "offset_yx": warp.offset.astype(float).tolist(),
                "delta_matrix_fro": float(np.linalg.norm(warp.matrix - np.eye(2, dtype=np.float32))),
                "offset_norm_px": float(np.linalg.norm(warp.offset)),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize generated-DMD affine warp theta and closed-form A on a crop.")
    parser.add_argument("--config", default="configs/ossim_dataset.yaml")
    parser.add_argument("--sample", default="细台阶")
    parser.add_argument("--groups", default="ossim_t6_h_3step,ossim_t6_v_3step")
    parser.add_argument("--exclude-samples", default="台阶曝光1")
    parser.add_argument("--crop", default="center:256")
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--q-source", choices=["fft", "group"], default="fft")
    parser.add_argument("--pattern-kind", choices=["binary", "cos"], default="binary")
    parser.add_argument("--duty", type=float, default=0.5)
    parser.add_argument("--lambda-affine", type=float, default=0.2)
    parser.add_argument("--lambda-sec", type=float, default=0.01)
    parser.add_argument("--max-matrix-delta", type=float, default=0.04)
    parser.add_argument("--max-offset-px", type=float, default=3.0)
    parser.add_argument("--platform-low-mask", default="outputs/height/baseline_t6_hv/platform_low_mask.npy")
    parser.add_argument("--platform-high-mask", default="outputs/height/baseline_t6_hv/platform_high_mask.npy")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="outputs/affine_warp_opt/t6_hv")
    args = parser.parse_args()

    np.random.seed(args.seed)
    config = data_io.load_config(args.config)
    excluded = data_io.parse_excluded_samples(args.exclude_samples)
    group_names = data_io.parse_groups(args.groups, config)
    crop = data_io.infer_crop_for_sample(config, args.sample, args.crop)
    bundle = data_io.load_os_sim_stack(config, args.sample, group_names, crop, excluded_samples=excluded)
    Y = bundle["Y"]
    z_values = bundle["z_values"]
    demod = demodulate_multigroup(Y)
    A_ref = demod["A"]
    D0 = demod["D0"]
    q_init, q_info = q_initializers(Y, bundle["groups"], args.q_source)

    initial_params = np.zeros(len(group_names) * PARAMS_PER_GROUP, dtype=np.float32)
    initial_loss, initial_A, initial_M = objective(
        initial_params, Y, bundle["groups"], q_init, A_ref, args.pattern_kind, args.duty, args.lambda_affine, args.lambda_sec
    )
    params, A_opt, M_opt, curve = coordinate_search(
        Y,
        bundle["groups"],
        q_init,
        A_ref,
        args.passes,
        args.pattern_kind,
        args.duty,
        args.lambda_affine,
        args.lambda_sec,
        args.max_matrix_delta,
        args.max_offset_px,
    )
    final_loss, _, _ = objective(params, Y, bundle["groups"], q_init, A_ref, args.pattern_kind, args.duty, args.lambda_affine, args.lambda_sec)

    A_base_fused = fuse_hv(A_ref[:, 0], A_ref[:, 1] if A_ref.shape[1] > 1 else A_ref[:, 0])
    A_fused = fuse_hv(A_opt[:, 0], A_opt[:, 1] if A_opt.shape[1] > 1 else A_opt[:, 0])
    height_base = height_parabolic(A_base_fused, z_values)
    height_opt = height_parabolic(A_fused, z_values)
    conf_base = confidence_metrics(A_base_fused)
    conf_opt = confidence_metrics(A_fused)

    platform = {"status": "not_evaluated", "reason": "platform masks unavailable or shape mismatch"}
    low_path = ROOT / args.platform_low_mask
    high_path = ROOT / args.platform_high_mask
    if low_path.exists() and high_path.exists():
        low_mask = np.load(low_path).astype(bool)
        high_mask = np.load(high_path).astype(bool)
        platform = platform_metrics_from_masks(height_opt, low_mask, high_mask)

    out_dir = ROOT / args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "z_values.npy", z_values)
    np.save(out_dir / "A_h_stack.npy", A_opt[:, 0])
    np.save(out_dir / "A_v_stack.npy", A_opt[:, 1] if A_opt.shape[1] > 1 else A_opt[:, 0])
    np.save(out_dir / "A_fused_stack.npy", A_fused)
    np.save(out_dir / "height_affine_peakfit.npy", height_opt)
    np.save(out_dir / "height_baseline_peakfit.npy", height_base)
    mid = Y.shape[0] // 2
    Y_hat = D0[:, :, None, :, :] + A_opt[:, :, None, :, :] * M_opt[None, :, :, :, :]
    save_loss_curve(curve, out_dir / "affine_warp_loss_curve.png")
    save_montage(
        [A_base_fused[mid], A_fused[mid], height_base, height_opt, np.abs(height_opt - height_base)],
        out_dir / "affine_A_height_compare.png",
        ["A baseline fused", "A affine fused", "height baseline", "height affine", "height abs diff"],
        cols=3,
    )
    save_montage(
        [
            Y[mid, 0, 0],
            Y_hat[mid, 0, 0],
            np.abs(Y[mid, 0, 0] - Y_hat[mid, 0, 0]),
            Y[mid, 1, 0],
            Y_hat[mid, 1, 0],
            np.abs(Y[mid, 1, 0] - Y_hat[mid, 1, 0]),
        ],
        out_dir / "affine_reprojection_examples.png",
        ["real H", "hat H", "abs H", "real V", "hat V", "abs V"],
        cols=3,
    )
    save_scalar_image(height_opt, out_dir / "affine_height_map.png", "Generated-DMD affine fused height")

    metrics = {
        "sample": args.sample,
        "groups": group_names,
        "crop": crop.label if crop else "full",
        "passes": args.passes,
        "q_source": args.q_source,
        "pattern_kind": args.pattern_kind,
        "initial_objective": float(initial_loss),
        "final_objective": float(final_loss),
        "objective_delta": float(final_loss - initial_loss),
        "initial_A_tv": tv_l1(A_base_fused),
        "affine_A_tv": tv_l1(A_fused),
        "height_baseline": summarize_height(height_base, conf_base),
        "height_affine": summarize_height(height_opt, conf_opt),
        "platform": platform,
        "affine_theta": params_to_report(params, bundle["groups"]),
        "q_initialization": q_info,
        "regularization": {
            "lambda_affine": args.lambda_affine,
            "lambda_sec": args.lambda_sec,
            "max_matrix_delta": args.max_matrix_delta,
            "max_offset_px": args.max_offset_px,
        },
        "interpretation": "This optimizes a constrained affine warp over generated DMD patterns. It is a generated-pattern control and does not consume the measured DMD bitmaps or dense DMD-CCD calibration map.",
    }
    data_io.write_json(out_dir / "metrics.json", metrics)
    data_io.write_json(out_dir / "config.json", vars(args))
    report = [
        "# Generated-DMD Affine Warp Optimization Report",
        "",
        f"- Sample: `{args.sample}`",
        f"- Groups: `{', '.join(group_names)}`",
        f"- Crop: `{crop.label if crop else 'full'}`",
        f"- Pattern kind: `{args.pattern_kind}`",
        f"- Objective: `{initial_loss:.6g}` -> `{final_loss:.6g}`",
        f"- Height std: `{metrics['height_affine']['height_std']:.6g}`",
        f"- Platform RMS: `{platform.get('pooled_platform_rms', 'not_evaluated')}`",
        "",
        "This stage validates the low-dimensional affine warp fitting path using generated DMD patterns. It does not claim calibrated DMD-CCD mapping superiority.",
    ]
    (out_dir / "affine_warp_report.md").write_text("\n".join(report), encoding="utf-8")
    data_io.append_report(
        ROOT / "outputs/nightly_report.md",
        "P5f Generated-DMD Affine Warp Optimization",
        [
            f"- Optimized constrained affine warp theta for `{args.sample}` on `{crop.label if crop else 'full'}`.",
            f"- Objective `{initial_loss:.6g}` -> `{final_loss:.6g}`.",
            f"- Outputs saved under `{out_dir}`.",
        ],
    )
    print(f"Wrote affine warp optimization outputs to {out_dir}")


if __name__ == "__main__":
    main()
