from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src import data_io
from src.height import confidence_metrics, height_parabolic, platform_metrics_from_masks, summarize_height
from src.losses import tv_l1
from src.os_sim import demodulate_multigroup, fuse_hv
from src.visualization import save_loss_curve, save_montage, save_scalar_image

from optimize_modulation_latents import (  # noqa: E402
    build_grid_masks,
    build_patterns_by_source,
    grid_energy_mean,
    grid_penalty_gradient,
    modulation_loss,
)
from sweep_latent_hyperparams import score_candidate  # noqa: E402


def step_A(
    A: np.ndarray,
    A_init: np.ndarray,
    Y_tilde: np.ndarray,
    M: np.ndarray,
    grid_masks: np.ndarray,
    denom: np.ndarray,
    lr: float,
    lambda_sec: float,
    lambda_smooth: float,
    lambda_hv: float,
    lambda_grid: float,
) -> np.ndarray:
    residual = A[:, :, None, :, :] * M[None, :, :, :, :] - Y_tilde
    grad = np.mean(residual * M[None, :, :, :, :], axis=2) / denom
    if lambda_sec:
        grad += lambda_sec * (A - A_init)
    if lambda_smooth:
        smooth_target = ndimage.gaussian_filter(A, sigma=(0, 0, 1.0, 1.0), mode="reflect")
        grad += lambda_smooth * (A - smooth_target)
    if lambda_hv and A.shape[1] == 2:
        shared = A.mean(axis=1, keepdims=True)
        grad += lambda_hv * (A - shared)
    if lambda_grid:
        grad += lambda_grid * grid_penalty_gradient(A, grid_masks)
    return np.maximum(A - lr * grad, 0.0).astype(np.float32)


def evaluate(
    name: str,
    iteration: int,
    A: np.ndarray,
    Y: np.ndarray,
    M: np.ndarray,
    grid_masks: np.ndarray,
    z_values: np.ndarray,
    low_mask: np.ndarray,
    high_mask: np.ndarray,
) -> dict:
    A_fused = fuse_hv(A[:, 0], A[:, 1] if A.shape[1] > 1 else A[:, 0])
    height = height_parabolic(A_fused, z_values)
    conf = confidence_metrics(A_fused)
    return {
        "name": name,
        "iteration": int(iteration),
        "modulation_loss": modulation_loss(Y, A, M),
        "grid_energy": grid_energy_mean(A, grid_masks),
        "A_fused_tv": tv_l1(A_fused),
        "height": summarize_height(height, conf),
        "platform": platform_metrics_from_masks(height, low_mask, high_mask),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="A-latent optimization with fixed-platform early stopping.")
    parser.add_argument("--config", default="configs/ossim_dataset.yaml")
    parser.add_argument("--sample", default="细台阶")
    parser.add_argument("--groups", default="ossim_t6_h_3step,ossim_t6_v_3step")
    parser.add_argument("--exclude-samples", default="台阶曝光1")
    parser.add_argument("--crop", default="center:256")
    parser.add_argument("--platform-low-mask", default="outputs/height/baseline_t6_hv/platform_low_mask.npy")
    parser.add_argument("--platform-high-mask", default="outputs/height/baseline_t6_hv/platform_high_mask.npy")
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--lambda-sec", type=float, default=0.80)
    parser.add_argument("--lambda-smooth", type=float, default=0.60)
    parser.add_argument("--lambda-hv", type=float, default=0.20)
    parser.add_argument("--lambda-grid", type=float, default=0.001)
    parser.add_argument("--pattern-source", choices=["auto", "cos_fft", "measured_calibrated"], default="auto")
    parser.add_argument("--projection-sequence", default="saomiao3_6")
    parser.add_argument("--calibration-label", default="3-6")
    parser.add_argument("--calibration-mat", default="DMD_CCD_Calibration_Dict.mat")
    parser.add_argument("--max-platform-rms-multiplier", type=float, default=1.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="outputs/platform_opt/t6_hv")
    args = parser.parse_args()

    np.random.seed(args.seed)
    config = data_io.load_config(args.config)
    excluded = data_io.parse_excluded_samples(args.exclude_samples)
    groups = data_io.parse_groups(args.groups, config)
    crop = data_io.infer_crop_for_sample(config, args.sample, args.crop)
    bundle = data_io.load_os_sim_stack(config, args.sample, groups, crop, excluded_samples=excluded)
    Y = bundle["Y"]
    z_values = bundle["z_values"]
    demod = demodulate_multigroup(Y)
    A_init = demod["A"]
    M, pattern_infos = build_patterns_by_source(
        Y,
        bundle["groups"],
        groups,
        config,
        crop,
        pattern_source=args.pattern_source,
        sequence_stem=args.projection_sequence,
        calibration_label=args.calibration_label,
        calibration_mat=args.calibration_mat,
    )
    actual_pattern_source = pattern_infos[0].get("source", args.pattern_source) if pattern_infos else args.pattern_source
    grid_masks, grid_peaks = build_grid_masks(A_init)
    low_mask = np.load(ROOT / args.platform_low_mask).astype(bool)
    high_mask = np.load(ROOT / args.platform_high_mask).astype(bool)

    out_dir = ROOT / args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "z_values.npy", z_values)

    Y_tilde = Y - Y.mean(axis=2, keepdims=True)
    denom = np.mean(M[None, :, :, :, :] ** 2, axis=2) + 1e-6
    A = A_init.copy().astype(np.float32)
    baseline = evaluate("baseline_no_opt", 0, A, Y, M, grid_masks, z_values, low_mask, high_mask)
    baseline["score"], baseline["feasible_platform_rms"] = score_candidate(baseline, baseline, args.max_platform_rms_multiplier)
    best = baseline
    best_A = A.copy()
    history = [baseline]
    loss_curve = [baseline["modulation_loss"]]

    for iteration in range(1, args.iterations + 1):
        A = step_A(
            A,
            A_init,
            Y_tilde,
            M,
            grid_masks,
            denom,
            lr=args.lr,
            lambda_sec=args.lambda_sec,
            lambda_smooth=args.lambda_smooth,
            lambda_hv=args.lambda_hv,
            lambda_grid=args.lambda_grid,
        )
        if iteration % args.eval_every == 0 or iteration == args.iterations:
            metrics = evaluate("platform_constrained", iteration, A, Y, M, grid_masks, z_values, low_mask, high_mask)
            metrics["score"], metrics["feasible_platform_rms"] = score_candidate(metrics, baseline, args.max_platform_rms_multiplier)
            history.append(metrics)
            loss_curve.append(metrics["modulation_loss"])
            if metrics["score"] > best["score"]:
                best = metrics
                best_A = A.copy()
            print(
                f"iter {iteration}: score={metrics['score']:.4g}, feasible={metrics['feasible_platform_rms']}, "
                f"loss={metrics['modulation_loss']:.6g}, platform_rms={metrics['platform'].get('pooled_platform_rms')}"
            )

    A_h = best_A[:, 0]
    A_v = best_A[:, 1] if best_A.shape[1] > 1 else best_A[:, 0]
    A_fused = fuse_hv(A_h, A_v)
    height = height_parabolic(A_fused, z_values)
    np.save(out_dir / "A_h_best_stack.npy", A_h)
    np.save(out_dir / "A_v_best_stack.npy", A_v)
    np.save(out_dir / "A_fused_best_stack.npy", A_fused)
    np.save(out_dir / "height_best_peakfit.npy", height)
    save_loss_curve(loss_curve, out_dir / "platform_opt_loss_curve.png")
    mid = A_fused.shape[0] // 2
    baseline_A_fused = fuse_hv(A_init[:, 0], A_init[:, 1] if A_init.shape[1] > 1 else A_init[:, 0])
    save_montage(
        [baseline_A_fused[mid], A_fused[mid], height],
        out_dir / "platform_opt_summary.png",
        ["baseline A fused", f"best A iter {best['iteration']}", "best height"],
        cols=3,
        cmap="viridis",
    )
    save_scalar_image(height, out_dir / "height_best_map.png", f"Platform-constrained best iter {best['iteration']}")

    output = {
        "sample": args.sample,
        "groups": groups,
        "crop": crop.label if crop else "full",
        "best_iteration": best["iteration"],
        "best_score": best["score"],
        "baseline": baseline,
        "best": best,
        "history": history,
        "pattern_source": actual_pattern_source,
        "requested_pattern_source": args.pattern_source,
        "pattern_infos": pattern_infos,
        "grid_peaks": grid_peaks,
        "config": {
            "iterations": args.iterations,
            "eval_every": args.eval_every,
            "lr": args.lr,
            "lambda_sec": args.lambda_sec,
            "lambda_smooth": args.lambda_smooth,
            "lambda_hv": args.lambda_hv,
            "lambda_grid": args.lambda_grid,
            "pattern_source": actual_pattern_source,
            "requested_pattern_source": args.pattern_source,
            "projection_sequence": args.projection_sequence,
            "calibration_label": args.calibration_label,
            "calibration_mat": args.calibration_mat,
            "max_platform_rms_multiplier": args.max_platform_rms_multiplier,
            "platform_low_mask": args.platform_low_mask,
            "platform_high_mask": args.platform_high_mask,
            "random_seed": args.seed,
        },
        "interpretation": "The saved A stack is the best evaluated iteration under the fixed-platform score, not necessarily the final iteration.",
    }
    data_io.write_json(out_dir / "metrics.json", output)
    data_io.write_json(out_dir / "config.json", vars(args))
    report = [
        "# Platform-Constrained Latent Optimization Report",
        "",
        f"- Pattern source: `{actual_pattern_source}` (requested `{args.pattern_source}`)",
        f"- Best iteration: `{best['iteration']}`",
        f"- Baseline modulation loss: `{baseline['modulation_loss']:.6g}`",
        f"- Best modulation loss: `{best['modulation_loss']:.6g}`",
        f"- Baseline platform RMS: `{baseline['platform']['pooled_platform_rms']:.6g}`",
        f"- Best platform RMS: `{best['platform']['pooled_platform_rms']:.6g}`",
        f"- Best height std: `{best['height']['height_std']:.6g}`",
    ]
    (out_dir / "platform_opt_report.md").write_text("\n".join(report), encoding="utf-8")
    data_io.append_report(
        ROOT / "outputs/nightly_report.md",
        "P5d Platform-Constrained Early Stop",
        [
            f"- Ran platform-constrained latent optimization for `{args.sample}` on `{crop.label if crop else 'full'}`.",
            f"- Best iteration `{best['iteration']}` with loss `{best['modulation_loss']:.6g}` and platform RMS `{best['platform']['pooled_platform_rms']:.6g}`.",
            f"- Outputs saved under `{out_dir}`.",
        ],
    )
    print(f"Wrote platform-constrained optimization outputs to {out_dir}")


if __name__ == "__main__":
    main()
