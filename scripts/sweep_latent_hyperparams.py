from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src import data_io
from src.height import confidence_metrics, height_parabolic, platform_metrics_from_masks, summarize_height
from src.losses import tv_l1
from src.os_sim import demodulate_multigroup, fuse_hv
from src.visualization import save_montage, save_scalar_image

from optimize_modulation_latents import (  # noqa: E402
    build_grid_masks,
    build_patterns,
    grid_energy_mean,
    modulation_loss,
    optimize_A,
)


def candidate_grid() -> list[dict[str, float | int | str]]:
    return [
        {
            "name": "baseline_no_opt",
            "iterations": 0,
            "lr": 0.0,
            "lambda_sec": 0.0,
            "lambda_smooth": 0.0,
            "lambda_hv": 0.0,
            "lambda_grid": 0.0,
        },
        {
            "name": "conservative_sec_smooth",
            "iterations": 35,
            "lr": 0.08,
            "lambda_sec": 0.20,
            "lambda_smooth": 0.25,
            "lambda_hv": 0.05,
            "lambda_grid": 0.004,
        },
        {
            "name": "micro_stability_probe",
            "iterations": 12,
            "lr": 0.02,
            "lambda_sec": 0.80,
            "lambda_smooth": 0.60,
            "lambda_hv": 0.20,
            "lambda_grid": 0.001,
        },
        {
            "name": "tiny_grid_probe",
            "iterations": 12,
            "lr": 0.03,
            "lambda_sec": 0.50,
            "lambda_smooth": 0.50,
            "lambda_hv": 0.15,
            "lambda_grid": 0.004,
        },
        {
            "name": "moderate_balanced",
            "iterations": 35,
            "lr": 0.12,
            "lambda_sec": 0.10,
            "lambda_smooth": 0.15,
            "lambda_hv": 0.03,
            "lambda_grid": 0.008,
        },
        {
            "name": "grid_damped",
            "iterations": 35,
            "lr": 0.10,
            "lambda_sec": 0.12,
            "lambda_smooth": 0.30,
            "lambda_hv": 0.06,
            "lambda_grid": 0.018,
        },
        {
            "name": "no_grid_smooth_control",
            "iterations": 35,
            "lr": 0.12,
            "lambda_sec": 0.12,
            "lambda_smooth": 0.30,
            "lambda_hv": 0.06,
            "lambda_grid": 0.0,
        },
        {
            "name": "previous_aggressive",
            "iterations": 35,
            "lr": 0.25,
            "lambda_sec": 0.02,
            "lambda_smooth": 0.05,
            "lambda_hv": 0.01,
            "lambda_grid": 0.02,
        },
    ]


def evaluate_candidate(
    name: str,
    A: np.ndarray,
    Y: np.ndarray,
    M: np.ndarray,
    masks: np.ndarray,
    z_values: np.ndarray,
    low_mask: np.ndarray,
    high_mask: np.ndarray,
) -> dict:
    A_fused = fuse_hv(A[:, 0], A[:, 1] if A.shape[1] > 1 else A[:, 0])
    height = height_parabolic(A_fused, z_values)
    conf = confidence_metrics(A_fused)
    platform = platform_metrics_from_masks(height, low_mask, high_mask)
    return {
        "name": name,
        "modulation_loss": modulation_loss(Y, A, M),
        "grid_energy": grid_energy_mean(A, masks),
        "A_fused_tv": tv_l1(A_fused),
        "height": summarize_height(height, conf),
        "platform": platform,
    }


def score_candidate(metrics: dict, baseline: dict, max_platform_rms_multiplier: float) -> tuple[float, bool]:
    platform = metrics.get("platform", {})
    if platform.get("status") != "ok":
        return float("inf"), False
    baseline_rms = baseline["platform"]["pooled_platform_rms"]
    rms = platform["pooled_platform_rms"]
    feasible = rms <= baseline_rms * max_platform_rms_multiplier
    loss_improve = (baseline["modulation_loss"] - metrics["modulation_loss"]) / (baseline["modulation_loss"] + 1e-8)
    grid_improve = (baseline["grid_energy"] - metrics["grid_energy"]) / (baseline["grid_energy"] + 1e-8)
    rms_penalty = max(0.0, (rms / (baseline_rms + 1e-8)) - 1.0)
    height_penalty = max(0.0, (metrics["height"]["height_std"] / (baseline["height"]["height_std"] + 1e-8)) - 1.0)
    score = 2.0 * loss_improve + grid_improve - 1.5 * rms_penalty - 0.5 * height_penalty
    if not feasible:
        score -= 10.0
    return float(score), bool(feasible)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep A-latent hyperparameters with fixed platform ROI evaluation.")
    parser.add_argument("--config", default="configs/ossim_dataset.yaml")
    parser.add_argument("--sample", default="细台阶")
    parser.add_argument("--groups", default="ossim_t6_h_3step,ossim_t6_v_3step")
    parser.add_argument("--exclude-samples", default="台阶曝光1")
    parser.add_argument("--crop", default="center:256")
    parser.add_argument("--platform-low-mask", default="outputs/height/baseline_t6_hv/platform_low_mask.npy")
    parser.add_argument("--platform-high-mask", default="outputs/height/baseline_t6_hv/platform_high_mask.npy")
    parser.add_argument("--max-platform-rms-multiplier", type=float, default=1.25)
    parser.add_argument("--output", default="outputs/latent_sweep/t6_hv")
    args = parser.parse_args()

    config = data_io.load_config(args.config)
    excluded = data_io.parse_excluded_samples(args.exclude_samples)
    groups = data_io.parse_groups(args.groups, config)
    crop = data_io.infer_crop_for_sample(config, args.sample, args.crop)
    bundle = data_io.load_os_sim_stack(config, args.sample, groups, crop, excluded_samples=excluded)
    Y = bundle["Y"]
    z_values = bundle["z_values"]
    demod = demodulate_multigroup(Y)
    A_init = demod["A"]
    M, q_infos = build_patterns(Y, bundle["groups"])
    grid_masks, grid_peaks = build_grid_masks(A_init)
    low_mask = np.load(ROOT / args.platform_low_mask).astype(bool)
    high_mask = np.load(ROOT / args.platform_high_mask).astype(bool)

    out_dir = ROOT / args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "z_values.npy", z_values)

    results = []
    best_A = None
    best_score = -float("inf")
    best_name = None
    baseline_metrics = None
    curves: dict[str, list[float]] = {}
    for cand in candidate_grid():
        name = str(cand["name"])
        if int(cand["iterations"]) == 0:
            A = A_init.copy()
            curves[name] = [modulation_loss(Y, A, M)]
        else:
            A, losses = optimize_A(
                Y,
                A_init,
                M,
                grid_masks,
                iterations=int(cand["iterations"]),
                lr=float(cand["lr"]),
                lambda_sec=float(cand["lambda_sec"]),
                lambda_smooth=float(cand["lambda_smooth"]),
                lambda_hv=float(cand["lambda_hv"]),
                lambda_grid=float(cand["lambda_grid"]),
            )
            curves[name] = losses
        metrics = evaluate_candidate(name, A, Y, M, grid_masks, z_values, low_mask, high_mask)
        metrics["params"] = cand
        if baseline_metrics is None:
            baseline_metrics = metrics
        score, feasible = score_candidate(metrics, baseline_metrics, args.max_platform_rms_multiplier)
        metrics["score"] = score
        metrics["feasible_platform_rms"] = feasible
        results.append(metrics)
        if score > best_score:
            best_score = score
            best_name = name
            best_A = A
        print(f"{name}: score={score:.4g}, feasible={feasible}, loss={metrics['modulation_loss']:.6g}, platform_rms={metrics['platform'].get('pooled_platform_rms')}")

    if best_A is None:
        raise RuntimeError("No sweep candidate was evaluated")
    A_h = best_A[:, 0]
    A_v = best_A[:, 1] if best_A.shape[1] > 1 else best_A[:, 0]
    A_fused = fuse_hv(A_h, A_v)
    height = height_parabolic(A_fused, z_values)
    np.save(out_dir / "A_h_best_stack.npy", A_h)
    np.save(out_dir / "A_v_best_stack.npy", A_v)
    np.save(out_dir / "A_fused_best_stack.npy", A_fused)
    np.save(out_dir / "height_best_peakfit.npy", height)
    save_scalar_image(height, out_dir / "height_best_map.png", f"Best sweep height: {best_name}")
    mid = A_fused.shape[0] // 2
    baseline_A_fused = fuse_hv(A_init[:, 0], A_init[:, 1] if A_init.shape[1] > 1 else A_init[:, 0])
    save_montage(
        [baseline_A_fused[mid], A_fused[mid], height],
        out_dir / "best_candidate_summary.png",
        ["baseline A fused", f"best A fused: {best_name}", "best height"],
        cols=3,
        cmap="viridis",
    )
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    names = [r["name"] for r in results]
    axes[0].bar(names, [r["modulation_loss"] for r in results])
    axes[0].set_title("Modulation loss")
    axes[1].bar(names, [r["grid_energy"] for r in results])
    axes[1].set_title("A grid energy")
    axes[2].bar(names, [r["platform"].get("pooled_platform_rms", np.nan) for r in results])
    axes[2].set_title("Fixed-ROI platform RMS")
    for ax in axes:
        ax.tick_params(axis="x", labelrotation=35)
        ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "sweep_metrics.png", dpi=150)
    plt.close(fig)

    output = {
        "sample": args.sample,
        "groups": groups,
        "crop": crop.label if crop else "full",
        "platform_low_mask": args.platform_low_mask,
        "platform_high_mask": args.platform_high_mask,
        "max_platform_rms_multiplier": args.max_platform_rms_multiplier,
        "best_name": best_name,
        "best_score": best_score,
        "q_infos": q_infos,
        "grid_peaks": grid_peaks,
        "results": results,
        "interpretation": "A candidate is feasible only if fixed-ROI platform RMS stays within the configured multiplier of the baseline RMS. This prevents accepting lower modulation/grid loss when platform flatness collapses.",
    }
    data_io.write_json(out_dir / "sweep_metrics.json", output)
    data_io.write_json(out_dir / "config.json", vars(args))
    report = [
        "# Latent Hyperparameter Sweep Report",
        "",
        f"- Sample: `{args.sample}`",
        f"- Crop: `{crop.label if crop else 'full'}`",
        f"- Best candidate by constrained score: `{best_name}`",
        "",
        "| candidate | feasible | score | mod loss | grid energy | platform RMS |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        report.append(
            f"| {r['name']} | {r['feasible_platform_rms']} | {r['score']:.6g} | {r['modulation_loss']:.6g} | {r['grid_energy']:.6g} | {r['platform'].get('pooled_platform_rms', float('nan')):.6g} |"
        )
    (out_dir / "sweep_report.md").write_text("\n".join(report), encoding="utf-8")
    data_io.append_report(
        ROOT / "outputs/nightly_report.md",
        "P5c Platform-Constrained Latent Sweep",
        [
            f"- Ran {len(results)} latent hyperparameter candidates for `{args.sample}` on `{crop.label if crop else 'full'}`.",
            f"- Best candidate by constrained score: `{best_name}`.",
            f"- Outputs saved under `{out_dir}`.",
        ],
    )
    print(f"Wrote latent sweep outputs to {out_dir}")


if __name__ == "__main__":
    main()
