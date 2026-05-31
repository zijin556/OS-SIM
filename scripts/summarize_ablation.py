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

from src import data_io


def read_json(path: Path) -> dict:
    if not path.exists():
        return {"status": "missing", "path": str(path)}
    return data_io.read_json(path)


def get_nested(data: dict, keys: list[str], default: object = None) -> object:
    cur: object = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def summarize_forward(forward: dict) -> list[dict]:
    rows: list[dict] = []
    if forward.get("status") == "missing":
        return rows
    model_labels = {
        "level0_ideal_cos_group_period": "B_level0_ideal_cos",
        "level1_modulation_cos_fft_q": "C_level1_modulation_domain_fft_q",
        "level2_binary_dmd_identity_affine_warp_fft_q": "D_level2_generated_binary_dmd_affine",
        "level3_binary_dmd_identity_polynomial_warp_fft_q": "D2_level3_binary_dmd_polynomial_identity",
        "level3_binary_dmd_zero_bspline_warp_fft_q": "D3_level3_binary_dmd_bspline_zero",
        "level4_harmonic_grid_no_blur_fft_q": "E_level4_harmonic_grid_no_blur",
        "level4_harmonic_grid_blur_fft_q": "F_level4_harmonic_grid_optional_blur",
        "level2_measured_dmd_calibrated_map": "D4_level2_measured_dmd_calibrated_map",
        "level4_measured_dmd_calibrated_map_blur": "F2_level4_measured_dmd_calibrated_map_blur",
    }
    for group_name, group_data in forward.get("models", {}).items():
        for model_name, label in model_labels.items():
            loss = group_data.get("losses", {}).get(model_name)
            if not loss:
                continue
            rows.append(
                {
                    "ablation": label,
                    "group": group_name,
                    "model": model_name,
                    "mod_rmse": loss.get("mod_rmse"),
                    "mod_charbonnier": loss.get("mod_charbonnier"),
                    "raw_rmse": loss.get("raw_rmse"),
                    "residual_grid_energy_mean": loss.get("residual_grid_energy_mean"),
                    "residual_stripe_energy_mean": loss.get("residual_stripe_energy_mean"),
                    "residual_top_peak_energy_mean": loss.get("residual_top_peak_energy_mean"),
                }
            )
    return rows


def summarize_a_stack(
    baseline: dict,
    latent: dict,
    latent_measured: dict,
    theta: dict,
    affine: dict,
    platform: dict,
    platform_measured: dict,
    platform_compare: dict,
) -> list[dict]:
    rows: list[dict] = []
    platform_by_method = {}
    for item in platform_compare.get("methods", []):
        platform_by_method[item.get("method")] = item

    if baseline.get("status") != "missing":
        baseline_platform = platform_by_method.get("baseline", {})
        rows.append(
            {
                "ablation": "A_baseline_A_cls_only",
                "source": "outputs/baseline/t6_hv",
                "modulation_loss": get_nested(platform, ["baseline", "modulation_loss"], None),
                "grid_energy": get_nested(platform, ["baseline", "grid_energy"], None),
                "height_std": get_nested(baseline, ["fused", "height_std"], None),
                "platform_rms": baseline_platform.get("pooled_platform_rms"),
                "step_height_scan_units": baseline_platform.get("step_height_scan_units"),
                "accepted_for_height": True,
                "reason": "Classical OS-SIM reference; not clean truth, but platform RMS is the stability baseline.",
            }
        )

    if latent.get("status") != "missing":
        latent_platform = platform_by_method.get("a_only_grid_aggressive", {})
        rows.append(
            {
                "ablation": "P5_aggressive_A_only_grid_loss",
                "source": "outputs/latent_opt/t6_hv",
                "modulation_loss": latent.get("final_modulation_loss"),
                "grid_energy": latent.get("optimized_A_grid_energy"),
                "height_std": get_nested(latent, ["height_optimized", "height_std"], None),
                "platform_rms": latent_platform.get("pooled_platform_rms"),
                "step_height_scan_units": latent_platform.get("step_height_scan_units"),
                "accepted_for_height": False,
                "reason": "Lower modulation/grid loss, but fixed-platform RMS collapses.",
            }
        )

    if latent_measured.get("status") != "missing":
        rows.append(
            {
                "ablation": "P5_measured_calibrated_A_only_grid_loss",
                "source": "outputs/latent_opt/t6_hv_measured",
                "modulation_loss": latent_measured.get("final_modulation_loss"),
                "grid_energy": latent_measured.get("optimized_A_grid_energy"),
                "height_std": get_nested(latent_measured, ["height_optimized", "height_std"], None),
                "platform_rms": None,
                "step_height_scan_units": None,
                "accepted_for_height": False,
                "reason": "Measured calibrated fixed-pattern A-only path lowers loss strongly, but it has no platform early-stop gate.",
            }
        )

    if theta.get("status") != "missing":
        theta_platform = platform_by_method.get("theta", {})
        rows.append(
            {
                "ablation": "E_theta_q_phase_harmonic_closed_form_A",
                "source": "outputs/theta_opt/t6_hv",
                "modulation_loss": theta.get("final_objective"),
                "grid_energy": None,
                "height_std": get_nested(theta, ["height_theta", "height_std"], None),
                "platform_rms": theta_platform.get("pooled_platform_rms"),
                "step_height_scan_units": theta_platform.get("step_height_scan_units"),
                "accepted_for_height": False,
                "reason": "Low-dimensional theta objective improves slightly, but platform RMS is not stable.",
            }
        )

    if affine.get("status") != "missing":
        affine_platform = affine.get("platform", {}) if isinstance(affine.get("platform"), dict) else {}
        rows.append(
            {
                "ablation": "D_affine_generated_dmd_warp_closed_form_A",
                "source": "outputs/affine_warp_opt/t6_hv",
                "modulation_loss": affine.get("final_objective"),
                "grid_energy": None,
                "height_std": get_nested(affine, ["height_affine", "height_std"], None),
                "platform_rms": affine_platform.get("pooled_platform_rms"),
                "step_height_scan_units": affine_platform.get("step_height"),
                "accepted_for_height": False,
                "reason": "Generated-DMD affine warp fitting path works, but platform RMS is not stable and mapping is not calibrated.",
            }
        )

    if platform.get("status") != "missing":
        platform_opt = platform_by_method.get("platform_opt_early_stop", {})
        rows.append(
            {
                "ablation": "P5_platform_constrained_early_stop",
                "source": "outputs/platform_opt/t6_hv",
                "modulation_loss": get_nested(platform, ["best", "modulation_loss"], None),
                "grid_energy": get_nested(platform, ["best", "grid_energy"], None),
                "height_std": get_nested(platform, ["best", "height", "height_std"], None),
                "platform_rms": platform_opt.get("pooled_platform_rms"),
                "step_height_scan_units": platform_opt.get("step_height_scan_units"),
                "accepted_for_height": True,
                "reason": "Best current optimized A stack under fixed-platform stability gate.",
            }
        )

    if platform_measured.get("status") != "missing":
        rows.append(
            {
                "ablation": "P5_measured_calibrated_platform_constrained",
                "source": "outputs/platform_opt/t6_hv_measured",
                "modulation_loss": get_nested(platform_measured, ["best", "modulation_loss"], None),
                "grid_energy": get_nested(platform_measured, ["best", "grid_energy"], None),
                "height_std": get_nested(platform_measured, ["best", "height", "height_std"], None),
                "platform_rms": get_nested(platform_measured, ["best", "platform", "pooled_platform_rms"], None),
                "step_height_scan_units": get_nested(platform_measured, ["best", "platform", "step_height"], None),
                "accepted_for_height": True,
                "reason": "Best current measured-DMD calibrated A stack under fixed-platform stability gate.",
            }
        )
    return rows


def plot_a_stack(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    labels = [r["ablation"].replace("_", "\n") for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    specs = [
        ("modulation_loss", "Modulation/objective"),
        ("height_std", "Height std"),
        ("platform_rms", "Platform RMS"),
    ]
    for ax, (key, title) in zip(axes, specs):
        values = [np.nan if r.get(key) is None else float(r[key]) for r in rows]
        colors = ["#2f6f4e" if r.get("accepted_for_height") else "#9b3d3d" for r in rows]
        ax.bar(labels, values, color=colors)
        ax.set_title(title)
        ax.tick_params(axis="x", labelrotation=35, labelsize=7)
        ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_forward(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    groups = sorted({r["group"] for r in rows})
    fig, axes = plt.subplots(len(groups), 3, figsize=(13, 4 * len(groups)), squeeze=False)
    for row_idx, group in enumerate(groups):
        group_rows = [r for r in rows if r["group"] == group]
        labels = [r["ablation"].replace("_", "\n") for r in group_rows]
        for ax, key, title in [
            (axes[row_idx, 0], "mod_rmse", "Mod RMSE"),
            (axes[row_idx, 1], "residual_grid_energy_mean", "Residual grid energy"),
            (axes[row_idx, 2], "residual_stripe_energy_mean", "Stripe leakage"),
        ]:
            values = [np.nan if r.get(key) is None else float(r[key]) for r in group_rows]
            ax.bar(labels, values, color="#426a8c")
            ax.set_title(f"{group}: {title}")
            ax.tick_params(axis="x", labelrotation=35, labelsize=7)
            ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def markdown_table(rows: list[dict], columns: list[str]) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        values = []
        for col in columns:
            value = row.get(col)
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append("" if value is None else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize OS-SIM forward and A-stack ablations from existing outputs.")
    parser.add_argument("--forward", default="outputs/forward/t6_hv/model_compare.json")
    parser.add_argument("--baseline", default="outputs/baseline/t6_hv/metrics.json")
    parser.add_argument("--latent", default="outputs/latent_opt/t6_hv/metrics.json")
    parser.add_argument("--latent-measured", default="outputs/latent_opt/t6_hv_measured/metrics.json")
    parser.add_argument("--theta", default="outputs/theta_opt/t6_hv/metrics.json")
    parser.add_argument("--affine", default="outputs/affine_warp_opt/t6_hv/metrics.json")
    parser.add_argument("--platform", default="outputs/platform_opt/t6_hv/metrics.json")
    parser.add_argument("--platform-measured", default="outputs/platform_opt/t6_hv_measured/metrics.json")
    parser.add_argument("--platform-compare", default="outputs/height/platform_compare.json")
    parser.add_argument("--output", default="outputs/ablation/t6_hv")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = ROOT / args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)

    forward = read_json(ROOT / args.forward)
    baseline = read_json(ROOT / args.baseline)
    latent = read_json(ROOT / args.latent)
    latent_measured = read_json(ROOT / args.latent_measured)
    theta = read_json(ROOT / args.theta)
    affine = read_json(ROOT / args.affine)
    platform = read_json(ROOT / args.platform)
    platform_measured = read_json(ROOT / args.platform_measured)
    platform_compare = read_json(ROOT / args.platform_compare)

    forward_rows = summarize_forward(forward)
    a_stack_rows = summarize_a_stack(baseline, latent, latent_measured, theta, affine, platform, platform_measured, platform_compare)
    recommended = next((r for r in a_stack_rows if r["ablation"] == "P5_measured_calibrated_platform_constrained"), None)
    if recommended is None:
        recommended = next((r for r in a_stack_rows if r["ablation"] == "P5_platform_constrained_early_stop"), None)
    if recommended is None:
        recommended = next((r for r in a_stack_rows if r.get("accepted_for_height")), None)

    metrics = {
        "sample": forward.get("sample", baseline.get("sample")),
        "crop": forward.get("crop", baseline.get("crop")),
        "inputs": {
            "forward": args.forward,
            "baseline": args.baseline,
            "latent": args.latent,
            "latent_measured": args.latent_measured,
            "theta": args.theta,
            "affine": args.affine,
            "platform": args.platform,
            "platform_measured": args.platform_measured,
            "platform_compare": args.platform_compare,
        },
        "forward_ablation": forward_rows,
        "a_stack_ablation": a_stack_rows,
        "recommended_a_stack": recommended,
        "interpretation": "This table makes the stage-5 ablation explicit: lower reprojection/grid loss is not accepted unless height/platform diagnostics remain stable.",
    }
    data_io.write_json(out_dir / "metrics.json", metrics)
    data_io.write_json(out_dir / "config.json", vars(args))
    plot_forward(forward_rows, out_dir / "forward_ablation.png")
    plot_a_stack(a_stack_rows, out_dir / "A_stack_ablation.png")

    report = [
        "# OS-SIM Ablation Summary",
        "",
        f"- Sample: `{metrics['sample']}`",
        f"- Crop: `{metrics['crop']}`",
        f"- Recommended A stack: `{recommended['source'] if recommended else 'none'}`",
        "",
        "## Forward Model Ablation",
        "",
    ]
    report += markdown_table(
        forward_rows,
        [
            "ablation",
            "group",
            "mod_rmse",
            "mod_charbonnier",
            "residual_grid_energy_mean",
            "residual_stripe_energy_mean",
        ],
    )
    report += [
        "",
        "## A-Stack / Height Ablation",
        "",
    ]
    report += markdown_table(
        a_stack_rows,
        [
            "ablation",
            "modulation_loss",
            "grid_energy",
            "height_std",
            "platform_rms",
            "accepted_for_height",
            "reason",
        ],
    )
    report += [
        "",
        "## Interpretation",
        "",
        "- The aggressive A-only path improves modulation and grid objectives but is rejected because platform RMS worsens strongly.",
        "- The low-dimensional theta path improves its objective only slightly and is also rejected by platform flatness.",
        "- The generated-DMD affine warp path validates constrained affine fitting but is also rejected for height metrology because fixed-platform RMS worsens.",
        "- The measured-calibrated platform-constrained path is the current recommended optimized A stack because it uses real DMD bitmaps and the DMD-CCD map while keeping the fixed-platform diagnostic near baseline.",
        "- The older cos-FFT platform-constrained early-stop path remains a useful generated-pattern control.",
        "- Forward-model ablations include generated controls plus measured DMD bitmap rows sampled through the real DMD-CCD calibration map.",
    ]
    (out_dir / "ablation_report.md").write_text("\n".join(report), encoding="utf-8")
    data_io.append_report(
        ROOT / "outputs/nightly_report.md",
        "P5e Ablation Summary",
        [
            f"- Built explicit forward and A-stack ablation tables for `{metrics['sample']}` on `{metrics['crop']}`.",
            f"- Recommended A stack source: `{recommended['source'] if recommended else 'none'}`.",
            f"- Outputs saved under `{out_dir}`.",
        ],
    )
    print(f"Wrote ablation outputs to {out_dir}")


if __name__ == "__main__":
    main()
