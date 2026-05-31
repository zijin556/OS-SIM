from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src import calibration, data_io
from src.fft_analysis import circular_peak_mask, detect_frequency_peaks, masked_fft_energy
from src.forward_models import normalize_m
from src.height import auto_two_platform_metrics, confidence_metrics, height_parabolic, summarize_height
from src.losses import charbonnier
from src.os_sim import demodulate_multigroup, fuse_hv
from src.visualization import save_montage, save_profiles, save_scalar_image


def choose_complete_samples(manifest: dict[str, Any], requested: str | None) -> list[str]:
    if requested:
        return [name.strip() for name in requested.split(",") if name.strip()]
    samples = []
    for row in manifest.get("samples", []):
        if row.get("excluded"):
            continue
        if row.get("incomplete_layer_count") == 0:
            samples.append(row["name"])
    return samples


def modulation_metrics(Y: np.ndarray, A: np.ndarray, M: np.ndarray) -> dict[str, Any]:
    y_tilde = Y - Y.mean(axis=2, keepdims=True)
    pred = A[:, :, None, :, :] * M[None, :, :, :, :]
    residual = y_tilde - pred
    group_rows = []
    for g in range(Y.shape[1]):
        rg = residual[:, g]
        group_rows.append(
            {
                "group_index": g,
                "mod_rmse": float(np.sqrt(np.mean(rg * rg))),
                "mod_charbonnier": charbonnier(rg),
                "residual_mean_abs": float(np.mean(np.abs(rg))),
            }
        )
    return {
        "mod_rmse": float(np.sqrt(np.mean(residual * residual))),
        "mod_charbonnier": charbonnier(residual),
        "residual_mean_abs": float(np.mean(np.abs(residual))),
        "groups": group_rows,
    }


def a_grid_energy(A_stack: np.ndarray) -> tuple[float, list[dict[str, float]]]:
    mid = A_stack.shape[0] // 2
    peaks = detect_frequency_peaks(A_stack[mid], top_n=4, min_radius=4)
    if not peaks:
        return 0.0, []
    mask = circular_peak_mask(A_stack.shape[-2:], peaks, radius=4)
    vals = [masked_fft_energy(A_stack[k], mask) for k in range(A_stack.shape[0])]
    return float(np.mean(vals)), peaks


def plot_summary(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    labels = [r["sample_id"] for r in rows]
    specs = [
        ("mod_rmse_measured_calibrated", "Measured-DMD mod RMSE"),
        ("height_std", "Height std"),
        ("confidence_sharpness_mean", "Sharpness"),
        ("a_fused_grid_energy", "A_fused grid energy"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
    for ax, (key, title) in zip(axes.ravel(), specs):
        values = [float(r.get(key, np.nan)) for r in rows]
        ax.bar(labels, values, color="#426a8c")
        ax.set_title(title)
        ax.tick_params(axis="x", labelrotation=25)
        ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def markdown_table(rows: list[dict[str, Any]]) -> list[str]:
    columns = [
        "sample_id",
        "sample",
        "z_layers",
        "mod_rmse_measured_calibrated",
        "height_std",
        "confidence_sharpness_mean",
        "a_fused_grid_energy",
        "platform_status",
        "platform_rms",
        "step_height_scan_units",
    ]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        vals = []
        for col in columns:
            val = row.get(col)
            if isinstance(val, float):
                vals.append(f"{val:.6g}")
            elif val is None:
                vals.append("")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare all complete non-excluded samples under the same T6 H/V measured-calibrated pipeline.")
    parser.add_argument("--config", default="configs/ossim_dataset.yaml")
    parser.add_argument("--samples", default=None)
    parser.add_argument("--groups", default="ossim_t6_h_3step,ossim_t6_v_3step")
    parser.add_argument("--exclude-samples", default="台阶曝光1")
    parser.add_argument("--crop", default="center:256")
    parser.add_argument("--projection-sequence", default="saomiao3_6")
    parser.add_argument("--calibration-label", default="3-6")
    parser.add_argument("--calibration-mat", default="DMD_CCD_Calibration_Dict.mat")
    parser.add_argument("--output", default="outputs/complete_samples/t6_hv_measured")
    args = parser.parse_args()

    config = data_io.load_config(args.config)
    excluded = data_io.parse_excluded_samples(args.exclude_samples)
    manifest = data_io.scan_dataset(config, excluded_samples=excluded)
    groups = data_io.parse_groups(args.groups, config)
    samples = choose_complete_samples(manifest, args.samples)
    if not samples:
        raise ValueError("No complete non-excluded samples selected")

    out_dir = ROOT / args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    montage_images = []
    montage_titles = []
    first_crop = None
    measured_M = None
    pattern_infos = None

    for sample_index, sample in enumerate(samples, start=1):
        sample_id = f"S{sample_index}"
        crop = data_io.infer_crop_for_sample(config, sample, args.crop)
        if first_crop is None:
            first_crop = crop
            measured_patterns = []
            pattern_infos = []
            for group_name in groups:
                M_g, info = calibration.measured_camera_patterns_for_group(
                    config,
                    group_name,
                    crop=crop,
                    sequence_stem=args.projection_sequence,
                    calibration_label_prefix=args.calibration_label,
                    mat_name=args.calibration_mat,
                )
                measured_patterns.append(M_g)
                pattern_infos.append(info)
            measured_M = normalize_m(np.stack(measured_patterns, axis=0))
        elif (crop.label if crop else "full") != (first_crop.label if first_crop else "full"):
            raise ValueError("All samples must use the same crop for this comparison")
        assert measured_M is not None

        bundle = data_io.load_os_sim_stack(config, sample, groups, crop, excluded_samples=excluded)
        Y = bundle["Y"]
        z_values = bundle["z_values"]
        demod = demodulate_multigroup(Y)
        A = demod["A"]
        A_h = A[:, 0]
        A_v = A[:, 1] if A.shape[1] > 1 else A[:, 0]
        A_fused = fuse_hv(A_h, A_v)
        height = height_parabolic(A_fused, z_values)
        conf = confidence_metrics(A_fused)
        height_summary = summarize_height(height, conf)
        mod = modulation_metrics(Y, A, measured_M)
        grid_energy, grid_peaks = a_grid_energy(A_fused)
        platform = auto_two_platform_metrics(height, invalid_mask=conf["invalid_mask"], erode_px=5)
        platform_json = {k: v for k, v in platform.items() if k not in {"low_mask", "high_mask"}}

        sample_dir = out_dir / sample
        sample_dir.mkdir(parents=True, exist_ok=True)
        np.save(sample_dir / "z_values.npy", z_values)
        np.save(sample_dir / "A_h_stack.npy", A_h)
        np.save(sample_dir / "A_v_stack.npy", A_v)
        np.save(sample_dir / "A_fused_stack.npy", A_fused)
        np.save(sample_dir / "height_parabolic.npy", height)
        np.save(sample_dir / "confidence_sharpness.npy", conf["sharpness"])
        save_montage([A_h[A_h.shape[0] // 2], A_v[A_v.shape[0] // 2], A_fused[A_fused.shape[0] // 2], height], sample_dir / "summary_montage.png", ["A H mid", "A V mid", "A fused mid", "height"], cols=2, cmap="viridis")
        save_scalar_image(height, sample_dir / "height_map.png", f"{sample_id} height")
        save_profiles(A_fused, z_values, sample_dir / "profiles.png", f"{sample_id} A_fused profiles")
        data_io.write_json(
            sample_dir / "metrics.json",
            {
                "sample": sample,
                "sample_id": sample_id,
                "groups": groups,
                "crop": crop.label if crop else "full",
                "z_layers": int(Y.shape[0]),
                "z_min": float(z_values.min()),
                "z_max": float(z_values.max()),
                "measured_calibrated_modulation": mod,
                "height": height_summary,
                "a_fused_grid_energy": grid_energy,
                "a_fused_grid_peaks": grid_peaks,
                "platform": platform_json,
            },
        )

        row = {
            "sample": sample,
            "sample_id": sample_id,
            "z_layers": int(Y.shape[0]),
            "z_min": float(z_values.min()),
            "z_max": float(z_values.max()),
            "mod_rmse_measured_calibrated": mod["mod_rmse"],
            "mod_charbonnier_measured_calibrated": mod["mod_charbonnier"],
            "a_fused_grid_energy": grid_energy,
            "platform_status": platform.get("status"),
            "platform_rms": platform.get("pooled_platform_rms") if platform.get("status") == "ok" else None,
            "step_height_scan_units": platform.get("step_height") if platform.get("status") == "ok" else None,
            **height_summary,
        }
        rows.append(row)
        montage_images.extend([A_fused[A_fused.shape[0] // 2], height])
        montage_titles.extend([f"{sample_id} A", f"{sample_id} h"])

    data_io.write_json(
        out_dir / "metrics.json",
        {
            "samples": samples,
            "excluded_samples": sorted(excluded),
            "groups": groups,
            "crop": first_crop.label if first_crop else "full",
            "pattern_source": "measured_calibrated",
            "pattern_infos": pattern_infos,
            "rows": rows,
        },
    )
    data_io.write_json(out_dir / "config.json", vars(args))
    plot_summary(rows, out_dir / "sample_metric_summary.png")
    save_montage(montage_images, out_dir / "sample_montage.png", montage_titles, cols=2, cmap="viridis")
    report = [
        "# Complete Sample Comparison",
        "",
        f"- Samples: `{', '.join(samples)}`",
        f"- Excluded samples: `{', '.join(sorted(excluded))}`",
        f"- Groups: `{', '.join(groups)}`",
        f"- Crop: `{first_crop.label if first_crop else 'full'}`",
        "- Pattern source: `measured_calibrated`",
        "",
        "## Metrics",
        "",
    ]
    report += markdown_table(rows)
    report += [
        "",
        "## Notes",
        "",
        "- `platform_*` metrics are automatically estimated from the height distribution and are meaningful mainly for step-like samples.",
        "- Height values are still in unknown scan-coordinate units.",
    ]
    (out_dir / "complete_samples_report.md").write_text("\n".join(report), encoding="utf-8")
    data_io.append_report(
        ROOT / "outputs/nightly_report.md",
        "P11 Complete Sample Comparison",
        [
            f"- Compared complete non-excluded samples `{', '.join(samples)}` using T6 H/V and measured-calibrated patterns.",
            f"- Outputs saved under `{out_dir}`.",
        ],
    )
    print(f"Wrote complete-sample comparison outputs to {out_dir}")


if __name__ == "__main__":
    main()
