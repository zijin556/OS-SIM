from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src import data_io
from src.fft_analysis import fft_magnitude
from src.height import confidence_metrics, height_parabolic, platform_metrics_from_masks, summarize_height
from src.os_sim import demodulate_multigroup, fuse_hv
from src.visualization import save_montage, save_scalar_image

from optimize_modulation_latents import build_grid_masks, build_patterns_by_source, grid_energy_mean, modulation_loss


def ls_projection(Y: np.ndarray, M: np.ndarray) -> np.ndarray:
    y_tilde = Y - Y.mean(axis=2, keepdims=True)
    numer = np.sum(y_tilde * M[None, :, :, :, :], axis=2)
    denom = np.sum(M[None, :, :, :, :] ** 2, axis=2) + 1e-8
    return np.maximum(numer / denom, 0.0).astype(np.float32)


def method_metrics(name: str, A: np.ndarray, Y: np.ndarray, M: np.ndarray, z_values: np.ndarray, grid_masks: np.ndarray, low_mask: np.ndarray | None, high_mask: np.ndarray | None) -> dict[str, Any]:
    A_fused = fuse_hv(A[:, 0], A[:, 1] if A.shape[1] > 1 else A[:, 0])
    height = height_parabolic(A_fused, z_values)
    conf = confidence_metrics(A_fused)
    row = {
        "method": name,
        "modulation_loss": modulation_loss(Y, A, M),
        "grid_energy": grid_energy_mean(A, grid_masks),
        "height": summarize_height(height, conf),
    }
    if low_mask is not None and high_mask is not None and low_mask.shape == height.shape:
        platform = platform_metrics_from_masks(height, low_mask, high_mask)
        row["platform"] = {k: v for k, v in platform.items() if k not in {"low_mask", "high_mask"}}
    return row


def optional_mask(path_text: str | None) -> np.ndarray | None:
    if not path_text:
        return None
    path = ROOT / path_text
    if not path.exists():
        return None
    return np.load(path).astype(bool)


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate A by closed-form LS projection with fixed M_theta.")
    parser.add_argument("--config", default="configs/ossim_dataset.yaml")
    parser.add_argument("--sample", default="细台阶")
    parser.add_argument("--groups", default="ossim_t6_h_3step,ossim_t6_v_3step")
    parser.add_argument("--exclude-samples", default="台阶曝光1")
    parser.add_argument("--crop", default="center:256")
    parser.add_argument("--pattern-source", choices=["auto", "cos_fft", "measured_calibrated"], default="auto")
    parser.add_argument("--projection-sequence", default="saomiao3_6")
    parser.add_argument("--calibration-label", default="3-6")
    parser.add_argument("--calibration-mat", default="DMD_CCD_Calibration_Dict.mat")
    parser.add_argument("--platform-low-mask", default=None)
    parser.add_argument("--platform-high-mask", default=None)
    parser.add_argument("--output", default="outputs/ls_projection")
    args = parser.parse_args()

    config = data_io.load_config(args.config)
    excluded = data_io.parse_excluded_samples(args.exclude_samples)
    groups = data_io.parse_groups(args.groups, config)
    crop = data_io.infer_crop_for_sample(config, args.sample, args.crop)
    bundle = data_io.load_os_sim_stack(config, args.sample, groups, crop, excluded_samples=excluded)
    Y = bundle["Y"].astype(np.float32)
    z_values = bundle["z_values"].astype(np.float32)
    demod = demodulate_multigroup(Y)
    A_cls = demod["A"].astype(np.float32)
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
    M = M.astype(np.float32)
    actual_source = pattern_infos[0].get("source", args.pattern_source) if pattern_infos else args.pattern_source
    A_ls = ls_projection(Y, M)
    A_cls_fused = fuse_hv(A_cls[:, 0], A_cls[:, 1] if A_cls.shape[1] > 1 else A_cls[:, 0])
    A_ls_fused = fuse_hv(A_ls[:, 0], A_ls[:, 1] if A_ls.shape[1] > 1 else A_ls[:, 0])
    height_cls = height_parabolic(A_cls_fused, z_values)
    height_ls = height_parabolic(A_ls_fused, z_values)
    grid_masks, grid_peaks = build_grid_masks(A_cls)
    low_mask = optional_mask(args.platform_low_mask)
    high_mask = optional_mask(args.platform_high_mask)

    out_dir = ROOT / args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "z_values.npy", z_values)
    np.save(out_dir / "A_h_ls_stack.npy", A_ls[:, 0])
    np.save(out_dir / "A_v_ls_stack.npy", A_ls[:, 1] if A_ls.shape[1] > 1 else A_ls[:, 0])
    np.save(out_dir / "A_fused_ls_stack.npy", A_ls_fused)
    np.save(out_dir / "A_fused_cls_stack.npy", A_cls_fused)
    np.save(out_dir / "height_ls_peakfit.npy", height_ls)
    np.save(out_dir / "height_cls_peakfit.npy", height_cls)

    y_tilde = Y - Y.mean(axis=2, keepdims=True)
    residual_cls = y_tilde - A_cls[:, :, None, :, :] * M[None, :, :, :, :]
    residual_ls = y_tilde - A_ls[:, :, None, :, :] * M[None, :, :, :, :]
    mid = Y.shape[0] // 2
    save_montage([A_cls_fused[mid], A_ls_fused[mid], A_ls_fused[mid] - A_cls_fused[mid]], out_dir / "A_cls_vs_A_ls.png", ["A_cls", "A_ls", "A_ls - A_cls"], cols=3, cmap="viridis")
    save_montage([height_cls, height_ls, height_ls - height_cls], out_dir / "height_cls_vs_ls.png", ["height cls", "height ls", "delta"], cols=3, cmap="viridis")
    save_montage([fft_magnitude(residual_cls[mid, 0, 0]), fft_magnitude(residual_ls[mid, 0, 0]), fft_magnitude(residual_cls[mid, -1, 0]), fft_magnitude(residual_ls[mid, -1, 0])], out_dir / "residual_fft.png", ["cls H FFT", "ls H FFT", "cls V FFT", "ls V FFT"], cols=2, cmap="magma")
    save_scalar_image(height_ls, out_dir / "height_ls_map.png", "LS-projection height")

    rows = [
        method_metrics("A_cls", A_cls, Y, M, z_values, grid_masks, low_mask, high_mask),
        method_metrics("A_ls_fixed_Mtheta", A_ls, Y, M, z_values, grid_masks, low_mask, high_mask),
    ]
    metrics = {
        "sample": args.sample,
        "groups": groups,
        "crop": crop.label if crop else "full",
        "pattern_source": actual_source,
        "requested_pattern_source": args.pattern_source,
        "pattern_infos": pattern_infos,
        "grid_peaks": grid_peaks,
        "methods": rows,
        "interpretation": "A_ls is a closed-form fixed-M_theta baseline. If it matches or beats network outputs, the network is not yet necessary for the core inverse step.",
    }
    data_io.write_json(out_dir / "metrics.json", metrics)
    data_io.write_json(out_dir / "config.json", vars(args))
    report = [
        "# Fixed-M_theta LS Projection Report",
        "",
        f"- Sample: `{args.sample}`",
        f"- Groups: `{', '.join(groups)}`",
        f"- Crop: `{crop.label if crop else 'full'}`",
        f"- Pattern source: `{actual_source}` (requested `{args.pattern_source}`)",
        "",
        "| method | modulation loss | grid energy | height std | invalid fraction |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        report.append(
            f"| {row['method']} | {row['modulation_loss']:.6g} | {row['grid_energy']:.6g} | {row['height']['height_std']:.6g} | {row['height']['invalid_fraction']:.6g} |"
        )
    report += [
        "",
        "A_cls remains a diagnostic baseline, not ground truth. A_ls tests how far a fixed measured/calibrated pattern can go without a network.",
    ]
    (out_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    data_io.append_report(
        ROOT / "outputs/nightly_report.md",
        "P5 v1 Fixed-M_theta LS Projection",
        [
            f"- Estimated closed-form `A_ls` for `{args.sample}` under `{out_dir}`.",
            f"- A_cls/LS modulation losses: `{rows[0]['modulation_loss']:.6g}` / `{rows[1]['modulation_loss']:.6g}`.",
        ],
    )
    print(f"Wrote LS projection outputs to {out_dir}")


if __name__ == "__main__":
    main()
