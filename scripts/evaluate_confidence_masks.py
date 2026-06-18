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
from src.confidence import REASON_LABELS, axial_confidence_maps
from src.forward_models import normalize_m
from src.metrics_v1 import modulation_residual, project_a_nonnegative
from src.os_sim import demodulate_multigroup, fuse_hv
from src.visualization import save_montage, save_scalar_image

from optimize_modulation_latents import build_patterns_by_source
from recompute_all_metrics_v1 import choose_sample, phase_aligned_m


def reason_summary(reason_map: np.ndarray) -> dict[str, Any]:
    total = int(reason_map.size)
    out = {}
    for code, label in REASON_LABELS.items():
        count = int(np.sum(reason_map == code))
        out[str(code)] = {"label": label, "count": count, "fraction": float(count / max(total, 1))}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate v1 confidence, invalid, and reason maps.")
    parser.add_argument("--config", default="configs/ossim_dataset.yaml")
    parser.add_argument("--sample", default=None)
    parser.add_argument("--groups", default="ossim_t6_h_3step,ossim_t6_v_3step")
    parser.add_argument("--exclude-samples", default="鍙伴樁鏇濆厜1")
    parser.add_argument("--crop", default="center:256")
    parser.add_argument("--pattern-source", choices=["auto", "cos_fft", "measured_calibrated"], default="auto")
    parser.add_argument("--projection-sequence", default="saomiao3_6")
    parser.add_argument("--calibration-label", default="3-6")
    parser.add_argument("--calibration-mat", default="DMD_CCD_Calibration_Dict.mat")
    parser.add_argument("--phase-order-metrics", default="outputs/v1_next/phase_order/metrics.json")
    parser.add_argument("--output", default="outputs/v1_next/confidence")
    args = parser.parse_args()

    config = data_io.load_config(args.config)
    sample = choose_sample(config, args.sample)
    excluded = data_io.parse_excluded_samples(args.exclude_samples)
    groups = data_io.parse_groups(args.groups, config)
    crop = data_io.infer_crop_for_sample(config, sample, args.crop)
    bundle = data_io.load_os_sim_stack(config, sample, groups, crop, excluded_samples=excluded)
    Y = bundle["Y"].astype(np.float32)
    z_values = bundle["z_values"].astype(np.float32)
    demod = demodulate_multigroup(Y)
    A = demod["A"].astype(np.float32)
    A_h = A[:, 0]
    A_v = A[:, 1] if A.shape[1] > 1 else A[:, 0]
    A_fused = fuse_hv(A_h, A_v)

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
    M = normalize_m(M.astype(np.float32))
    M_phase, phase_info = phase_aligned_m(M, groups, ROOT / args.phase_order_metrics)
    A_ls = project_a_nonnegative(Y, M_phase)
    residual = modulation_residual(Y, A_ls, M_phase)
    residual_energy_map = np.sqrt(np.mean(residual * residual, axis=(0, 1, 2))).astype(np.float32)

    maps = axial_confidence_maps(
        A_fused,
        z_values,
        A_h=A_h,
        A_v=A_v,
        residual_energy_map=residual_energy_map,
        raw_stack=Y,
    )

    out_dir = ROOT / args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "confidence.npy", maps["confidence"])
    np.save(out_dir / "invalid_mask.npy", maps["invalid_mask"])
    np.save(out_dir / "reason_map.npy", maps["reason_map"])
    np.save(out_dir / "residual_energy_map.npy", residual_energy_map)

    save_montage(
        [
            maps["confidence"],
            maps["peak_to_background"],
            maps["sharpness"],
            maps["fwhm_layers"],
            maps["softmax_entropy"],
            residual_energy_map,
        ],
        out_dir / "confidence_maps.png",
        ["confidence", "peak/background", "sharpness", "FWHM layers", "entropy", "residual RMS"],
        cols=3,
        cmap="magma",
    )
    save_montage(
        [maps["invalid_mask"].astype(np.float32), maps["number_of_peaks"].astype(np.float32), maps["hv_peak_z_difference"]],
        out_dir / "invalid_masks.png",
        ["invalid mask", "number of peaks", "H/V peak z diff"],
        cols=3,
        cmap="magma",
    )
    save_scalar_image(maps["reason_map"].astype(np.float32), out_dir / "reason_map.png", "Reason map", cmap="tab10")

    summary = reason_summary(maps["reason_map"])
    metrics = {
        "sample": sample,
        "groups": groups,
        "crop": crop.label if crop else "full",
        "pattern_infos": pattern_infos,
        "phase_alignment": phase_info,
        "reason_labels": {str(k): v for k, v in REASON_LABELS.items()},
        "summary": summary,
        "global": {
            "invalid_fraction": float(np.mean(maps["invalid_mask"])),
            "mean_confidence": float(np.mean(maps["confidence"])),
            "median_confidence": float(np.median(maps["confidence"])),
            "mean_peak_to_background": float(np.mean(maps["peak_to_background"])),
            "mean_sharpness": float(np.mean(maps["sharpness"])),
            "mean_fwhm_layers": float(np.mean(maps["fwhm_layers"])),
            "mean_entropy": float(np.mean(maps["softmax_entropy"])),
            "mean_hv_peak_z_difference": float(np.mean(maps["hv_peak_z_difference"])),
            "mean_residual_energy": float(np.mean(residual_energy_map)),
        },
        "interpretation": "Invalid masks are diagnostic exclusions for internal consistency/metrology summaries; they are not independent truth labels.",
    }
    data_io.write_json(out_dir / "metrics.json", metrics)
    data_io.write_json(out_dir / "confidence_metrics.json", metrics)

    dominant = sorted(summary.items(), key=lambda item: item[1]["fraction"], reverse=True)[:4]
    report = [
        "# Confidence and Invalid Mask Report",
        "",
        f"- Sample: `{sample}`",
        f"- Groups: `{', '.join(groups)}`",
        f"- Crop: `{crop.label if crop else 'full'}`",
        "",
        "## Required Answers",
        "",
        f"- Invalid fraction: `{metrics['global']['invalid_fraction']:.6g}`.",
        f"- Mean confidence: `{metrics['global']['mean_confidence']:.6g}`.",
        f"- Mean H/V peak-z difference: `{metrics['global']['mean_hv_peak_z_difference']:.6g}`.",
        "- Dominant reason classes:",
    ]
    for code, row in dominant:
        report.append(f"  - `{code}` `{row['label']}`: `{row['fraction']:.6g}`")
    report += [
        "",
        "## Reason Fractions",
        "",
        "| code | label | fraction | count |",
        "|---:|---|---:|---:|",
    ]
    for code, row in summary.items():
        report.append(f"| {code} | {row['label']} | {row['fraction']:.6g} | {row['count']} |")
    report += [
        "",
        "Reason priority is exclusive: later checks can overwrite earlier classes, so the map is useful for screening but not for causal proof.",
    ]
    (out_dir / "confidence_report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote confidence outputs to {out_dir}")


if __name__ == "__main__":
    main()
