from __future__ import annotations

import argparse
import itertools
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
from src.forward_models import normalize_m
from src.height import confidence_metrics, height_parabolic
from src.metrics_v1 import (
    a_grid_energy,
    grid_mask_from_a,
    modulation_loss,
    modulation_residual,
    project_a_nonnegative,
    residual_energy,
    residual_fft_l2,
    stripe_mask_from_raw,
)
from src.os_sim import demodulate_phase_stack
from src.visualization import save_montage

from optimize_modulation_latents import build_patterns_by_source


def choose_sample(config: dict[str, Any], sample: str | None) -> str:
    if sample:
        return sample
    excluded = data_io.parse_excluded_samples(None)
    for row in reversed(config.get("samples", [])):
        name = row.get("name")
        if name and name not in excluded:
            return name
    raise ValueError("No usable sample found in config")


def safe_group_label(name: str) -> str:
    return name.replace("ossim_", "").replace("_3step", "").replace("_6step", "")


def template_variants(M: np.ndarray) -> dict[str, np.ndarray]:
    variants = {"default_template": M}
    variants["contrast_inverted"] = -M
    if M.shape[0] == 3:
        variants["phase_sign_flip"] = M[[0, 2, 1]]
    return {name: normalize_m(arr.astype(np.float32)) for name, arr in variants.items()}


def candidate_score(row: dict[str, Any]) -> float:
    return (
        float(row["modulation_loss"])
        + 0.35 * float(row["residual_grid_energy"])
        + 0.35 * float(row["residual_stripe_leakage"])
        + 0.02 * float(row["height_std"])
        - 0.15 * float(row["peak_sharpness_mean"])
    )


def analyze_group(
    Y_group: np.ndarray,
    M_group: np.ndarray,
    z_values: np.ndarray,
    group_name: str,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    rows: list[dict[str, Any]] = []
    saved_arrays: dict[str, np.ndarray] = {}
    perms = list(itertools.permutations(range(Y_group.shape[1])))
    for perm in perms:
        Y_perm = Y_group[:, perm, :, :]
        A_cls = np.stack([demodulate_phase_stack(Y_perm[k])["A"] for k in range(Y_perm.shape[0])], axis=0).astype(np.float32)
        grid_mask, grid_peaks = grid_mask_from_a(A_cls)
        stripe_mask, stripe_peaks = stripe_mask_from_raw(Y_perm)
        conf = confidence_metrics(A_cls)
        height = height_parabolic(A_cls, z_values)
        for variant_name, M_variant in template_variants(M_group).items():
            A_ls = project_a_nonnegative(Y_perm[:, None, :, :, :], M_variant[None, :, :, :])[:, 0]
            residual = modulation_residual(Y_perm[:, None, :, :, :], A_ls[:, None, :, :], M_variant[None, :, :, :])[:, 0]
            row = {
                "group": group_name,
                "perm": list(perm),
                "variant": variant_name,
                "is_default_order": list(perm) == list(range(Y_group.shape[1])) and variant_name == "default_template",
                "is_cyclic_shift": tuple(perm) in {(0, 1, 2), (1, 2, 0), (2, 0, 1)} if Y_group.shape[1] == 3 else False,
                "modulation_loss": modulation_loss(Y_perm[:, None, :, :, :], A_ls[:, None, :, :], M_variant[None, :, :, :]),
                "A_grid_energy": a_grid_energy(A_cls, grid_mask),
                "residual_grid_energy": residual_energy(residual, grid_mask),
                "residual_stripe_leakage": residual_energy(residual, stripe_mask),
                "residual_fft_l2": residual_fft_l2(residual),
                "peak_sharpness_mean": float(np.mean(conf["sharpness"])),
                "peak_to_background_mean": float(np.mean(conf["peak_to_background"])),
                "fwhm_layers_mean": float(np.mean(conf["fwhm_layers"])),
                "height_std": float(np.std(height)),
                "height_mean": float(np.mean(height)),
                "grid_peaks": grid_peaks,
                "stripe_peaks": stripe_peaks,
            }
            row["score"] = candidate_score(row)
            rows.append(row)
            key = f"{group_name}|{','.join(map(str, perm))}|{variant_name}"
            if row["is_default_order"] or len(saved_arrays) < 6:
                saved_arrays[f"A_cls::{key}"] = A_cls
                saved_arrays[f"A_ls::{key}"] = A_ls
                saved_arrays[f"residual::{key}"] = residual
    rows.sort(key=lambda r: r["score"])
    best = rows[0]
    best_key = f"{group_name}|{','.join(map(str, best['perm']))}|{best['variant']}"
    if f"A_cls::{best_key}" not in saved_arrays:
        Y_best = Y_group[:, best["perm"], :, :]
        M_best = template_variants(M_group)[best["variant"]]
        A_cls_best = np.stack([demodulate_phase_stack(Y_best[k])["A"] for k in range(Y_best.shape[0])], axis=0).astype(np.float32)
        A_ls_best = project_a_nonnegative(Y_best[:, None, :, :, :], M_best[None, :, :, :])[:, 0]
        residual_best = modulation_residual(Y_best[:, None, :, :, :], A_ls_best[:, None, :, :], M_best[None, :, :, :])[:, 0]
        saved_arrays[f"A_cls::{best_key}"] = A_cls_best
        saved_arrays[f"A_ls::{best_key}"] = A_ls_best
        saved_arrays[f"residual::{best_key}"] = residual_best
    return rows, saved_arrays


def stack_for_candidate(Y_group: np.ndarray, M_group: np.ndarray, candidate: dict[str, Any]) -> np.ndarray:
    Y_perm = Y_group[:, candidate["perm"], :, :]
    M_variant = template_variants(M_group)[candidate["variant"]]
    return project_a_nonnegative(Y_perm[:, None, :, :, :], M_variant[None, :, :, :])[:, 0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Enumerate phase order, cyclic shifts, contrast inversion, and phase-sign flips.")
    parser.add_argument("--config", default="configs/ossim_dataset.yaml")
    parser.add_argument("--sample", default=None)
    parser.add_argument("--groups", default="ossim_t6_h_3step,ossim_t6_v_3step")
    parser.add_argument("--exclude-samples", default="鍙伴樁鏇濆厜1")
    parser.add_argument("--crop", default="center:256")
    parser.add_argument("--pattern-source", choices=["auto", "cos_fft", "measured_calibrated"], default="auto")
    parser.add_argument("--projection-sequence", default="saomiao3_6")
    parser.add_argument("--calibration-label", default="3-6")
    parser.add_argument("--calibration-mat", default="DMD_CCD_Calibration_Dict.mat")
    parser.add_argument("--output", default="outputs/v1_next/phase_order")
    args = parser.parse_args()

    config = data_io.load_config(args.config)
    sample = choose_sample(config, args.sample)
    excluded = data_io.parse_excluded_samples(args.exclude_samples)
    groups = data_io.parse_groups(args.groups, config)
    crop = data_io.infer_crop_for_sample(config, sample, args.crop)
    bundle = data_io.load_os_sim_stack(config, sample, groups, crop, excluded_samples=excluded)
    Y = bundle["Y"].astype(np.float32)
    z_values = bundle["z_values"].astype(np.float32)
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

    out_dir = ROOT / args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: dict[str, list[dict[str, Any]]] = {}
    best_by_group: dict[str, dict[str, Any]] = {}
    default_by_group: dict[str, dict[str, Any]] = {}
    arrays: dict[str, np.ndarray] = {}
    for g_idx, group_name in enumerate(groups):
        rows, saved = analyze_group(Y[:, g_idx], M[g_idx], z_values, group_name)
        all_rows[group_name] = rows
        best_by_group[group_name] = rows[0]
        default = [row for row in rows if row["is_default_order"]]
        default_by_group[group_name] = default[0] if default else rows[0]
        arrays.update(saved)

    hv_consistency: dict[str, Any] = {}
    if len(groups) >= 2:
        best_h = stack_for_candidate(Y[:, 0], M[0], best_by_group[groups[0]])
        best_v = stack_for_candidate(Y[:, 1], M[1], best_by_group[groups[1]])
        default_h = stack_for_candidate(Y[:, 0], M[0], default_by_group[groups[0]])
        default_v = stack_for_candidate(Y[:, 1], M[1], default_by_group[groups[1]])
        diff_best = height_parabolic(best_h, z_values) - height_parabolic(best_v, z_values)
        diff_default = height_parabolic(default_h, z_values) - height_parabolic(default_v, z_values)
        hv_consistency = {
            "best_pair_median_abs_height_diff": float(np.median(np.abs(diff_best))),
            "best_pair_mean_abs_height_diff": float(np.mean(np.abs(diff_best))),
            "default_pair_median_abs_height_diff": float(np.median(np.abs(diff_default))),
            "default_pair_mean_abs_height_diff": float(np.mean(np.abs(diff_default))),
        }

    mid = Y.shape[0] // 2
    montage_images = []
    montage_titles = []
    fft_images = []
    fft_titles = []
    for g_idx, group_name in enumerate(groups):
        for label, cand in [("default", default_by_group[group_name]), ("best", best_by_group[group_name])]:
            A = stack_for_candidate(Y[:, g_idx], M[g_idx], cand)
            residual = modulation_residual(
                Y[:, g_idx][:, cand["perm"], :, :][:, None, :, :, :],
                A[:, None, :, :],
                template_variants(M[g_idx])[cand["variant"]][None, :, :, :],
            )[:, 0]
            montage_images.append(A[mid])
            montage_titles.append(f"{safe_group_label(group_name)} {label}")
            fft_images.append(fft_magnitude(residual[mid, 0]))
            fft_titles.append(f"{safe_group_label(group_name)} {label} FFT")
    save_montage(montage_images, out_dir / "A_by_order.png", montage_titles, cols=2, cmap="viridis")
    save_montage(fft_images, out_dir / "residual_by_order.png", fft_titles, cols=2, cmap="magma")

    metrics = {
        "sample": sample,
        "groups": groups,
        "crop": crop.label if crop else "full",
        "pattern_infos": pattern_infos,
        "best_by_group": best_by_group,
        "default_by_group": default_by_group,
        "hv_consistency": hv_consistency,
        "all_rows": all_rows,
        "interpretation": "Lower score prefers lower modulation loss, residual grid/stripe leakage, and height std with higher peak sharpness. This is a diagnostic, not height truth.",
    }
    data_io.write_json(out_dir / "metrics.json", metrics)
    data_io.write_json(out_dir / "phase_order_metrics.json", metrics)

    report = [
        "# Phase Order Diagnostic",
        "",
        f"- Sample: `{sample}`",
        f"- Groups: `{', '.join(groups)}`",
        f"- Crop: `{crop.label if crop else 'full'}`",
        "",
        "## Answer",
        "",
    ]
    for group_name in groups:
        best = best_by_group[group_name]
        default = default_by_group[group_name]
        report += [
            f"### {group_name}",
            "",
            f"- Best candidate: perm `{best['perm']}`, variant `{best['variant']}`, score `{best['score']:.6g}`, modulation loss `{best['modulation_loss']:.6g}`.",
            f"- Default candidate: perm `{default['perm']}`, variant `{default['variant']}`, score `{default['score']:.6g}`, modulation loss `{default['modulation_loss']:.6g}`.",
            f"- Default rank: `{1 + [row for row in all_rows[group_name]].index(default)}` of `{len(all_rows[group_name])}` after score sorting.",
            "",
        ]
    if hv_consistency:
        report += [
            "## H/V Consistency",
            "",
            f"- Default median |H-V height|: `{hv_consistency['default_pair_median_abs_height_diff']:.6g}`.",
            f"- Best-candidate median |H-V height|: `{hv_consistency['best_pair_median_abs_height_diff']:.6g}`.",
            "",
        ]
    report += [
        "## Interpretation",
        "",
        "- If a non-default permutation wins by only a small margin, the V anomaly should not be attributed to phase order alone.",
        "- If contrast inversion wins, the physical sign can be absorbed by a signed amplitude model but not by a nonnegative amplitude model; this matters for fixed-template LS.",
        "- See `A_by_order.png` and `residual_by_order.png` for visual comparison.",
    ]
    (out_dir / "phase_order_report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote phase-order diagnostics to {out_dir}")


if __name__ == "__main__":
    main()
