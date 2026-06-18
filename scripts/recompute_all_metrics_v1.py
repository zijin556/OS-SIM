from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src import data_io
from src.forward_models import normalize_m
from src.height import height_parabolic
from src.losses import tv_l1
from src.metrics_v1 import (
    grid_mask_from_a,
    height_summary,
    method_metrics,
    project_a_nonnegative,
    project_a_signed,
    stripe_mask_from_raw,
)
from src.os_sim import demodulate_multigroup, fuse_hv
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


def inverse_perm(perm: list[int]) -> list[int]:
    inv = [0] * len(perm)
    for logical, raw_idx in enumerate(perm):
        inv[raw_idx] = logical
    return inv


def phase_aligned_m(M: np.ndarray, groups: list[str], phase_metrics_path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    out = M.copy()
    info: dict[str, Any] = {"source": "default_no_phase_metrics"}
    if not phase_metrics_path.exists():
        return out, info
    metrics = data_io.read_json(phase_metrics_path)
    info = {"source": str(phase_metrics_path), "applied": {}}
    for g_idx, group_name in enumerate(groups):
        best = metrics.get("best_by_group", {}).get(group_name)
        if not best:
            continue
        perm = [int(v) for v in best.get("perm", [])]
        variant = best.get("variant")
        if len(perm) == M.shape[1] and variant == "default_template":
            inv = inverse_perm(perm)
            out[g_idx] = M[g_idx, inv]
            info["applied"][group_name] = {"raw_perm": perm, "template_perm": inv, "variant": variant}
    return normalize_m(out), info


def metric_masks(Y: np.ndarray, A_ref: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    grid_masks = []
    stripe_masks = []
    peaks: dict[str, Any] = {"grid": [], "stripe": []}
    for g in range(A_ref.shape[1]):
        grid_mask, grid_peak = grid_mask_from_a(A_ref[:, g])
        stripe_mask, stripe_peak = stripe_mask_from_raw(Y[:, g])
        grid_masks.append(grid_mask)
        stripe_masks.append(stripe_mask)
        peaks["grid"].append(grid_peak)
        peaks["stripe"].append(stripe_peak)
    return np.stack(grid_masks, axis=0), np.stack(stripe_masks, axis=0), peaks


def flatten_row(row: dict[str, Any]) -> dict[str, Any]:
    height = row.get("height_fused") or {}
    hv = row.get("hv_consistency") or {}
    return {
        "method": row.get("method"),
        "kind": row.get("kind", "full_hv"),
        "modulation_loss": row.get("modulation_loss"),
        "A_grid_energy_mean": row.get("A_grid_energy_mean"),
        "residual_grid_energy_mean": row.get("residual_grid_energy_mean"),
        "residual_stripe_leakage_mean": row.get("residual_stripe_leakage_mean"),
        "residual_fft_l2": row.get("residual_fft_l2"),
        "A_fused_grid_energy": row.get("A_fused_grid_energy"),
        "A_fused_tv": row.get("A_fused_tv"),
        "height_std": height.get("height_std"),
        "height_mean": height.get("height_mean"),
        "invalid_fraction": height.get("invalid_fraction"),
        "hv_median_abs_diff": hv.get("height_diff_median_abs"),
        "hv_mean_abs_diff": hv.get("height_diff_mean_abs"),
        "note": row.get("note", ""),
    }


def fused_only_metrics(name: str, A_fused: np.ndarray, z_values: np.ndarray, reference_mask: np.ndarray, note: str) -> dict[str, Any]:
    from src.metrics_v1 import a_grid_energy

    return {
        "method": name,
        "kind": "fused_only",
        "modulation_loss": None,
        "A_grid_energy_mean": None,
        "residual_grid_energy_mean": None,
        "residual_stripe_leakage_mean": None,
        "residual_fft_l2": None,
        "A_fused_grid_energy": a_grid_energy(A_fused, reference_mask),
        "A_fused_tv": tv_l1(A_fused),
        "height_fused": height_summary(A_fused, z_values),
        "hv_consistency": None,
        "note": note,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute all v1 method metrics with one metric implementation.")
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
    parser.add_argument("--latent-dir", default="outputs/latent_opt/v1_t6_hv_no_platform")
    parser.add_argument("--network-fused", default="outputs/inverse_net_sweep/tx_p015_s012_g003_256_selectbest/A_fused_pred_stack.npy")
    parser.add_argument("--output", default="outputs/v1_next/metrics")
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
    M = normalize_m(M.astype(np.float32))
    M_phase, phase_info = phase_aligned_m(M, groups, ROOT / args.phase_order_metrics)
    grid_masks, stripe_masks, mask_peaks = metric_masks(Y, A_cls)
    fused_ref = fuse_hv(A_cls[:, 0], A_cls[:, 1] if A_cls.shape[1] > 1 else A_cls[:, 0])
    fused_mask, fused_peaks = grid_mask_from_a(fused_ref)

    methods: list[tuple[str, np.ndarray, np.ndarray, str]] = []
    A_signed = project_a_signed(Y, M)
    A_nn = np.maximum(A_signed, 0.0).astype(np.float32)
    A_phase_nn = project_a_nonnegative(Y, M_phase)
    methods.extend(
        [
            ("A_cls_traditional_3step", A_cls, M, "Original three-step/cos-sin demodulated amplitude. This is the raw baseline, not ground truth."),
            ("A_ls_signed_measured_template", A_signed, M, "Signed fixed-M projection; negative amplitudes are allowed and indicate template/sign mismatch."),
            ("A_ls_nonnegative_measured_template", A_nn, M, "Physical nonnegative fixed-M projection with measured-calibrated template."),
            ("A_ls_cos_sin_nominal", A_cls, M, "For 3-step data this amplitude equals the traditional cos/sin demodulation; residual is still evaluated against measured M."),
            ("A_ls_nonnegative_phase_aligned_V", A_phase_nn, M_phase, "Fixed-M projection after applying phase-order diagnostic template permutation."),
        ]
    )
    methods.append(
        (
            "A_cls_lowpass_sigma1",
            ndimage.gaussian_filter(A_cls, sigma=(0, 0, 1.0, 1.0), mode="reflect").astype(np.float32),
            M,
            "Simple spatial lowpass on A_cls; can reduce visible grid by smoothing.",
        )
    )
    methods.append(
        (
            "A_cls_median3",
            ndimage.median_filter(A_cls, size=(1, 1, 3, 3), mode="reflect").astype(np.float32),
            M,
            "Simple median filter on A_cls; can suppress pixel/grid artifacts but may distort peaks.",
        )
    )

    latent_dir = ROOT / args.latent_dir
    if (latent_dir / "A_h_stack.npy").exists() and (latent_dir / "A_v_stack.npy").exists():
        A_latent = np.stack([np.load(latent_dir / "A_h_stack.npy"), np.load(latent_dir / "A_v_stack.npy")], axis=1).astype(np.float32)
        if A_latent.shape == A_cls.shape:
            methods.append(("latent_no_platform", A_latent, M, "Existing v1 latent optimization without platform loss."))

    rows: list[dict[str, Any]] = []
    for name, A, M_eval, note in methods:
        row = method_metrics(name, Y, A, M_eval, z_values, grid_masks, stripe_masks)
        row["note"] = note
        if name == "A_ls_signed_measured_template":
            row["negative_fraction"] = float(np.mean(A < 0.0))
        rows.append(row)

    fused_only: list[dict[str, Any]] = []
    if (latent_dir / "A_fused_stack.npy").exists():
        A_latent_fused = np.load(latent_dir / "A_fused_stack.npy").astype(np.float32)
        if A_latent_fused.shape == fused_ref.shape and "latent_no_platform" not in [r["method"] for r in rows]:
            fused_only.append(fused_only_metrics("latent_no_platform_fused_only", A_latent_fused, z_values, fused_mask, "Only fused stack found; no modulation residual computed."))
    network_path = ROOT / args.network_fused
    if network_path.exists():
        A_net = np.load(network_path).astype(np.float32)
        if A_net.shape == fused_ref.shape:
            fused_only.append(
                fused_only_metrics(
                    "existing_network_fused_only",
                    A_net,
                    z_values,
                    fused_mask,
                    "Network output is fused-only in this run; grid/height can be compared, modulation residual cannot.",
                )
            )
    rows.extend(fused_only)

    flat_rows = [flatten_row(row) for row in rows]
    out_dir = ROOT / args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(flat_rows)
    with (out_dir / "unified_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(flat_rows)
    metrics = {
        "sample": sample,
        "groups": groups,
        "crop": crop.label if crop else "full",
        "pattern_infos": pattern_infos,
        "phase_alignment": phase_info,
        "mask_peaks": mask_peaks,
        "fused_grid_peaks": fused_peaks,
        "methods": rows,
        "flat_rows": flat_rows,
        "interpretation": "All rows use the same metrics_v1 implementation. Older v1 grid numbers are not directly comparable unless they used the same masks and signal domain.",
    }
    data_io.write_json(out_dir / "metrics.json", metrics)
    data_io.write_json(out_dir / "unified_metrics.json", metrics)

    mid = Y.shape[0] // 2
    montage = []
    titles = []
    for row in rows[:8]:
        name = row["method"]
        if row.get("kind") == "fused_only":
            if name == "existing_network_fused_only" and network_path.exists():
                A_show = np.load(network_path).astype(np.float32)
            else:
                A_show = np.load(latent_dir / "A_fused_stack.npy").astype(np.float32)
        else:
            A_src = next(A for n, A, _, _ in methods if n == name)
            A_show = fuse_hv(A_src[:, 0], A_src[:, 1] if A_src.shape[1] > 1 else A_src[:, 0])
        montage.append(A_show[mid])
        titles.append(name.replace("_", " "))
    save_montage(montage, out_dir / "method_A_compare.png", titles, cols=3, cmap="viridis")

    ranked_mod = [r for r in flat_rows if r["modulation_loss"] is not None]
    ranked_mod.sort(key=lambda r: float(r["modulation_loss"]))
    ranked_grid = sorted(flat_rows, key=lambda r: float("inf") if r["A_fused_grid_energy"] is None else float(r["A_fused_grid_energy"]))

    report = [
        "# Unified v1 Metrics Report",
        "",
        f"- Sample: `{sample}`",
        f"- Groups: `{', '.join(groups)}`",
        f"- Crop: `{crop.label if crop else 'full'}`",
        "",
        "## Required Answers",
        "",
        "- Older grid numbers are not directly comparable because previous scripts mixed A-domain, fused-domain, residual-domain, and stripe-frequency masks.",
        f"- Best modulation-domain fit: `{ranked_mod[0]['method']}` with loss `{float(ranked_mod[0]['modulation_loss']):.6g}`.",
        f"- Lowest fused A-grid energy: `{ranked_grid[0]['method']}` with `{float(ranked_grid[0]['A_fused_grid_energy']):.6g}`.",
        "- Rows marked `fused_only` cannot be compared by modulation residual because the saved artifact has no H/V phase-domain stack.",
        "",
        "## Table",
        "",
        "| method | kind | mod loss | A grid | residual grid | stripe leakage | fused grid | A TV | height std | HV median diff |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in flat_rows:
        def fmt(v: Any) -> str:
            return "" if v is None else f"{float(v):.6g}"

        report.append(
            f"| {row['method']} | {row['kind']} | {fmt(row['modulation_loss'])} | {fmt(row['A_grid_energy_mean'])} | {fmt(row['residual_grid_energy_mean'])} | {fmt(row['residual_stripe_leakage_mean'])} | {fmt(row['A_fused_grid_energy'])} | {fmt(row['A_fused_tv'])} | {fmt(row['height_std'])} | {fmt(row['hv_median_abs_diff'])} |"
        )
    report += [
        "",
        "## Notes",
        "",
        "- Simple filters mostly lower visible A variation by smoothing; they do not prove a better forward model.",
        "- LS rows are the strongest non-network baseline because they directly minimize the fixed-template modulation residual.",
        "- Network rows with extremely low TV should be read as smoothing risk unless independent truth/repeatability supports them.",
    ]
    (out_dir / "metrics_report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote unified metrics to {out_dir}")


if __name__ == "__main__":
    main()
