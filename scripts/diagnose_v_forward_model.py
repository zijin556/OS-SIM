from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src import calibration, data_io
from src.fft_analysis import fft_magnitude
from src.forward_models import (
    HarmonicGridForward,
    OptionalBlur,
    binary_grating_patterns,
    cos_patterns,
    estimate_q_from_frames,
    normalize_m,
    phase_dependent_grid_basis,
)
from src.metrics_v1 import (
    grid_mask_from_a,
    method_metrics,
    modulation_loss,
    modulation_residual,
    project_a_nonnegative,
    project_a_signed,
    residual_energy,
    stripe_mask_from_raw,
)
from src.os_sim import demodulate_phase_stack
from src.visualization import save_montage


def choose_sample(config: dict[str, Any], sample: str | None) -> str:
    if sample:
        return sample
    excluded = data_io.parse_excluded_samples(None)
    for row in reversed(config.get("samples", [])):
        name = row.get("name")
        if name and name not in excluded:
            return name
    raise ValueError("No usable sample found in config")


def measured_patterns(
    config: dict[str, Any],
    group_name: str,
    crop: data_io.CropSpec | None,
    sequence_stem: str,
    calibration_label: str,
    calibration_mat: str,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    try:
        M, info = calibration.measured_camera_patterns_for_group(
            config,
            group_name,
            crop=crop,
            sequence_stem=sequence_stem,
            calibration_label_prefix=calibration_label,
            mat_name=calibration_mat,
        )
        return normalize_m(M.astype(np.float32)), {"status": "loaded", **info}
    except Exception as exc:
        return None, {"status": "unavailable", "error": type(exc).__name__, "message": str(exc)}


def shift_patterns(M: np.ndarray, dy: float, dx: float) -> np.ndarray:
    shifted = np.stack(
        [ndimage.shift(frame, shift=(dy, dx), order=1, mode="reflect", prefilter=False) for frame in M],
        axis=0,
    )
    return normalize_m(shifted.astype(np.float32))


def blur_patterns(M: np.ndarray, sigma: float) -> np.ndarray:
    blurred = np.stack([ndimage.gaussian_filter(frame, sigma=sigma, mode="reflect") for frame in M], axis=0)
    return normalize_m(blurred.astype(np.float32))


def evaluate_candidate(
    Y_group: np.ndarray,
    M_group: np.ndarray,
    z_values: np.ndarray,
    group_label: str,
    model_name: str,
    params: dict[str, Any],
    grid_mask: np.ndarray,
    stripe_mask: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    Y = Y_group[:, None, :, :, :].astype(np.float32)
    M = normalize_m(M_group.astype(np.float32))[None, :, :, :]
    A_signed = project_a_signed(Y, M)
    A = np.maximum(A_signed, 0.0).astype(np.float32)
    row = method_metrics(model_name, Y, A, M, z_values, grid_mask[None, :, :], stripe_mask[None, :, :])
    row["group"] = group_label
    row["params"] = params
    row["negative_fraction"] = float(np.mean(A_signed < 0.0))
    residual = modulation_residual(Y, A, M)
    row["residual_grid_energy_check"] = residual_energy(residual, grid_mask)
    row["residual_stripe_leakage_check"] = residual_energy(residual, stripe_mask)
    return row, A[:, 0], residual[:, 0]


def best_of(
    Y_group: np.ndarray,
    z_values: np.ndarray,
    group_label: str,
    model_name: str,
    candidates: list[tuple[np.ndarray, dict[str, Any]]],
    grid_mask: np.ndarray,
    stripe_mask: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    quick_rows = []
    Y = Y_group[:, None, :, :, :].astype(np.float32)
    for M, params in candidates:
        M_norm = normalize_m(M.astype(np.float32))[None, :, :, :]
        A_quick = project_a_nonnegative(Y, M_norm)
        quick_rows.append(
            {
                "params": params,
                "modulation_loss": modulation_loss(Y, A_quick, M_norm),
                "M": M_norm[0],
            }
        )
    best_idx = min(
        range(len(quick_rows)),
        key=lambda i: quick_rows[i]["modulation_loss"],
    )
    best_quick = quick_rows[best_idx]
    best, A, residual = evaluate_candidate(
        Y_group,
        best_quick["M"],
        z_values,
        group_label,
        model_name,
        best_quick["params"],
        grid_mask,
        stripe_mask,
    )
    best["candidate_count"] = len(quick_rows)
    best["candidate_losses"] = [
        {
            "params": row["params"],
            "modulation_loss": row["modulation_loss"],
        }
        for row in quick_rows
    ]
    return best, A, residual


def phase_candidates(M: np.ndarray) -> list[tuple[np.ndarray, dict[str, Any]]]:
    out = []
    for perm in itertools.permutations(range(M.shape[0])):
        out.append((normalize_m(M[list(perm)]), {"perm": list(perm)}))
    return out


def shift_candidates(M: np.ndarray, radius: int = 2) -> list[tuple[np.ndarray, dict[str, Any]]]:
    out = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            out.append((shift_patterns(M, float(dy), float(dx)), {"dy": float(dy), "dx": float(dx)}))
    return out


def phase_shift_candidates(M: np.ndarray, radius: int = 1) -> list[tuple[np.ndarray, dict[str, Any]]]:
    out = []
    for perm in itertools.permutations(range(M.shape[0])):
        base = normalize_m(M[list(perm)])
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                out.append((shift_patterns(base, float(dy), float(dx)), {"perm": list(perm), "dy": float(dy), "dx": float(dx)}))
    return out


def harmonic_patterns(group: dict[str, Any], shape: tuple[int, int], qxy: tuple[float, float], A_probe: np.ndarray) -> np.ndarray:
    peaks = grid_mask_from_a(A_probe)[1]
    basis = phase_dependent_grid_basis(shape, peaks[:2], m_count=int(group.get("phase_steps", 3)), max_basis=2)
    model = HarmonicGridForward(
        group,
        qxy=qxy,
        a3=0.03,
        a5=0.01,
        grid_basis=basis,
        grid_coeff=np.zeros(basis.shape[1], dtype=np.float32),
        blur=OptionalBlur(0.0),
    )
    return model.patterns(shape)


def diagnose_group(
    config: dict[str, Any],
    group_name: str,
    group: dict[str, Any],
    Y_group: np.ndarray,
    z_values: np.ndarray,
    crop: data_io.CropSpec | None,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    mid = Y_group.shape[0] // 2
    A_cls = np.stack([demodulate_phase_stack(Y_group[k])["A"] for k in range(Y_group.shape[0])], axis=0).astype(np.float32)
    grid_mask, grid_peaks = grid_mask_from_a(A_cls)
    stripe_mask, stripe_peaks = stripe_mask_from_raw(Y_group)
    qx, qy, q_info = estimate_q_from_frames(Y_group[mid], group)
    shape = Y_group.shape[-2:]
    M_measured, cal_info = measured_patterns(
        config,
        group_name,
        crop,
        sequence_stem=args.projection_sequence,
        calibration_label=args.calibration_label,
        calibration_mat=args.calibration_mat,
    )
    M_cos_nominal = cos_patterns(shape, group)
    M_cos_q = cos_patterns(shape, group, qxy=(qx, qy))
    M_binary_q = binary_grating_patterns(shape, group, qxy=(qx, qy), duty=0.5)
    model_defs: list[tuple[str, list[tuple[np.ndarray, dict[str, Any]]]]] = [
        ("V0_ideal_cos_nominal", [(M_cos_nominal, {"source": "nominal_period"})]),
        ("V3_cos_fft_q_correction", [(M_cos_q, {"qx": qx, "qy": qy, "q_info": q_info})]),
        ("V6_binary_DMD_affine_shift", shift_candidates(M_binary_q, radius=2)),
        ("V7_harmonic_fft_q", [(harmonic_patterns(group, shape, (qx, qy), A_cls), {"qx": qx, "qy": qy, "a3": 0.03, "a5": 0.01})]),
    ]
    if M_measured is not None:
        model_defs.extend(
            [
                ("V1_measured_calibrated", [(M_measured, {"source": "measured_calibrated"})]),
                ("V2_measured_phase_offset", phase_candidates(M_measured)),
                ("V4_measured_affine_shift", shift_candidates(M_measured, radius=2)),
                ("V5_measured_phase_plus_affine", phase_shift_candidates(M_measured, radius=1)),
                ("V8_measured_blur", [(blur_patterns(M_measured, sigma), {"sigma": sigma}) for sigma in (0.2, 0.35, 0.5, 0.8, 1.2)]),
            ]
        )

    rows: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}
    for model_name, candidates in model_defs:
        row, A, residual = best_of(Y_group, z_values, group_name, model_name, candidates, grid_mask, stripe_mask)
        rows[model_name] = row
        arrays[f"A::{group_name}::{model_name}"] = A
        arrays[f"residual::{group_name}::{model_name}"] = residual

    ranked = sorted(rows.items(), key=lambda item: item[1]["modulation_loss"])
    return (
        {
            "group": group_name,
            "q_estimate": {"qx": qx, "qy": qy, "info": q_info},
            "calibration": cal_info,
            "grid_peaks": grid_peaks,
            "stripe_peaks": stripe_peaks,
            "models": rows,
            "ranking_by_modulation_loss": [name for name, _ in ranked],
            "best_by_modulation_loss": ranked[0][0],
            "best_by_residual_grid": min(rows.items(), key=lambda item: item[1]["residual_grid_energy_mean"])[0],
            "best_by_stripe_leakage": min(rows.items(), key=lambda item: item[1]["residual_stripe_leakage_mean"])[0],
        },
        arrays,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose H/V forward-template mismatch with phase, q, affine, binary, harmonic, and blur variants.")
    parser.add_argument("--config", default="configs/ossim_dataset.yaml")
    parser.add_argument("--sample", default=None)
    parser.add_argument("--groups", default="ossim_t6_h_3step,ossim_t6_v_3step")
    parser.add_argument("--exclude-samples", default="鍙伴樁鏇濆厜1")
    parser.add_argument("--crop", default="center:256")
    parser.add_argument("--projection-sequence", default="saomiao3_6")
    parser.add_argument("--calibration-label", default="3-6")
    parser.add_argument("--calibration-mat", default="DMD_CCD_Calibration_Dict.mat")
    parser.add_argument("--output", default="outputs/v1_next/v_forward")
    args = parser.parse_args()

    config = data_io.load_config(args.config)
    sample = choose_sample(config, args.sample)
    excluded = data_io.parse_excluded_samples(args.exclude_samples)
    groups = data_io.parse_groups(args.groups, config)
    crop = data_io.infer_crop_for_sample(config, sample, args.crop)
    bundle = data_io.load_os_sim_stack(config, sample, groups, crop, excluded_samples=excluded)
    Y = bundle["Y"].astype(np.float32)
    z_values = bundle["z_values"].astype(np.float32)

    out_dir = ROOT / args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}
    for g_idx, group_name in enumerate(groups):
        result, group_arrays = diagnose_group(config, group_name, bundle["groups"][g_idx], Y[:, g_idx], z_values, crop, args)
        results[group_name] = result
        arrays.update(group_arrays)

    mid = Y.shape[0] // 2
    repro_images = []
    repro_titles = []
    residual_images = []
    residual_titles = []
    fft_images = []
    fft_titles = []
    for g_idx, group_name in enumerate(groups):
        best_name = results[group_name]["best_by_modulation_loss"]
        measured_name = "V1_measured_calibrated" if "V1_measured_calibrated" in results[group_name]["models"] else best_name
        for model_name in dict.fromkeys([measured_name, best_name]):
            A = arrays[f"A::{group_name}::{model_name}"]
            residual = arrays[f"residual::{group_name}::{model_name}"]
            repro_images.append(A[mid])
            repro_titles.append(f"{group_name} {model_name}")
            residual_images.append(np.abs(residual[mid, 0]))
            residual_titles.append(f"{group_name} {model_name} abs residual")
            fft_images.append(fft_magnitude(residual[mid, 0]))
            fft_titles.append(f"{group_name} {model_name} resid FFT")
    save_montage(repro_images, out_dir / "reprojection_A_examples.png", repro_titles, cols=2, cmap="viridis")
    save_montage(repro_images, out_dir / "reprojection_HV.png", repro_titles, cols=2, cmap="viridis")
    save_montage(residual_images, out_dir / "residual_HV.png", residual_titles, cols=2, cmap="magma")
    save_montage(fft_images, out_dir / "residual_fft_examples.png", fft_titles, cols=2, cmap="magma")
    save_montage(fft_images, out_dir / "residual_fft_HV.png", fft_titles, cols=2, cmap="magma")

    metrics = {
        "sample": sample,
        "groups": groups,
        "crop": crop.label if crop else "full",
        "results": results,
        "interpretation": "Models are compared in the phase-centered modulation domain after closed-form nonnegative A projection. Lower modulation loss and structured residual energy are preferred.",
    }
    data_io.write_json(out_dir / "metrics.json", metrics)
    data_io.write_json(out_dir / "v_forward_model_compare.json", metrics)
    data_io.write_json(out_dir / "model_compare_HV.json", metrics)

    report = [
        "# V Forward Model Diagnostic",
        "",
        f"- Sample: `{sample}`",
        f"- Groups: `{', '.join(groups)}`",
        f"- Crop: `{crop.label if crop else 'full'}`",
        "",
        "## Results",
        "",
    ]
    for group_name, result in results.items():
        report += [
            f"### {group_name}",
            "",
            f"- Best by modulation loss: `{result['best_by_modulation_loss']}`.",
            f"- Best by residual grid: `{result['best_by_residual_grid']}`.",
            f"- Best by stripe leakage: `{result['best_by_stripe_leakage']}`.",
            "",
            "| model | modulation loss | residual grid | stripe leakage | A grid | negative frac | best params |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
        ranked = sorted(result["models"].items(), key=lambda item: item[1]["modulation_loss"])
        for name, row in ranked:
            report.append(
                f"| {name} | {row['modulation_loss']:.6g} | {row['residual_grid_energy_mean']:.6g} | {row['residual_stripe_leakage_mean']:.6g} | {row['A_grid_energy_mean']:.6g} | {row['negative_fraction']:.6g} | `{row['params']}` |"
            )
        report.append("")
    report += [
        "## Interpretation",
        "",
        "- If V improves mainly under `V2` or `V5`, the V issue is largely phase alignment/phase-zero mismatch.",
        "- If V still needs affine/binary/harmonic variants after phase correction, the forward template is not yet physically sufficient.",
        "- H-only may be a diagnostic fallback, but it should not become the mainline unless V remains inconsistent after phase/order correction.",
    ]
    (out_dir / "v_forward_report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote V forward diagnostics to {out_dir}")


if __name__ == "__main__":
    main()
