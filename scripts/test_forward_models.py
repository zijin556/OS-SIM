from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import data_io
from src import calibration
from src.fft_analysis import circular_peak_mask, detect_frequency_peaks, masked_fft_energy
from src.forward_models import (
    AffineWarp,
    BSplineWarp,
    DMDPatternWarpForward,
    FixedPatternForward,
    HarmonicGridForward,
    IdealCosForward,
    ModulationDomainForward,
    OptionalBlur,
    PolynomialWarp,
    estimate_q_from_frames,
    model_losses,
    phase_dependent_grid_basis,
)
from src.os_sim import demodulate_phase_stack
from src.visualization import save_montage


def mean_masked_fft_energy(stack: np.ndarray, mask: np.ndarray) -> float:
    values = []
    for image in np.asarray(stack):
        values.append(masked_fft_energy(image, mask))
    return float(np.mean(values)) if values else 0.0


def residual_spectral_metrics(residual: np.ndarray, grid_mask: np.ndarray, stripe_mask: np.ndarray) -> dict[str, object]:
    """Summarize structured residual energy over K x M residual images."""
    flat = residual.reshape((-1,) + residual.shape[-2:])
    mid_resid = residual[residual.shape[0] // 2, 0]
    residual_peaks = detect_frequency_peaks(mid_resid, top_n=6, min_radius=4)
    peak_mask = circular_peak_mask(mid_resid.shape, residual_peaks[:4], radius=4) if residual_peaks else np.zeros(mid_resid.shape, bool)
    fft_norms = []
    for image in flat:
        centered = image - np.mean(image)
        F = np.fft.fft2(centered)
        fft_norms.append(np.linalg.norm(F.ravel()) / np.sqrt(centered.size))
    return {
        "residual_fft_l2_mean": float(np.mean(fft_norms)),
        "residual_top_peak_energy_mean": mean_masked_fft_energy(flat, peak_mask) if residual_peaks else 0.0,
        "residual_grid_energy_mean": mean_masked_fft_energy(flat, grid_mask) if np.any(grid_mask) else 0.0,
        "residual_stripe_energy_mean": mean_masked_fft_energy(flat, stripe_mask) if np.any(stripe_mask) else 0.0,
        "residual_peaks_mid_z_phase0": residual_peaks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Level 0/1/2 forward models on a real crop.")
    parser.add_argument("--config", default="configs/ossim_dataset.yaml")
    parser.add_argument("--sample", default="细台阶")
    parser.add_argument("--groups", default="ossim_t6_h_3step,ossim_t6_v_3step")
    parser.add_argument("--exclude-samples", default="台阶曝光1")
    parser.add_argument("--crop", default="center:256")
    parser.add_argument("--output", default="outputs/forward/t6_hv")
    parser.add_argument("--measured-patterns", choices=["auto", "off", "required"], default="auto")
    parser.add_argument("--projection-sequence", default="saomiao3_6")
    parser.add_argument("--calibration-label", default="3-6")
    parser.add_argument("--calibration-mat", default="DMD_CCD_Calibration_Dict.mat")
    args = parser.parse_args()

    config = data_io.load_config(args.config)
    excluded = data_io.parse_excluded_samples(args.exclude_samples)
    groups = data_io.parse_groups(args.groups, config)
    crop = data_io.infer_crop_for_sample(config, args.sample, args.crop)
    bundle = data_io.load_os_sim_stack(config, args.sample, groups, crop, excluded_samples=excluded)
    Y = bundle["Y"]  # K,G,M,H,W
    out_dir = ROOT / args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "sample": args.sample,
        "groups": groups,
        "crop": crop.label if crop else "full",
        "measured_patterns_mode": args.measured_patterns,
        "calibration": {},
        "models": {},
    }
    measured_patterns: dict[str, np.ndarray] = {}
    if args.measured_patterns != "off":
        for group_name in groups:
            try:
                patterns, info = calibration.measured_camera_patterns_for_group(
                    config,
                    group_name,
                    crop=crop,
                    sequence_stem=args.projection_sequence,
                    calibration_label_prefix=args.calibration_label,
                    mat_name=args.calibration_mat,
                )
                measured_patterns[group_name] = patterns
                metrics["calibration"][group_name] = {"status": "loaded", **info}
            except Exception as exc:
                metrics["calibration"][group_name] = {
                    "status": "unavailable",
                    "error": type(exc).__name__,
                    "message": str(exc),
                }
                if args.measured_patterns == "required":
                    raise
    images = []
    titles = []
    for g_idx, group in enumerate(bundle["groups"]):
        group_name = groups[g_idx]
        frames = Y[:, g_idx]
        demod = demodulate_phase_stack(frames)
        D0, A = demod["D0"], demod["A"]
        qx, qy, q_info = estimate_q_from_frames(frames[len(frames) // 2], group)
        grid_peaks = detect_frequency_peaks(A[len(A) // 2], top_n=4, min_radius=4)
        grid_mask = circular_peak_mask(A.shape[-2:], grid_peaks, radius=4) if grid_peaks else np.zeros(A.shape[-2:], bool)
        stripe_peak = q_info.get("peak")
        stripe_mask = circular_peak_mask(A.shape[-2:], [stripe_peak], radius=4) if stripe_peak else np.zeros(A.shape[-2:], bool)
        phase_grid_basis = phase_dependent_grid_basis(A.shape[-2:], grid_peaks[:2], m_count=frames.shape[1], max_basis=2)
        models = {
            "level0_ideal_cos_group_period": IdealCosForward(group),
            "level1_modulation_cos_fft_q": ModulationDomainForward(group, qxy=(qx, qy)),
            "level2_cos_identity_affine_warp_fft_q": DMDPatternWarpForward(group, warp=AffineWarp.identity(), qxy=(qx, qy), pattern_kind="cos"),
            "level2_binary_dmd_identity_affine_warp_fft_q": DMDPatternWarpForward(
                group, warp=AffineWarp.identity(), qxy=(qx, qy), pattern_kind="binary", duty=0.5
            ),
            "level3_binary_dmd_identity_polynomial_warp_fft_q": DMDPatternWarpForward(
                group, warp=PolynomialWarp.identity(), qxy=(qx, qy), pattern_kind="binary", duty=0.5
            ),
            "level3_binary_dmd_zero_bspline_warp_fft_q": DMDPatternWarpForward(
                group, warp=BSplineWarp.zeros((8, 8)), qxy=(qx, qy), pattern_kind="binary", duty=0.5
            ),
            "level4_harmonic_grid_no_blur_fft_q": HarmonicGridForward(
                group,
                qxy=(qx, qy),
                a3=0.03,
                a5=0.01,
                grid_basis=phase_grid_basis,
                grid_coeff=np.zeros(phase_grid_basis.shape[1], dtype=np.float32),
                blur=OptionalBlur(0.0),
            ),
            "level4_harmonic_grid_blur_fft_q": HarmonicGridForward(
                group,
                qxy=(qx, qy),
                a3=0.03,
                a5=0.01,
                grid_basis=phase_grid_basis,
                grid_coeff=np.zeros(phase_grid_basis.shape[1], dtype=np.float32),
                blur=OptionalBlur(0.35),
            ),
        }
        if group_name in measured_patterns:
            models["level2_measured_dmd_calibrated_map"] = FixedPatternForward(measured_patterns[group_name])
            models["level4_measured_dmd_calibrated_map_blur"] = FixedPatternForward(
                measured_patterns[group_name], blur=OptionalBlur(0.35)
            )
        metrics["models"][group_name] = {
            "q_estimate": {"qx": qx, "qy": qy, "info": q_info},
            "baseline_A_grid_peaks": grid_peaks,
            "losses": {},
        }
        mid = len(frames) // 2
        images.append(frames[mid, 0])
        titles.append(f"real {group_name}")
        for model_name, model in models.items():
            M = model.patterns(A.shape[-2:])
            losses = model_losses(frames, D0, A, M)
            Y_hat = D0[:, None, :, :] + A[:, None, :, :] * M[None, :, :, :]
            residual = frames - Y_hat
            losses.update(residual_spectral_metrics(residual, grid_mask, stripe_mask))
            metrics["models"][group_name]["losses"][model_name] = losses
            images.append(Y_hat[mid, 0])
            titles.append(model_name.replace("_", " "))
            images.append(np.abs(frames[mid, 0] - Y_hat[mid, 0]))
            titles.append(f"resid {model_name}")

    save_montage(images[:12], out_dir / "reprojection_examples.png", titles[:12], cols=3)
    # Residual examples are every third entry after the real frame per group.
    residuals = [img for title, img in zip(titles, images) if title.startswith("resid")]
    residual_titles = [title for title in titles if title.startswith("resid")]
    save_montage(residuals[:8], out_dir / "residual_examples.png", residual_titles[:8], cols=2)
    data_io.write_json(out_dir / "model_compare.json", metrics)
    data_io.write_json(
        out_dir / "config.json",
        {
            "sample": args.sample,
            "groups": groups,
            "exclude_samples": sorted(excluded),
            "crop": crop.label if crop else "full",
            "measured_patterns": args.measured_patterns,
            "projection_sequence": args.projection_sequence,
            "calibration_label": args.calibration_label,
            "calibration_mat": args.calibration_mat,
        },
    )
    lines = [
        "# Forward Model Report",
        "",
        f"- Sample: `{args.sample}`",
        f"- Groups: `{', '.join(groups)}`",
        f"- Crop: `{crop.label if crop else 'full'}`",
        "",
        "## Notes",
        "",
        "- Level 0 uses ideal cos patterns from nominal period.",
        "- Level 1 uses FFT-estimated q in the modulation domain.",
        "- Level 2 compares generated cos patterns and generated binary DMD stripe patterns through the same identity affine warp interface.",
        "- If available, Level 2 also samples measured DMD bitmaps through the loaded DMD-CCD calibration map.",
        "- Level 3 validates low-order polynomial and coarse B-spline residual warp interfaces.",
        "- Level 4 validates harmonic/grid nuisance terms both without blur and with optional Gaussian blur.",
    ]
    lines += [
        "",
        "## Calibration Assets",
        "",
    ]
    for group_name in groups:
        info = metrics["calibration"].get(group_name, {})
        if info.get("status") == "loaded":
            lines.append(
                f"- `{group_name}` measured patterns loaded from frames `{info.get('frame_ids')}` with crop valid fraction `{info.get('valid_fraction'):.6g}`."
            )
        elif args.measured_patterns == "off":
            lines.append(f"- `{group_name}` measured patterns were disabled by CLI.")
        else:
            lines.append(f"- `{group_name}` measured patterns unavailable: `{info.get('message')}`.")
    lines += [
        "",
        "## Metric Fields",
        "",
        "- `raw_rmse`, `mod_rmse`, `raw_charbonnier`, and `mod_charbonnier` measure image-domain and modulation-domain reprojection error.",
        "- `residual_grid_energy_mean` measures residual energy on baseline A-stack grid/Moire FFT peaks.",
        "- `residual_stripe_energy_mean` measures residual energy on the selected nominal stripe-frequency peak.",
        "- `residual_top_peak_energy_mean` measures energy on the strongest residual FFT peaks from the middle z layer.",
        "",
        "## Summary",
        "",
    ]
    for group_name, detail in metrics["models"].items():
        ranked = sorted(detail["losses"].items(), key=lambda item: item[1]["mod_rmse"])
        best_name, best_loss = ranked[0]
        best_grid_name, best_grid_loss = min(detail["losses"].items(), key=lambda item: item[1]["residual_grid_energy_mean"])
        best_stripe_name, best_stripe_loss = min(detail["losses"].items(), key=lambda item: item[1]["residual_stripe_energy_mean"])
        lines += [
            f"### {group_name}",
            "",
            f"- Best modulation RMSE: `{best_name}` with `{best_loss['mod_rmse']:.6g}`.",
            f"- Best residual grid energy: `{best_grid_name}` with `{best_grid_loss['residual_grid_energy_mean']:.6g}`.",
            f"- Best residual stripe leakage: `{best_stripe_name}` with `{best_stripe_loss['residual_stripe_energy_mean']:.6g}`.",
            f"- Top residual FFT peak energy for best-RMSE model: `{best_loss['residual_top_peak_energy_mean']:.6g}`.",
            "",
        ]
    (out_dir / "forward_report.md").write_text("\n".join(lines), encoding="utf-8")
    data_io.append_report(
        ROOT / "outputs/nightly_report.md",
        "P3 Forward Model Crop Reprojection",
        [
            f"- Ran Level 0/1/2/3/4 forward comparisons for `{args.sample}` T6 H/V crop.",
            f"- Saved model comparison JSON and reprojection examples under `{out_dir}`.",
        ],
    )
    print(f"Wrote forward outputs to {out_dir}")


if __name__ == "__main__":
    main()
