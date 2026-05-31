from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import calibration, data_io
from src.fft_analysis import circular_peak_mask, detect_frequency_peaks, fft_magnitude, masked_fft_energy
from src.forward_models import estimate_q_from_frames, normalize_m, cos_patterns
from src.height import confidence_metrics, height_parabolic, summarize_height
from src.losses import charbonnier, tv_l1
from src.os_sim import demodulate_multigroup, fuse_hv
from src.visualization import save_loss_curve, save_montage, save_scalar_image


def build_patterns(Y: np.ndarray, groups: list[dict]) -> tuple[np.ndarray, list[dict]]:
    patterns = []
    infos = []
    for g_idx, group in enumerate(groups):
        frames_mid = Y[Y.shape[0] // 2, g_idx]
        qx, qy, q_info = estimate_q_from_frames(frames_mid, group)
        M = cos_patterns(Y.shape[-2:], group, qxy=(qx, qy))
        patterns.append(M)
        infos.append({"qx": qx, "qy": qy, "q_info": q_info})
    return np.stack(patterns, axis=0), infos  # G,M,H,W


def build_measured_calibrated_patterns(
    config: dict,
    group_names: list[str],
    crop: data_io.CropSpec | None,
    sequence_stem: str,
    calibration_label: str,
    calibration_mat: str,
) -> tuple[np.ndarray, list[dict]]:
    patterns = []
    infos = []
    for group_name in group_names:
        M, info = calibration.measured_camera_patterns_for_group(
            config,
            group_name,
            crop=crop,
            sequence_stem=sequence_stem,
            calibration_label_prefix=calibration_label,
            mat_name=calibration_mat,
        )
        patterns.append(M)
        infos.append({"source": "measured_calibrated", **info})
    return np.stack(patterns, axis=0), infos


def build_patterns_by_source(
    Y: np.ndarray,
    groups: list[dict],
    group_names: list[str],
    config: dict,
    crop: data_io.CropSpec | None,
    pattern_source: str,
    sequence_stem: str,
    calibration_label: str,
    calibration_mat: str,
) -> tuple[np.ndarray, list[dict]]:
    if pattern_source == "auto":
        try:
            return build_patterns_by_source(
                Y,
                groups,
                group_names,
                config,
                crop,
                pattern_source="measured_calibrated",
                sequence_stem=sequence_stem,
                calibration_label=calibration_label,
                calibration_mat=calibration_mat,
            )
        except Exception as exc:
            M, infos = build_patterns(Y, groups)
            for info in infos:
                info["source"] = "cos_fft"
                info["auto_fallback_reason"] = f"{type(exc).__name__}: {exc}"
            return M, infos
    if pattern_source == "cos_fft":
        M, infos = build_patterns(Y, groups)
        for info in infos:
            info["source"] = "cos_fft"
        return M, infos
    if pattern_source == "measured_calibrated":
        M, infos = build_measured_calibrated_patterns(
            config,
            group_names,
            crop,
            sequence_stem=sequence_stem,
            calibration_label=calibration_label,
            calibration_mat=calibration_mat,
        )
        if tuple(M.shape[-2:]) != tuple(Y.shape[-2:]):
            raise ValueError(f"Measured calibrated patterns have shape {M.shape[-2:]}, expected {Y.shape[-2:]}")
        return M, infos
    raise ValueError(f"Unknown pattern source: {pattern_source}")


def modulation_loss(Y: np.ndarray, A: np.ndarray, M: np.ndarray) -> float:
    Y_tilde = Y - Y.mean(axis=2, keepdims=True)
    pred = A[:, :, None, :, :] * M[None, :, :, :, :]
    return charbonnier(Y_tilde - pred)


def build_grid_masks(A_init: np.ndarray, top_n: int = 4) -> tuple[np.ndarray, list[list[dict]]]:
    masks = []
    peaks_by_group = []
    for g in range(A_init.shape[1]):
        mid = A_init.shape[0] // 2
        peaks = detect_frequency_peaks(A_init[mid, g], top_n=top_n, min_radius=4)
        peaks_by_group.append(peaks)
        masks.append(circular_peak_mask(A_init.shape[-2:], peaks, radius=4) if peaks else np.zeros(A_init.shape[-2:], dtype=bool))
    return np.stack(masks, axis=0), peaks_by_group


def grid_energy_mean(A: np.ndarray, masks: np.ndarray) -> float:
    values = []
    for k in range(A.shape[0]):
        for g in range(A.shape[1]):
            values.append(masked_fft_energy(A[k, g], masks[g]))
    return float(np.mean(values))


def grid_penalty_gradient(A: np.ndarray, masks: np.ndarray) -> np.ndarray:
    grad = np.zeros_like(A, dtype=np.float32)
    for k in range(A.shape[0]):
        for g in range(A.shape[1]):
            F = np.fft.fftshift(np.fft.fft2(A[k, g]))
            filtered = np.fft.ifft2(np.fft.ifftshift(F * masks[g])).real
            grad[k, g] = filtered.astype(np.float32)
    return grad


def optimize_A(
    Y: np.ndarray,
    A_init: np.ndarray,
    M: np.ndarray,
    grid_masks: np.ndarray,
    iterations: int,
    lr: float,
    lambda_sec: float,
    lambda_smooth: float,
    lambda_hv: float,
    lambda_grid: float,
) -> tuple[np.ndarray, list[float]]:
    A = np.maximum(A_init.copy().astype(np.float32), 0.0)
    Y_tilde = Y - Y.mean(axis=2, keepdims=True)
    denom = np.mean(M[None, :, :, :, :] ** 2, axis=2) + 1e-6
    losses = []
    for _ in range(iterations):
        residual = A[:, :, None, :, :] * M[None, :, :, :, :] - Y_tilde
        grad = np.mean(residual * M[None, :, :, :, :], axis=2) / denom
        if lambda_sec:
            grad += lambda_sec * (A - A_init)
        if lambda_smooth:
            # Smooth only spatial dimensions. This is a stable quadratic smoothness proxy for TV.
            smooth_target = ndimage.gaussian_filter(A, sigma=(0, 0, 1.0, 1.0), mode="reflect")
            grad += lambda_smooth * (A - smooth_target)
        if lambda_hv and A.shape[1] == 2:
            shared = A.mean(axis=1, keepdims=True)
            grad += lambda_hv * (A - shared)
        if lambda_grid:
            grad += lambda_grid * grid_penalty_gradient(A, grid_masks)
        A = np.maximum(A - lr * grad, 0.0)
        losses.append(modulation_loss(Y, A, M))
    return A.astype(np.float32), losses


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize H/V modulation latent maps on a crop using NumPy/SciPy.")
    parser.add_argument("--config", default="configs/ossim_dataset.yaml")
    parser.add_argument("--sample", default="细台阶")
    parser.add_argument("--groups", default="ossim_t6_h_3step,ossim_t6_v_3step")
    parser.add_argument("--exclude-samples", default="台阶曝光1")
    parser.add_argument("--crop", default="center:256")
    parser.add_argument("--backend", default="scipy", choices=["scipy"])
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--lr", type=float, default=0.25)
    parser.add_argument("--lambda-sec", type=float, default=0.02)
    parser.add_argument("--lambda-smooth", type=float, default=0.05)
    parser.add_argument("--lambda-hv", type=float, default=0.01)
    parser.add_argument("--lambda-grid", type=float, default=0.02)
    parser.add_argument("--pattern-source", choices=["auto", "cos_fft", "measured_calibrated"], default="auto")
    parser.add_argument("--projection-sequence", default="saomiao3_6")
    parser.add_argument("--calibration-label", default="3-6")
    parser.add_argument("--calibration-mat", default="DMD_CCD_Calibration_Dict.mat")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="outputs/latent_opt/t6_hv")
    args = parser.parse_args()

    np.random.seed(args.seed)
    config = data_io.load_config(args.config)
    excluded = data_io.parse_excluded_samples(args.exclude_samples)
    groups = data_io.parse_groups(args.groups, config)
    crop = data_io.infer_crop_for_sample(config, args.sample, args.crop)
    bundle = data_io.load_os_sim_stack(config, args.sample, groups, crop, excluded_samples=excluded)
    Y = bundle["Y"]  # K,G,M,H,W
    z_values = bundle["z_values"]
    out_dir = ROOT / args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "z_values.npy", z_values)

    demod = demodulate_multigroup(Y)
    A_init = demod["A"]
    D0 = demod["D0"]
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
    M = normalize_m(M)
    grid_masks, grid_peaks = build_grid_masks(A_init)
    baseline_loss = modulation_loss(Y, A_init, M)
    baseline_grid_energy = grid_energy_mean(A_init, grid_masks)
    A_opt, losses = optimize_A(
        Y,
        A_init,
        M,
        grid_masks,
        iterations=args.iterations,
        lr=args.lr,
        lambda_sec=args.lambda_sec,
        lambda_smooth=args.lambda_smooth,
        lambda_hv=args.lambda_hv,
        lambda_grid=args.lambda_grid,
    )
    A_h, A_v = A_opt[:, 0], A_opt[:, 1] if A_opt.shape[1] > 1 else A_opt[:, 0]
    A_base_h = A_init[:, 0]
    A_base_v = A_init[:, 1] if A_init.shape[1] > 1 else A_init[:, 0]
    A_fused = fuse_hv(A_h, A_v)
    A_base_fused = fuse_hv(A_base_h, A_base_v)
    np.save(out_dir / "A_h_stack.npy", A_h)
    np.save(out_dir / "A_v_stack.npy", A_v)
    np.save(out_dir / "A_fused_stack.npy", A_fused)
    np.save(out_dir / "A_baseline_fused_stack.npy", A_base_fused)

    height_opt = height_parabolic(A_fused, z_values)
    height_base = height_parabolic(A_base_fused, z_values)
    conf_opt = confidence_metrics(A_fused)
    conf_base = confidence_metrics(A_base_fused)
    np.save(out_dir / "height_optimized_peakfit.npy", height_opt)
    np.save(out_dir / "height_baseline_peakfit.npy", height_base)
    np.save(out_dir / "confidence_optimized_sharpness.npy", conf_opt["sharpness"])

    final_loss = modulation_loss(Y, A_opt, M)
    final_grid_energy = grid_energy_mean(A_opt, grid_masks)
    # Reprojection snapshots.
    mid = Y.shape[0] // 2
    Y_hat = D0[:, :, None, :, :] + A_opt[:, :, None, :, :] * M[None, :, :, :, :]
    residual = Y - Y_hat
    save_loss_curve([baseline_loss] + losses, out_dir / "loss_curve.png")
    save_montage(
        [A_base_h[mid], A_h[mid], A_base_v[mid], A_v[mid], A_base_fused[mid], A_fused[mid]],
        out_dir / "A_baseline_vs_optimized.png",
        ["base H", "opt H", "base V", "opt V", "base fused", "opt fused"],
        cols=2,
    )
    save_montage(
        [Y[mid, 0, 0], Y_hat[mid, 0, 0], Y[mid, 1, 0], Y_hat[mid, 1, 0]],
        out_dir / "reprojection_examples.png",
        ["real H", "hat H", "real V", "hat V"],
        cols=2,
    )
    save_montage(
        [np.abs(residual[mid, 0, 0]), np.abs(residual[mid, 1, 0])],
        out_dir / "residual_examples.png",
        ["abs residual H", "abs residual V"],
        cols=2,
    )
    from src.fft_analysis import fft_magnitude

    save_montage(
        [fft_magnitude(residual[mid, 0, 0]), fft_magnitude(residual[mid, 1, 0])],
        out_dir / "residual_fft.png",
        ["residual FFT H", "residual FFT V"],
        cols=2,
        cmap="magma",
    )
    save_montage(
        [height_base, height_opt, np.abs(height_opt - height_base)],
        out_dir / "height_baseline_vs_optimized.png",
        ["baseline height", "optimized height", "abs diff"],
        cols=3,
        cmap="viridis",
    )
    save_scalar_image(height_opt, out_dir / "height_optimized_map.png", "Optimized fused height")

    metrics = {
        "sample": args.sample,
        "groups": groups,
        "crop": crop.label if crop else "full",
        "backend": args.backend,
        "pattern_source": actual_pattern_source,
        "requested_pattern_source": args.pattern_source,
        "iterations": args.iterations,
        "baseline_modulation_loss": float(baseline_loss),
        "final_modulation_loss": float(final_loss),
        "loss_delta": float(final_loss - baseline_loss),
        "baseline_A_grid_energy": float(baseline_grid_energy),
        "optimized_A_grid_energy": float(final_grid_energy),
        "grid_energy_delta": float(final_grid_energy - baseline_grid_energy),
        "A_baseline_tv": tv_l1(A_base_fused),
        "A_optimized_tv": tv_l1(A_fused),
        "height_baseline": summarize_height(height_base, conf_base),
        "height_optimized": summarize_height(height_opt, conf_opt),
        "pattern_infos": pattern_infos,
        "grid_peaks": grid_peaks,
    }
    data_io.write_json(out_dir / "metrics.json", metrics)
    data_io.write_json(
        out_dir / "config.json",
        {
            "sample": args.sample,
            "groups": groups,
            "exclude_samples": sorted(excluded),
            "crop": crop.label if crop else "full",
            "backend": args.backend,
            "iterations": args.iterations,
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
            "random_seed": args.seed,
        },
    )
    report = [
        "# Latent Optimization Report",
        "",
        f"- Sample: `{args.sample}`",
        f"- Groups: `{', '.join(groups)}`",
        f"- Crop: `{crop.label if crop else 'full'}`",
        f"- Backend: `{args.backend}`",
        f"- Pattern source: `{actual_pattern_source}` (requested `{args.pattern_source}`)",
        f"- Baseline modulation loss: `{baseline_loss:.6g}`",
        f"- Final modulation loss: `{final_loss:.6g}`",
        f"- Baseline A grid energy: `{baseline_grid_energy:.6g}`",
        f"- Optimized A grid energy: `{final_grid_energy:.6g}`",
        "",
        "## Caveat",
        "",
        "This implementation uses fixed forward patterns and optimizes only A maps with smoothness, weak H/V consistency, and FFT grid-energy damping. With `measured_calibrated`, the fixed patterns are measured DMD bitmaps sampled through the DMD-CCD map; it is still a crop-level NumPy/SciPy validation path, not the full PyTorch differentiable warp optimizer.",
    ]
    (out_dir / "latent_opt_report.md").write_text("\n".join(report), encoding="utf-8")
    data_io.append_report(
        ROOT / "outputs/nightly_report.md",
        "P4/P5 Crop Latent Optimization",
        [
            f"- Completed SciPy crop optimization for `{args.sample}` T6 H/V.",
            f"- Baseline modulation loss `{baseline_loss:.6g}`, final `{final_loss:.6g}`.",
            f"- A grid energy `{baseline_grid_energy:.6g}` -> `{final_grid_energy:.6g}`.",
            f"- Saved optimized fused A stack and height under `{out_dir}`.",
        ],
    )
    print(f"Wrote latent optimization outputs to {out_dir}")


if __name__ == "__main__":
    main()
