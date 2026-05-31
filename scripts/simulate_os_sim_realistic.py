from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import data_io
from src.forward_models import normalize_m, phases_for_m
from src.height import confidence_metrics, height_parabolic, summarize_height
from src.os_sim import demodulate_phase_stack, fuse_hv
from src.visualization import save_montage, save_scalar_image


def make_height(shape: tuple[int, int], z_min: float, z_max: float) -> np.ndarray:
    h, w = shape
    yy, xx = np.indices(shape, dtype=np.float32)
    ramp = (xx / max(w - 1, 1)) * 0.45 + (yy / max(h - 1, 1)) * 0.20
    bump = 0.35 * np.exp(-(((xx - 0.55 * w) / (0.18 * w)) ** 2 + ((yy - 0.45 * h) / (0.16 * h)) ** 2))
    norm = np.clip(ramp + bump, 0.0, 1.0)
    return (z_min + norm * (z_max - z_min)).astype(np.float32)


def stripe_patterns(shape: tuple[int, int], qxy: tuple[float, float], m: int, phase_error: np.ndarray | None = None) -> np.ndarray:
    h, w = shape
    yy, xx = np.indices(shape, dtype=np.float32)
    phi = qxy[0] * xx + qxy[1] * yy
    phases = phases_for_m(m)
    if phase_error is not None:
        phases = phases + phase_error
    return normalize_m(np.stack([np.cos(phi + p) for p in phases], axis=0))


def synthesize(
    z_values: np.ndarray,
    z_gt: np.ndarray,
    qxy: tuple[float, float],
    mode: str,
    rng: np.random.Generator,
    m: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    shape = z_gt.shape
    sigma_z = 1.0
    flat = 0.18 + 0.04 * rng.normal(size=shape).astype(np.float32)
    D0 = np.clip(0.45 + ndimage.gaussian_filter(flat, 8.0), 0.25, 0.75)
    Y = []
    A_stack = []
    phase_error = np.zeros(m, dtype=np.float32)
    if mode == "realistic":
        phase_error = np.array([0.0, 0.08, -0.05], dtype=np.float32)
    for k, z in enumerate(z_values):
        A = np.exp(-((z - z_gt) ** 2) / (2.0 * sigma_z * sigma_z)).astype(np.float32)
        A *= 0.22 + 0.04 * np.sin(2.0 * np.pi * k / max(len(z_values) - 1, 1))
        M = stripe_patterns(shape, qxy, m=m, phase_error=phase_error)
        if mode == "realistic":
            yy, xx = np.indices(shape, dtype=np.float32)
            Phi = qxy[0] * xx + qxy[1] * yy
            harmonic = 0.05 * np.cos(3.0 * Phi) + 0.02 * np.cos(5.0 * Phi)
            grid = 0.025 * np.cos(2.0 * np.pi * (0.125 * xx + 0.03125 * yy))
            M = normalize_m(M + harmonic[None, :, :] + grid[None, :, :])
            M = normalize_m(np.stack([ndimage.gaussian_filter(f, sigma=0.25 + 0.02 * k, mode="reflect") for f in M], axis=0))
        frames = D0[None, :, :] + A[None, :, :] * M
        if mode == "realistic":
            frames += rng.normal(0.0, 0.015, size=frames.shape).astype(np.float32)
            frames = rng.poisson(np.clip(frames, 0.0, 1.0) * 255.0).astype(np.float32) / 255.0
        else:
            frames += rng.normal(0.0, 0.005, size=frames.shape).astype(np.float32)
        Y.append(np.clip(frames, 0.0, 1.0).astype(np.float32))
        A_stack.append(A)
    return np.stack(Y, axis=0), np.stack(A_stack, axis=0)


def run_mode(mode: str, z_values: np.ndarray, z_gt: np.ndarray, out_dir: Path, rng: np.random.Generator) -> dict:
    Y_h, A_gt = synthesize(z_values, z_gt, (0.0, 2.0 * np.pi / 6.0), mode, rng)
    Y_v, _ = synthesize(z_values, z_gt, (2.0 * np.pi / 6.0, 0.0), mode, rng)
    demod_h = demodulate_phase_stack(Y_h)
    demod_v = demodulate_phase_stack(Y_v)
    A_fused = fuse_hv(demod_h["A"], demod_v["A"])
    z_hat = height_parabolic(A_fused, z_values)
    conf = confidence_metrics(A_fused)
    err = z_hat - z_gt
    metrics = summarize_height(z_hat, conf)
    metrics.update(
        {
            "mode": mode,
            "height_mae": float(np.mean(np.abs(err))),
            "height_rmse": float(np.sqrt(np.mean(err * err))),
            "height_bias": float(np.mean(err)),
        }
    )
    mid = len(z_values) // 2
    save_montage(
        [Y_h[mid, 0], Y_h[mid, 1], Y_h[mid, 2], A_gt[mid], A_fused[mid], z_hat],
        out_dir / f"{mode}_results.png",
        [f"{mode} frame 1", "frame 2", "frame 3", "A_gt mid", "A_fused mid", "z_hat"],
        cols=3,
    )
    save_montage([z_gt, z_hat, np.abs(err)], out_dir / f"{mode}_height_error.png", ["z_gt", "z_hat", "abs error"], cols=3, cmap="viridis")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate ideal and realistic OS-SIM data for sanity checks.")
    parser.add_argument("--shape", default="128x128")
    parser.add_argument("--z-count", type=int, default=31)
    parser.add_argument("--z-min", type=float, default=25.0)
    parser.add_argument("--z-max", type=float, default=40.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="outputs/sim")
    args = parser.parse_args()

    h, w = [int(v) for v in args.shape.lower().split("x", 1)]
    rng = np.random.default_rng(args.seed)
    out_dir = ROOT / args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    z_values = np.linspace(args.z_min, args.z_max, args.z_count, dtype=np.float32)
    z_gt = make_height((h, w), args.z_min + 2.0, args.z_max - 2.0)
    ideal_metrics = run_mode("ideal", z_values, z_gt, out_dir, rng)
    realistic_metrics = run_mode("realistic", z_values, z_gt, out_dir, rng)
    save_scalar_image(z_gt, out_dir / "z_gt.png", "Synthetic z_gt")
    (out_dir / "sim_height_error.png").write_bytes((out_dir / "realistic_height_error.png").read_bytes())
    metrics = {"seed": args.seed, "shape": [h, w], "z_count": args.z_count, "ideal": ideal_metrics, "realistic": realistic_metrics}
    data_io.write_json(out_dir / "sim_metrics.json", metrics)
    data_io.write_json(out_dir / "config.json", vars(args))
    report = [
        "# OS-SIM Simulation Report",
        "",
        f"- Ideal RMSE: `{ideal_metrics['height_rmse']:.6g}`",
        f"- Realistic RMSE: `{realistic_metrics['height_rmse']:.6g}`",
        "",
        "The simulation is only a sanity check for formulas, demodulation, height readout, and degradation sensitivity. It is not used as real training evidence.",
    ]
    (out_dir / "sim_report.md").write_text("\n".join(report), encoding="utf-8")
    data_io.append_report(
        ROOT / "outputs/nightly_report.md",
        "P7 Simulation Sanity Check",
        [
            f"- Ran ideal and realistic synthetic OS-SIM sanity checks at shape `{h}x{w}`.",
            f"- Ideal RMSE `{ideal_metrics['height_rmse']:.6g}`, realistic RMSE `{realistic_metrics['height_rmse']:.6g}`.",
            f"- Outputs saved under `{out_dir}`.",
        ],
    )
    print(f"Wrote simulation outputs to {out_dir}")


if __name__ == "__main__":
    main()
