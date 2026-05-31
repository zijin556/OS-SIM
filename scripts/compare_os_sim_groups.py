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
from src.fft_analysis import circular_peak_mask, detect_frequency_peaks, masked_fft_energy
from src.height import confidence_metrics, height_parabolic
from src.os_sim import demodulate_phase_stack, fuse_hv
from src.visualization import save_montage


PAIR_DEFS = {
    "t12_3step": ("ossim_t12_h_3step", "ossim_t12_v_3step"),
    "t12_6step": ("ossim_t12_h_6step", "ossim_t12_v_6step"),
    "t6_3step": ("ossim_t6_h_3step", "ossim_t6_v_3step"),
}


def group_short(name: str) -> str:
    return name.replace("ossim_", "")


def stack_for_group(config: dict, sample: str, group_name: str, crop, excluded: set[str]) -> tuple[np.ndarray, np.ndarray]:
    known = data_io.frame_groups_by_name(config)
    layers = data_io.complete_z_layers(config, sample, [group_name], excluded)
    z_values = np.array([z for z, _ in layers], dtype=np.float32)
    A = []
    for z, z_path in layers:
        frames = data_io.read_frame_group(z_path, known[group_name], crop=crop, normalize=True)
        A.append(demodulate_phase_stack(frames)["A"])
    return np.stack(A, axis=0).astype(np.float32), z_values


def group_metrics(A: np.ndarray, z_values: np.ndarray) -> dict:
    height = height_parabolic(A, z_values)
    conf = confidence_metrics(A)
    mid = A.shape[0] // 2
    peaks = detect_frequency_peaks(A[mid], top_n=8, min_radius=4)
    residual_peaks = peaks[:4]
    mask = circular_peak_mask(A[mid].shape, residual_peaks, radius=4) if residual_peaks else np.zeros(A[mid].shape, dtype=bool)
    grid_energy = [masked_fft_energy(A[k], mask) if residual_peaks else 0.0 for k in range(A.shape[0])]
    contrast = (np.percentile(A, 95) - np.percentile(A, 5)) / (float(np.mean(A)) + 1e-8)
    return {
        "shape": list(A.shape),
        "A_contrast_p95_p5_over_mean": float(contrast),
        "grid_energy_mean": float(np.mean(grid_energy)),
        "grid_energy_max": float(np.max(grid_energy)),
        "peak_sharpness_mean": float(np.mean(conf["sharpness"])),
        "peak_to_background_mean": float(np.mean(conf["peak_to_background"])),
        "height_std": float(np.std(height)),
        "height_min": float(np.min(height)),
        "height_max": float(np.max(height)),
        "residual_fft_peaks": residual_peaks,
    }


def rank_pairs(pair_metrics: dict[str, dict]) -> dict:
    if not pair_metrics:
        return {}
    score = {}
    grid_vals = np.array([v["grid_energy_mean"] for v in pair_metrics.values()], dtype=np.float32)
    sharp_vals = np.array([v["peak_sharpness_mean"] for v in pair_metrics.values()], dtype=np.float32)
    # Lower grid energy and higher sharpness are better; use normalized rank-like score.
    for (name, metrics), grid, sharp in zip(pair_metrics.items(), grid_vals, sharp_vals):
        grid_norm = (grid - grid_vals.min()) / (np.ptp(grid_vals) + 1e-8)
        sharp_norm = (sharp - sharp_vals.min()) / (np.ptp(sharp_vals) + 1e-8)
        score[name] = float((1.0 - grid_norm) + sharp_norm)
    best = max(score, key=score.get)
    return {"score": score, "best_by_grid_sharpness_score": best}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare T6/T12 and 3-step/6-step OS-SIM groups on a real crop.")
    parser.add_argument("--config", default="configs/ossim_dataset.yaml")
    parser.add_argument("--sample", default="细台阶")
    parser.add_argument("--exclude-samples", default="台阶曝光1")
    parser.add_argument("--crop", default="center:256")
    parser.add_argument("--output", default="outputs/frequency/group_compare")
    args = parser.parse_args()

    config = data_io.load_config(args.config)
    excluded = data_io.parse_excluded_samples(args.exclude_samples)
    crop = data_io.infer_crop_for_sample(config, args.sample, args.crop)
    out_dir = ROOT / args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    group_names = sorted({name for pair in PAIR_DEFS.values() for name in pair})
    stacks: dict[str, np.ndarray] = {}
    z_ref: np.ndarray | None = None
    group_results = {}
    for group_name in group_names:
        A, z_values = stack_for_group(config, args.sample, group_name, crop, excluded)
        stacks[group_name] = A
        z_ref = z_values if z_ref is None else z_ref
        group_results[group_name] = group_metrics(A, z_values)

    pair_results = {}
    montage_images = []
    montage_titles = []
    for pair_name, (h_name, v_name) in PAIR_DEFS.items():
        A_fused = fuse_hv(stacks[h_name], stacks[v_name])
        pair_results[pair_name] = group_metrics(A_fused, z_ref)
        np.save(out_dir / f"A_{pair_name}_fused_stack.npy", A_fused)
        montage_images.append(A_fused[A_fused.shape[0] // 2])
        montage_titles.append(pair_name)

    ranking = rank_pairs(pair_results)
    save_montage(montage_images, out_dir / "group_fused_sections.png", montage_titles, cols=3)

    names = list(pair_results)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    axes[0].bar(names, [pair_results[n]["grid_energy_mean"] for n in names])
    axes[0].set_title("Grid/Moire energy")
    axes[1].bar(names, [pair_results[n]["peak_sharpness_mean"] for n in names])
    axes[1].set_title("Peak sharpness")
    axes[2].bar(names, [pair_results[n]["height_std"] for n in names])
    axes[2].set_title("Height std")
    for ax in axes:
        ax.tick_params(axis="x", labelrotation=25)
        ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "t6_vs_t12_compare.png", dpi=150)
    plt.close(fig)

    compare = {
        "sample": args.sample,
        "crop": crop.label if crop else "full",
        "excluded_samples": sorted(excluded),
        "group_metrics": group_results,
        "pair_metrics": pair_results,
        "ranking": ranking,
        "interpretation": "Scores are diagnostic only: lower residual grid energy and higher peak sharpness are preferred. No external height truth was used.",
    }
    data_io.write_json(out_dir / "t6_vs_t12_compare.json", compare)
    # Also update the filename expected by the original phase-3 output list.
    legacy_dir = ROOT / "outputs/frequency/t6_hv"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    data_io.write_json(legacy_dir / "t6_vs_t12_compare.json", compare)
    data_io.write_json(out_dir / "config.json", vars(args))
    report = [
        "# T6/T12 OS-SIM Group Comparison",
        "",
        f"- Sample: `{args.sample}`",
        f"- Crop: `{crop.label if crop else 'full'}`",
        f"- Best diagnostic pair: `{ranking.get('best_by_grid_sharpness_score', 'n/a')}`",
        "",
        "## Pair Metrics",
        "",
        "| pair | grid energy mean | peak sharpness mean | height std |",
        "|---|---:|---:|---:|",
    ]
    for name, metrics in pair_results.items():
        report.append(f"| {name} | {metrics['grid_energy_mean']:.6g} | {metrics['peak_sharpness_mean']:.6g} | {metrics['height_std']:.6g} |")
    (out_dir / "frequency_compare_report.md").write_text("\n".join(report), encoding="utf-8")
    data_io.append_report(
        ROOT / "outputs/nightly_report.md",
        "P2b T6/T12 Group Comparison",
        [
            f"- Compared `t12_3step`, `t12_6step`, and `t6_3step` fused H/V stacks for `{args.sample}`.",
            f"- Best diagnostic pair by grid/sharpness score: `{ranking.get('best_by_grid_sharpness_score', 'n/a')}`.",
            f"- Outputs saved under `{out_dir}` and `outputs/frequency/t6_hv/t6_vs_t12_compare.json`.",
        ],
    )
    print(f"Wrote group comparison outputs to {out_dir}")


if __name__ == "__main__":
    main()
