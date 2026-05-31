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
from src.fft_analysis import circular_peak_mask, detect_frequency_peaks, fft_magnitude, masked_fft_energy, stripe_orientation_from_peak
from src.forward_models import period_from_group
from src.height import confidence_metrics, height_parabolic
from src.os_sim import demodulate_phase_stack, fuse_hv
from src.visualization import save_montage


PAIR_DEFS = {
    "t12_3step": ("ossim_t12_h_3step", "ossim_t12_v_3step"),
    "t12_6step": ("ossim_t12_h_6step", "ossim_t12_v_6step"),
    "t6_3step": ("ossim_t6_h_3step", "ossim_t6_v_3step"),
}


def safe_label(group_name: str) -> str:
    return group_name.replace("ossim_", "").replace("_3step", "")


def select_nominal_stripe_peak(peaks: list[dict], group: dict) -> dict | None:
    if not peaks:
        return None
    target = 1.0 / period_from_group(group)
    direction = group.get("direction", "horizontal")
    if direction == "horizontal":
        candidates = [p for p in peaks if abs(p["freq_y_cycles_per_px"]) >= abs(p["freq_x_cycles_per_px"])]
        key = lambda p: abs(abs(p["freq_y_cycles_per_px"]) - target) + 0.25 * abs(p["freq_x_cycles_per_px"])
    else:
        candidates = [p for p in peaks if abs(p["freq_x_cycles_per_px"]) >= abs(p["freq_y_cycles_per_px"])]
        key = lambda p: abs(abs(p["freq_x_cycles_per_px"]) - target) + 0.25 * abs(p["freq_y_cycles_per_px"])
    return min(candidates or peaks, key=key)


def load_baseline_stack(baseline_dir: Path, group_name: str) -> np.ndarray:
    path = baseline_dir / f"A_{safe_label(group_name)}_stack.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing baseline stack {path}. Run baseline_os_sim.py first.")
    return np.load(path, mmap_mode="r")


def align_to_probe(section: np.ndarray, probe_shape: tuple[int, int], crop: data_io.CropSpec | None) -> np.ndarray:
    """Align a saved baseline section to the currently loaded raw probe size."""
    arr = np.asarray(section)
    if arr.shape == probe_shape:
        return arr
    if crop is not None and crop.y.stop <= arr.shape[0] and crop.x.stop <= arr.shape[1]:
        cropped = arr[crop.y, crop.x]
        if cropped.shape == probe_shape:
            return cropped
    if arr.shape[0] >= probe_shape[0] and arr.shape[1] >= probe_shape[1]:
        y0 = (arr.shape[0] - probe_shape[0]) // 2
        x0 = (arr.shape[1] - probe_shape[1]) // 2
        return arr[y0 : y0 + probe_shape[0], x0 : x0 + probe_shape[1]]
    raise ValueError(f"Cannot align baseline section shape {arr.shape} to raw probe shape {probe_shape}.")


def stack_for_group(config: dict, sample: str, group_name: str, crop, excluded: set[str]) -> tuple[np.ndarray, np.ndarray]:
    known = data_io.frame_groups_by_name(config)
    layers = data_io.complete_z_layers(config, sample, [group_name], excluded)
    z_values = np.array([z for z, _ in layers], dtype=np.float32)
    A = []
    for _, z_path in layers:
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
    return {
        "shape": list(A.shape),
        "grid_energy_mean": float(np.mean(grid_energy)),
        "grid_energy_max": float(np.max(grid_energy)),
        "peak_sharpness_mean": float(np.mean(conf["sharpness"])),
        "peak_to_background_mean": float(np.mean(conf["peak_to_background"])),
        "fwhm_layers_mean": float(np.mean(conf["fwhm_layers"])),
        "height_std": float(np.std(height)),
        "height_min": float(np.min(height)),
        "height_max": float(np.max(height)),
        "residual_fft_peaks": residual_peaks,
    }


def compare_pairs(config: dict, sample: str, crop, excluded: set[str]) -> dict:
    group_names = sorted({name for pair in PAIR_DEFS.values() for name in pair})
    stacks: dict[str, np.ndarray] = {}
    z_ref: np.ndarray | None = None
    group_results = {}
    for group_name in group_names:
        try:
            A, z_values = stack_for_group(config, sample, group_name, crop, excluded)
            stacks[group_name] = A
            z_ref = z_values if z_ref is None else z_ref
            group_results[group_name] = group_metrics(A, z_values)
        except Exception as exc:
            group_results[group_name] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    pair_results = {}
    for pair_name, (h_name, v_name) in PAIR_DEFS.items():
        if h_name not in stacks or v_name not in stacks or z_ref is None:
            pair_results[pair_name] = {"status": "failed", "reason": "missing_stack"}
            continue
        A_fused = fuse_hv(stacks[h_name], stacks[v_name])
        pair_results[pair_name] = {"status": "ok", **group_metrics(A_fused, z_ref)}
    ok = {name: row for name, row in pair_results.items() if row.get("status") == "ok"}
    ranking: dict[str, object] = {}
    if ok:
        grid_vals = np.array([v["grid_energy_mean"] for v in ok.values()], dtype=np.float32)
        sharp_vals = np.array([v["peak_sharpness_mean"] for v in ok.values()], dtype=np.float32)
        fwhm_vals = np.array([v["fwhm_layers_mean"] for v in ok.values()], dtype=np.float32)
        score = {}
        for (name, metrics), grid, sharp, fwhm in zip(ok.items(), grid_vals, sharp_vals, fwhm_vals):
            grid_norm = (grid - grid_vals.min()) / (np.ptp(grid_vals) + 1e-8)
            sharp_norm = (sharp - sharp_vals.min()) / (np.ptp(sharp_vals) + 1e-8)
            fwhm_norm = (fwhm - fwhm_vals.min()) / (np.ptp(fwhm_vals) + 1e-8)
            score[name] = float((1.0 - grid_norm) + sharp_norm + (1.0 - fwhm_norm))
        ranking = {"score": score, "best_by_grid_sharpness_fwhm_score": max(score, key=score.get)}
    return {
        "sample": sample,
        "crop": crop.label if crop else "full",
        "group_metrics": group_results,
        "pair_metrics": pair_results,
        "ranking": ranking,
        "interpretation": "Diagnostic only: lower residual grid energy, higher peak sharpness, and narrower FWHM are preferred. No external height truth was used.",
    }


def save_pair_compare(compare: dict, out_dir: Path) -> None:
    rows = [(name, row) for name, row in compare.get("pair_metrics", {}).items() if row.get("status") == "ok"]
    if not rows:
        return
    names = [name for name, _ in rows]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    axes[0].bar(names, [row["grid_energy_mean"] for _, row in rows])
    axes[0].set_title("Grid/Moire energy")
    axes[1].bar(names, [row["peak_sharpness_mean"] for _, row in rows])
    axes[1].set_title("Peak sharpness")
    axes[2].bar(names, [row["fwhm_layers_mean"] for _, row in rows])
    axes[2].set_title("FWHM layers")
    for ax in axes:
        ax.tick_params(axis="x", labelrotation=25)
        ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "t6_vs_t12_compare.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze raw/A-stack FFT residual stripe and grid energy.")
    parser.add_argument("--config", default="configs/ossim_dataset.yaml")
    parser.add_argument("--sample", default="细台阶")
    parser.add_argument("--groups", default="ossim_t6_h_3step,ossim_t6_v_3step")
    parser.add_argument("--exclude-samples", default="台阶曝光1")
    parser.add_argument("--crop", default="center:512")
    parser.add_argument("--baseline-dir", default="outputs/baseline/t6_hv")
    parser.add_argument("--output", default="outputs/frequency/t6_hv")
    args = parser.parse_args()

    config = data_io.load_config(args.config)
    excluded = data_io.parse_excluded_samples(args.exclude_samples)
    groups = data_io.parse_groups(args.groups, config)
    crop = data_io.infer_crop_for_sample(config, args.sample, args.crop)
    known = data_io.frame_groups_by_name(config)
    layers = data_io.complete_z_layers(config, args.sample, groups, excluded)
    out_dir = ROOT / args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline_dir = ROOT / args.baseline_dir
    z_values = np.load(baseline_dir / "z_values.npy") if (baseline_dir / "z_values.npy").exists() else np.array([z for z, _ in layers])

    metrics = {"sample": args.sample, "groups": groups, "crop": crop.label if crop else "full", "groups_detail": {}}
    raw_fft_images = []
    raw_fft_titles = []
    section_fft_images = []
    section_fft_titles = []

    fig_grid, ax_grid = plt.subplots(figsize=(7, 4))
    fig_stripe, ax_stripe = plt.subplots(figsize=(7, 4))

    mid_layer = layers[len(layers) // 2]
    for group_name in groups:
        group = known[group_name]
        frames = data_io.read_frame_group(mid_layer[1], group, crop=crop, normalize=True)
        y_tilde = frames - frames.mean(axis=0, keepdims=True)
        raw_probe = y_tilde[0]
        raw_peaks = detect_frequency_peaks(raw_probe, top_n=8, min_radius=4)
        main_peak = select_nominal_stripe_peak(raw_peaks, group)
        A_stack = load_baseline_stack(baseline_dir, group_name)
        # If baseline was generated on a different crop/full image, crop it here for comparable FFT size.
        A_mid = align_to_probe(A_stack[len(A_stack) // 2], raw_probe.shape, crop)
        section_peaks = detect_frequency_peaks(A_mid, top_n=8, min_radius=4)
        residual_peaks = section_peaks[:4]
        mask = circular_peak_mask(A_mid.shape, residual_peaks, radius=4) if residual_peaks else np.zeros_like(A_mid, dtype=bool)
        grid_energy = []
        stripe_energy = []
        if main_peak is not None:
            stripe_mask = circular_peak_mask(A_mid.shape, [main_peak], radius=4)
        else:
            stripe_mask = np.zeros_like(A_mid, dtype=bool)
        for k in range(A_stack.shape[0]):
            section = align_to_probe(A_stack[k], raw_probe.shape, crop)
            grid_energy.append(masked_fft_energy(section, mask) if residual_peaks else 0.0)
            stripe_energy.append(masked_fft_energy(section, stripe_mask) if main_peak is not None else 0.0)
        ax_grid.plot(z_values[: len(grid_energy)], grid_energy, marker="o", ms=2, label=group_name)
        ax_stripe.plot(z_values[: len(stripe_energy)], stripe_energy, marker="o", ms=2, label=group_name)
        raw_fft_images.append(fft_magnitude(raw_probe))
        raw_fft_titles.append(f"raw {group_name}")
        section_fft_images.append(fft_magnitude(A_mid))
        section_fft_titles.append(f"A {group_name}")
        metrics["groups_detail"][group_name] = {
            "main_raw_peaks": raw_peaks,
            "selected_nominal_main_peak": main_peak,
            "main_peak_orientation": stripe_orientation_from_peak(main_peak) if main_peak else "not_detected",
            "section_residual_peaks": section_peaks,
            "grid_energy_mean": float(np.mean(grid_energy)),
            "grid_energy_max": float(np.max(grid_energy)),
            "stripe_energy_mean": float(np.mean(stripe_energy)),
        }

    ax_grid.set_title("A-stack residual grid/Moire energy over z")
    ax_grid.set_xlabel("z")
    ax_grid.set_ylabel("masked FFT energy ratio")
    ax_grid.grid(True, alpha=0.3)
    ax_grid.legend()
    fig_grid.tight_layout()
    fig_grid.savefig(out_dir / "grid_energy_over_z.png", dpi=150)
    plt.close(fig_grid)

    ax_stripe.set_title("Main stripe-frequency leakage over z")
    ax_stripe.set_xlabel("z")
    ax_stripe.set_ylabel("masked FFT energy ratio")
    ax_stripe.grid(True, alpha=0.3)
    ax_stripe.legend()
    fig_stripe.tight_layout()
    fig_stripe.savefig(out_dir / "stripe_peak_over_z.png", dpi=150)
    plt.close(fig_stripe)

    save_montage(raw_fft_images, out_dir / "raw_fft_examples.png", raw_fft_titles, cols=2, cmap="magma")
    save_montage(section_fft_images, out_dir / "section_fft_examples.png", section_fft_titles, cols=2, cmap="magma")
    compare = compare_pairs(config, args.sample, crop, excluded)
    save_pair_compare(compare, out_dir)
    data_io.write_json(out_dir / "metrics.json", metrics)
    data_io.write_json(out_dir / "t6_vs_t12_compare.json", compare)
    data_io.write_json(
        out_dir / "config.json",
        {
            "sample": args.sample,
            "groups": groups,
            "exclude_samples": sorted(excluded),
            "crop": crop.label if crop else "full",
            "baseline_dir": str(baseline_dir),
        },
    )
    report_lines = [
        "# Frequency Analysis Report",
        "",
        f"- Sample: `{args.sample}`",
        f"- Groups: `{', '.join(groups)}`",
        f"- Crop: `{crop.label if crop else 'full'}`",
        "",
        "## Detected Peaks",
        "",
    ]
    for group_name, detail in metrics["groups_detail"].items():
        report_lines += [
            f"### {group_name}",
            "",
            f"- Raw main orientation: `{detail['main_peak_orientation']}`",
            f"- Selected nominal main peak: `{detail['selected_nominal_main_peak']}`",
            f"- Mean residual grid energy: `{detail['grid_energy_mean']:.6g}`",
            f"- Mean stripe leakage energy: `{detail['stripe_energy_mean']:.6g}`",
            f"- Top raw peaks: `{detail['main_raw_peaks'][:4]}`",
            f"- Top A-stack residual peaks: `{detail['section_residual_peaks'][:4]}`",
            "",
        ]
    report_lines += [
        "## Interpretation",
        "",
        "- T6 H/V directions are analyzed independently; direction-dependent residual peaks should be treated as nuisance terms before any network stage.",
        f"- T6-vs-T12 diagnostic winner: `{compare.get('ranking', {}).get('best_by_grid_sharpness_fwhm_score', 'n/a')}`.",
    ]
    (out_dir / "frequency_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    data_io.append_report(
        ROOT / "outputs/nightly_report.md",
        "P2 Frequency Residual Analysis",
        [
            f"- Completed FFT analysis for `{args.sample}` T6 H/V.",
            f"- Saved frequency figures and metrics under `{out_dir}`.",
            f"- T6-vs-T12 diagnostic winner: `{compare.get('ranking', {}).get('best_by_grid_sharpness_fwhm_score', 'n/a')}`.",
        ],
    )
    print(f"Wrote frequency outputs to {out_dir}")


if __name__ == "__main__":
    main()
