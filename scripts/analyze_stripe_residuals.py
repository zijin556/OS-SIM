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
from src.visualization import save_montage


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
    data_io.write_json(out_dir / "metrics.json", metrics)
    data_io.write_json(out_dir / "t6_vs_t12_compare.json", {"status": "not_run", "reason": "This run was limited to T6 H/V 3-step groups."})
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
        "- T6-vs-T12 comparison is not claimed in this run because only T6 groups were requested.",
    ]
    (out_dir / "frequency_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    data_io.append_report(
        ROOT / "outputs/nightly_report.md",
        "P2 Frequency Residual Analysis",
        [
            f"- Completed FFT analysis for `{args.sample}` T6 H/V.",
            f"- Saved frequency figures and metrics under `{out_dir}`.",
            "- T6-vs-T12 comparison is intentionally marked `not_run` for this first pass.",
        ],
    )
    print(f"Wrote frequency outputs to {out_dir}")


if __name__ == "__main__":
    main()
