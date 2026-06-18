from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import data_io
from src.fft_analysis import circular_peak_mask, detect_frequency_peaks, masked_fft_energy
from src.height import auto_two_platform_metrics, confidence_metrics, height_parabolic
from src.os_sim import demodulate_multigroup, fuse_hv
from src.visualization import save_montage, save_scalar_image


PAIR_DEFS = {
    "t12_3step": ("ossim_t12_h_3step", "ossim_t12_v_3step"),
    "t12_6step": ("ossim_t12_h_6step", "ossim_t12_v_6step"),
    "t6_3step": ("ossim_t6_h_3step", "ossim_t6_v_3step"),
}


def safe_groups_label(groups: list[str]) -> str:
    if groups == ["ossim_t6_h_3step", "ossim_t6_v_3step"]:
        return "t6_hv"
    return "_".join(g.replace("ossim_", "").replace("_3step", "").replace("_6step", "") for g in groups)


def baseline_candidates(groups: list[str], crop_label: str | None) -> list[Path]:
    label = safe_groups_label(groups)
    smoke = crop_label and "128" in crop_label
    return [
        ROOT / f"outputs/baseline/v1_{label}_smoke",
        ROOT / f"outputs/baseline/v1_{label}",
        ROOT / ("outputs/baseline/t6_hv_smoke128" if smoke else "outputs/baseline/t6_hv"),
        ROOT / "outputs/baseline/t6_hv",
    ]


def load_or_compute_a_stack(
    config: dict[str, Any],
    sample: str,
    groups: list[str],
    crop: data_io.CropSpec | None,
    excluded: set[str],
    baseline_dir: Path | None,
    out_dir: Path,
) -> tuple[np.ndarray, np.ndarray, str]:
    candidates = [baseline_dir] if baseline_dir else []
    candidates += baseline_candidates(groups, crop.label if crop else None)
    for path in candidates:
        if path is None:
            continue
        a_path = path / "A_fused_stack.npy"
        z_path = path / "z_values.npy"
        if a_path.exists() and z_path.exists():
            A = np.load(a_path)
            z = np.load(z_path)
            return A.astype(np.float32), z.astype(np.float32), str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)

    bundle = data_io.load_os_sim_stack(config, sample, groups, crop, excluded_samples=excluded)
    demod = demodulate_multigroup(bundle["Y"])
    A = demod["A"]
    fused = fuse_hv(A[:, 0], A[:, 1] if A.shape[1] > 1 else A[:, 0])
    np.save(out_dir / "input_A_fused_stack.npy", fused)
    np.save(out_dir / "z_values.npy", bundle["z_values"])
    return fused.astype(np.float32), bundle["z_values"].astype(np.float32), "computed_from_raw"


def rect_mask(shape: tuple[int, int], cy: int, cx: int, size: int) -> np.ndarray:
    h, w = shape
    half = size // 2
    y0, y1 = max(0, cy - half), min(h, cy + half)
    x0, x1 = max(0, cx - half), min(w, cx + half)
    mask = np.zeros(shape, dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def best_tile_mask(score: np.ndarray, tile: int, maximize: bool = True) -> np.ndarray:
    h, w = score.shape
    best_value = -np.inf if maximize else np.inf
    best = (h // 2, w // 2)
    step = max(tile, 1)
    for y0 in range(0, h - tile + 1, step):
        for x0 in range(0, w - tile + 1, step):
            value = float(np.nanmean(score[y0 : y0 + tile, x0 : x0 + tile]))
            if (maximize and value > best_value) or (not maximize and value < best_value):
                best_value = value
                best = (y0 + tile // 2, x0 + tile // 2)
    return rect_mask((h, w), best[0], best[1], tile)


def roi_dict(A: np.ndarray, z_values: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    conf = confidence_metrics(A)
    height = height_parabolic(A, z_values)
    h, w = A.shape[-2:]
    tile = max(12, min(h, w) // 8)
    rois: dict[str, np.ndarray] = {
        "center_between_surfaces": rect_mask((h, w), h // 2, w // 2, tile),
        "left_lower_quasi_focus": rect_mask((h, w), int(h * 0.78), int(w * 0.18), tile),
        "right_upper_single_focus": rect_mask((h, w), int(h * 0.22), int(w * 0.78), tile),
        "background_low_return": best_tile_mask(conf["peak"], tile, maximize=False),
        "high_conf_single_peak": best_tile_mask(conf["sharpness"] * conf["peak_to_background"], tile, maximize=True),
    }
    platform = auto_two_platform_metrics(height, invalid_mask=conf["invalid_mask"], erode_px=max(1, tile // 4), min_fraction=0.005)
    if platform.get("status") == "ok":
        rois["platform_low_surface"] = platform["low_mask"]
        rois["platform_high_surface"] = platform["high_mask"]
    return rois, conf


def curve_metrics(curve: np.ndarray, z_values: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(curve, dtype=np.float32)
    z = np.asarray(z_values, dtype=np.float32)
    idx = int(np.argmax(y))
    peak = float(y[idx])
    background = float(np.percentile(y, 10.0))
    half = background + 0.5 * (peak - background)
    above = np.where(y >= half)[0]
    dz = float(np.median(np.diff(z))) if z.size > 1 else 1.0
    fwhm = float((above[-1] - above[0] + 1) * dz) if above.size else 0.0
    sharpness = float((np.max(y) - np.partition(y, -2)[-2]) / (np.max(y) + 1e-8)) if y.size > 1 else 0.0
    p2b = float(peak / (float(np.mean(y)) + 1e-8))
    local = 0
    threshold = background + 0.2 * (peak - background)
    for i in range(1, len(y) - 1):
        if y[i] >= y[i - 1] and y[i] >= y[i + 1] and y[i] >= threshold:
            local += 1
    logits = 20.0 * ((y - y.min()) / (np.ptp(y) + 1e-8))
    probs = np.exp(logits - logits.max())
    probs /= probs.sum() + 1e-8
    entropy = float(-np.sum(probs * np.log(probs + 1e-8)) / np.log(max(len(y), 2)))
    return {
        "peak_z": float(z[idx]),
        "peak_value": peak,
        "fwhm_z_units": fwhm,
        "peak_sharpness": sharpness,
        "peak_to_background": p2b,
        "number_of_peaks": int(local),
        "softmax_entropy": entropy,
        "confidence": float((1.0 - entropy) * max(sharpness, 0.0) * max(p2b, 0.0)),
    }


def summarize_rois(A: np.ndarray, z_values: np.ndarray, rois: dict[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name, mask in rois.items():
        if mask.sum() == 0:
            continue
        pixels = A[:, mask]
        mean_curve = pixels.mean(axis=1)
        median_curve = np.median(pixels, axis=1)
        std_curve = pixels.std(axis=1)
        out[name] = {
            "pixel_count": int(mask.sum()),
            "mean": mean_curve.tolist(),
            "median": median_curve.tolist(),
            "std": std_curve.tolist(),
            "metrics": curve_metrics(mean_curve, z_values),
        }
    return out


def save_roi_locations(A_mid: np.ndarray, rois: dict[str, np.ndarray], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    ax.imshow(A_mid, cmap="gray")
    ax.axis("off")
    colors = plt.cm.tab10(np.linspace(0, 1, len(rois)))
    for color, (name, mask) in zip(colors, rois.items()):
        ys, xs = np.where(mask)
        if ys.size == 0:
            continue
        ax.plot([xs.min(), xs.max(), xs.max(), xs.min(), xs.min()], [ys.min(), ys.min(), ys.max(), ys.max(), ys.min()], color=color, lw=1.8)
        ax.text(xs.min(), ys.min(), name, color=color, fontsize=7, va="bottom")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_curves(summary: dict[str, dict[str, Any]], z_values: np.ndarray, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5))
    for name, row in summary.items():
        mean = np.asarray(row["mean"], dtype=np.float32)
        std = np.asarray(row["std"], dtype=np.float32)
        axes[0].plot(z_values, mean, marker="o", ms=2, label=name)
        axes[0].fill_between(z_values, mean - std, mean + std, alpha=0.12)
        axes[1].plot(z_values, row["median"], marker="o", ms=2, label=name)
    axes[0].set_title("ROI A mean +/- std")
    axes[1].set_title("ROI A median")
    for ax in axes:
        ax.set_xlabel("z")
        ax.set_ylabel("A")
        ax.grid(True, alpha=0.3)
    axes[1].legend(fontsize=7, loc="best")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def group_pair_metrics(config: dict[str, Any], sample: str, crop: data_io.CropSpec | None, excluded: set[str]) -> dict[str, Any]:
    pair_results = {}
    for name, groups in PAIR_DEFS.items():
        try:
            bundle = data_io.load_os_sim_stack(config, sample, list(groups), crop, excluded_samples=excluded)
            demod = demodulate_multigroup(bundle["Y"])
            A = fuse_hv(demod["A"][:, 0], demod["A"][:, 1])
            conf = confidence_metrics(A)
            mid = A.shape[0] // 2
            peaks = detect_frequency_peaks(A[mid], top_n=4, min_radius=4)
            mask = circular_peak_mask(A[mid].shape, peaks, radius=4) if peaks else np.zeros(A[mid].shape, bool)
            grid = [masked_fft_energy(A[k], mask) for k in range(A.shape[0])] if peaks else [0.0]
            pair_results[name] = {
                "status": "ok",
                "mean_fwhm_layers": float(np.mean(conf["fwhm_layers"])),
                "mean_peak_sharpness": float(np.mean(conf["sharpness"])),
                "mean_peak_to_background": float(np.mean(conf["peak_to_background"])),
                "mean_grid_energy": float(np.mean(grid)),
            }
        except Exception as exc:
            pair_results[name] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    ok = {k: v for k, v in pair_results.items() if v.get("status") == "ok"}
    best = None
    if ok:
        score = {
            k: float(v["mean_peak_sharpness"] / (v["mean_grid_energy"] + 1e-8) / (v["mean_fwhm_layers"] + 1e-8))
            for k, v in ok.items()
        }
        best = max(score, key=score.get)
    return {"pair_metrics": pair_results, "best_by_sharp_grid_fwhm": best}


def save_pair_compare(compare: dict[str, Any], path: Path) -> None:
    rows = [(k, v) for k, v in compare["pair_metrics"].items() if v.get("status") == "ok"]
    if not rows:
        return
    names = [r[0] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8))
    axes[0].bar(names, [r[1]["mean_fwhm_layers"] for r in rows])
    axes[0].set_title("FWHM layers")
    axes[1].bar(names, [r[1]["mean_peak_sharpness"] for r in rows])
    axes[1].set_title("Peak sharpness")
    axes[2].bar(names, [r[1]["mean_grid_energy"] for r in rows])
    axes[2].set_title("Residual grid energy")
    for ax in axes:
        ax.tick_params(axis="x", labelrotation=25)
        ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze continuous axial modulation response from real OS-SIM A stacks.")
    parser.add_argument("--config", default="configs/ossim_dataset.yaml")
    parser.add_argument("--sample", default="细台阶")
    parser.add_argument("--groups", default="ossim_t6_h_3step,ossim_t6_v_3step")
    parser.add_argument("--exclude-samples", default="台阶曝光1")
    parser.add_argument("--crop", default="center:256")
    parser.add_argument("--baseline-dir", default=None)
    parser.add_argument("--output", default="outputs/axial_response")
    args = parser.parse_args()

    config = data_io.load_config(args.config)
    groups = data_io.parse_groups(args.groups, config)
    excluded = data_io.parse_excluded_samples(args.exclude_samples)
    crop = data_io.infer_crop_for_sample(config, args.sample, args.crop)
    out_dir = ROOT / args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline_dir = ROOT / args.baseline_dir if args.baseline_dir else None
    A, z_values, source = load_or_compute_a_stack(config, args.sample, groups, crop, excluded, baseline_dir, out_dir)
    np.save(out_dir / "A_fused_stack.npy", A)
    np.save(out_dir / "z_values.npy", z_values)

    rois, conf = roi_dict(A, z_values)
    summary = summarize_rois(A, z_values, rois)
    height = height_parabolic(A, z_values)
    confidence_map = (1.0 - conf["softmax_entropy"]) * np.maximum(conf["sharpness"], 0.0)
    compare = group_pair_metrics(config, args.sample, crop, excluded)

    mid = A.shape[0] // 2
    save_roi_locations(A[mid], rois, out_dir / "roi_locations.png")
    save_curves(summary, z_values, out_dir / "axial_curves.png")
    save_scalar_image(conf["fwhm_layers"], out_dir / "fwhm_map.png", "FWHM layers", cmap="magma")
    save_scalar_image(height, out_dir / "peak_z_map.png", "Peak z map", cmap="viridis")
    save_scalar_image(confidence_map, out_dir / "confidence_map.png", "Axial confidence", cmap="magma")
    save_pair_compare(compare, out_dir / "t6_t12_response_compare.png")
    save_montage([A[0], A[mid], A[-1], height, conf["fwhm_layers"], confidence_map], out_dir / "axial_summary.png", ["A first", "A mid", "A last", "peak z", "FWHM", "confidence"], cols=3, cmap="viridis")

    metrics = {
        "sample": args.sample,
        "groups": groups,
        "crop": crop.label if crop else "full",
        "input_source": source,
        "z_values": z_values.tolist(),
        "roi_summary": summary,
        "global": {
            "mean_fwhm_layers": float(np.mean(conf["fwhm_layers"])),
            "mean_peak_sharpness": float(np.mean(conf["sharpness"])),
            "mean_peak_to_background": float(np.mean(conf["peak_to_background"])),
            "invalid_fraction": float(np.mean(conf["invalid_mask"])),
        },
        "t6_t12_compare": compare,
    }
    data_io.write_json(out_dir / "metrics.json", metrics)
    data_io.write_json(out_dir / "config.json", vars(args))

    best_pair = compare.get("best_by_sharp_grid_fwhm")
    strong_single = summary.get("right_upper_single_focus", {}).get("metrics", {})
    left = summary.get("left_lower_quasi_focus", {}).get("metrics", {})
    report = [
        "# Axial Modulation Response Report",
        "",
        f"- Sample: `{args.sample}`",
        f"- Groups: `{', '.join(groups)}`",
        f"- Crop: `{crop.label if crop else 'full'}`",
        f"- Input source: `{source}`",
        "",
        "## Required Answers",
        "",
        f"- Continuous response support: ROI curves are saved in `axial_curves.png`; mean global FWHM is `{metrics['global']['mean_fwhm_layers']:.6g}` z layers, so the response is not treated as binary in this diagnostic.",
        f"- Quasi-focus state: left-lower ROI peak count `{left.get('number_of_peaks', 'missing')}`, FWHM `{left.get('fwhm_z_units', 'missing')}`, entropy `{left.get('softmax_entropy', 'missing')}`.",
        f"- Right-upper single-focus state: peak count `{strong_single.get('number_of_peaks', 'missing')}`, sharpness `{strong_single.get('peak_sharpness', 'missing')}`, peak/background `{strong_single.get('peak_to_background', 'missing')}`.",
        f"- T6/T12 diagnostic winner by sharpness/grid/FWHM score: `{best_pair}`.",
        "- This evidence can support introducing a continuous Stokseth-style modulation-response interpretation only after ROI curves are inspected against sample geometry and residual FFT diagnostics.",
        "",
        "## ROI Metrics",
        "",
        "| ROI | pixels | peak z | FWHM | peaks | entropy | confidence |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in summary.items():
        m = row["metrics"]
        report.append(
            f"| {name} | {row['pixel_count']} | {m['peak_z']:.6g} | {m['fwhm_z_units']:.6g} | {m['number_of_peaks']} | {m['softmax_entropy']:.6g} | {m['confidence']:.6g} |"
        )
    (out_dir / "axial_response_report.md").write_text("\n".join(report), encoding="utf-8")
    data_io.append_report(
        ROOT / "outputs/nightly_report.md",
        "P2 v1 Axial Modulation Response",
        [
            f"- Generated ROI A(z) curves and axial maps for `{args.sample}` under `{out_dir}`.",
            f"- T6/T12 diagnostic winner: `{best_pair}`.",
        ],
    )
    print(f"Wrote axial response outputs to {out_dir}")


if __name__ == "__main__":
    main()
