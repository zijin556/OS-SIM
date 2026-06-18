from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import data_io
from src.height import height_parabolic
from src.metrology import repeatability_metrics, standard_step_metrics
from src.visualization import save_montage


def load_z(input_path: Path, z_arg: str | None) -> np.ndarray:
    candidates = []
    if z_arg:
        candidates.append(ROOT / z_arg)
    candidates.append(input_path.parent / "z_values.npy")
    candidates.append(ROOT / "outputs/baseline/t6_hv/z_values.npy")
    for path in candidates:
        if path.exists():
            return np.load(path).astype(np.float32)
    A = np.load(input_path, mmap_mode="r")
    return np.arange(A.shape[0], dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate metrology diagnostics without claiming absolute truth unless a nominal standard is supplied.")
    parser.add_argument("--inputs", default="outputs/baseline/t6_hv/A_fused_stack.npy", help="Comma-separated A_stack .npy paths.")
    parser.add_argument("--labels", default=None, help="Comma-separated labels matching --inputs.")
    parser.add_argument("--z-values", default=None)
    parser.add_argument("--platform-low-mask", default=None)
    parser.add_argument("--platform-high-mask", default=None)
    parser.add_argument("--confidence-invalid-mask", default=None)
    parser.add_argument("--add-simple-filters", action="store_true", help="Add lowpass/median filtered variants of the first input A stack.")
    parser.add_argument("--nominal-step", type=float, default=None)
    parser.add_argument("--output", default="outputs/metrology/v1")
    args = parser.parse_args()

    input_paths = [ROOT / p.strip() for p in args.inputs.split(",") if p.strip()]
    labels = [p.stem for p in input_paths] if not args.labels else [s.strip() for s in args.labels.split(",")]
    if len(labels) != len(input_paths):
        raise ValueError("--labels must match --inputs")
    out_dir = ROOT / args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    invalid_mask = None
    if args.confidence_invalid_mask:
        mask_path = ROOT / args.confidence_invalid_mask
        if mask_path.exists():
            invalid_mask = np.load(mask_path).astype(bool)

    height_maps = {}
    method_rows = []
    loaded_a_stacks: dict[str, tuple[np.ndarray, np.ndarray, Path]] = {}
    for label, path in zip(labels, input_paths):
        A = np.load(path).astype(np.float32)
        z_values = load_z(path, args.z_values)
        loaded_a_stacks[label] = (A, z_values, path)
        height = height_parabolic(A, z_values)
        height_maps[label] = height
        row = {"label": label, "input": str(path), "height_mean": float(np.mean(height)), "height_std": float(np.std(height))}
        if invalid_mask is not None and invalid_mask.shape == height.shape:
            valid = ~invalid_mask
            row.update(
                {
                    "confidence_masked_height_mean": float(np.mean(height[valid])) if np.any(valid) else None,
                    "confidence_masked_height_std": float(np.std(height[valid])) if np.any(valid) else None,
                    "confidence_valid_fraction": float(np.mean(valid)),
                }
            )
        method_rows.append(row)

    if args.add_simple_filters and loaded_a_stacks:
        first_label = labels[0]
        A0, z0, p0 = loaded_a_stacks[first_label]
        filter_defs = {
            f"{first_label}_lowpass_sigma1": ndimage.gaussian_filter(A0, sigma=(0, 1.0, 1.0), mode="reflect").astype(np.float32),
            f"{first_label}_median3": ndimage.median_filter(A0, size=(1, 3, 3), mode="reflect").astype(np.float32),
        }
        for label, A in filter_defs.items():
            height = height_parabolic(A, z0)
            height_maps[label] = height
            row = {
                "label": label,
                "input": f"generated_filter_from:{p0}",
                "height_mean": float(np.mean(height)),
                "height_std": float(np.std(height)),
            }
            if invalid_mask is not None and invalid_mask.shape == height.shape:
                valid = ~invalid_mask
                row.update(
                    {
                        "confidence_masked_height_mean": float(np.mean(height[valid])) if np.any(valid) else None,
                        "confidence_masked_height_std": float(np.std(height[valid])) if np.any(valid) else None,
                        "confidence_valid_fraction": float(np.mean(valid)),
                    }
                )
            method_rows.append(row)

    standard = {"status": "not_run", "reason": "platform masks not supplied"}
    low_mask = high_mask = None
    if args.platform_low_mask and args.platform_high_mask:
        low_mask = np.load(ROOT / args.platform_low_mask).astype(bool)
        high_mask = np.load(ROOT / args.platform_high_mask).astype(bool)
        standard = {
            label: standard_step_metrics(height, low_mask, high_mask, args.nominal_step)
            for label, height in height_maps.items()
        }

    repeatability = repeatability_metrics(height_maps)
    save_montage(list(height_maps.values()), out_dir / "height_maps.png", list(height_maps.keys()), cols=min(3, len(height_maps)), cmap="viridis")
    # Center-line profiles make oversmoothing and step-position shifts visible without requiring ground truth.
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    for label, height in height_maps.items():
        y = height.shape[0] // 2
        ax.plot(height[y], lw=1.1, label=label)
    ax.set_title("Center-row height profiles")
    ax.set_xlabel("x pixel")
    ax.set_ylabel("height coordinate")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "profile_compare.png", dpi=150)
    plt.close(fig)

    if repeatability.get("status") == "ok":
        stack = np.stack([np.asarray(v, dtype=np.float32) for v in height_maps.values()], axis=0)
        std_map = np.std(stack, axis=0)
        fig, ax = plt.subplots(figsize=(6.0, 5.0))
        im = ax.imshow(std_map, cmap="magma")
        ax.set_title("Internal consistency: pixelwise std across methods")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(out_dir / "repeatability.png", dpi=150)
        plt.close(fig)
    else:
        fig, ax = plt.subplots(figsize=(7.0, 4.0))
        ax.bar([row["label"] for row in method_rows], [row["height_std"] for row in method_rows])
        ax.set_title("Height std by method")
        ax.tick_params(axis="x", labelrotation=25)
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / "repeatability.png", dpi=150)
        plt.close(fig)

    if low_mask is not None and high_mask is not None:
        roi_vis = np.zeros_like(next(iter(height_maps.values())), dtype=np.float32)
        roi_vis[low_mask] = 1.0
        roi_vis[high_mask] = 2.0
        save_montage([next(iter(height_maps.values())), roi_vis], out_dir / "platform_rois.png", ["height", "platform ROIs"], cols=2, cmap="viridis")

    metrics = {
        "inputs": method_rows,
        "repeatability": repeatability,
        "standard_step": standard,
        "absolute_metrology_claim": bool(args.nominal_step is not None and args.platform_low_mask and args.platform_high_mask),
        "confidence_invalid_mask": {
            "path": args.confidence_invalid_mask,
            "loaded": bool(invalid_mask is not None),
            "invalid_fraction": None if invalid_mask is None else float(np.mean(invalid_mask)),
        },
        "interpretation": "Without a nominal standard step or independent truth, report only repeatability/internal consistency.",
    }
    data_io.write_json(out_dir / "metrology_metrics.json", metrics)
    data_io.write_json(out_dir / "config.json", vars(args))
    report = [
        "# Metrology Report",
        "",
        f"- Methods evaluated: `{len(method_rows)}`",
        f"- Repeatability status: `{repeatability.get('status')}`",
        f"- Absolute metrology claim allowed: `{metrics['absolute_metrology_claim']}`",
        f"- Confidence mask loaded: `{metrics['confidence_invalid_mask']['loaded']}`",
        "",
        "| method | height mean | height std | masked mean | masked std |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in method_rows:
        masked_mean = row.get("confidence_masked_height_mean")
        masked_std = row.get("confidence_masked_height_std")
        report.append(
            f"| {row['label']} | {row['height_mean']:.6g} | {row['height_std']:.6g} | {'' if masked_mean is None else f'{masked_mean:.6g}'} | {'' if masked_std is None else f'{masked_std:.6g}'} |"
        )
    report.append("")
    report.append("No absolute step-height error is reported unless `--nominal-step` and platform masks are supplied. Cross-method pixelwise std is internal consistency, not true repeatability across independent scans.")
    (out_dir / "metrology_report.md").write_text("\n".join(report), encoding="utf-8")
    data_io.append_report(
        ROOT / "outputs/nightly_report.md",
        "P9 v1 Metrology Diagnostics",
        [
            f"- Evaluated metrology diagnostics under `{out_dir}`.",
            f"- Absolute metrology claim allowed: `{metrics['absolute_metrology_claim']}`.",
        ],
    )
    print(f"Wrote metrology outputs to {out_dir}")


if __name__ == "__main__":
    main()
