from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

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
    parser.add_argument("--nominal-step", type=float, default=None)
    parser.add_argument("--output", default="outputs/metrology/v1")
    args = parser.parse_args()

    input_paths = [ROOT / p.strip() for p in args.inputs.split(",") if p.strip()]
    labels = [p.stem for p in input_paths] if not args.labels else [s.strip() for s in args.labels.split(",")]
    if len(labels) != len(input_paths):
        raise ValueError("--labels must match --inputs")
    out_dir = ROOT / args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    height_maps = {}
    method_rows = []
    for label, path in zip(labels, input_paths):
        A = np.load(path).astype(np.float32)
        z_values = load_z(path, args.z_values)
        height = height_parabolic(A, z_values)
        height_maps[label] = height
        method_rows.append({"label": label, "input": str(path), "height_mean": float(np.mean(height)), "height_std": float(np.std(height))})

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
        "",
        "| method | height mean | height std |",
        "|---|---:|---:|",
    ]
    for row in method_rows:
        report.append(f"| {row['label']} | {row['height_mean']:.6g} | {row['height_std']:.6g} |")
    report.append("")
    report.append("No absolute step-height error is reported unless `--nominal-step` and platform masks are supplied.")
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
