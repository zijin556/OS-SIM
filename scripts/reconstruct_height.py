from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import data_io
from src.height import (
    auto_two_platform_metrics,
    confidence_metrics,
    height_argmax,
    height_gaussian,
    height_parabolic,
    height_softargmax,
    platform_metrics_from_masks,
    summarize_height,
)
from src.visualization import save_montage, save_profiles, save_scalar_image


def find_z_values(input_path: Path, z_arg: str | None) -> np.ndarray | None:
    candidates = []
    if z_arg:
        candidates.append(ROOT / z_arg)
        candidates.append(Path(z_arg))
    candidates.append(input_path.parent / "z_values.npy")
    candidates.append(ROOT / "outputs/baseline/t6_hv/z_values.npy")
    for path in candidates:
        if path.exists():
            return np.load(path)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruct height maps from an A_stack.")
    parser.add_argument("--sample", default="细台阶")
    parser.add_argument("--input", default="outputs/latent_opt/t6_hv/A_fused_stack.npy")
    parser.add_argument("--z-values", default=None)
    parser.add_argument("--output", default="outputs/height/t6_hv")
    parser.add_argument("--auto-platform", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--platform-erode-px", type=int, default=5)
    parser.add_argument("--platform-low-mask", default=None)
    parser.add_argument("--platform-high-mask", default=None)
    args = parser.parse_args()

    input_path = ROOT / args.input
    A = np.load(input_path)
    z_values = find_z_values(input_path, args.z_values)
    if z_values is None:
        z_values = np.arange(A.shape[0], dtype=np.float32)
    out_dir = ROOT / args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "z_values.npy", z_values)

    h_arg = height_argmax(A, z_values)
    h_par = height_parabolic(A, z_values)
    h_gau = height_gaussian(A, z_values)
    h_soft = height_softargmax(A, z_values)
    conf = confidence_metrics(A)
    np.save(out_dir / "height_argmax.npy", h_arg)
    np.save(out_dir / "height_parabolic.npy", h_par)
    np.save(out_dir / "height_gaussian.npy", h_gau)
    np.save(out_dir / "height_softargmax.npy", h_soft)
    np.save(out_dir / "confidence_sharpness.npy", conf["sharpness"])
    np.save(out_dir / "confidence_peak_to_background.npy", conf["peak_to_background"])
    np.save(out_dir / "invalid_mask.npy", conf["invalid_mask"])
    save_montage([h_arg, h_par, h_gau, h_soft], out_dir / "height_maps.png", ["argmax", "parabolic", "gaussian", "softargmax"], cols=2, cmap="viridis")
    save_scalar_image(conf["sharpness"], out_dir / "confidence_map.png", "Peak sharpness", cmap="magma")
    save_profiles(A, z_values, out_dir / "profiles.png", "Height reconstruction A profiles")
    metrics = summarize_height(h_par, conf)
    metrics.update({"sample": args.sample, "input": str(input_path), "z_unit": "unknown_scan_coordinate"})
    platform = {"status": "not_run"}
    if args.platform_low_mask and args.platform_high_mask:
        low_mask = np.load(ROOT / args.platform_low_mask).astype(bool)
        high_mask = np.load(ROOT / args.platform_high_mask).astype(bool)
        platform = platform_metrics_from_masks(h_par, low_mask, high_mask)
        if platform.get("status") == "ok":
            np.save(out_dir / "platform_low_mask.npy", low_mask)
            np.save(out_dir / "platform_high_mask.npy", high_mask)
            roi_vis = np.zeros_like(h_par, dtype=np.float32)
            roi_vis[low_mask] = 1.0
            roi_vis[high_mask] = 2.0
            save_montage([h_par, roi_vis], out_dir / "platform_rois.png", ["height", "provided platform ROIs"], cols=2, cmap="viridis")
    elif args.auto_platform:
        platform = auto_two_platform_metrics(h_par, invalid_mask=conf["invalid_mask"], erode_px=args.platform_erode_px)
        platform_json = {k: v for k, v in platform.items() if k not in {"low_mask", "high_mask"}}
        metrics["platform"] = platform_json
        if platform.get("status") == "ok":
            low_mask = platform["low_mask"]
            high_mask = platform["high_mask"]
            np.save(out_dir / "platform_low_mask.npy", low_mask)
            np.save(out_dir / "platform_high_mask.npy", high_mask)
            roi_vis = np.zeros_like(h_par, dtype=np.float32)
            roi_vis[low_mask] = 1.0
            roi_vis[high_mask] = 2.0
            save_montage([h_par, roi_vis], out_dir / "platform_rois.png", ["height", "auto platform ROIs"], cols=2, cmap="viridis")
    platform_json = {k: v for k, v in platform.items() if k not in {"low_mask", "high_mask"}}
    metrics["platform"] = platform_json
    data_io.write_json(out_dir / "platform_metrics.json", metrics)
    data_io.write_json(out_dir / "metrics.json", metrics)
    data_io.write_json(
        out_dir / "config.json",
        {
            "sample": args.sample,
            "input": str(input_path),
            "z_values": "auto",
            "output": str(out_dir),
            "auto_platform": args.auto_platform,
            "platform_erode_px": args.platform_erode_px,
            "platform_low_mask": args.platform_low_mask,
            "platform_high_mask": args.platform_high_mask,
        },
    )
    report = [
        "# Height Reconstruction Report",
        "",
        f"- Sample: `{args.sample}`",
        f"- Input A stack: `{input_path}`",
        f"- z coordinate unit: `unknown_scan_coordinate`",
        f"- Height mean/std: `{metrics['height_mean']:.6g}` / `{metrics['height_std']:.6g}`",
        "",
    ]
    if platform.get("status") == "ok":
        report += [
            "## Auto Platform Metrics",
            "",
            f"- Step height: `{platform['step_height']:.6g}` scan-coordinate units.",
            f"- Low platform RMS: `{platform['low_platform']['rms']:.6g}`.",
            f"- High platform RMS: `{platform['high_platform']['rms']:.6g}`.",
            f"- Pooled platform RMS: `{platform['pooled_platform_rms']:.6g}`.",
            "",
            "No calibrated reference step height was provided, so absolute step-height error is not reported.",
        ]
    else:
        report.append(f"Auto platform ROI status: `{platform.get('status')}` ({platform.get('reason', 'not requested')}).")
    (out_dir / "height_report.md").write_text("\n".join(report), encoding="utf-8")
    data_io.append_report(
        ROOT / "outputs/nightly_report.md",
        "P6 Height Reconstruction",
        [
            f"- Reconstructed height maps from `{input_path}`.",
            f"- Saved height report and confidence maps under `{out_dir}`.",
            f"- Auto platform status: `{platform.get('status')}`.",
        ],
    )
    print(f"Wrote height outputs to {out_dir}")


if __name__ == "__main__":
    main()
