from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src import data_io
from src.height import confidence_metrics, height_parabolic, platform_metrics_from_masks, summarize_height
from src.losses import tv_l1
from src.visualization import save_montage, save_scalar_image

from optimize_modulation_latents import build_grid_masks, grid_energy_mean


def load_optional(path: Path) -> np.ndarray | None:
    return np.load(path) if path.exists() else None


def metric_row(
    name: str,
    A_fused: np.ndarray,
    z_values: np.ndarray,
    grid_masks: np.ndarray,
    low_mask: np.ndarray | None,
    high_mask: np.ndarray | None,
) -> dict[str, Any]:
    height = height_parabolic(A_fused, z_values)
    conf = confidence_metrics(A_fused)
    platform = {"status": "not_run"}
    if low_mask is not None and high_mask is not None and low_mask.shape == height.shape:
        platform = platform_metrics_from_masks(height, low_mask, high_mask)
    A_for_grid = A_fused[:, None, :, :]
    masks = grid_masks[:1] if grid_masks.shape[0] != 1 else grid_masks
    return {
        "name": name,
        "A_shape": list(A_fused.shape),
        "A_tv": tv_l1(A_fused),
        "grid_energy": grid_energy_mean(A_for_grid, masks),
        "height": summarize_height(height, conf),
        "platform": {k: v for k, v in platform.items() if k not in {"low_mask", "high_mask"}},
    }


def write_missing_predictions(out_dir: Path, args: argparse.Namespace, missing: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "missing_predictions",
        "missing": missing,
        "message": "Run train_inverse_lightsection_net.py after installing PyTorch to generate A_pred stacks first.",
    }
    data_io.write_json(out_dir / "evaluation_metrics.json", result)
    report = [
        "# Inverse Net Evaluation Report",
        "",
        "Status: `missing_predictions`",
        "",
        "Missing required prediction files:",
    ]
    report += [f"- `{item}`" for item in missing]
    report += [
        "",
        "Install PyTorch in the project virtual environment, then run:",
        "",
        "```powershell",
        "python scripts/train_inverse_lightsection_net.py --crop center:128 --epochs 30 --pattern-source measured_calibrated",
        "python scripts/evaluate_inverse_lightsection_net.py --input outputs/inverse_net/t6_hv/A_fused_pred_stack.npy",
        "```",
    ]
    (out_dir / "inverse_net_evaluation_report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Missing inverse-net predictions; wrote evaluation report to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate inverse-net A-stack and height reconstruction against baseline and latent optimization.")
    parser.add_argument("--input", default="outputs/inverse_net/t6_hv/A_fused_pred_stack.npy")
    parser.add_argument("--a-h", default="outputs/inverse_net/t6_hv/A_h_pred_stack.npy")
    parser.add_argument("--a-v", default="outputs/inverse_net/t6_hv/A_v_pred_stack.npy")
    parser.add_argument("--z-values", default="outputs/inverse_net/t6_hv/z_values.npy")
    parser.add_argument("--baseline", default="outputs/baseline/t6_hv/A_fused_stack.npy")
    parser.add_argument("--latent", default="outputs/platform_opt/t6_hv_measured/A_fused_best_stack.npy")
    parser.add_argument("--platform-low-mask", default="outputs/height/baseline_t6_hv/platform_low_mask.npy")
    parser.add_argument("--platform-high-mask", default="outputs/height/baseline_t6_hv/platform_high_mask.npy")
    parser.add_argument("--output", default="outputs/inverse_net/t6_hv")
    args = parser.parse_args()

    out_dir = ROOT / args.output
    pred_path = ROOT / args.input
    missing = [path for path in [args.input, args.a_h, args.a_v, args.z_values] if not (ROOT / path).exists()]
    if missing:
        write_missing_predictions(out_dir, args, missing)
        return

    A_pred = np.load(pred_path).astype(np.float32)
    A_h = np.load(ROOT / args.a_h).astype(np.float32)
    A_v = np.load(ROOT / args.a_v).astype(np.float32)
    z_values = np.load(ROOT / args.z_values).astype(np.float32)
    baseline = load_optional(ROOT / args.baseline)
    latent = load_optional(ROOT / args.latent)
    low_mask = load_optional(ROOT / args.platform_low_mask)
    high_mask = load_optional(ROOT / args.platform_high_mask)
    if low_mask is not None:
        low_mask = low_mask.astype(bool)
    if high_mask is not None:
        high_mask = high_mask.astype(bool)

    grid_source = baseline if baseline is not None and baseline.shape == A_pred.shape else A_pred
    grid_masks, peaks = build_grid_masks(grid_source[:, None, :, :])
    rows = [metric_row("inverse_net", A_pred, z_values, grid_masks, low_mask, high_mask)]
    if baseline is not None and baseline.shape == A_pred.shape:
        rows.append(metric_row("baseline_A_cls", baseline.astype(np.float32), z_values, grid_masks, low_mask, high_mask))
    if latent is not None and latent.shape == A_pred.shape:
        rows.append(metric_row("latent_platform_opt", latent.astype(np.float32), z_values, grid_masks, low_mask, high_mask))

    height_pred = height_parabolic(A_pred, z_values)
    np.save(out_dir / "height_pred.npy", height_pred)
    save_scalar_image(height_pred, out_dir / "height_pred.png", "inverse_net height from A_fused")
    mid = A_pred.shape[0] // 2
    a_images = [A_pred[mid]]
    a_titles = ["inverse_net A_fused"]
    h_images = [height_pred]
    h_titles = ["inverse_net height"]
    if baseline is not None and baseline.shape == A_pred.shape:
        base_h = height_parabolic(baseline, z_values)
        a_images += [baseline[mid], A_pred[mid] - baseline[mid]]
        a_titles += ["baseline A_cls", "inverse - baseline"]
        h_images += [base_h, height_pred - base_h]
        h_titles += ["baseline height", "inverse - baseline"]
    if latent is not None and latent.shape == A_pred.shape:
        latent_h = height_parabolic(latent, z_values)
        a_images += [latent[mid], A_pred[mid] - latent[mid]]
        a_titles += ["latent A", "inverse - latent"]
        h_images += [latent_h, height_pred - latent_h]
        h_titles += ["latent height", "inverse - latent"]
    save_montage(a_images, out_dir / "A_comparison.png", a_titles, cols=3, cmap="viridis")
    save_montage(h_images, out_dir / "height_compare.png", h_titles, cols=3, cmap="viridis")

    pred_row = rows[0]
    baseline_row = next((row for row in rows if row["name"] == "baseline_A_cls"), None)
    latent_row = next((row for row in rows if row["name"] == "latent_platform_opt"), None)
    result = {
        "status": "completed",
        "input": args.input,
        "A_h_shape": list(A_h.shape),
        "A_v_shape": list(A_v.shape),
        "A_fused_shape": list(A_pred.shape),
        "grid_peaks": peaks,
        "rows": rows,
        "grid_energy_vs_baseline": None if baseline_row is None else pred_row["grid_energy"] - baseline_row["grid_energy"],
        "grid_energy_vs_latent": None if latent_row is None else pred_row["grid_energy"] - latent_row["grid_energy"],
        "height_std": pred_row["height"]["height_std"],
        "platform": pred_row["platform"],
    }
    data_io.write_json(out_dir / "evaluation_metrics.json", result)
    report = [
        "# Inverse Net Evaluation Report",
        "",
        "Status: `completed`",
        "",
        "| method | grid_energy | height_std | platform_rms |",
        "|---|---:|---:|---:|",
    ]
    for item in rows:
        report.append(
            "| {name} | {grid:.6g} | {hstd:.6g} | {rms} |".format(
                name=item["name"],
                grid=item["grid_energy"],
                hstd=item["height"]["height_std"],
                rms=item["platform"].get("pooled_platform_rms", "n/a"),
            )
        )
    (out_dir / "inverse_net_evaluation_report.md").write_text("\n".join(report), encoding="utf-8")
    data_io.append_report(
        ROOT / "outputs/nightly_report.md",
        "P18b Inverse Net Evaluation",
        [
            f"- Evaluated inverse-net A stack `{args.input}`.",
            f"- Grid energy `{pred_row['grid_energy']:.6g}`; height std `{pred_row['height']['height_std']:.6g}`.",
            f"- Outputs saved under `{out_dir}`.",
        ],
    )
    print(f"Wrote inverse-net evaluation outputs to {out_dir}")


if __name__ == "__main__":
    main()
