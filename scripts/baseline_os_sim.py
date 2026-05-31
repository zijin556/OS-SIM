from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import data_io
from src.height import (
    confidence_metrics,
    height_argmax,
    height_gaussian,
    height_parabolic,
    height_softargmax,
    summarize_height,
)
from src.os_sim import demodulate_phase_stack, fuse_hv
from src.visualization import save_montage, save_profiles, save_scalar_image


def main() -> None:
    parser = argparse.ArgumentParser(description="Run traditional OS-SIM H/V baseline.")
    parser.add_argument("--config", default="configs/ossim_dataset.yaml")
    parser.add_argument("--sample", default="细台阶")
    parser.add_argument("--groups", default="ossim_t6_h_3step,ossim_t6_v_3step")
    parser.add_argument("--exclude-samples", default="台阶曝光1")
    parser.add_argument("--crop", default="full")
    parser.add_argument("--output", default="outputs/baseline/t6_hv")
    args = parser.parse_args()

    config = data_io.load_config(args.config)
    excluded = data_io.parse_excluded_samples(args.exclude_samples)
    groups = data_io.parse_groups(args.groups, config)
    crop = data_io.infer_crop_for_sample(config, args.sample, args.crop)
    known = data_io.frame_groups_by_name(config)
    layers = data_io.complete_z_layers(config, args.sample, groups, excluded)
    if not layers:
        raise RuntimeError("No complete layers available for baseline")
    out_dir = ROOT / args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    first_frames = data_io.read_frame_group(layers[0][1], known[groups[0]], crop=crop, normalize=True)
    k_count, h, w = len(layers), first_frames.shape[-2], first_frames.shape[-1]
    z_values = np.array([z for z, _ in layers], dtype=np.float32)
    np.save(out_dir / "z_values.npy", z_values)

    group_outputs: dict[str, dict[str, np.ndarray]] = {}
    for group_name in groups:
        safe = group_name.replace("ossim_", "").replace("_3step", "")
        group_outputs[group_name] = {
            "D0": np.lib.format.open_memmap(out_dir / f"D0_{safe}_stack.npy", mode="w+", dtype=np.float32, shape=(k_count, h, w)),
            "Dc": np.lib.format.open_memmap(out_dir / f"Dc_{safe}_stack.npy", mode="w+", dtype=np.float32, shape=(k_count, h, w)),
            "Ds": np.lib.format.open_memmap(out_dir / f"Ds_{safe}_stack.npy", mode="w+", dtype=np.float32, shape=(k_count, h, w)),
            "A": np.lib.format.open_memmap(out_dir / f"A_{safe}_stack.npy", mode="w+", dtype=np.float32, shape=(k_count, h, w)),
        }

    for k_idx, (z, z_path) in enumerate(layers):
        for group_name in groups:
            frames = data_io.read_frame_group(z_path, known[group_name], crop=crop, normalize=True)
            demod = demodulate_phase_stack(frames)
            for key in ("D0", "Dc", "Ds", "A"):
                group_outputs[group_name][key][k_idx] = demod[key]
        print(f"baseline z {k_idx + 1}/{k_count}: {z:g}")

    # Flush memmaps before reading them back.
    for outputs in group_outputs.values():
        for arr in outputs.values():
            arr.flush()

    if len(groups) >= 2:
        A_h = np.asarray(group_outputs[groups[0]]["A"])
        A_v = np.asarray(group_outputs[groups[1]]["A"])
        A_fused = np.lib.format.open_memmap(out_dir / "A_fused_stack.npy", mode="w+", dtype=np.float32, shape=(k_count, h, w))
        A_fused[:] = fuse_hv(A_h, A_v)
        A_fused.flush()
    else:
        A_fused = np.asarray(group_outputs[groups[0]]["A"])
        np.save(out_dir / "A_fused_stack.npy", A_fused)

    heights = {}
    metrics = {
        "sample": args.sample,
        "groups": groups,
        "crop": crop.label if crop else "full",
        "z_count": int(k_count),
        "shape": [int(h), int(w)],
    }
    stacks = {"fused": np.asarray(A_fused)}
    for idx, group_name in enumerate(groups):
        label = "h" if idx == 0 else "v" if idx == 1 else group_name
        stacks[label] = np.asarray(group_outputs[group_name]["A"])
    for label, stack in stacks.items():
        h_arg = height_argmax(stack, z_values)
        h_par = height_parabolic(stack, z_values)
        h_gau = height_gaussian(stack, z_values)
        h_soft = height_softargmax(stack, z_values)
        conf = confidence_metrics(stack)
        np.save(out_dir / f"height_{label}_argmax.npy", h_arg)
        np.save(out_dir / f"height_{label}_peakfit.npy", h_par)
        np.save(out_dir / f"height_{label}_gaussian.npy", h_gau)
        np.save(out_dir / f"height_{label}_softargmax.npy", h_soft)
        np.save(out_dir / f"confidence_{label}_sharpness.npy", conf["sharpness"])
        heights[label] = h_par
        metrics[label] = summarize_height(h_par, conf)

    mid = k_count // 2
    save_montage(
        [stacks.get("h", stacks["fused"])[mid], stacks.get("v", stacks["fused"])[mid], stacks["fused"][mid]],
        out_dir / "section_montage.png",
        ["A_h mid-z", "A_v mid-z", "A_fused mid-z"],
        cols=3,
    )
    save_scalar_image(heights["fused"], out_dir / "height_map.png", "Baseline fused height")
    save_profiles(stacks["fused"], z_values, out_dir / "axis_profiles.png", "Baseline fused A profiles")

    data_io.write_json(out_dir / "metrics.json", metrics)
    data_io.write_json(
        out_dir / "config.json",
        {
            "config": str((ROOT / args.config).resolve()),
            "sample": args.sample,
            "groups": groups,
            "exclude_samples": sorted(excluded),
            "crop": crop.label if crop else "full",
            "output": str(out_dir),
        },
    )
    report = [
        "# Baseline OS-SIM Report",
        "",
        f"- Sample: `{args.sample}`",
        f"- Groups: `{', '.join(groups)}`",
        f"- Crop: `{crop.label if crop else 'full'}`",
        f"- z layers: `{k_count}`",
        f"- Output shape: `{h} x {w}`",
        "",
        "## Outputs",
        "",
        "- `A_fused_stack.npy`",
        "- `height_fused_peakfit.npy`",
        "- `section_montage.png`",
        "- `height_map.png`",
        "- `axis_profiles.png`",
    ]
    (out_dir / "baseline_report.md").write_text("\n".join(report), encoding="utf-8")
    data_io.append_report(
        ROOT / "outputs/nightly_report.md",
        "P1 Traditional OS-SIM Baseline",
        [
            f"- Completed H/V baseline for `{args.sample}` using `{', '.join(groups)}`.",
            f"- Saved fused A stack and height maps under `{out_dir}`.",
            f"- Fused height std: `{metrics['fused']['height_std']:.4g}` in scan-coordinate units.",
        ],
    )
    print(f"Wrote baseline outputs to {out_dir}")


if __name__ == "__main__":
    main()

