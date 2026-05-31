from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import calibration, data_io
from src.visualization import save_montage


def summarize_array(arr: np.ndarray) -> dict[str, Any]:
    finite = np.asarray(arr)[np.isfinite(arr)]
    if not finite.size:
        return {"shape": list(arr.shape), "dtype": str(arr.dtype), "finite_count": 0}
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "finite_count": int(finite.size),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
    }


def summarize_mat(path: Path) -> dict[str, Any]:
    mat = loadmat(path)
    arrays = {}
    for key, value in mat.items():
        if key.startswith("__") or not hasattr(value, "shape"):
            continue
        arrays[key] = summarize_array(np.asarray(value))
    out = {"path": str(path), "size_bytes": path.stat().st_size, "arrays": arrays}
    if {"Map_DMD_X", "Map_DMD_Y", "mask_valid"} <= set(arrays):
        mask = np.asarray(mat["mask_valid"], dtype=bool)
        out["mapping_status"] = "usable_map_xy_mask"
        out["valid_fraction"] = float(np.mean(mask))
    return out


def inventory_tree(root: Path) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    examples: dict[str, list[str]] = defaultdict(list)
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        ext = path.suffix.lower() or "<no_ext>"
        counts[ext] += 1
        if len(examples[ext]) < 12:
            examples[ext].append(str(path))
    return {"root": str(root), "counts_by_extension": dict(sorted(counts.items())), "examples_by_extension": dict(examples)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect DMD calibration maps and measured projection bitmaps.")
    parser.add_argument("--config", default="configs/ossim_dataset.yaml")
    parser.add_argument("--groups", default="ossim_t6_h_3step,ossim_t6_v_3step")
    parser.add_argument("--crop", default="center:256")
    parser.add_argument("--projection-sequence", default="saomiao3_6")
    parser.add_argument("--calibration-label", default="3-6")
    parser.add_argument("--calibration-mat", default="DMD_CCD_Calibration_Dict.mat")
    parser.add_argument("--output", default="outputs/calibration")
    args = parser.parse_args()

    config = data_io.load_config(args.config)
    groups = data_io.parse_groups(args.groups, config)
    h = int(config["dataset"]["file_format"]["height_px"])
    w = int(config["dataset"]["file_format"]["width_px"])
    crop = data_io.parse_crop(args.crop, height=h, width=w)
    out_dir = ROOT / args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    cal_root = calibration.calibration_root_from_config(config)
    proj_root = calibration.projection_root_from_config(config)
    cal_dir = next(p for p in cal_root.iterdir() if p.is_dir() and p.name.startswith(args.calibration_label)) / "Z_0.000"
    sequence = calibration.read_projection_sequence(config, stem=args.projection_sequence)

    mat_summaries = [summarize_mat(path) for path in sorted(cal_dir.glob("*.mat"))]
    preferred_map = calibration.load_calibration_map(
        config,
        label_prefix=args.calibration_label,
        mat_name=args.calibration_mat,
        crop=None,
        roi_left=sequence.roi_left,
        roi_top=sequence.roi_top,
        roi_width=sequence.roi_width,
        roi_height=sequence.roi_height,
    )
    save_montage(
        [preferred_map.map_x, preferred_map.map_y, preferred_map.mask_valid.astype(np.float32)],
        out_dir / "mapping_preview.png",
        ["Map_DMD_X", "Map_DMD_Y", "mask_valid"],
        cols=3,
        cmap="viridis",
        max_side=900,
    )

    measured_groups = {}
    mapped_images = []
    mapped_titles = []
    dmd_images = []
    dmd_titles = []
    for group_name in groups:
        entries = calibration.projection_paths_for_group(config, group_name, sequence=sequence)
        raw = calibration.read_projection_bitmaps(entries)
        mapped, info = calibration.measured_camera_patterns_for_group(
            config,
            group_name,
            crop=crop,
            sequence_stem=args.projection_sequence,
            calibration_label_prefix=args.calibration_label,
            mat_name=args.calibration_mat,
        )
        np.save(out_dir / f"{group_name}_mapped_patterns.npy", mapped)
        measured_groups[group_name] = {
            "frame_ids": [entry.frame_id for entry in entries],
            "bitmap_names": [entry.bitmap_name for entry in entries],
            "bitmap_paths": [entry.bitmap_path for entry in entries],
            "raw_bitmap_shape": [int(v) for v in raw.shape],
            "mapped_pattern_shape": [int(v) for v in mapped.shape],
            "mapped_info": info,
        }
        for idx, entry in enumerate(entries):
            dmd_images.append(raw[idx])
            dmd_titles.append(f"{group_name}:{entry.frame_id}")
            mapped_images.append(mapped[idx])
            mapped_titles.append(f"{group_name}:{entry.frame_id}")

    if dmd_images:
        save_montage(dmd_images, out_dir / "t6_dmd_patterns.png", dmd_titles, cols=3, max_side=700)
        save_montage(mapped_images, out_dir / "t6_mapped_camera_patterns.png", mapped_titles, cols=3, max_side=700)

    expected_t6_names = [
        "Fringe_T6_3Step_H_Deg0.bmp",
        "Fringe_T6_3Step_H_Deg120.bmp",
        "Fringe_T6_3Step_H_Deg240.bmp",
        "Fringe_T6_3Step_V_Deg0.bmp",
        "Fringe_T6_3Step_V_Deg120.bmp",
        "Fringe_T6_3Step_V_Deg240.bmp",
    ]
    actual_t6_names = [sequence.entries[idx].bitmap_name for idx in range(100, 106) if idx in sequence.entries]
    matched_t6 = actual_t6_names == expected_t6_names

    result = {
        "calibration_root": str(cal_root),
        "projection_patterns_root": str(proj_root),
        "calibration_inventory": inventory_tree(cal_root),
        "projection_inventory": inventory_tree(proj_root),
        "calibration_dir": str(cal_dir),
        "mat_files": mat_summaries,
        "preferred_mapping": {
            "path": preferred_map.source_path,
            "shape": list(preferred_map.shape),
            "coordinate_frame": preferred_map.coordinate_frame,
            "roi_left": preferred_map.roi_left,
            "roi_top": preferred_map.roi_top,
            "roi_width": preferred_map.roi_width,
            "roi_height": preferred_map.roi_height,
            "valid_fraction_full": float(np.mean(preferred_map.mask_valid)),
        },
        "projection_sequence": {
            "path": sequence.path,
            "encoding": sequence.encoding,
            "entry_count": len(sequence.entries),
            "roi_left": sequence.roi_left,
            "roi_top": sequence.roi_top,
            "roi_width": sequence.roi_width,
            "roi_height": sequence.roi_height,
            "frames_100_105": actual_t6_names,
            "matched_t6_hv_frames_100_105": matched_t6,
        },
        "crop": crop.label if crop else "full",
        "measured_groups": measured_groups,
    }
    data_io.write_json(out_dir / "assets_inventory.json", result)
    data_io.write_json(
        out_dir / "config.json",
        {
            "groups": groups,
            "crop": crop.label if crop else "full",
            "projection_sequence": args.projection_sequence,
            "calibration_label": args.calibration_label,
            "calibration_mat": args.calibration_mat,
        },
    )

    lines = [
        "# Calibration Asset Report",
        "",
        f"- Calibration root: `{cal_root}`",
        f"- Projection root: `{proj_root}`",
        f"- Preferred mapping: `{preferred_map.source_path}`",
        f"- Projection sequence: `{sequence.path}`",
        f"- Sequence entries: `{len(sequence.entries)}`",
        f"- ROI offset/size from sequence: left `{sequence.roi_left}`, top `{sequence.roi_top}`, width `{sequence.roi_width}`, height `{sequence.roi_height}`",
        f"- Frames 100-105 match T6 H/V three-step patterns: `{matched_t6}`",
        f"- Mapping coordinate frame inferred as: `{preferred_map.coordinate_frame}`",
        f"- Full-map valid fraction: `{float(np.mean(preferred_map.mask_valid)):.6g}`",
        f"- Crop: `{crop.label if crop else 'full'}`",
        "",
        "## Group Mapping",
        "",
        "| group | frames | bitmaps | mapped shape | valid fraction |",
        "|---|---|---|---|---:|",
    ]
    for group_name, info in measured_groups.items():
        mapped_info = info["mapped_info"]
        lines.append(
            f"| {group_name} | {info['frame_ids']} | {info['bitmap_names']} | {info['mapped_pattern_shape']} | {mapped_info['valid_fraction']:.6g} |"
        )
    lines += [
        "",
        "## Outputs",
        "",
        "- `assets_inventory.json`",
        "- `mapping_preview.png`",
        "- `t6_dmd_patterns.png`",
        "- `t6_mapped_camera_patterns.png`",
        "- `*_mapped_patterns.npy`",
    ]
    (out_dir / "calibration_report.md").write_text("\n".join(lines), encoding="utf-8")
    data_io.append_report(
        ROOT / "outputs/nightly_report.md",
        "P3a Calibration And Measured DMD Asset Inspection",
        [
            f"- Loaded DMD-CCD mapping `{preferred_map.source_path}` with full valid fraction `{float(np.mean(preferred_map.mask_valid)):.6g}`.",
            f"- Confirmed projection sequence frames `100-105` match T6 H/V 3-step patterns: `{matched_t6}`.",
            f"- Saved mapped measured DMD patterns and previews under `{out_dir}`.",
        ],
    )
    print(f"Wrote calibration asset outputs to {out_dir}")


if __name__ == "__main__":
    main()
