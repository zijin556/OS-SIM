from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import data_io
from src.visualization import save_montage


SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".png", ".bmp", ".npy", ".npz", ".mat", ".h5", ".hdf5"}


def describe_example(path: Path) -> dict[str, Any]:
    ext = path.suffix.lower()
    out: dict[str, Any] = {"path": str(path), "extension": ext, "reader": "unknown"}
    try:
        if ext in {".tif", ".tiff", ".png", ".bmp"}:
            arr = iio.imread(path)
            out.update({"reader": "imageio", "shape": list(arr.shape), "dtype": str(arr.dtype)})
        elif ext == ".npy":
            arr = np.load(path, mmap_mode="r")
            out.update({"reader": "numpy", "shape": list(arr.shape), "dtype": str(arr.dtype)})
        elif ext == ".npz":
            with np.load(path) as z:
                out.update(
                    {
                        "reader": "numpy_npz",
                        "arrays": {name: {"shape": list(z[name].shape), "dtype": str(z[name].dtype)} for name in z.files[:8]},
                    }
                )
        elif ext in {".h5", ".hdf5"}:
            import h5py

            datasets = {}
            with h5py.File(path, "r") as f:
                def visitor(name: str, obj: Any) -> None:
                    if hasattr(obj, "shape") and len(datasets) < 16:
                        datasets[name] = {"shape": list(obj.shape), "dtype": str(obj.dtype)}

                f.visititems(visitor)
            out.update({"reader": "h5py", "datasets": datasets})
        elif ext == ".mat":
            try:
                from scipy.io import loadmat

                mat = loadmat(path, simplify_cells=False)
                arrays = {}
                for name, value in mat.items():
                    if name.startswith("__") or not hasattr(value, "shape"):
                        continue
                    arrays[name] = {"shape": list(value.shape), "dtype": str(value.dtype)}
                    if len(arrays) >= 16:
                        break
                out.update({"reader": "scipy.io.loadmat", "arrays": arrays})
            except NotImplementedError:
                import h5py

                datasets = {}
                with h5py.File(path, "r") as f:
                    def visitor(name: str, obj: Any) -> None:
                        if hasattr(obj, "shape") and len(datasets) < 16:
                            datasets[name] = {"shape": list(obj.shape), "dtype": str(obj.dtype)}

                    f.visititems(visitor)
                out.update({"reader": "h5py_mat", "datasets": datasets})
    except Exception as exc:
        out.update({"reader_error": type(exc).__name__, "reader_error_message": str(exc)})
    return out


def scan_format_inventory(root: Path) -> dict[str, Any]:
    counts = {ext: 0 for ext in sorted(SUPPORTED_EXTENSIONS)}
    examples: dict[str, list[str]] = {ext: [] for ext in sorted(SUPPORTED_EXTENSIONS)}
    total_supported = 0
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            continue
        counts[ext] += 1
        total_supported += 1
        if len(examples[ext]) < 5:
            examples[ext].append(str(path))
    described = {}
    for ext, paths in examples.items():
        if paths:
            described[ext] = describe_example(Path(paths[0]))
    return {
        "root": str(root),
        "supported_extensions": sorted(SUPPORTED_EXTENSIONS),
        "counts_by_extension": counts,
        "total_supported_files": total_supported,
        "examples_by_extension": examples,
        "example_descriptions": described,
    }


def infer_dimension_candidates(
    sample: str,
    groups: list[str],
    layers: list[tuple[float, Path]],
    known: dict[str, dict],
    crop: data_io.CropSpec | None,
) -> dict[str, Any]:
    first_frames = data_io.read_frame_group(layers[0][1], known[groups[0]], crop=crop, normalize=False)
    h, w = int(first_frames.shape[-2]), int(first_frames.shape[-1])
    candidates = []
    for group_name in groups:
        group = known[group_name]
        candidates.append(
            {
                "interpretation": f"{group_name} as Y_real[k,m,y,x]",
                "sample": sample,
                "K": len(layers),
                "M": len(data_io.group_frame_ids(group)),
                "H": h,
                "W": w,
                "z_values_from_folders": [float(z) for z, _ in layers],
                "frame_ids": data_io.group_frame_ids(group),
            }
        )
    candidates.append(
        {
            "interpretation": "H/V multigroup stack as Y_real[k,g,m,y,x]",
            "sample": sample,
            "K": len(layers),
            "G": len(groups),
            "M_per_group": [len(data_io.group_frame_ids(known[g])) for g in groups],
            "H": h,
            "W": w,
            "groups": groups,
        }
    )
    return {
        "selected_crop": crop.label if crop else "full",
        "primary_interpretation": "Each H/V group is a separate Y_real[k,m,y,x] stack; H/V can also be treated as Y_real[k,g,m,y,x].",
        "candidates": candidates,
    }


def frame_stats(arr: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "saturated_low_fraction": float(np.mean(arr <= 0.0)),
        "saturated_high_fraction": float(np.mean(arr >= 1.0)),
    }


def choose_sample(manifest: dict, requested: str | None) -> str:
    if requested:
        return requested
    names = [s["name"] for s in manifest["samples"] if not s["excluded"]]
    if "细台阶" in names:
        return "细台阶"
    if not names:
        raise ValueError("No included samples are available")
    return names[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect OS-SIM raw data and write a manifest/report.")
    parser.add_argument("--config", default="configs/ossim_dataset.yaml")
    parser.add_argument("--sample", default=None)
    parser.add_argument("--groups", default="ossim_t6_h_3step,ossim_t6_v_3step")
    parser.add_argument("--exclude-samples", default="台阶曝光1")
    parser.add_argument("--crop", default="center:512")
    parser.add_argument("--output", default="outputs")
    args = parser.parse_args()

    config = data_io.load_config(args.config)
    excluded = data_io.parse_excluded_samples(args.exclude_samples)
    manifest = data_io.scan_dataset(config, excluded_samples=excluded)
    out_root = ROOT / args.output
    preview_dir = out_root / "previews"
    data_io.write_json(out_root / "config.json", vars(args))
    data_io.write_json(out_root / "data_manifest.json", manifest)
    format_inventory = scan_format_inventory(data_io.raw_root_from_config(config))
    data_io.write_json(out_root / "data_format_inventory.json", format_inventory)

    sample = choose_sample(manifest, args.sample)
    groups = data_io.parse_groups(args.groups, config)
    crop = data_io.infer_crop_for_sample(config, sample, args.crop)
    known = data_io.frame_groups_by_name(config)
    layers = data_io.complete_z_layers(config, sample, groups, excluded)
    if not layers:
        raise RuntimeError(f"No complete layers for sample={sample}, groups={groups}")
    dimension_inference = infer_dimension_candidates(sample, groups, layers, known, crop)
    data_io.write_json(out_root / "data_dimension_inference.json", dimension_inference)

    stats: list[dict] = []
    raw_examples: list[np.ndarray] = []
    raw_titles: list[str] = []
    mid_layer = layers[len(layers) // 2]
    for group_name in groups:
        group = known[group_name]
        frames = data_io.read_frame_group(mid_layer[1], group, crop=crop, normalize=True)
        for m_idx, frame_id in enumerate(data_io.group_frame_ids(group)):
            raw_examples.append(frames[m_idx])
            raw_titles.append(f"{group_name}:{frame_id}")
        for z, z_path in layers:
            frames_z = data_io.read_frame_group(z_path, group, crop=crop, normalize=True)
            for m_idx, frame_id in enumerate(data_io.group_frame_ids(group)):
                item = {
                    "sample": sample,
                    "z": float(z),
                    "z_folder": z_path.name,
                    "group": group_name,
                    "frame_id": int(frame_id),
                }
                item.update(frame_stats(frames_z[m_idx]))
                stats.append(item)

    save_montage(raw_examples, preview_dir / "raw_frames.png", raw_titles, cols=3)
    save_montage(raw_examples[:3], preview_dir / "phase_frames_montage.png", raw_titles[:3], cols=3)

    z_examples = []
    z_titles = []
    for z, z_path in layers[:: max(1, len(layers) // 8)]:
        frame = data_io.read_frame_group(z_path, known[groups[0]], crop=crop, normalize=True)[0]
        z_examples.append(frame)
        z_titles.append(f"z={z:g}")
    save_montage(z_examples, preview_dir / "z_scan_montage.png", z_titles, cols=4)
    stack = np.stack(z_examples, axis=0)
    save_montage([stack.mean(axis=0), stack.var(axis=0)], preview_dir / "mean_variance.png", ["mean", "variance"], cols=2)

    data_io.write_json(out_root / "data_frame_stats.json", stats)
    included = [s["name"] for s in manifest["samples"] if not s["excluded"]]
    excluded_rows = [s for s in manifest["samples"] if s["excluded"]]
    report = [
        "# Data Inspection Report",
        "",
        f"- Raw root: `{manifest['raw_root']}`",
        f"- Excluded samples: `{', '.join(manifest['excluded_samples'])}`",
        f"- Included experiment samples: `{', '.join(included)}`",
        f"- Inspection sample: `{sample}`",
        f"- Inspection groups: `{', '.join(groups)}`",
        f"- Crop: `{crop.label if crop else 'full'}`",
        "",
        "## Sample Summary",
        "",
        "| sample | excluded | z layers | incomplete layers | BMP files |",
        "|---|---:|---:|---:|---:|",
    ]
    for s in manifest["samples"]:
        report.append(
            f"| {s['name']} | {s['excluded']} | {s['z_layer_count']} | {s['incomplete_layer_count']} | {s['bmp_count']} |"
        )
    if excluded_rows:
        report += ["", "## Exclusions", ""]
        for s in excluded_rows:
            report.append(f"- `{s['name']}`: `{s['exclude_reason']}`.")
    report += [
        "",
        "## Format Inventory",
        "",
        f"- Supported extensions: `{', '.join(format_inventory['supported_extensions'])}`",
        f"- Supported files found: `{format_inventory['total_supported_files']}`",
        f"- Counts by extension: `{format_inventory['counts_by_extension']}`",
        "",
        "## Dimension Inference",
        "",
        f"- Primary interpretation: {dimension_inference['primary_interpretation']}",
        f"- Selected crop: `{dimension_inference['selected_crop']}`",
        f"- Candidate count: `{len(dimension_inference['candidates'])}`",
        "",
        "## Preview Outputs",
        "",
        "- `outputs/previews/raw_frames.png`",
        "- `outputs/previews/phase_frames_montage.png`",
        "- `outputs/previews/z_scan_montage.png`",
        "- `outputs/previews/mean_variance.png`",
        "- `outputs/data_format_inventory.json`",
        "- `outputs/data_dimension_inference.json`",
        "",
        "## Frame Statistics",
        "",
        f"- Wrote `{len(stats)}` per-frame statistics rows to `outputs/data_frame_stats.json`.",
    ]
    (out_root / "data_report.md").write_text("\n".join(report), encoding="utf-8")
    data_io.append_report(
        out_root / "nightly_report.md",
        "P0 Data Inspection",
        [
            f"- Completed data manifest with `{len(manifest['samples'])}` samples.",
            "- `台阶曝光1` is excluded from experiment scripts as `excluded_incomplete_sample`.",
            f"- First-round sample is `{sample}` with groups `{', '.join(groups)}`.",
            "- Generated data report and preview figures.",
        ],
    )
    print(f"Wrote inspection outputs to {out_root}")


if __name__ == "__main__":
    main()
