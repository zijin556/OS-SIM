from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import imageio.v3 as iio
import numpy as np
import yaml


DEFAULT_CONFIG = Path("configs/ossim_dataset.yaml")
DEFAULT_EXCLUDED_SAMPLES = ("台阶曝光1",)


@dataclass(frozen=True)
class CropSpec:
    y: slice
    x: slice
    label: str


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(path: str | Path, base: Path | None = None) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (base or repo_root() / p).resolve()


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    cfg_path = resolve_path(path)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(cfg_path)
    return cfg


def write_json(path: str | Path, data: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def raw_root_from_config(config: dict[str, Any]) -> Path:
    return Path(config["dataset"]["raw_root"])


def frame_groups_by_name(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {g["name"]: g for g in config["frame_groups"]}


def parse_groups(groups: str | Iterable[str] | None, config: dict[str, Any]) -> list[str]:
    if groups is None:
        return ["ossim_t6_h_3step", "ossim_t6_v_3step"]
    if isinstance(groups, str):
        names = [g.strip() for g in groups.split(",") if g.strip()]
    else:
        names = list(groups)
    known = frame_groups_by_name(config)
    missing = [g for g in names if g not in known]
    if missing:
        raise ValueError(f"Unknown frame group(s): {missing}. Known groups: {sorted(known)}")
    return names


def parse_excluded_samples(value: str | Iterable[str] | None) -> set[str]:
    if value is None:
        return set(DEFAULT_EXCLUDED_SAMPLES)
    if isinstance(value, str):
        return {s.strip() for s in value.split(",") if s.strip()}
    return {str(s) for s in value}


def parse_z_folder(name: str) -> float | None:
    m = re.match(r"^Z_(?P<z>[0-9]+(?:\.[0-9]+)?)$", name)
    if not m:
        return None
    return float(m.group("z"))


def z_dirs(sample_dir: Path) -> list[tuple[float, Path]]:
    out: list[tuple[float, Path]] = []
    for p in sample_dir.iterdir():
        if not p.is_dir():
            continue
        z = parse_z_folder(p.name)
        if z is not None:
            out.append((z, p))
    return sorted(out, key=lambda item: item[0])


def frame_id(path: Path) -> int | None:
    try:
        return int(path.stem)
    except ValueError:
        return None


def frame_files(z_dir_path: Path) -> dict[int, Path]:
    files: dict[int, Path] = {}
    for p in z_dir_path.glob("*.bmp"):
        idx = frame_id(p)
        if idx is not None:
            files[idx] = p
    return dict(sorted(files.items()))


def group_frame_ids(group: dict[str, Any]) -> list[int]:
    start, end = group["frames"]
    return list(range(int(start), int(end) + 1))


def sample_names(config: dict[str, Any]) -> list[str]:
    root = raw_root_from_config(config)
    return sorted([p.name for p in root.iterdir() if p.is_dir()])


def validate_sample(sample: str, excluded_samples: set[str]) -> None:
    if sample in excluded_samples:
        raise ValueError(f"Sample {sample!r} is excluded by policy because it is incomplete.")


def scan_dataset(config: dict[str, Any], excluded_samples: set[str] | None = None) -> dict[str, Any]:
    excluded_samples = excluded_samples or set(DEFAULT_EXCLUDED_SAMPLES)
    root = raw_root_from_config(config)
    expected_all = set(range(1, 106))
    known_groups = frame_groups_by_name(config)
    samples: list[dict[str, Any]] = []
    for sample_dir in sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name):
        layers: list[dict[str, Any]] = []
        bmp_count = 0
        for z, z_path in z_dirs(sample_dir):
            files = frame_files(z_path)
            have = set(files)
            missing_all = sorted(expected_all - have)
            group_status = {}
            for name, group in known_groups.items():
                ids = group_frame_ids(group)
                missing = [i for i in ids if i not in have]
                group_status[name] = {
                    "frames": ids,
                    "complete": not missing,
                    "missing_frames": missing,
                }
            bmp_count += len(files)
            layers.append(
                {
                    "folder": z_path.name,
                    "z": z,
                    "frame_count": len(files),
                    "missing_frames_1_105": missing_all,
                    "groups": group_status,
                }
            )
        samples.append(
            {
                "name": sample_dir.name,
                "path": str(sample_dir),
                "z_layer_count": len(layers),
                "z_min": layers[0]["z"] if layers else None,
                "z_max": layers[-1]["z"] if layers else None,
                "bmp_count": bmp_count,
                "excluded": sample_dir.name in excluded_samples,
                "exclude_reason": "excluded_incomplete_sample" if sample_dir.name in excluded_samples else None,
                "incomplete_layer_count": sum(1 for layer in layers if layer["missing_frames_1_105"]),
                "layers": layers,
            }
        )
    return {
        "raw_root": str(root),
        "excluded_samples": sorted(excluded_samples),
        "samples": samples,
    }


def complete_z_layers(
    config: dict[str, Any],
    sample: str,
    groups: list[str],
    excluded_samples: set[str] | None = None,
    z_min: float | None = None,
    z_max: float | None = None,
) -> list[tuple[float, Path]]:
    excluded_samples = excluded_samples or set(DEFAULT_EXCLUDED_SAMPLES)
    validate_sample(sample, excluded_samples)
    root = raw_root_from_config(config)
    sample_dir = root / sample
    if not sample_dir.exists():
        raise FileNotFoundError(f"Sample directory not found: {sample_dir}")
    known = frame_groups_by_name(config)
    required = [group_frame_ids(known[g]) for g in groups]
    layers: list[tuple[float, Path]] = []
    for z, z_path in z_dirs(sample_dir):
        if z_min is not None and z < z_min:
            continue
        if z_max is not None and z > z_max:
            continue
        have = set(frame_files(z_path))
        if all(all(idx in have for idx in ids) for ids in required):
            layers.append((z, z_path))
    return layers


def parse_crop(crop: str | None, height: int | None = None, width: int | None = None) -> CropSpec | None:
    if crop is None or crop.strip().lower() in {"", "none", "full"}:
        return None
    text = crop.strip().lower()
    if text.startswith("center:"):
        if height is None or width is None:
            raise ValueError("center crop requires image height and width")
        size_text = text.split(":", 1)[1]
        if "x" in size_text:
            h, w = [int(v) for v in size_text.split("x", 1)]
        else:
            h = w = int(size_text)
        if h > height or w > width:
            raise ValueError(f"Crop {h}x{w} is larger than image {height}x{width}")
        y0 = (height - h) // 2
        x0 = (width - w) // 2
        return CropSpec(slice(y0, y0 + h), slice(x0, x0 + w), f"center_{h}x{w}")
    m = re.match(r"^(?P<y0>\d+):(?P<y1>\d+),(?P<x0>\d+):(?P<x1>\d+)$", text)
    if not m:
        raise ValueError("Crop must be 'center:N', 'center:HxW', 'y0:y1,x0:x1', or 'full'")
    y0, y1, x0, x1 = [int(m.group(k)) for k in ("y0", "y1", "x0", "x1")]
    if height is not None and (y0 < 0 or y1 > height):
        raise ValueError(f"Y crop {y0}:{y1} outside image height {height}")
    if width is not None and (x0 < 0 or x1 > width):
        raise ValueError(f"X crop {x0}:{x1} outside image width {width}")
    return CropSpec(slice(y0, y1), slice(x0, x1), f"y{y0}_{y1}_x{x0}_{x1}")


def image_shape(path: Path) -> tuple[int, int]:
    arr = iio.imread(path)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return int(arr.shape[0]), int(arr.shape[1])


def read_image(path: Path, crop: CropSpec | None = None, normalize: bool = True) -> np.ndarray:
    arr = iio.imread(path)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if crop is not None:
        arr = arr[crop.y, crop.x]
    arr = np.asarray(arr)
    if normalize:
        return arr.astype(np.float32) / 255.0
    return arr


def read_frame_group(
    z_path: Path,
    group: dict[str, Any],
    crop: CropSpec | None = None,
    normalize: bool = True,
) -> np.ndarray:
    files = frame_files(z_path)
    frames = []
    missing = []
    for idx in group_frame_ids(group):
        if idx not in files:
            missing.append(idx)
        else:
            frames.append(read_image(files[idx], crop=crop, normalize=normalize))
    if missing:
        raise FileNotFoundError(f"Missing frames {missing} in {z_path}")
    return np.stack(frames, axis=0)


def infer_crop_for_sample(config: dict[str, Any], sample: str, crop_arg: str | None) -> CropSpec | None:
    root = raw_root_from_config(config)
    first_z = z_dirs(root / sample)[0][1]
    first_frame = next(iter(frame_files(first_z).values()))
    h, w = image_shape(first_frame)
    return parse_crop(crop_arg, height=h, width=w)


def load_os_sim_stack(
    config: dict[str, Any],
    sample: str,
    groups: list[str],
    crop: CropSpec | None,
    excluded_samples: set[str] | None = None,
    z_min: float | None = None,
    z_max: float | None = None,
    normalize: bool = True,
) -> dict[str, Any]:
    known = frame_groups_by_name(config)
    layers = complete_z_layers(config, sample, groups, excluded_samples, z_min=z_min, z_max=z_max)
    if not layers:
        raise ValueError(f"No complete z layers found for sample={sample}, groups={groups}")
    data = []
    for _, z_path in layers:
        group_frames = [read_frame_group(z_path, known[g], crop=crop, normalize=normalize) for g in groups]
        data.append(np.stack(group_frames, axis=0))
    return {
        "Y": np.stack(data, axis=0),  # K, G, M, H, W
        "z_values": np.array([z for z, _ in layers], dtype=np.float32),
        "z_folders": [p.name for _, p in layers],
        "groups": [known[g] for g in groups],
        "group_names": groups,
        "crop": None if crop is None else {"y": [crop.y.start, crop.y.stop], "x": [crop.x.start, crop.x.stop], "label": crop.label},
    }


def append_report(path: str | Path, title: str, lines: list[str]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(f"\n\n## {title}\n\n")
        for line in lines:
            f.write(f"{line}\n")

