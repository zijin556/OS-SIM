from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
from scipy import ndimage
from scipy.io import loadmat

from .data_io import CropSpec, frame_groups_by_name, group_frame_ids, raw_root_from_config


@dataclass(frozen=True)
class CalibrationMap:
    map_x: np.ndarray
    map_y: np.ndarray
    mask_valid: np.ndarray
    source_path: str
    coordinate_frame: str
    roi_left: int
    roi_top: int
    roi_width: int | None
    roi_height: int | None

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.map_y.shape[0]), int(self.map_y.shape[1])


@dataclass(frozen=True)
class ProjectionEntry:
    frame_id: int
    bitmap_name: str
    bitmap_path: str
    exposure_us: int | None
    dark_time_us: int | None
    bit_depth: int | None


@dataclass(frozen=True)
class ProjectionSequence:
    path: str
    entries: dict[int, ProjectionEntry]
    roi_left: int | None
    roi_top: int | None
    roi_width: int | None
    roi_height: int | None
    encoding: str


def dlp6500_root_from_config(config: dict[str, Any]) -> Path:
    assets = config.get("calibration_assets", {})
    configured = assets.get("dlp6500_root")
    if configured:
        return Path(configured)
    return raw_root_from_config(config).parents[1]


def calibration_root_from_config(config: dict[str, Any]) -> Path:
    assets = config.get("calibration_assets", {})
    configured = assets.get("calibration_root")
    if configured:
        return Path(configured)
    return dlp6500_root_from_config(config) / "calibration"


def projection_root_from_config(config: dict[str, Any]) -> Path:
    assets = config.get("calibration_assets", {})
    configured = assets.get("projection_patterns_root")
    if configured:
        return Path(configured)
    return dlp6500_root_from_config(config) / "projection_patterns"


def _find_dir_by_prefix(root: Path, prefix: str) -> Path:
    matches = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith(prefix)], key=lambda p: p.name)
    if not matches:
        raise FileNotFoundError(f"No directory starting with {prefix!r} under {root}")
    return matches[0]


def _read_text_fallback(path: Path) -> tuple[str, str]:
    data = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "gbk", "latin1"):
        try:
            return data.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    return data.decode("latin1", errors="replace"), "latin1-replace"


def _extract_roi_from_text(text: str) -> tuple[int | None, int | None, int | None, int | None]:
    match = re.search(r"ROI_(?P<w>\d+)x(?P<h>\d+)_T(?P<t>\d+)_L(?P<l>\d+)", text)
    if not match:
        return None, None, None, None
    return (
        int(match.group("l")),
        int(match.group("t")),
        int(match.group("w")),
        int(match.group("h")),
    )


def find_projection_sequence(config: dict[str, Any], stem: str = "saomiao3_6") -> Path:
    root = projection_root_from_config(config)
    matches = sorted(root.rglob(f"{stem}.txt"), key=lambda p: str(p))
    if not matches:
        raise FileNotFoundError(f"Projection sequence {stem}.txt not found under {root}")
    return matches[0]


def read_projection_sequence(config: dict[str, Any], stem: str = "saomiao3_6") -> ProjectionSequence:
    path = find_projection_sequence(config, stem=stem)
    text, encoding = _read_text_fallback(path)
    roi_left, roi_top, roi_width, roi_height = _extract_roi_from_text(text)
    entries: dict[int, ProjectionEntry] = {}
    frame_id = 0
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.lower().startswith("normal mode"):
            continue
        parts = [part.strip() for part in line.split(",")]
        if not parts or not parts[0]:
            continue
        frame_id += 1
        bitmap_name = Path(parts[0]).name
        bitmap_path = path.parent / bitmap_name
        bit_depth = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
        exposure_us = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else None
        dark_time_us = int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else None
        entries[frame_id] = ProjectionEntry(
            frame_id=frame_id,
            bitmap_name=bitmap_name,
            bitmap_path=str(bitmap_path),
            exposure_us=exposure_us,
            dark_time_us=dark_time_us,
            bit_depth=bit_depth,
        )
    return ProjectionSequence(
        path=str(path),
        entries=entries,
        roi_left=roi_left,
        roi_top=roi_top,
        roi_width=roi_width,
        roi_height=roi_height,
        encoding=encoding,
    )


def load_calibration_map(
    config: dict[str, Any],
    label_prefix: str = "3-6",
    mat_name: str = "DMD_CCD_Calibration_Dict.mat",
    z_folder: str = "Z_0.000",
    crop: CropSpec | None = None,
    roi_left: int | None = None,
    roi_top: int | None = None,
    roi_width: int | None = None,
    roi_height: int | None = None,
) -> CalibrationMap:
    cal_root = calibration_root_from_config(config)
    cal_dir = _find_dir_by_prefix(cal_root, label_prefix) / z_folder
    mat_path = cal_dir / mat_name
    if not mat_path.exists():
        raise FileNotFoundError(f"Calibration mat not found: {mat_path}")
    mat = loadmat(mat_path)
    map_x = np.asarray(mat["Map_DMD_X"], dtype=np.float32)
    map_y = np.asarray(mat["Map_DMD_Y"], dtype=np.float32)
    mask_valid = np.asarray(mat["mask_valid"], dtype=bool)
    if crop is not None:
        map_x = map_x[crop.y, crop.x]
        map_y = map_y[crop.y, crop.x]
        mask_valid = mask_valid[crop.y, crop.x]

    finite_x = map_x[np.isfinite(map_x)]
    finite_y = map_y[np.isfinite(map_y)]
    max_x = float(np.max(finite_x)) if finite_x.size else np.inf
    max_y = float(np.max(finite_y)) if finite_y.size else np.inf
    coordinate_frame = "full_dmd"
    if roi_width is not None and roi_height is not None and max_x <= roi_width + 16 and max_y <= roi_height + 16:
        coordinate_frame = "roi_local"
    return CalibrationMap(
        map_x=map_x,
        map_y=map_y,
        mask_valid=mask_valid,
        source_path=str(mat_path),
        coordinate_frame=coordinate_frame,
        roi_left=int(roi_left or 0),
        roi_top=int(roi_top or 0),
        roi_width=roi_width,
        roi_height=roi_height,
    )


def projection_paths_for_group(
    config: dict[str, Any],
    group_name: str,
    sequence: ProjectionSequence | None = None,
    sequence_stem: str = "saomiao3_6",
) -> list[ProjectionEntry]:
    sequence = sequence or read_projection_sequence(config, stem=sequence_stem)
    group = frame_groups_by_name(config)[group_name]
    entries = []
    for idx in group_frame_ids(group):
        if idx not in sequence.entries:
            raise KeyError(f"Frame {idx} is absent from projection sequence {sequence.path}")
        entry = sequence.entries[idx]
        if not Path(entry.bitmap_path).exists():
            raise FileNotFoundError(f"Projection bitmap for frame {idx} not found: {entry.bitmap_path}")
        entries.append(entry)
    return entries


def read_projection_bitmaps(entries: list[ProjectionEntry]) -> np.ndarray:
    frames = []
    for entry in entries:
        arr = iio.imread(entry.bitmap_path)
        if arr.ndim == 3:
            arr = arr[..., 0]
        frames.append(np.asarray(arr, dtype=np.float32))
    stack = np.stack(frames, axis=0)
    if stack.max(initial=0.0) > 1.0:
        stack = stack / 255.0
    return stack.astype(np.float32)


def normalize_phase_patterns(patterns: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    arr = np.asarray(patterns, dtype=np.float32)
    centered = arr - arr.mean(axis=0, keepdims=True)
    rms = np.sqrt(np.mean(centered * centered, axis=0, keepdims=True))
    return (centered / (rms + eps)).astype(np.float32)


def sample_patterns_to_camera(patterns: np.ndarray, cal_map: CalibrationMap) -> tuple[np.ndarray, dict[str, Any]]:
    sampled = []
    y = cal_map.map_y.astype(np.float32)
    x = cal_map.map_x.astype(np.float32)
    if cal_map.coordinate_frame == "roi_local":
        y = y + float(cal_map.roi_top)
        x = x + float(cal_map.roi_left)
    valid = cal_map.mask_valid & np.isfinite(x) & np.isfinite(y)
    h_dmd, w_dmd = patterns.shape[-2:]
    valid &= x >= 0
    valid &= y >= 0
    valid &= x <= (w_dmd - 1)
    valid &= y <= (h_dmd - 1)
    coords = [np.where(valid, y, 0.0), np.where(valid, x, 0.0)]
    for frame in np.asarray(patterns, dtype=np.float32):
        mapped = ndimage.map_coordinates(frame, coords, order=1, mode="constant", cval=0.0, prefilter=False)
        mapped = mapped.astype(np.float32)
        mapped[~valid] = 0.0
        sampled.append(mapped)
    sampled_stack = normalize_phase_patterns(np.stack(sampled, axis=0))
    sampled_stack[:, ~valid] = 0.0
    info = {
        "calibration_source": cal_map.source_path,
        "coordinate_frame": cal_map.coordinate_frame,
        "roi_left": cal_map.roi_left,
        "roi_top": cal_map.roi_top,
        "roi_width": cal_map.roi_width,
        "roi_height": cal_map.roi_height,
        "valid_fraction": float(np.mean(valid)),
        "dmd_shape": [int(h_dmd), int(w_dmd)],
        "camera_shape": [int(sampled_stack.shape[-2]), int(sampled_stack.shape[-1])],
    }
    return sampled_stack, info


def measured_camera_patterns_for_group(
    config: dict[str, Any],
    group_name: str,
    crop: CropSpec | None = None,
    sequence_stem: str = "saomiao3_6",
    calibration_label_prefix: str = "3-6",
    mat_name: str = "DMD_CCD_Calibration_Dict.mat",
) -> tuple[np.ndarray, dict[str, Any]]:
    sequence = read_projection_sequence(config, stem=sequence_stem)
    entries = projection_paths_for_group(config, group_name, sequence=sequence)
    cal_map = load_calibration_map(
        config,
        label_prefix=calibration_label_prefix,
        mat_name=mat_name,
        crop=crop,
        roi_left=sequence.roi_left,
        roi_top=sequence.roi_top,
        roi_width=sequence.roi_width,
        roi_height=sequence.roi_height,
    )
    bitmaps = read_projection_bitmaps(entries)
    sampled, sample_info = sample_patterns_to_camera(bitmaps, cal_map)
    info = {
        "sequence_path": sequence.path,
        "sequence_encoding": sequence.encoding,
        "frame_ids": [entry.frame_id for entry in entries],
        "bitmap_names": [entry.bitmap_name for entry in entries],
        "bitmap_paths": [entry.bitmap_path for entry in entries],
        "bitmap_shape": [int(v) for v in bitmaps.shape],
        **sample_info,
    }
    return sampled, info
