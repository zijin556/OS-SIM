from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import data_io


NAME_PATTERNS = [
    "*三维形貌*.mat",
    "*三维形貌*.txt",
    "*三维点云*.txt",
    "*点云*.txt",
    "*profile*.csv",
    "*boundary*_profile*.csv",
    "*height*.csv",
    "*height*.txt",
    "*RMS*.m",
]

EXTERNAL_TRUTH_KEYWORDS = [
    "标准",
    "真值",
    "ground",
    "truth",
    "reference",
    "profilometer",
    "轮廓仪",
    "白光",
    "AFM",
    "台阶高度",
    "step_height",
]

DERIVED_KEYWORDS = [
    "square_stripes_output",
    "Analysis_Output",
    "SHT",
    "HiLo",
    "FUIHT",
    "公式",
    "os_sim",
    "confocal",
]


def rel_or_str(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def classify(path: Path) -> tuple[str, str]:
    text = str(path).lower()
    if any(keyword.lower() in text for keyword in EXTERNAL_TRUTH_KEYWORDS):
        return (
            "possible_external_truth_needs_manual_review",
            "The filename/path contains ground-truth or metrology keywords, but the file still needs provenance review.",
        )
    if any(keyword.lower() in text for keyword in DERIVED_KEYWORDS):
        return (
            "derived_algorithm_or_analysis_output_not_independent_truth",
            "The path indicates a derived algorithm output or local analysis result, so it can be used as a pseudo-reference/diagnostic comparison but not as independent calibrated height truth.",
        )
    return (
        "unclassified_candidate_needs_manual_review",
        "The path may contain height-like data but lacks explicit provenance in the filename/path.",
    )


def inspect_mat(path: Path) -> dict[str, Any]:
    try:
        from scipy.io import whosmat

        variables = [
            {"name": name, "shape": list(shape), "class": cls}
            for name, shape, cls in whosmat(path)
        ]
        return {"reader": "scipy.io.whosmat", "variables": variables}
    except Exception as exc:
        return {"reader": "scipy.io.whosmat", "error": f"{type(exc).__name__}: {exc}"}


def inspect_text_table(path: Path, max_rows: int = 2000) -> dict[str, Any]:
    try:
        arr = np.loadtxt(path, max_rows=max_rows)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return {
            "reader": "numpy.loadtxt",
            "sampled_rows": int(arr.shape[0]),
            "columns": int(arr.shape[1]),
            "sample_min_by_column": np.nanmin(arr, axis=0).astype(float).tolist(),
            "sample_max_by_column": np.nanmax(arr, axis=0).astype(float).tolist(),
        }
    except Exception as exc:
        return {"reader": "numpy.loadtxt", "error": f"{type(exc).__name__}: {exc}"}


def inspect_csv(path: Path, max_rows: int = 2000) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            rows = []
            for idx, row in enumerate(reader):
                if idx >= max_rows:
                    break
                rows.append(row)
        numeric_summary: dict[str, dict[str, float]] = {}
        if rows:
            for key in rows[0]:
                values = []
                for row in rows:
                    try:
                        values.append(float(row[key]))
                    except (TypeError, ValueError):
                        pass
                if values:
                    arr = np.asarray(values, dtype=float)
                    numeric_summary[key] = {
                        "min": float(np.nanmin(arr)),
                        "max": float(np.nanmax(arr)),
                        "mean": float(np.nanmean(arr)),
                    }
        return {
            "reader": "csv.DictReader",
            "sampled_rows": len(rows),
            "fieldnames": list(rows[0].keys()) if rows else [],
            "numeric_summary": numeric_summary,
        }
    except Exception as exc:
        return {"reader": "csv.DictReader", "error": f"{type(exc).__name__}: {exc}"}


def inspect_candidate(path: Path, max_rows: int) -> dict[str, Any]:
    status, rationale = classify(path)
    suffix = path.suffix.lower()
    if suffix == ".mat":
        details = inspect_mat(path)
    elif suffix == ".csv":
        details = inspect_csv(path, max_rows=max_rows)
    elif suffix in {".txt", ".m", ".md", ".json"}:
        details = inspect_text_table(path, max_rows=max_rows) if suffix == ".txt" else {"reader": "text_metadata_only"}
    else:
        details = {"reader": "not_attempted"}
    return {
        "path": str(path),
        "name": path.name,
        "suffix": suffix,
        "size_bytes": path.stat().st_size,
        "classification": status,
        "classification_rationale": rationale,
        "details": details,
    }


def unique_paths(paths: list[Path]) -> list[Path]:
    seen = set()
    out = []
    for path in sorted(paths, key=lambda p: str(p).lower()):
        key = str(path.resolve()).lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def find_candidates(search_roots: list[Path]) -> list[Path]:
    candidates: list[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        for pattern in NAME_PATTERNS:
            candidates.extend(path for path in root.rglob(pattern) if path.is_file())
    return unique_paths(candidates)


def write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Height Truth Asset Search",
        "",
        "## Summary",
        "",
        f"- Search status: `{result['search_status']}`",
        f"- Candidate files found: `{result['candidate_count']}`",
        f"- Independently calibrated truth found: `{result['independent_truth_found']}`",
        f"- Usable pseudo-reference candidates: `{result['pseudo_reference_candidate_count']}`",
        f"- Search roots: `{result['search_roots']}`",
        "",
        "## Interpretation",
        "",
        result["interpretation"],
        "",
        "## Candidate Classes",
        "",
        "| class | count |",
        "|---|---:|",
    ]
    for cls, count in sorted(result["classification_counts"].items()):
        lines.append(f"| {cls} | {count} |")
    lines += ["", "## Inspected Examples", ""]
    for item in result["inspected_candidates"][:40]:
        detail = item["details"]
        summary = ""
        if "variables" in detail:
            vars_text = ", ".join(f"{v['name']}:{v['shape']}" for v in detail["variables"][:5])
            summary = f" variables `{vars_text}`"
        elif "columns" in detail:
            summary = f" sampled `{detail.get('sampled_rows')}` rows x `{detail.get('columns')}` cols"
        elif "fieldnames" in detail:
            summary = f" fields `{detail.get('fieldnames')}`"
        elif "error" in detail:
            summary = f" error `{detail['error']}`"
        lines.append(
            f"- `{item['classification']}`: `{item['path']}` ({item['size_bytes']} bytes){summary}"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Search local project data for independent height-truth or pseudo-reference assets.")
    parser.add_argument("--config", default="configs/ossim_dataset.yaml")
    parser.add_argument("--output", default="outputs/height_truth_assets")
    parser.add_argument("--max-inspect", type=int, default=80)
    parser.add_argument("--max-rows", type=int, default=2000)
    args = parser.parse_args()

    config = data_io.load_config(args.config)
    dlp_root = Path(config["calibration_assets"]["dlp6500_root"])
    project_root = data_io.raw_root_from_config(config).parents[3]
    search_roots = [
        dlp_root / "square_stripes_output",
        project_root / "02_analyse_data_programs",
        project_root / "03_simulation",
    ]
    candidates = find_candidates(search_roots)
    inspected = [inspect_candidate(path, max_rows=args.max_rows) for path in candidates[: args.max_inspect]]
    classification_counts: dict[str, int] = {}
    for item in inspected:
        cls = item["classification"]
        classification_counts[cls] = classification_counts.get(cls, 0) + 1
    independent_truth_found = any(item["classification"] == "possible_external_truth_needs_manual_review" for item in inspected)
    pseudo_reference_count = sum(
        1 for item in inspected if item["classification"] == "derived_algorithm_or_analysis_output_not_independent_truth"
    )
    result = {
        "search_status": "completed",
        "search_roots": [str(path) for path in search_roots],
        "candidate_count": len(candidates),
        "inspected_count": len(inspected),
        "max_inspect": args.max_inspect,
        "independent_truth_found": bool(independent_truth_found),
        "pseudo_reference_candidate_count": pseudo_reference_count,
        "classification_counts": classification_counts,
        "inspected_candidates": inspected,
        "interpretation": (
            "The local tree contains many height-like morphology maps and point clouds, especially under "
            "`square_stripes_output` and MATLAB analysis output folders. These are derived algorithm outputs "
            "(SHT/HiLo/formula/OS-SIM/confocal analysis) and are useful as pseudo-reference or diagnostic "
            "comparisons, but they do not document an independent calibrated step-height or external profilometer "
            "measurement for the current `3-6扫描` raw dataset. Therefore the absolute metrology blocker remains."
        ),
    }

    out_dir = ROOT / args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    data_io.write_json(out_dir / "assets.json", result)
    data_io.write_json(out_dir / "config.json", vars(args))
    write_report(out_dir / "height_truth_assets_report.md", result)
    data_io.append_report(
        ROOT / "outputs/nightly_report.md",
        "P15 Height Truth Asset Search",
        [
            f"- Searched local data and analysis roots for independent height-truth candidates; found `{len(candidates)}` height-like files.",
            f"- Inspected `{len(inspected)}` candidates; independent calibrated truth found: `{independent_truth_found}`.",
            "- Derived morphology maps/point clouds can be used as pseudo-reference diagnostics, not absolute metrology truth.",
            f"- Outputs saved under `{out_dir}`.",
        ],
    )
    print(f"Wrote height-truth asset search outputs to {out_dir}")
    print(json.dumps({"candidate_count": len(candidates), "independent_truth_found": independent_truth_found}, ensure_ascii=False))


if __name__ == "__main__":
    main()
