from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import data_io


def read_text(path: Path) -> tuple[str, str]:
    data = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-8", "gbk", "latin1"):
        try:
            return data.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    return data.decode("latin1", errors="replace"), "latin1-replace"


def line_hits(path: Path, patterns: list[str]) -> list[dict[str, Any]]:
    text, encoding = read_text(path)
    rows = []
    for idx, line in enumerate(text.splitlines(), start=1):
        if any(re.search(pattern, line, flags=re.IGNORECASE) for pattern in patterns):
            rows.append({"path": str(path), "line": idx, "encoding": encoding, "text": line.strip()})
    return rows


def inspect_profile_csv(path: Path, z_max: float) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        first_rows = [row for _, row in zip(range(8), reader)]
    checks = []
    for row in first_rows:
        raw_height = float(row["raw_height"])
        fitted_height = float(row["fitted_height"])
        raw_peak = float(row["raw_pzt_peak"])
        fitted_peak = float(row["fitted_pzt_peak"])
        checks.append(
            {
                "sample_index": int(float(row["sample_index"])),
                "raw_height": raw_height,
                "raw_pzt_peak": raw_peak,
                "z_max_minus_raw_pzt_peak": z_max - raw_peak,
                "raw_height_abs_error": abs(raw_height - (z_max - raw_peak)),
                "fitted_height": fitted_height,
                "fitted_pzt_peak": fitted_peak,
                "z_max_minus_fitted_pzt_peak": z_max - fitted_peak,
                "fitted_height_abs_error": abs(fitted_height - (z_max - fitted_peak)),
            }
        )
    return {
        "path": str(path),
        "z_max_used_for_height_check": z_max,
        "rows_checked": len(checks),
        "max_raw_height_identity_error": max(item["raw_height_abs_error"] for item in checks) if checks else None,
        "max_fitted_height_identity_error": max(item["fitted_height_abs_error"] for item in checks) if checks else None,
        "checks": checks,
    }


def inspect_z_summary_csv(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        z_values = [float(row["z"]) for row in reader]
    diffs = [round(z_values[i + 1] - z_values[i], 10) for i in range(len(z_values) - 1)]
    return {
        "path": str(path),
        "z_min": min(z_values) if z_values else None,
        "z_max": max(z_values) if z_values else None,
        "z_count": len(z_values),
        "unique_z_steps": sorted(set(diffs)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect local evidence for z coordinate unit and height convention.")
    parser.add_argument("--config", default="configs/ossim_dataset.yaml")
    parser.add_argument("--output", default="outputs/z_unit")
    args = parser.parse_args()

    config = data_io.load_config(args.config)
    project_data_root = data_io.raw_root_from_config(config).parents[3]
    analysis_root = project_data_root / "02_analyse_data_programs" / "02_dlp6500_processing_matlab" / "Confocal&OS-SIM"
    square_root = data_io.raw_root_from_config(config).parents[1] / "square_stripes_output"
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    data_io.write_json(output / "config.json", vars(args))

    matlab_evidence_files = [
        analysis_root / "confocal_SIM.m",
        analysis_root / "confocal_cutoff_9vs16vs25vs81_lmd.m",
        analysis_root / "confocal_zangwu_plot.m",
        square_root / "auto" / "RMS.m",
    ]
    patterns = [
        r"z_step\s*=",
        r"物理高度",
        r"\\mum",
        r"pixelSize\s*=",
        r"zScale\s*=",
    ]
    text_evidence = []
    for path in matlab_evidence_files:
        if path.exists():
            text_evidence.extend(line_hits(path, patterns))

    profile_csv = (
        analysis_root
        / "共聚焦判断异常点并输出gif"
        / "Analysis_Output"
        / "细台阶"
        / "os_sim"
        / "analysis"
        / "T6_3Step"
        / "data"
        / "boundary_line_profile.csv"
    )
    z_summary_csv = (
        analysis_root
        / "共聚焦判断异常点并输出gif"
        / "Analysis_Output"
        / "细台阶"
        / "os_sim"
        / "analysis"
        / "T6_3Step"
        / "data"
        / "z_slice_intensity_summary.csv"
    )
    profile_check = inspect_profile_csv(profile_csv, z_max=40.0) if profile_csv.exists() else None
    z_summary = inspect_z_summary_csv(z_summary_csv) if z_summary_csv.exists() else None

    evidence = {
        "inferred_z_unit": "micrometer",
        "confidence": "local_analysis_convention_high",
        "not_an_external_height_truth": True,
        "interpretation": (
            "Local MATLAB analysis scripts label reconstructed profiles as physical height in micrometers, "
            "and existing CSV outputs compute raw/fitted height as z_max minus PZT peak in the same z coordinate. "
            "This supports using the Z folder values as micrometers for reporting, but it does not provide an "
            "independent calibrated step-height truth for absolute error."
        ),
        "project_data_root": str(project_data_root),
        "text_evidence": text_evidence,
        "profile_identity_check": profile_check,
        "z_summary": z_summary,
    }
    data_io.write_json(output / "evidence.json", evidence)
    report = [
        "# Z Unit Evidence Report",
        "",
        "- Inferred z unit: `micrometer`",
        "- Confidence: `local_analysis_convention_high`",
        "- This is not an independent step-height truth.",
        "",
        "## Text Evidence",
        "",
    ]
    for item in text_evidence:
        report.append(f"- `{item['path']}` line `{item['line']}`: `{item['text']}`")
    if z_summary:
        report += [
            "",
            "## Z Summary CSV",
            "",
            f"- Path: `{z_summary['path']}`",
            f"- z range/count: `{z_summary['z_min']}` to `{z_summary['z_max']}`, count `{z_summary['z_count']}`",
            f"- unique z steps: `{z_summary['unique_z_steps']}`",
        ]
    if profile_check:
        report += [
            "",
            "## Height Identity Check",
            "",
            f"- Path: `{profile_check['path']}`",
            f"- Checked first `{profile_check['rows_checked']}` profile rows.",
            f"- max |raw_height - (40 - raw_pzt_peak)|: `{profile_check['max_raw_height_identity_error']:.6g}`",
            f"- max |fitted_height - (40 - fitted_pzt_peak)|: `{profile_check['max_fitted_height_identity_error']:.6g}`",
        ]
    report += [
        "",
        "## Conclusion",
        "",
        "Use micrometers as the local z-coordinate reporting unit. Keep absolute metrology error blocked until an independent height reference or calibrated step artifact is supplied.",
    ]
    (output / "z_unit_report.md").write_text("\n".join(report), encoding="utf-8")
    data_io.append_report(
        ROOT / "outputs/nightly_report.md",
        "P12 Z Unit Evidence",
        [
            "- Local MATLAB analysis evidence supports reporting z coordinates in micrometers.",
            "- This does not supply an independent step-height truth for absolute error.",
            f"- Outputs saved under `{output}`.",
        ],
    )
    print(f"Wrote z-unit evidence outputs to {output}")


if __name__ == "__main__":
    main()
