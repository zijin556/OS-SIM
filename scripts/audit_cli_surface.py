from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import data_io


def load_json(rel_path: str) -> Any | None:
    path = ROOT / rel_path
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def is_outputs_path(path_text: str) -> bool:
    normalized = path_text.replace("\\", "/")
    path = Path(normalized)
    if path.is_absolute():
        try:
            path.resolve().relative_to((ROOT / "outputs").resolve())
            return True
        except ValueError:
            return False
    return normalized == "outputs" or normalized.startswith("outputs/")


def run_help(script: Path, timeout: int) -> dict[str, Any]:
    rel = script.relative_to(ROOT).as_posix()
    cmd = [sys.executable, str(script), "--help"]
    try:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
        combined = f"{proc.stdout}\n{proc.stderr}"
        return {
            "script": rel,
            "returncode": proc.returncode,
            "help_ok": proc.returncode == 0 and "usage:" in combined.lower(),
            "stdout_head": proc.stdout[:500],
            "stderr_head": proc.stderr[:500],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "script": rel,
            "returncode": None,
            "help_ok": False,
            "timeout_seconds": timeout,
            "stdout_head": (exc.stdout or "")[:500] if isinstance(exc.stdout, str) else "",
            "stderr_head": (exc.stderr or "")[:500] if isinstance(exc.stderr, str) else "",
        }


def inspect_output_defaults(script: Path) -> dict[str, Any]:
    text = script.read_text(encoding="utf-8")
    rel = script.relative_to(ROOT).as_posix()
    matches = re.findall(
        r"add_argument\(\s*['\"]--output['\"].*?default\s*=\s*['\"]([^'\"]+)['\"]",
        text,
        flags=re.DOTALL,
    )
    return {
        "script": rel,
        "has_output_argument": "--output" in text,
        "default_outputs": matches,
        "defaults_under_outputs": all(is_outputs_path(match) for match in matches) if matches else "--output" in text,
    }


def audit_contract_output_scope() -> dict[str, Any]:
    contracts = load_json("outputs/reproducibility/contracts.json")
    if not contracts:
        return {"status": "missing", "reason": "outputs/reproducibility/contracts.json not found"}
    rows = contracts.get("contracts", [])
    bad = [
        {"id": row.get("id"), "output_under_outputs": row.get("output_under_outputs")}
        for row in rows
        if row.get("output_under_outputs") is not True
    ]
    return {
        "status": "satisfied" if not bad and rows else "missing",
        "contract_count": len(rows),
        "bad_contracts": bad,
    }


def audit_raw_root_policy() -> dict[str, Any]:
    config = data_io.load_config("configs/ossim_dataset.yaml")
    raw_root = Path(config["dataset"]["raw_root"]).resolve()
    outputs_root = (ROOT / "outputs").resolve()
    repo_root = ROOT.resolve()
    raw_under_outputs = False
    try:
        raw_root.relative_to(outputs_root)
        raw_under_outputs = True
    except ValueError:
        pass
    raw_under_repo = False
    try:
        raw_root.relative_to(repo_root)
        raw_under_repo = True
    except ValueError:
        pass
    return {
        "status": "satisfied" if raw_root.exists() and not raw_under_outputs else "missing",
        "raw_root": str(raw_root),
        "raw_root_exists": raw_root.exists(),
        "raw_root_under_repo": raw_under_repo,
        "raw_root_under_outputs": raw_under_outputs,
        "policy": "raw data is read-only input; generated artifacts are written under the repository outputs directory",
    }


def audit_dependency_graceful_degradation() -> dict[str, Any]:
    tiny = load_json("outputs/tiny_net/metrics.json")
    report_exists = (ROOT / "outputs/tiny_net/report.md").exists()
    status = tiny.get("status") if isinstance(tiny, dict) else None
    ok = status in {"skipped_missing_torch", "initialized_only", "completed"} and report_exists
    return {
        "status": "satisfied" if ok else "missing",
        "tiny_net_status": status,
        "report_exists": report_exists,
        "metrics_path": "outputs/tiny_net/metrics.json",
    }


def summarize_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = row["status"]
        counts[status] = counts.get(status, 0) + 1
    return counts


def write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# CLI And Safety Surface Audit",
        "",
        "## Summary",
        "",
        f"- Overall: `{result['overall']}`",
        f"- Status counts: `{result['status_counts']}`",
        f"- Scripts checked: `{len(result['cli_help_rows'])}`",
        f"- Contract output scope: `{result['contract_output_scope']['status']}`",
        f"- Raw data policy: `{result['raw_root_policy']['status']}`",
        f"- Dependency degradation: `{result['dependency_graceful_degradation']['status']}`",
        "",
        "## CLI Help",
        "",
        "| script | help_ok | returncode |",
        "|---|---:|---:|",
    ]
    for row in result["cli_help_rows"]:
        lines.append(f"| `{row['script']}` | `{row['help_ok']}` | `{row['returncode']}` |")
    lines += [
        "",
        "## Output Defaults",
        "",
        "| script | has --output | defaults | defaults under outputs |",
        "|---|---:|---|---:|",
    ]
    for row in result["output_default_rows"]:
        defaults = ", ".join(f"`{item}`" for item in row["default_outputs"])
        lines.append(
            f"| `{row['script']}` | `{row['has_output_argument']}` | {defaults} | `{row['defaults_under_outputs']}` |"
        )
    lines += [
        "",
        "## Policy Checks",
        "",
        f"- Contract output scope: `{result['contract_output_scope']}`",
        f"- Raw root policy: `{result['raw_root_policy']}`",
        f"- Dependency degradation: `{result['dependency_graceful_degradation']}`",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit command-line and safety surface for OS-SIM scripts.")
    parser.add_argument("--output", default="outputs/cli_surface")
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()

    out_dir = ROOT / args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    scripts = sorted((ROOT / "scripts").glob("*.py"), key=lambda p: p.name)
    cli_help_rows = [run_help(script, args.timeout) for script in scripts]
    output_default_rows = [inspect_output_defaults(script) for script in scripts]
    contract_output_scope = audit_contract_output_scope()
    raw_root_policy = audit_raw_root_policy()
    dependency_graceful_degradation = audit_dependency_graceful_degradation()
    status_rows = [
        {"id": "cli_help", "status": "satisfied" if all(row["help_ok"] for row in cli_help_rows) else "missing"},
        {
            "id": "output_defaults",
            "status": "satisfied"
            if all(row["has_output_argument"] and row["defaults_under_outputs"] for row in output_default_rows)
            else "missing",
        },
        {"id": "contract_output_scope", "status": contract_output_scope["status"]},
        {"id": "raw_root_policy", "status": raw_root_policy["status"]},
        {"id": "dependency_graceful_degradation", "status": dependency_graceful_degradation["status"]},
    ]
    counts = summarize_counts(status_rows)
    result = {
        "overall": "satisfied" if counts.get("missing", 0) == 0 else "active_not_complete",
        "status_counts": counts,
        "config": vars(args),
        "status_rows": status_rows,
        "cli_help_rows": cli_help_rows,
        "output_default_rows": output_default_rows,
        "contract_output_scope": contract_output_scope,
        "raw_root_policy": raw_root_policy,
        "dependency_graceful_degradation": dependency_graceful_degradation,
    }
    data_io.write_json(out_dir / "cli_surface.json", result)
    data_io.write_json(out_dir / "config.json", vars(args))
    write_report(out_dir / "cli_surface_report.md", result)
    data_io.append_report(
        ROOT / "outputs/nightly_report.md",
        "P16 CLI And Safety Surface Audit",
        [
            f"- Checked `{len(cli_help_rows)}` scripts for `--help` support; status counts `{counts}`.",
            f"- Contract output scope: `{contract_output_scope['status']}`.",
            f"- Raw root policy: `{raw_root_policy['status']}`.",
            f"- Outputs saved under `{out_dir}`.",
        ],
    )
    print(f"Wrote CLI surface audit to {out_dir}")
    print(json.dumps({"overall": result["overall"], "status_counts": counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
