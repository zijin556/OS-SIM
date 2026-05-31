from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import data_io


CONTRACTS: list[dict[str, Any]] = [
    {
        "id": "data_inspection",
        "description": "Raw data manifest, format inventory, dimension inference, frame stats, and previews.",
        "config": "outputs/config.json",
        "results": [
            "outputs/data_manifest.json",
            "outputs/data_report.md",
            "outputs/data_frame_stats.json",
            "outputs/data_format_inventory.json",
            "outputs/data_dimension_inference.json",
            "outputs/previews/raw_frames.png",
            "outputs/previews/phase_frames_montage.png",
            "outputs/previews/z_scan_montage.png",
        ],
    },
    {
        "id": "calibration_assets",
        "description": "Measured DMD bitmap and DMD-CCD map inventory.",
        "config": "outputs/calibration/config.json",
        "results": [
            "outputs/calibration/assets_inventory.json",
            "outputs/calibration/calibration_report.md",
            "outputs/calibration/ossim_t6_h_3step_mapped_patterns.npy",
            "outputs/calibration/ossim_t6_v_3step_mapped_patterns.npy",
        ],
    },
    {
        "id": "z_unit_evidence",
        "description": "Local z-unit evidence and report.",
        "config": "outputs/z_unit/config.json",
        "results": ["outputs/z_unit/evidence.json", "outputs/z_unit/z_unit_report.md"],
    },
    {
        "id": "height_truth_assets",
        "description": "Local search for independent height-truth assets and derived pseudo-references.",
        "config": "outputs/height_truth_assets/config.json",
        "metrics": "outputs/height_truth_assets/assets.json",
        "results": ["outputs/height_truth_assets/height_truth_assets_report.md"],
    },
    {
        "id": "cli_surface",
        "description": "Command-line help, output-scope, raw-root policy, and graceful dependency-degradation audit.",
        "config": "outputs/cli_surface/config.json",
        "metrics": "outputs/cli_surface/cli_surface.json",
        "results": ["outputs/cli_surface/cli_surface_report.md"],
    },
    {
        "id": "baseline_t6_hv",
        "description": "Traditional H/V/fused OS-SIM baseline.",
        "config": "outputs/baseline/t6_hv/config.json",
        "metrics": "outputs/baseline/t6_hv/metrics.json",
        "results": ["outputs/baseline/t6_hv/A_fused_stack.npy", "outputs/baseline/t6_hv/height_map.png"],
    },
    {
        "id": "frequency_t6_hv",
        "description": "T6 H/V FFT residual analysis.",
        "config": "outputs/frequency/t6_hv/config.json",
        "metrics": "outputs/frequency/t6_hv/metrics.json",
        "results": ["outputs/frequency/t6_hv/frequency_report.md"],
    },
    {
        "id": "frequency_group_compare",
        "description": "T6/T12 group comparison.",
        "config": "outputs/frequency/group_compare/config.json",
        "metrics": "outputs/frequency/group_compare/t6_vs_t12_compare.json",
        "results": ["outputs/frequency/group_compare/frequency_compare_report.md"],
    },
    {
        "id": "forward_t6_hv",
        "description": "Forward model crop comparison.",
        "config": "outputs/forward/t6_hv/config.json",
        "metrics": "outputs/forward/t6_hv/model_compare.json",
        "results": ["outputs/forward/t6_hv/forward_report.md"],
    },
    {
        "id": "latent_opt_generated",
        "description": "Generated-pattern A-latent optimization.",
        "config": "outputs/latent_opt/t6_hv/config.json",
        "metrics": "outputs/latent_opt/t6_hv/metrics.json",
        "results": ["outputs/latent_opt/t6_hv/A_fused_stack.npy"],
        "requires_seed": True,
    },
    {
        "id": "latent_opt_measured",
        "description": "Measured-calibrated A-latent optimization.",
        "config": "outputs/latent_opt/t6_hv_measured/config.json",
        "metrics": "outputs/latent_opt/t6_hv_measured/metrics.json",
        "results": ["outputs/latent_opt/t6_hv_measured/A_fused_stack.npy"],
        "requires_seed": True,
    },
    {
        "id": "theta_opt",
        "description": "Low-dimensional q/phase/harmonic theta optimization.",
        "config": "outputs/theta_opt/t6_hv/config.json",
        "metrics": "outputs/theta_opt/t6_hv/metrics.json",
        "results": ["outputs/theta_opt/t6_hv/A_fused_stack.npy"],
    },
    {
        "id": "affine_warp_opt",
        "description": "Generated-DMD affine warp theta optimization.",
        "config": "outputs/affine_warp_opt/t6_hv/config.json",
        "metrics": "outputs/affine_warp_opt/t6_hv/metrics.json",
        "results": ["outputs/affine_warp_opt/t6_hv/A_fused_stack.npy"],
        "requires_seed": True,
    },
    {
        "id": "platform_opt_generated",
        "description": "Generated-pattern platform-gated A optimization.",
        "config": "outputs/platform_opt/t6_hv/config.json",
        "metrics": "outputs/platform_opt/t6_hv/metrics.json",
        "results": ["outputs/platform_opt/t6_hv/A_fused_best_stack.npy"],
        "requires_seed": True,
    },
    {
        "id": "platform_opt_measured",
        "description": "Measured-calibrated platform-gated A optimization.",
        "config": "outputs/platform_opt/t6_hv_measured/config.json",
        "metrics": "outputs/platform_opt/t6_hv_measured/metrics.json",
        "results": ["outputs/platform_opt/t6_hv_measured/A_fused_best_stack.npy"],
        "requires_seed": True,
    },
    {
        "id": "height_baseline",
        "description": "Baseline height reconstruction.",
        "config": "outputs/height/baseline_t6_hv/config.json",
        "metrics": "outputs/height/baseline_t6_hv/metrics.json",
        "results": ["outputs/height/baseline_t6_hv/height_maps.png", "outputs/height/baseline_t6_hv/platform_metrics.json"],
    },
    {
        "id": "height_platform_measured",
        "description": "Measured-calibrated platform-gated height reconstruction.",
        "config": "outputs/height/platform_opt_t6_hv_measured/config.json",
        "metrics": "outputs/height/platform_opt_t6_hv_measured/metrics.json",
        "results": [
            "outputs/height/platform_opt_t6_hv_measured/height_maps.png",
            "outputs/height/platform_opt_t6_hv_measured/platform_metrics.json",
        ],
    },
    {
        "id": "complete_samples",
        "description": "Complete non-excluded sample comparison.",
        "config": "outputs/complete_samples/t6_hv_measured/config.json",
        "metrics": "outputs/complete_samples/t6_hv_measured/metrics.json",
        "results": ["outputs/complete_samples/t6_hv_measured/complete_samples_report.md"],
    },
    {
        "id": "latent_sweep",
        "description": "A-latent hyperparameter sweep.",
        "config": "outputs/latent_sweep/t6_hv/config.json",
        "metrics": "outputs/latent_sweep/t6_hv/sweep_metrics.json",
        "results": ["outputs/latent_sweep/t6_hv/sweep_report.md"],
    },
    {
        "id": "simulation",
        "description": "Synthetic ideal/realistic sanity check.",
        "config": "outputs/sim/config.json",
        "metrics": "outputs/sim/sim_metrics.json",
        "results": ["outputs/sim/sim_report.md"],
        "requires_seed": True,
    },
    {
        "id": "tiny_net",
        "description": "Optional tiny-net stage or graceful missing-torch report.",
        "config": "outputs/tiny_net/config.json",
        "metrics": "outputs/tiny_net/metrics.json",
        "results": ["outputs/tiny_net/report.md"],
    },
    {
        "id": "inverse_net_training",
        "description": "Neural inverse light-section network training or graceful missing-torch report.",
        "config": "outputs/inverse_net/t6_hv/config.json",
        "metrics": "outputs/inverse_net/t6_hv/metrics.json",
        "results": ["outputs/inverse_net/t6_hv/inverse_net_report.md"],
        "requires_seed": True,
    },
    {
        "id": "inverse_net_evaluation",
        "description": "Inverse-net A-stack/height evaluation or missing-prediction report.",
        "config": "outputs/inverse_net/t6_hv/config.json",
        "metrics": "outputs/inverse_net/t6_hv/evaluation_metrics.json",
        "results": ["outputs/inverse_net/t6_hv/inverse_net_evaluation_report.md"],
    },
    {
        "id": "ablation",
        "description": "Forward and A-stack ablation summary.",
        "config": "outputs/ablation/t6_hv/config.json",
        "metrics": "outputs/ablation/t6_hv/metrics.json",
        "results": ["outputs/ablation/t6_hv/ablation_report.md"],
        "requires_seed": True,
    },
    {
        "id": "final_questions",
        "description": "Evidence-linked answers to final required research questions.",
        "config": "outputs/final_questions/config.json",
        "metrics": "outputs/final_questions/answers.json",
        "results": ["outputs/final_questions/final_question_answers.md"],
    },
    {
        "id": "requirement_traceability",
        "description": "Original task requirement-to-evidence traceability matrix.",
        "config": "outputs/traceability/config.json",
        "metrics": "outputs/traceability/requirements_traceability.json",
        "results": ["outputs/traceability/requirements_traceability.md"],
    },
    {
        "id": "coverage_audit",
        "description": "Requirement-level reproducibility audit.",
        "config": "outputs/audit/config.json",
        "metrics": "outputs/audit/coverage.json",
        "results": ["outputs/audit/coverage_report.md"],
    },
]

SEED_SCRIPT_CHECKS = {
    "scripts/optimize_modulation_latents.py": ["--seed", "np.random.seed(args.seed)"],
    "scripts/optimize_platform_constrained_latents.py": ["--seed", "np.random.seed(args.seed)"],
    "scripts/optimize_affine_warp_latents.py": ["--seed", "np.random.seed(args.seed)"],
    "scripts/summarize_ablation.py": ["--seed", "np.random.seed(args.seed)"],
    "scripts/simulate_os_sim_realistic.py": ["--seed", "np.random.default_rng(args.seed)"],
    "scripts/train_inverse_lightsection_net.py": ["--seed", "torch.manual_seed(args.seed)", "np.random.seed(args.seed)"],
}


def exists(rel_path: str) -> bool:
    return (ROOT / rel_path).exists()


def parse_json(rel_path: str) -> tuple[bool, Any | None, str | None]:
    path = ROOT / rel_path
    if not path.exists():
        return False, None, "missing"
    try:
        with path.open("r", encoding="utf-8") as f:
            return True, json.load(f), None
    except Exception as exc:
        return False, None, f"{type(exc).__name__}: {exc}"


def is_outputs_path(rel_path: str) -> bool:
    p = Path(rel_path)
    if p.is_absolute():
        try:
            p.resolve().relative_to((ROOT / "outputs").resolve())
            return True
        except ValueError:
            return False
    return p.parts and p.parts[0] == "outputs"


def config_output_ok(config: Any) -> bool:
    if not isinstance(config, dict):
        return False
    output = config.get("output")
    if output is None:
        return True
    return is_outputs_path(str(output).replace("\\", "/"))


def has_seed(config: Any, metrics: Any) -> bool:
    candidates = []
    if isinstance(config, dict):
        candidates.extend([config.get("seed"), config.get("random_seed")])
        nested = config.get("config")
        if isinstance(nested, dict):
            candidates.extend([nested.get("seed"), nested.get("random_seed")])
    if isinstance(metrics, dict):
        candidates.extend([metrics.get("seed"), metrics.get("random_seed")])
        nested = metrics.get("config")
        if isinstance(nested, dict):
            candidates.extend([nested.get("seed"), nested.get("random_seed")])
    return any(value is not None for value in candidates)


def audit_contract(contract: dict[str, Any]) -> dict[str, Any]:
    required_paths = []
    if "config" in contract:
        required_paths.append(contract["config"])
    if "metrics" in contract:
        required_paths.append(contract["metrics"])
    required_paths.extend(contract.get("results", []))
    missing = [path for path in required_paths if not exists(path)]
    json_checks = {}
    config_ok, config, config_error = parse_json(contract["config"]) if "config" in contract else (True, {}, None)
    metrics_ok, metrics, metrics_error = parse_json(contract["metrics"]) if "metrics" in contract else (True, {}, None)
    if "config" in contract:
        json_checks["config"] = {"path": contract["config"], "ok": config_ok, "error": config_error}
    if "metrics" in contract:
        json_checks["metrics"] = {"path": contract["metrics"], "ok": metrics_ok, "error": metrics_error}
    output_ok = config_output_ok(config)
    seed_ok = True
    if contract.get("requires_seed"):
        seed_ok = has_seed(config, metrics)
    ok = not missing and all(item["ok"] for item in json_checks.values()) and output_ok and seed_ok
    return {
        "id": contract["id"],
        "description": contract["description"],
        "status": "satisfied" if ok else "missing",
        "required_paths": required_paths,
        "missing": missing,
        "json_checks": json_checks,
        "output_under_outputs": output_ok,
        "seed_required": bool(contract.get("requires_seed")),
        "seed_recorded": seed_ok,
    }


def audit_exclusion_policy() -> dict[str, Any]:
    manifest_ok, manifest, manifest_error = parse_json("outputs/data_manifest.json")
    complete_ok, complete_metrics, complete_error = parse_json("outputs/complete_samples/t6_hv_measured/metrics.json")
    excluded = set(manifest.get("excluded_samples", [])) if manifest_ok and isinstance(manifest, dict) else set()
    complete_samples = {
        sample.get("name")
        for sample in manifest.get("samples", [])
        if isinstance(sample, dict) and not sample.get("excluded") and sample.get("incomplete_layer_count") == 0
    } if manifest_ok and isinstance(manifest, dict) else set()
    compared_samples = set(complete_metrics.get("samples", [])) if complete_ok and isinstance(complete_metrics, dict) else set()
    compared_excluded = set(complete_metrics.get("excluded_samples", [])) if complete_ok and isinstance(complete_metrics, dict) else set()
    ok = (
        manifest_ok
        and complete_ok
        and "台阶曝光1" in excluded
        and "台阶曝光1" not in complete_samples
        and "台阶曝光1" not in compared_samples
        and "台阶曝光1" in compared_excluded
    )
    return {
        "id": "excluded_incomplete_sample",
        "status": "satisfied" if ok else "missing",
        "manifest_error": manifest_error,
        "complete_metrics_error": complete_error,
        "manifest_excluded_samples": sorted(excluded),
        "complete_nonexcluded_samples": sorted(s for s in complete_samples if s),
        "compared_samples": sorted(compared_samples),
        "compared_excluded_samples": sorted(compared_excluded),
    }


def audit_seed_scripts() -> dict[str, Any]:
    rows = []
    for rel_path, needles in SEED_SCRIPT_CHECKS.items():
        path = ROOT / rel_path
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        rows.append(
            {
                "path": rel_path,
                "exists": path.exists(),
                "required_snippets": needles,
                "snippets_present": all(needle in text for needle in needles),
            }
        )
    ok = all(row["exists"] and row["snippets_present"] for row in rows)
    return {"id": "seeded_random_processes", "status": "satisfied" if ok else "missing", "rows": rows}


def summarize_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = row["status"]
        counts[status] = counts.get(status, 0) + 1
    return counts


def write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Experiment Output Contract Audit",
        "",
        "## Summary",
        "",
        f"- Overall: `{result['overall']}`",
        f"- Status counts: `{result['status_counts']}`",
        "- This audit checks experiment output contracts: config/result files, parseable JSON metrics, output path scope, incomplete-sample exclusion, and random seed visibility.",
        "",
        "## Contract Table",
        "",
        "| id | status | missing | seed | output scope |",
        "|---|---|---|---|---|",
    ]
    for row in result["contracts"]:
        missing = ", ".join(row["missing"])
        seed = "required/ok" if row["seed_required"] and row["seed_recorded"] else "required/missing" if row["seed_required"] else "not_required"
        lines.append(f"| {row['id']} | {row['status']} | {missing} | {seed} | {row['output_under_outputs']} |")
    lines += [
        "",
        "## Policy Checks",
        "",
        f"- Exclusion policy: `{result['exclusion_policy']['status']}`",
        f"- Seeded scripts: `{result['seeded_random_processes']['status']}`",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit OS-SIM experiment output contracts.")
    parser.add_argument("--output", default="outputs/reproducibility")
    args = parser.parse_args()

    out_dir = ROOT / args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    contracts = [audit_contract(contract) for contract in CONTRACTS]
    exclusion_policy = audit_exclusion_policy()
    seeded_random_processes = audit_seed_scripts()
    status_rows = contracts + [exclusion_policy, seeded_random_processes]
    counts = summarize_counts(status_rows)
    result = {
        "overall": "satisfied" if counts.get("missing", 0) == 0 else "active_not_complete",
        "status_counts": counts,
        "contracts": contracts,
        "exclusion_policy": exclusion_policy,
        "seeded_random_processes": seeded_random_processes,
    }
    data_io.write_json(out_dir / "contracts.json", result)
    data_io.write_json(out_dir / "config.json", vars(args))
    write_report(out_dir / "contract_report.md", result)
    data_io.append_report(
        ROOT / "outputs/nightly_report.md",
        "P14 Experiment Output Contract Audit",
        [
            f"- Audited `{len(contracts)}` experiment output contracts with counts `{counts}`.",
            f"- Overall contract status: `{result['overall']}`.",
            f"- Outputs saved under `{out_dir}`.",
        ],
    )
    print(f"Wrote experiment contract audit to {out_dir}")
    print(json.dumps({"overall": result["overall"], "status_counts": counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
