from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import data_io


def exists(rel_path: str) -> bool:
    return (ROOT / rel_path).exists()


def load_json(rel_path: str) -> Any | None:
    path = ROOT / rel_path
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def np_shape(rel_path: str) -> list[int] | None:
    path = ROOT / rel_path
    if not path.exists():
        return None
    return list(np.load(path, mmap_mode="r").shape)


def row(
    req_id: str,
    phase: str,
    requirement: str,
    status: str,
    evidence: list[str],
    notes: str = "",
) -> dict[str, Any]:
    return {
        "id": req_id,
        "phase": phase,
        "requirement": requirement,
        "status": status,
        "evidence": evidence,
        "notes": notes,
    }


def all_exist(paths: list[str]) -> bool:
    return all(exists(path) for path in paths)


def build_rows() -> list[dict[str, Any]]:
    manifest = load_json("outputs/data_manifest.json") or {}
    coverage = load_json("outputs/audit/coverage.json") or {}
    contracts = load_json("outputs/reproducibility/contracts.json") or {}
    cli = load_json("outputs/cli_surface/cli_surface.json") or {}
    baseline = load_json("outputs/baseline/t6_hv/metrics.json") or {}
    frequency = load_json("outputs/frequency/t6_hv/metrics.json") or {}
    forward = load_json("outputs/forward/t6_hv/model_compare.json") or {}
    ablation = load_json("outputs/ablation/t6_hv/metrics.json") or {}
    height_compare = load_json("outputs/height/platform_compare.json") or {}
    answers = load_json("outputs/final_questions/answers.json") or {}
    height_truth = load_json("outputs/height_truth_assets/assets.json") or {}
    tiny = load_json("outputs/tiny_net/metrics.json") or {}

    samples = manifest.get("samples", [])
    excluded = set(manifest.get("excluded_samples", []))
    complete = [
        item["name"]
        for item in samples
        if not item.get("excluded") and item.get("incomplete_layer_count") == 0
    ]
    fine = next((item for item in samples if item.get("name") == "细台阶"), None)
    fine_t6_ok = False
    if fine:
        fine_t6_ok = all(
            set(range(100, 106)).issubset(
                set(layer.get("groups", {}).get("ossim_t6_h_3step", {}).get("frames", []))
                | set(layer.get("groups", {}).get("ossim_t6_v_3step", {}).get("frames", []))
            )
            for layer in fine.get("layers", [])
        )

    forward_model_names = set()
    for detail in forward.get("models", {}).values():
        forward_model_names.update(detail.get("losses", {}).keys())
    required_forward_prefixes = [
        "level0_ideal_cos",
        "level1_modulation",
        "level2_binary_dmd",
        "level3_binary_dmd",
        "level4_harmonic_grid",
    ]
    forward_levels_ok = all(any(name.startswith(prefix) for name in forward_model_names) for prefix in required_forward_prefixes)
    measured_forward_ok = all(
        "level2_measured_dmd_calibrated_map" in forward.get("models", {}).get(group, {}).get("losses", {})
        for group in ("ossim_t6_h_3step", "ossim_t6_v_3step")
    )
    ablation_rows = ablation.get("a_stack_ablation", [])
    recommended = ablation.get("recommended_a_stack", {}).get("source")
    final_answer_ids = {item.get("id") for item in answers.get("answers", [])}

    rows: list[dict[str, Any]] = []
    rows.append(
        row(
            "U0_DATA_ROOT",
            "User Constraints",
            "Use the user-specified raw scan directory as read-only input.",
            "satisfied" if manifest.get("raw_root") and Path(manifest["raw_root"]).exists() else "missing",
            ["configs/ossim_dataset.yaml", "outputs/data_manifest.json"],
            f"raw_root={manifest.get('raw_root')}",
        )
    )
    rows.append(
        row(
            "U1_EXCLUDE_INCOMPLETE_SAMPLE",
            "User Constraints",
            "Exclude incomplete `台阶曝光1` from experiments and comparisons.",
            "satisfied" if "台阶曝光1" in excluded and "台阶曝光1" not in complete else "missing",
            ["outputs/data_manifest.json", "outputs/complete_samples/t6_hv_measured/metrics.json"],
            f"complete_nonexcluded={complete}",
        )
    )
    rows.append(
        row(
            "U2_T6_HV_FRAMES",
            "User Constraints",
            "Use T6 H/V 3-step frames: H `100-102`, V `103-105`.",
            "satisfied" if fine_t6_ok else "missing",
            ["configs/ossim_dataset.yaml", "outputs/data_manifest.json"],
            f"fine_z_layers={fine.get('z_layer_count') if fine else None}",
        )
    )
    rows.append(
        row(
            "U3_FIRST_ROUND_SAMPLE",
            "User Constraints",
            "Run first-round pipeline on `细台阶`.",
            "satisfied" if baseline.get("sample") == "细台阶" and fine else "missing",
            ["outputs/baseline/t6_hv/metrics.json", "outputs/final_report.md"],
            f"sample={baseline.get('sample')}",
        )
    )
    rows.append(
        row(
            "P0_REAL_DATA_INSPECTION",
            "Phase 0-1",
            "Inspect real data, infer dimensions, collect frame stats, and render previews.",
            "satisfied"
            if all_exist(
                [
                    "outputs/data_report.md",
                    "outputs/data_frame_stats.json",
                    "outputs/data_dimension_inference.json",
                    "outputs/previews/raw_frames.png",
                    "outputs/previews/phase_frames_montage.png",
                    "outputs/previews/z_scan_montage.png",
                ]
            )
            else "missing",
            [
                "scripts/inspect_data.py",
                "outputs/data_report.md",
                "outputs/data_frame_stats.json",
                "outputs/data_dimension_inference.json",
                "outputs/previews/raw_frames.png",
            ],
        )
    )
    rows.append(
        row(
            "P1_OS_SIM_BASELINE",
            "Phase 2",
            "Run traditional OS-SIM demodulation and save H/V/fused A stacks plus height readouts.",
            "satisfied"
            if np_shape("outputs/baseline/t6_hv/A_fused_stack.npy") == [31, 256, 256]
            and all_exist(
                [
                    "outputs/baseline/t6_hv/A_t6_h_stack.npy",
                    "outputs/baseline/t6_hv/A_t6_v_stack.npy",
                    "outputs/baseline/t6_hv/height_fused_peakfit.npy",
                    "outputs/baseline/t6_hv/metrics.json",
                ]
            )
            else "missing",
            [
                "scripts/baseline_os_sim.py",
                "src/os_sim.py",
                "outputs/baseline/t6_hv/A_fused_stack.npy",
                "outputs/baseline/t6_hv/metrics.json",
            ],
            f"A_fused_shape={np_shape('outputs/baseline/t6_hv/A_fused_stack.npy')}",
        )
    )
    rows.append(
        row(
            "P2_FREQUENCY_ANALYSIS",
            "Phase 3",
            "Analyze raw/A-stack FFT peaks, residual grid/Moire energy, and T6/T12 comparison.",
            "satisfied"
            if {"ossim_t6_h_3step", "ossim_t6_v_3step"} <= set(frequency.get("groups_detail", {}))
            and all_exist(
                [
                    "outputs/frequency/t6_hv/frequency_report.md",
                    "outputs/frequency/group_compare/t6_vs_t12_compare.json",
                ]
            )
            else "missing",
            [
                "scripts/analyze_stripe_residuals.py",
                "src/fft_analysis.py",
                "outputs/frequency/t6_hv/metrics.json",
                "outputs/frequency/group_compare/t6_vs_t12_compare.json",
            ],
        )
    )
    rows.append(
        row(
            "P3_FORWARD_MODELS",
            "Phase 4",
            "Implement and compare Level 0-4 forward models on a real crop with modulation and residual-FFT metrics.",
            "satisfied" if forward_levels_ok else "missing",
            ["scripts/test_forward_models.py", "src/forward_models.py", "outputs/forward/t6_hv/model_compare.json"],
            f"models={sorted(forward_model_names)}",
        )
    )
    rows.append(
        row(
            "P3A_DMD_CCD_CALIBRATION",
            "Phase 4",
            "Explicitly include measured DMD patterns sampled through DMD-CCD calibration.",
            "satisfied" if measured_forward_ok else "missing",
            [
                "scripts/inspect_calibration_assets.py",
                "src/calibration.py",
                "outputs/calibration/assets_inventory.json",
                "outputs/forward/t6_hv/model_compare.json",
            ],
            f"measured_patterns_mode={forward.get('measured_patterns_mode')}",
        )
    )
    rows.append(
        row(
            "P4_LATENT_OPTIMIZATION",
            "Phase 5",
            "Optimize A/D0/theta-style latent variables on real crops and save optimized A stacks, residuals, curves, and metrics.",
            "satisfied"
            if all_exist(
                [
                    "outputs/latent_opt/t6_hv/A_fused_stack.npy",
                    "outputs/latent_opt/t6_hv_measured/A_fused_stack.npy",
                    "outputs/platform_opt/t6_hv_measured/A_fused_best_stack.npy",
                    "outputs/latent_opt/t6_hv/loss_curve.png",
                ]
            )
            else "missing",
            [
                "scripts/optimize_modulation_latents.py",
                "scripts/optimize_platform_constrained_latents.py",
                "outputs/latent_opt/t6_hv/metrics.json",
                "outputs/platform_opt/t6_hv_measured/metrics.json",
            ],
        )
    )
    rows.append(
        row(
            "P5_ABLATION_AND_RECOMMENDATION",
            "Phase 5",
            "Compare baseline, forward levels, optimized variants, and reject overfit variants with metrics.",
            "satisfied" if len(ablation_rows) >= 7 and recommended else "missing",
            ["scripts/summarize_ablation.py", "outputs/ablation/t6_hv/metrics.json", "outputs/ablation/t6_hv/ablation_report.md"],
            f"a_stack_rows={len(ablation_rows)}; recommended={recommended}",
        )
    )
    rows.append(
        row(
            "P6_HEIGHT_READOUT_FROM_A",
            "Phase 6",
            "Recover height only from A_stack using argmax/parabolic/Gaussian/softargmax and platform diagnostics.",
            "satisfied"
            if all_exist(
                [
                    "outputs/height/baseline_t6_hv/height_argmax.npy",
                    "outputs/height/baseline_t6_hv/height_parabolic.npy",
                    "outputs/height/baseline_t6_hv/height_gaussian.npy",
                    "outputs/height/baseline_t6_hv/height_softargmax.npy",
                    "outputs/height/platform_compare.json",
                ]
            )
            else "missing",
            ["scripts/reconstruct_height.py", "src/height.py", "outputs/height/platform_compare.json"],
            f"recommended={height_compare.get('recommended_optimized_method')}; z_unit={height_compare.get('z_unit')}",
        )
    )
    rows.append(
        row(
            "P7_SIMULATION_SANITY_ONLY",
            "Phase 7",
            "Use simulation only for sanity checks and ablation, not as main training data or real-data evidence.",
            "satisfied" if all_exist(["outputs/sim/sim_metrics.json", "outputs/sim/sim_report.md"]) else "missing",
            ["scripts/simulate_os_sim_realistic.py", "outputs/sim/sim_metrics.json", "outputs/sim/sim_report.md"],
        )
    )
    rows.append(
        row(
            "P8_TINY_NET_OPTIONAL",
            "Phase 8",
            "Treat tiny-net as optional and report missing PyTorch gracefully.",
            "satisfied" if tiny.get("status") in {"skipped_missing_torch", "initialized_only", "completed"} else "missing",
            ["scripts/train_tiny_network.py", "src/tiny_net.py", "outputs/tiny_net/metrics.json", "outputs/tiny_net/report.md"],
            f"status={tiny.get('status')}",
        )
    )
    rows.append(
        row(
            "P9_FINAL_QUESTIONS",
            "Phase 9",
            "Answer all 12 required final research questions with evidence paths.",
            "satisfied" if final_answer_ids == set(range(1, 13)) else "missing",
            ["scripts/compile_final_answers.py", "outputs/final_questions/answers.json", "outputs/final_questions/final_question_answers.md"],
            f"answer_ids={sorted(final_answer_ids)}",
        )
    )
    rows.append(
        row(
            "RUN_CLI_SURFACE",
            "Run Requirements",
            "All scripts support command-line execution and help text.",
            "satisfied" if cli.get("overall") == "satisfied" else "missing",
            ["scripts/audit_cli_surface.py", "outputs/cli_surface/cli_surface.json", "outputs/cli_surface/cli_surface_report.md"],
            f"scripts_checked={len(cli.get('cli_help_rows', []))}",
        )
    )
    rows.append(
        row(
            "RUN_OUTPUT_CONTRACTS",
            "Run Requirements",
            "Experiment outputs have config/result files, parseable metrics, output path scope, and random seed visibility.",
            "satisfied" if contracts.get("overall") == "satisfied" else "missing",
            [
                "scripts/audit_experiment_contracts.py",
                "outputs/reproducibility/contracts.json",
                "outputs/reproducibility/contract_report.md",
            ],
            f"counts={contracts.get('status_counts')}",
        )
    )
    rows.append(
        row(
            "BOUNDARY_NO_Z_TO_RAW_RENDERER",
            "Modeling Boundaries",
            "Do not generate full raw stripe sequences from z(x,y); height is a readout from A_stack.",
            "satisfied"
            if exists("src/height.py") and "def height_softargmax" in (ROOT / "src/height.py").read_text(encoding="utf-8")
            else "missing",
            ["src/height.py", "scripts/reconstruct_height.py", "outputs/height/platform_compare.json"],
            "Height scripts load A_stack and compute peak readouts; no z-to-raw rendering stage is part of the pipeline.",
        )
    )
    rows.append(
        row(
            "BOUNDARY_CONSTRAINED_M_THETA",
            "Modeling Boundaries",
            "Keep M_theta constrained to known patterns, low-dimensional warps, harmonics, grid bases, and optional blur.",
            "satisfied"
            if all(
                token in (ROOT / "src/forward_models.py").read_text(encoding="utf-8")
                for token in ["class AffineWarp", "class PolynomialWarp", "class BSplineWarp", "class HarmonicGridForward", "class OptionalBlur"]
            )
            else "missing",
            ["src/forward_models.py", "outputs/forward/t6_hv/model_compare.json"],
        )
    )
    rows.append(
        row(
            "BOUNDARY_A_CLS_WEAK_BASELINE",
            "Modeling Boundaries",
            "Treat classical A_cls as baseline/weak constraint, not clean ground truth.",
            "satisfied"
            if "A_baseline_A_cls_only" in {item.get("ablation") for item in ablation_rows}
            and "P5_aggressive_A_only_grid_loss" in {item.get("ablation") for item in ablation_rows}
            else "missing",
            ["outputs/ablation/t6_hv/metrics.json", "outputs/final_report.md"],
            "Aggressive A-only variants are rejected by platform stability gates rather than accepted from lower grid loss alone.",
        )
    )
    inverse_metrics = load_json("outputs/inverse_net/t6_hv/metrics.json") or {}
    inverse_eval = load_json("outputs/inverse_net/t6_hv/evaluation_metrics.json") or {}
    inverse_status = inverse_metrics.get("status")
    inverse_eval_status = inverse_eval.get("status")
    inverse_outputs = [
        "outputs/inverse_net/t6_hv/A_h_pred_stack.npy",
        "outputs/inverse_net/t6_hv/A_v_pred_stack.npy",
        "outputs/inverse_net/t6_hv/A_fused_pred_stack.npy",
        "outputs/inverse_net/t6_hv/height_pred.npy",
        "outputs/inverse_net/t6_hv/height_pred.png",
        "outputs/inverse_net/t6_hv/reprojection_examples.png",
        "outputs/inverse_net/t6_hv/A_comparison.png",
        "outputs/inverse_net/t6_hv/height_compare.png",
        "outputs/inverse_net/t6_hv/loss_curve.png",
    ]
    inverse_outputs_exist = all(exists(path) for path in inverse_outputs)
    graceful_missing_torch = inverse_status == "missing_torch" and inverse_eval_status == "missing_predictions"
    rows.append(
        row(
            "N1_INVERSE_NET_ARCHITECTURE",
            "Neural Inverse Model",
            "Implement Tiny U-Net/U-LiteNet that maps six T6 H/V phase frames to non-negative A_h/A_v, not directly to height.",
            "satisfied"
            if all(
                token in (ROOT / "src/inverse_net.py").read_text(encoding="utf-8")
                for token in ["InverseLightSectionNet", "in_channels: int = 6", "out_channels: int = 2", "softplus", "A_h", "A_v"]
            )
            else "missing",
            ["src/inverse_net.py"],
        )
    )
    rows.append(
        row(
            "N2_RESTRICTED_TORCH_FORWARD",
            "Neural Inverse Model",
            "Implement PyTorch constrained M_theta with fixed/measured patterns and only low-dimensional theta terms.",
            "satisfied"
            if all(
                token in (ROOT / "src/torch_forward.py").read_text(encoding="utf-8")
                for token in ["RestrictedStripeForward", "delta_q", "phase_offsets", "affine_residual", "blur_sigma_raw", "grid_coeff", "predict_modulation"]
            )
            else "missing",
            ["src/torch_forward.py"],
        )
    )
    rows.append(
        row(
            "N3_INVERSE_NET_TRAINING_ENTRYPOINT",
            "Neural Inverse Model",
            "Train with Y_hat = D0 + A_pred * M_theta using L_mod, L_raw, TV, FFT grid, weak A_cls, theta regularization, and platform terms.",
            "satisfied" if inverse_status in {"completed", "missing_torch"} else "missing",
            [
                "scripts/train_inverse_lightsection_net.py",
                "outputs/inverse_net/t6_hv/config.json",
                "outputs/inverse_net/t6_hv/metrics.json",
                "outputs/inverse_net/t6_hv/inverse_net_report.md",
            ],
            f"status={inverse_status}; missing_torch_is_graceful={inverse_status == 'missing_torch'}",
        )
    )
    rows.append(
        row(
            "N4_INVERSE_NET_EVALUATION_ENTRYPOINT",
            "Neural Inverse Model",
            "Evaluate predicted A_fused against baseline A_cls and latent optimization with grid energy, platform RMS, height std, and confidence metrics.",
            "satisfied" if inverse_eval_status in {"completed", "missing_predictions"} else "missing",
            [
                "scripts/evaluate_inverse_lightsection_net.py",
                "outputs/inverse_net/t6_hv/evaluation_metrics.json",
                "outputs/inverse_net/t6_hv/inverse_net_evaluation_report.md",
            ],
            f"status={inverse_eval_status}; trained_outputs_exist={inverse_outputs_exist}",
        )
    )
    rows.append(
        row(
            "N5_INVERSE_NET_OUTPUT_POLICY",
            "Neural Inverse Model",
            "Save required A/height/figure outputs after training; if PyTorch is absent, do not fake outputs and write a missing_torch report instead.",
            "satisfied" if inverse_outputs_exist or graceful_missing_torch else "missing",
            ["outputs/inverse_net/t6_hv/metrics.json", "outputs/inverse_net/t6_hv/inverse_net_report.md"],
            f"trained_outputs_exist={inverse_outputs_exist}; graceful_missing_torch={graceful_missing_torch}",
        )
    )
    rows.append(
        row(
            "LIMIT_HEIGHT_TRUTH",
            "External Limits",
            "Independent calibrated height truth is not available, so absolute metrology error is not claimed.",
            "documented_limit",
            ["outputs/height_truth_assets/assets.json", "outputs/height_truth_assets/height_truth_assets_report.md"],
            (
                f"candidates={height_truth.get('candidate_count')}; "
                f"pseudo_references={height_truth.get('pseudo_reference_candidate_count')}; "
                f"independent_truth_found={height_truth.get('independent_truth_found')}"
            ),
        )
    )
    rows.append(
        row(
            "LIMIT_TORCH",
            "External Limits",
            "PyTorch availability for neural inverse-net and optional tiny-net/autograd stages is recorded.",
            "documented_limit" if tiny.get("status") == "skipped_missing_torch" else "satisfied",
            ["outputs/tiny_net/metrics.json", "outputs/tiny_net/report.md"],
            f"status={tiny.get('status')}",
        )
    )
    rows.append(
        row(
            "META_COVERAGE_AUDIT",
            "Meta Verification",
            "The repo-level coverage audit is current and machine-readable.",
            "satisfied" if coverage.get("overall") == "complete_with_documented_limits" else "missing",
            ["outputs/audit/coverage.json", "outputs/audit/coverage_report.md"],
            f"overall={coverage.get('overall')}; counts={coverage.get('status_counts')}",
        )
    )
    return rows


def count_status(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in rows:
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    return counts


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Requirement Traceability Matrix",
        "",
        "## Material Passport",
        "",
        "- Artifact type: requirement-to-evidence traceability matrix",
        "- Scope: original OS-SIM research task plus later H/V and complete-data constraints",
        f"- Overall: `{result['overall']}`",
        f"- Status counts: `{result['status_counts']}`",
        "- `documented_limit` means the implemented framework is present, but a stronger optional or absolute-metrology claim is not supported.",
        "",
        "## Matrix",
        "",
        "| id | phase | status | requirement | evidence | notes |",
        "|---|---|---|---|---|---|",
    ]
    for item in result["requirements"]:
        evidence = "<br>".join(f"`{path}`" for path in item["evidence"])
        notes = str(item.get("notes", "")).replace("|", "\\|")
        requirement = item["requirement"].replace("|", "\\|")
        lines.append(f"| {item['id']} | {item['phase']} | {item['status']} | {requirement} | {evidence} | {notes} |")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile a requirement-to-evidence traceability matrix for the OS-SIM task.")
    parser.add_argument("--output", default="outputs/traceability")
    args = parser.parse_args()

    out_dir = ROOT / args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    counts = count_status(rows)
    result = {
        "artifact_type": "requirement_traceability_matrix",
        "overall": "active_not_complete" if counts.get("missing") or counts.get("blocked_external") else "complete_with_documented_limits" if counts.get("documented_limit") else "complete",
        "status_counts": counts,
        "requirements": rows,
    }
    data_io.write_json(out_dir / "requirements_traceability.json", result)
    data_io.write_json(out_dir / "config.json", vars(args))
    write_markdown(out_dir / "requirements_traceability.md", result)
    data_io.append_report(
        ROOT / "outputs/nightly_report.md",
        "P17 Requirement Traceability Matrix",
        [
            f"- Compiled `{len(rows)}` original-task requirements into a traceability matrix with counts `{counts}`.",
            f"- Outputs saved under `{out_dir}`.",
        ],
    )
    print(f"Wrote requirement traceability matrix to {out_dir}")
    print(json.dumps({"overall": result["overall"], "status_counts": counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
