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


Status = str


def exists(path: str | Path) -> bool:
    return (ROOT / path).exists()


def load_json(path: str | Path) -> Any | None:
    p = ROOT / path
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def array_shape(path: str | Path) -> list[int] | None:
    p = ROOT / path
    if not p.exists():
        return None
    return list(np.load(p, mmap_mode="r").shape)


def item(req_id: str, requirement: str, status: Status, evidence: list[str], notes: str = "") -> dict[str, Any]:
    return {
        "id": req_id,
        "requirement": requirement,
        "status": status,
        "evidence": evidence,
        "notes": notes,
    }


def status_from_files(paths: list[str]) -> Status:
    return "satisfied" if all(exists(p) for p in paths) else "missing"


def audit_files() -> list[dict[str, Any]]:
    required = [
        "README.md",
        "requirements.txt",
        "scripts/inspect_data.py",
        "scripts/inspect_calibration_assets.py",
        "scripts/inspect_z_unit_evidence.py",
        "scripts/inspect_height_truth_assets.py",
        "scripts/audit_cli_surface.py",
        "scripts/compile_requirement_traceability.py",
        "scripts/compile_final_answers.py",
        "scripts/audit_experiment_contracts.py",
        "scripts/compare_complete_samples.py",
        "scripts/baseline_os_sim.py",
        "scripts/analyze_stripe_residuals.py",
        "scripts/test_forward_models.py",
        "scripts/optimize_modulation_latents.py",
        "scripts/reconstruct_height.py",
        "scripts/simulate_os_sim_realistic.py",
        "scripts/train_tiny_network.py",
        "scripts/train_inverse_lightsection_net.py",
        "scripts/evaluate_inverse_lightsection_net.py",
        "src/data_io.py",
        "src/calibration.py",
        "src/os_sim.py",
        "src/fft_analysis.py",
        "src/forward_models.py",
        "src/torch_forward.py",
        "src/losses.py",
        "src/height.py",
        "src/visualization.py",
        "src/tiny_net.py",
        "src/inverse_net.py",
        "outputs/nightly_report.md",
    ]
    missing = [p for p in required if not exists(p)]
    return [
        item(
            "FILES",
            "Required project files and scripts exist.",
            "satisfied" if not missing else "missing",
            required,
            "" if not missing else f"Missing: {missing}",
        )
    ]


def audit_p0() -> list[dict[str, Any]]:
    manifest = load_json("outputs/data_manifest.json")
    stats = load_json("outputs/data_frame_stats.json")
    inventory = load_json("outputs/data_format_inventory.json")
    dims = load_json("outputs/data_dimension_inference.json")
    rows: list[dict[str, Any]] = []
    p0_files = [
        "outputs/data_manifest.json",
        "outputs/data_report.md",
        "outputs/data_frame_stats.json",
        "outputs/data_format_inventory.json",
        "outputs/data_dimension_inference.json",
        "outputs/previews/raw_frames.png",
        "outputs/previews/phase_frames_montage.png",
        "outputs/previews/z_scan_montage.png",
        "outputs/previews/mean_variance.png",
    ]
    rows.append(
        item(
            "P0_ARTIFACTS",
            "Data inspection artifacts, previews, and frame statistics exist.",
            status_from_files(p0_files),
            p0_files,
        )
    )
    if not manifest:
        rows.append(item("P0_MANIFEST", "Manifest is parseable.", "missing", ["outputs/data_manifest.json"]))
        return rows
    excluded = set(manifest.get("excluded_samples", []))
    samples = manifest.get("samples", [])
    complete = [s["name"] for s in samples if not s.get("excluded") and s.get("incomplete_layer_count") == 0]
    fine = next((s for s in samples if s.get("name") == "细台阶"), None)
    bad_t6_layers: list[str] = []
    if fine:
        for layer in fine.get("layers", []):
            groups = layer.get("groups", {})
            h = set(groups.get("ossim_t6_h_3step", {}).get("frames", []))
            v = set(groups.get("ossim_t6_v_3step", {}).get("frames", []))
            if not set(range(100, 106)).issubset(h | v):
                bad_t6_layers.append(layer.get("folder", "unknown"))
    rows.append(
        item(
            "P0_EXCLUSION",
            "`台阶曝光1` is excluded and complete experiment samples are identifiable.",
            "satisfied" if "台阶曝光1" in excluded and "台阶曝光1" not in complete else "missing",
            ["outputs/data_manifest.json"],
            f"complete_nonexcluded={complete}",
        )
    )
    rows.append(
        item(
            "P0_T6_FRAMES",
            "`细台阶` has complete T6 H/V frames `100-105` for every z layer.",
            "satisfied" if fine and not bad_t6_layers else "missing",
            ["outputs/data_manifest.json"],
            f"bad_layers={bad_t6_layers[:5]}, z_layers={fine.get('z_layer_count') if fine else None}",
        )
    )
    rows.append(
        item(
            "P0_FORMAT_SCOPE",
            "Inspector inventories supported tif/tiff/png/bmp/npy/npz/mat/h5/hdf5 formats and records example reader metadata.",
            "satisfied"
            if inventory and {".tif", ".tiff", ".png", ".bmp", ".npy", ".npz", ".mat", ".h5", ".hdf5"} <= set(inventory.get("supported_extensions", []))
            else "missing",
            ["scripts/inspect_data.py", "outputs/data_format_inventory.json"],
            f"counts_by_extension={inventory.get('counts_by_extension') if inventory else None}; current dataset contains BMP files.",
        )
    )
    rows.append(
        item(
            "P0_DIMENSIONS",
            "Inspector writes candidate K/M/H/W interpretations for the selected real OS-SIM stack.",
            "satisfied" if dims and len(dims.get("candidates", [])) >= 2 else "missing",
            ["outputs/data_dimension_inference.json"],
            f"candidate_count={len(dims.get('candidates', [])) if dims else 0}",
        )
    )
    rows.append(
        item(
            "P0_STATS_SCOPE",
            "Per-frame statistics are generated for the inspected first-round T6 H/V sample/groups.",
            "satisfied" if isinstance(stats, list) and len(stats) > 0 else "missing",
            ["outputs/data_frame_stats.json"],
            f"rows={len(stats) if isinstance(stats, list) else 0}; this is not a full all-sample/all-frame statistics pass.",
        )
    )
    return rows


def audit_p1() -> list[dict[str, Any]]:
    metrics = load_json("outputs/baseline/t6_hv/metrics.json")
    shape = array_shape("outputs/baseline/t6_hv/A_fused_stack.npy")
    files = [
        "outputs/baseline/t6_hv/config.json",
        "outputs/baseline/t6_hv/metrics.json",
        "outputs/baseline/t6_hv/D0_t6_h_stack.npy",
        "outputs/baseline/t6_hv/Dc_t6_h_stack.npy",
        "outputs/baseline/t6_hv/Ds_t6_h_stack.npy",
        "outputs/baseline/t6_hv/A_t6_h_stack.npy",
        "outputs/baseline/t6_hv/D0_t6_v_stack.npy",
        "outputs/baseline/t6_hv/Dc_t6_v_stack.npy",
        "outputs/baseline/t6_hv/Ds_t6_v_stack.npy",
        "outputs/baseline/t6_hv/A_t6_v_stack.npy",
        "outputs/baseline/t6_hv/A_fused_stack.npy",
        "outputs/baseline/t6_hv/height_fused_argmax.npy",
        "outputs/baseline/t6_hv/height_fused_peakfit.npy",
        "outputs/baseline/t6_hv/height_fused_gaussian.npy",
        "outputs/baseline/t6_hv/height_fused_softargmax.npy",
        "outputs/baseline/t6_hv/confidence_fused_sharpness.npy",
        "outputs/baseline/t6_hv/section_montage.png",
        "outputs/baseline/t6_hv/height_map.png",
        "outputs/baseline/t6_hv/axis_profiles.png",
    ]
    return [
        item(
            "P1_BASELINE",
            "Traditional OS-SIM baseline generates D0/Dc/Ds/A, H/V/fused stacks, height maps, confidence, figures, config, and metrics.",
            "satisfied" if status_from_files(files) == "satisfied" and shape == [31, 256, 256] and metrics else "missing",
            files,
            f"A_fused_shape={shape}; fused_height_std={metrics.get('fused', {}).get('height_std') if metrics else None}",
        )
    ]


def audit_p2() -> list[dict[str, Any]]:
    files = [
        "outputs/frequency/t6_hv/config.json",
        "outputs/frequency/t6_hv/metrics.json",
        "outputs/frequency/t6_hv/frequency_report.md",
        "outputs/frequency/t6_hv/raw_fft_examples.png",
        "outputs/frequency/t6_hv/section_fft_examples.png",
        "outputs/frequency/t6_hv/grid_energy_over_z.png",
        "outputs/frequency/t6_hv/stripe_peak_over_z.png",
        "outputs/frequency/group_compare/t6_vs_t12_compare.json",
        "outputs/frequency/group_compare/frequency_compare_report.md",
    ]
    metrics = load_json("outputs/frequency/t6_hv/metrics.json")
    compare = load_json("outputs/frequency/group_compare/t6_vs_t12_compare.json")
    has_hv = bool(metrics and {"ossim_t6_h_3step", "ossim_t6_v_3step"} <= set(metrics.get("groups_detail", {})))
    return [
        item(
            "P2_FREQUENCY",
            "Raw/A-stack FFT residual analysis and T6/T12 comparison are present.",
            "satisfied" if status_from_files(files) == "satisfied" and has_hv and compare else "missing",
            files,
            f"best_pair={compare.get('ranking', {}).get('best_by_grid_sharpness_score') if compare else None}",
        )
    ]


def audit_p3() -> list[dict[str, Any]]:
    forward = load_json("outputs/forward/t6_hv/model_compare.json")
    files = [
        "outputs/forward/t6_hv/config.json",
        "outputs/forward/t6_hv/model_compare.json",
        "outputs/forward/t6_hv/forward_report.md",
        "outputs/forward/t6_hv/reprojection_examples.png",
        "outputs/forward/t6_hv/residual_examples.png",
    ]
    required_models = {
        "level0_ideal_cos_group_period",
        "level1_modulation_cos_fft_q",
        "level2_binary_dmd_identity_affine_warp_fft_q",
        "level3_binary_dmd_identity_polynomial_warp_fft_q",
        "level3_binary_dmd_zero_bspline_warp_fft_q",
        "level4_harmonic_grid_no_blur_fft_q",
        "level4_harmonic_grid_blur_fft_q",
    }
    required_metrics = {
        "raw_rmse",
        "mod_rmse",
        "raw_charbonnier",
        "mod_charbonnier",
        "residual_grid_energy_mean",
        "residual_stripe_energy_mean",
        "residual_top_peak_energy_mean",
    }
    ok = False
    missing_details: list[str] = []
    if forward:
        ok = True
        for group, detail in forward.get("models", {}).items():
            losses = detail.get("losses", {})
            missing_models = sorted(required_models - set(losses))
            if missing_models:
                ok = False
                missing_details.append(f"{group} missing models {missing_models}")
            for model_name in required_models & set(losses):
                missing_metrics = sorted(required_metrics - set(losses[model_name]))
                if missing_metrics:
                    ok = False
                    missing_details.append(f"{group}/{model_name} missing metrics {missing_metrics}")
    return [
        item(
            "P3_FORWARD",
            "Forward model levels 0-4 run on real crop with RMSE, Charbonnier, residual FFT/grid/stripe metrics.",
            "satisfied" if status_from_files(files) == "satisfied" and ok else "missing",
            files,
            "; ".join(missing_details) if missing_details else "Level 2 includes generated pattern controls; measured calibrated DMD patterns are audited separately in P3B.",
        )
    ]


def audit_calibration_assets() -> list[dict[str, Any]]:
    assets = load_json("outputs/calibration/assets_inventory.json")
    forward = load_json("outputs/forward/t6_hv/model_compare.json")
    files = [
        "scripts/inspect_calibration_assets.py",
        "src/calibration.py",
        "outputs/calibration/assets_inventory.json",
        "outputs/calibration/calibration_report.md",
        "outputs/calibration/mapping_preview.png",
        "outputs/calibration/t6_dmd_patterns.png",
        "outputs/calibration/t6_mapped_camera_patterns.png",
        "outputs/calibration/ossim_t6_h_3step_mapped_patterns.npy",
        "outputs/calibration/ossim_t6_v_3step_mapped_patterns.npy",
    ]
    matched = bool(assets and assets.get("projection_sequence", {}).get("matched_t6_hv_frames_100_105"))
    preferred = assets.get("preferred_mapping", {}) if assets else {}
    groups = assets.get("measured_groups", {}) if assets else {}
    mapped_ok = {"ossim_t6_h_3step", "ossim_t6_v_3step"} <= set(groups)
    valid_ok = all(info.get("mapped_info", {}).get("valid_fraction", 0.0) >= 0.99 for info in groups.values()) if groups else False
    measured_forward_ok = False
    if forward:
        measured_forward_ok = all(
            "level2_measured_dmd_calibrated_map" in forward.get("models", {}).get(group, {}).get("losses", {})
            for group in ("ossim_t6_h_3step", "ossim_t6_v_3step")
        )
    return [
        item(
            "P3A_CALIBRATION_ASSETS",
            "3-6 calibration maps and measured projection bitmaps are inventoried, matched to frames 100-105, and sampled into the camera crop.",
            "satisfied" if status_from_files(files) == "satisfied" and matched and mapped_ok and valid_ok else "missing",
            files,
            f"mapping={preferred.get('path')}; coordinate_frame={preferred.get('coordinate_frame')}; matched_t6={matched}; mapped_groups={sorted(groups)}",
        ),
        item(
            "P3B_MEASURED_DMD_FORWARD",
            "Forward comparison includes measured DMD bitmap patterns sampled through the DMD-CCD calibration map.",
            "satisfied" if measured_forward_ok else "missing",
            ["outputs/forward/t6_hv/model_compare.json", "outputs/forward/t6_hv/forward_report.md"],
            f"measured_patterns_mode={forward.get('measured_patterns_mode') if forward else None}",
        ),
    ]


def audit_complete_samples() -> list[dict[str, Any]]:
    metrics = load_json("outputs/complete_samples/t6_hv_measured/metrics.json")
    files = [
        "scripts/compare_complete_samples.py",
        "outputs/complete_samples/t6_hv_measured/config.json",
        "outputs/complete_samples/t6_hv_measured/metrics.json",
        "outputs/complete_samples/t6_hv_measured/complete_samples_report.md",
        "outputs/complete_samples/t6_hv_measured/sample_metric_summary.png",
        "outputs/complete_samples/t6_hv_measured/sample_montage.png",
        "outputs/complete_samples/t6_hv_measured/台阶曝光2/metrics.json",
        "outputs/complete_samples/t6_hv_measured/回型曝光20000/metrics.json",
        "outputs/complete_samples/t6_hv_measured/小孔/metrics.json",
        "outputs/complete_samples/t6_hv_measured/细台阶/metrics.json",
    ]
    expected = {"台阶曝光2", "回型曝光20000", "小孔", "细台阶"}
    samples = set(metrics.get("samples", [])) if metrics else set()
    excluded = set(metrics.get("excluded_samples", [])) if metrics else set()
    rows = metrics.get("rows", []) if metrics else []
    row_samples = {row.get("sample") for row in rows}
    no_bad_sample = "台阶曝光1" not in samples and "台阶曝光1" in excluded
    metric_keys_ok = all(
        {
            "mod_rmse_measured_calibrated",
            "height_std",
            "confidence_sharpness_mean",
            "a_fused_grid_energy",
        }
        <= set(row)
        for row in rows
    )
    return [
        item(
            "P11_COMPLETE_SAMPLES",
            "All complete non-excluded samples are compared under T6 H/V measured-calibrated patterns, while the incomplete sample remains excluded.",
            "satisfied"
            if status_from_files(files) == "satisfied" and expected <= samples and expected <= row_samples and no_bad_sample and metric_keys_ok
            else "missing",
            files,
            f"samples={sorted(samples)}; excluded={sorted(excluded)}; row_count={len(rows)}",
        )
    ]


def audit_p4_p5() -> list[dict[str, Any]]:
    latent = load_json("outputs/latent_opt/t6_hv/metrics.json")
    theta = load_json("outputs/theta_opt/t6_hv/metrics.json")
    affine = load_json("outputs/affine_warp_opt/t6_hv/metrics.json")
    platform = load_json("outputs/platform_opt/t6_hv/metrics.json")
    latent_measured = load_json("outputs/latent_opt/t6_hv_measured/metrics.json")
    platform_measured = load_json("outputs/platform_opt/t6_hv_measured/metrics.json")
    ablation = load_json("outputs/ablation/t6_hv/metrics.json")
    files = [
        "outputs/latent_opt/t6_hv/config.json",
        "outputs/latent_opt/t6_hv/metrics.json",
        "outputs/latent_opt/t6_hv/A_fused_stack.npy",
        "outputs/latent_opt/t6_hv/loss_curve.png",
        "outputs/latent_opt/t6_hv/A_baseline_vs_optimized.png",
        "outputs/latent_opt/t6_hv/reprojection_examples.png",
        "outputs/latent_opt/t6_hv/residual_examples.png",
        "outputs/latent_opt/t6_hv/residual_fft.png",
        "outputs/latent_opt/t6_hv/height_baseline_vs_optimized.png",
        "outputs/affine_warp_opt/t6_hv/config.json",
        "outputs/affine_warp_opt/t6_hv/metrics.json",
        "outputs/affine_warp_opt/t6_hv/A_fused_stack.npy",
        "outputs/affine_warp_opt/t6_hv/affine_warp_loss_curve.png",
        "outputs/affine_warp_opt/t6_hv/affine_A_height_compare.png",
        "outputs/platform_opt/t6_hv/config.json",
        "outputs/platform_opt/t6_hv/metrics.json",
        "outputs/platform_opt/t6_hv/A_fused_best_stack.npy",
        "outputs/latent_opt/t6_hv_measured/config.json",
        "outputs/latent_opt/t6_hv_measured/metrics.json",
        "outputs/latent_opt/t6_hv_measured/A_fused_stack.npy",
        "outputs/platform_opt/t6_hv_measured/config.json",
        "outputs/platform_opt/t6_hv_measured/metrics.json",
        "outputs/platform_opt/t6_hv_measured/A_fused_best_stack.npy",
        "outputs/ablation/t6_hv/config.json",
        "outputs/ablation/t6_hv/metrics.json",
        "outputs/ablation/t6_hv/ablation_report.md",
    ]
    rows = []
    rows.append(
        item(
            "P4_P5_LATENT",
            "Crop latent optimization outputs generated-pattern and measured-calibrated optimized A stacks, height comparison, residual figures, config, metrics, and platform-constrained recommendations.",
            "satisfied" if status_from_files(files) == "satisfied" and latent and platform and latent_measured and platform_measured else "missing",
            files,
            f"aggressive_loss={latent.get('final_modulation_loss') if latent else None}; measured_loss={latent_measured.get('final_modulation_loss') if latent_measured else None}; platform_best_iter={platform.get('best_iteration') if platform else None}; measured_platform_best_iter={platform_measured.get('best_iteration') if platform_measured else None}",
        )
    )
    if ablation:
        forward_rows = len(ablation.get("forward_ablation", []))
        a_rows = len(ablation.get("a_stack_ablation", []))
        rows.append(
            item(
                "P5_ABLATION",
                "Stage-5 ablation table explicitly compares forward levels B-F and A-stack/height variants.",
                "satisfied" if forward_rows >= 14 and a_rows >= 5 else "partial",
                ["outputs/ablation/t6_hv/metrics.json", "outputs/ablation/t6_hv/ablation_report.md"],
                f"forward_rows={forward_rows}; a_stack_rows={a_rows}; recommended={ablation.get('recommended_a_stack', {}).get('source')}",
            )
        )
    else:
        rows.append(item("P5_ABLATION", "Stage-5 ablation table exists.", "missing", ["outputs/ablation/t6_hv/metrics.json"]))
    rows.append(
        item(
            "P5_THETA_SCOPE",
            "Low-dimensional theta optimization covers q/phase/harmonics and generated-DMD affine warp fitting.",
            "satisfied" if theta and affine else "missing",
            [
                "scripts/optimize_theta_latents.py",
                "scripts/optimize_affine_warp_latents.py",
                "outputs/theta_opt/t6_hv/metrics.json",
                "outputs/affine_warp_opt/t6_hv/metrics.json",
            ],
            "Generated-DMD affine warp fitting is implemented; measured DMD/calibration now cover forward comparison, while latent optimization still uses generated-pattern parameterizations.",
        )
    )
    return rows


def audit_p6() -> list[dict[str, Any]]:
    files = [
        "outputs/height/baseline_t6_hv/config.json",
        "outputs/height/baseline_t6_hv/metrics.json",
        "outputs/height/baseline_t6_hv/height_maps.png",
        "outputs/height/baseline_t6_hv/confidence_map.png",
        "outputs/height/baseline_t6_hv/profiles.png",
        "outputs/height/baseline_t6_hv/platform_metrics.json",
        "outputs/height/platform_opt_t6_hv/config.json",
        "outputs/height/platform_opt_t6_hv/metrics.json",
        "outputs/height/platform_opt_t6_hv/height_maps.png",
        "outputs/height/platform_opt_t6_hv/platform_metrics.json",
        "outputs/height/platform_opt_t6_hv_measured/config.json",
        "outputs/height/platform_opt_t6_hv_measured/metrics.json",
        "outputs/height/platform_opt_t6_hv_measured/height_maps.png",
        "outputs/height/platform_opt_t6_hv_measured/platform_metrics.json",
        "outputs/height/platform_compare.json",
    ]
    compare = load_json("outputs/height/platform_compare.json")
    z_unit = compare.get("z_unit") if compare else None
    return [
        item(
            "P6_HEIGHT",
            "Height reconstruction outputs argmax/parabolic/Gaussian/softargmax, confidence, profiles, platform metrics, and baseline-vs-generated/measured optimized comparison.",
            "satisfied" if status_from_files(files) == "satisfied" and compare else "missing",
            files,
            f"recommended={compare.get('recommended_optimized_method') if compare else None}; z_unit={z_unit}",
        )
    ]


def audit_z_unit_evidence() -> list[dict[str, Any]]:
    files = [
        "scripts/inspect_z_unit_evidence.py",
        "outputs/z_unit/evidence.json",
        "outputs/z_unit/z_unit_report.md",
    ]
    evidence = load_json("outputs/z_unit/evidence.json")
    ok = bool(
        status_from_files(files) == "satisfied"
        and evidence
        and evidence.get("inferred_z_unit") == "micrometer"
        and evidence.get("not_an_external_height_truth") is True
    )
    return [
        item(
            "P12_Z_UNIT_EVIDENCE",
            "Local analysis evidence supports reporting z scan coordinates in micrometers while explicitly not serving as independent height truth.",
            "satisfied" if ok else "missing",
            files,
            f"inferred_z_unit={evidence.get('inferred_z_unit') if evidence else None}; confidence={evidence.get('confidence') if evidence else None}; independent_truth={not evidence.get('not_an_external_height_truth') if evidence else None}",
        )
    ]


def audit_height_truth_asset_search() -> list[dict[str, Any]]:
    files = [
        "scripts/inspect_height_truth_assets.py",
        "outputs/height_truth_assets/config.json",
        "outputs/height_truth_assets/assets.json",
        "outputs/height_truth_assets/height_truth_assets_report.md",
    ]
    assets = load_json("outputs/height_truth_assets/assets.json")
    ok = bool(
        status_from_files(files) == "satisfied"
        and assets
        and assets.get("search_status") == "completed"
        and assets.get("candidate_count", 0) >= 1
        and assets.get("independent_truth_found") is False
        and assets.get("pseudo_reference_candidate_count", 0) >= 1
    )
    return [
        item(
            "P15_HEIGHT_TRUTH_ASSET_SEARCH",
            "Local height-like morphology and point-cloud assets are inventoried and explicitly rejected as independent calibrated height truth.",
            "satisfied" if ok else "missing",
            files,
            (
                f"candidates={assets.get('candidate_count') if assets else None}; "
                f"inspected={assets.get('inspected_count') if assets else None}; "
                f"pseudo_references={assets.get('pseudo_reference_candidate_count') if assets else None}; "
                f"independent_truth_found={assets.get('independent_truth_found') if assets else None}"
            ),
        )
    ]


def audit_cli_surface() -> list[dict[str, Any]]:
    files = [
        "scripts/audit_cli_surface.py",
        "outputs/cli_surface/config.json",
        "outputs/cli_surface/cli_surface.json",
        "outputs/cli_surface/cli_surface_report.md",
    ]
    cli = load_json("outputs/cli_surface/cli_surface.json")
    rows = cli.get("status_rows", []) if cli else []
    status_by_id = {row.get("id"): row.get("status") for row in rows}
    ok = bool(
        status_from_files(files) == "satisfied"
        and cli
        and cli.get("overall") == "satisfied"
        and status_by_id.get("cli_help") == "satisfied"
        and status_by_id.get("output_defaults") == "satisfied"
        and status_by_id.get("contract_output_scope") == "satisfied"
        and status_by_id.get("raw_root_policy") == "satisfied"
        and status_by_id.get("dependency_graceful_degradation") == "satisfied"
    )
    return [
        item(
            "P16_CLI_AND_SAFETY_SURFACE",
            "All scripts expose command-line help, default outputs stay under outputs, raw data remains input-only, and missing PyTorch is reported gracefully.",
            "satisfied" if ok else "missing",
            files,
            (
                f"overall={cli.get('overall') if cli else None}; "
                f"scripts_checked={len(cli.get('cli_help_rows', [])) if cli else 0}; "
                f"counts={cli.get('status_counts') if cli else None}; "
                f"tiny_net_status={cli.get('dependency_graceful_degradation', {}).get('tiny_net_status') if cli else None}"
            ),
        )
    ]


def audit_requirement_traceability() -> list[dict[str, Any]]:
    files = [
        "scripts/compile_requirement_traceability.py",
        "outputs/traceability/config.json",
        "outputs/traceability/requirements_traceability.json",
        "outputs/traceability/requirements_traceability.md",
    ]
    trace = load_json("outputs/traceability/requirements_traceability.json")
    counts = trace.get("status_counts", {}) if trace else {}
    rows = trace.get("requirements", []) if trace else []
    ids = {row.get("id") for row in rows}
    expected = {
        "U1_EXCLUDE_INCOMPLETE_SAMPLE",
        "U2_T6_HV_FRAMES",
        "P1_OS_SIM_BASELINE",
        "P3_FORWARD_MODELS",
        "P6_HEIGHT_READOUT_FROM_A",
        "RUN_CLI_SURFACE",
        "RUN_OUTPUT_CONTRACTS",
        "BOUNDARY_NO_Z_TO_RAW_RENDERER",
        "BOUNDARY_CONSTRAINED_M_THETA",
        "LIMIT_HEIGHT_TRUTH",
        "LIMIT_TORCH",
    }
    ok = bool(
        status_from_files(files) == "satisfied"
        and trace
        and trace.get("overall") == "complete_with_documented_limits"
        and counts.get("missing", 0) == 0
        and counts.get("blocked_external", 0) == 0
        and counts.get("documented_limit", 0) >= 1
        and expected <= ids
    )
    return [
        item(
            "P17_REQUIREMENT_TRACEABILITY",
            "Original task requirements, modeling boundaries, run requirements, and external limits are mapped to concrete evidence paths.",
            "satisfied" if ok else "missing",
            files,
            f"overall={trace.get('overall') if trace else None}; counts={counts}; rows={len(rows)}",
        )
    ]


def audit_p7_p8_p9() -> list[dict[str, Any]]:
    sim_files = [
        "outputs/sim/config.json",
        "outputs/sim/sim_metrics.json",
        "outputs/sim/sim_report.md",
        "outputs/sim/ideal_results.png",
        "outputs/sim/realistic_results.png",
        "outputs/sim/sim_height_error.png",
    ]
    tiny = load_json("outputs/tiny_net/metrics.json")
    rows = [
        item(
            "P7_SIM",
            "Simulation sanity check artifacts exist and distinguish ideal vs realistic modes.",
            "satisfied" if status_from_files(sim_files) == "satisfied" else "missing",
            sim_files,
        ),
        item(
            "P8_TINY_NET",
            "Tiny-net stage is optional and either initializes with PyTorch or gracefully reports missing PyTorch.",
            "satisfied" if tiny and tiny.get("status") in {"skipped_missing_torch", "initialized_only", "completed"} else "partial",
            ["scripts/train_tiny_network.py", "src/tiny_net.py", "outputs/tiny_net/metrics.json", "outputs/tiny_net/report.md"],
            f"status={tiny.get('status') if tiny else None}",
        ),
        item(
            "P9_REPORTS",
            "Nightly, final report, final question appendix, README, and ablation report are present.",
            status_from_files(
                [
                    "outputs/nightly_report.md",
                    "outputs/final_report.md",
                    "outputs/final_questions/final_question_answers.md",
                    "README.md",
                    "outputs/ablation/t6_hv/ablation_report.md",
                ]
            ),
            [
                "outputs/nightly_report.md",
                "outputs/final_report.md",
                "outputs/final_questions/final_question_answers.md",
                "README.md",
                "outputs/ablation/t6_hv/ablation_report.md",
            ],
        ),
    ]
    return rows


def audit_final_question_answers() -> list[dict[str, Any]]:
    files = [
        "scripts/compile_final_answers.py",
        "outputs/final_questions/config.json",
        "outputs/final_questions/answers.json",
        "outputs/final_questions/final_question_answers.md",
    ]
    answers = load_json("outputs/final_questions/answers.json")
    rows = answers.get("answers", []) if answers else []
    ids = {row.get("id") for row in rows}
    statuses = {str(row.get("status", "")) for row in rows}
    ok = (
        status_from_files(files) == "satisfied"
        and ids == set(range(1, 13))
        and all(status.startswith("answered") for status in statuses)
    )
    return [
        item(
            "P13_FINAL_QUESTIONS",
            "The final report appendix explicitly answers all 12 required research questions with evidence paths.",
            "satisfied" if ok else "missing",
            files,
            f"answer_ids={sorted(ids)}; statuses={sorted(statuses)}",
        )
    ]


def audit_experiment_contracts() -> list[dict[str, Any]]:
    files = [
        "scripts/audit_experiment_contracts.py",
        "outputs/reproducibility/config.json",
        "outputs/reproducibility/contracts.json",
        "outputs/reproducibility/contract_report.md",
    ]
    contracts = load_json("outputs/reproducibility/contracts.json")
    ok = bool(
        status_from_files(files) == "satisfied"
        and contracts
        and contracts.get("overall") == "satisfied"
        and contracts.get("status_counts", {}).get("missing", 0) == 0
        and contracts.get("exclusion_policy", {}).get("status") == "satisfied"
        and contracts.get("seeded_random_processes", {}).get("status") == "satisfied"
    )
    return [
        item(
            "P14_EXPERIMENT_CONTRACTS",
            "Experiment output contracts are audited for config/result files, parseable JSON metrics, output path scope, exclusion policy, and random seed visibility.",
            "satisfied" if ok else "missing",
            files,
            f"overall={contracts.get('overall') if contracts else None}; counts={contracts.get('status_counts') if contracts else None}",
        )
    ]


def audit_inverse_net() -> list[dict[str, Any]]:
    files = [
        "src/inverse_net.py",
        "src/torch_forward.py",
        "scripts/train_inverse_lightsection_net.py",
        "scripts/evaluate_inverse_lightsection_net.py",
        "outputs/inverse_net/t6_hv/config.json",
        "outputs/inverse_net/t6_hv/metrics.json",
        "outputs/inverse_net/t6_hv/inverse_net_report.md",
        "outputs/inverse_net/t6_hv/evaluation_metrics.json",
        "outputs/inverse_net/t6_hv/inverse_net_evaluation_report.md",
    ]
    metrics = load_json("outputs/inverse_net/t6_hv/metrics.json")
    evaluation = load_json("outputs/inverse_net/t6_hv/evaluation_metrics.json")
    source_ok = True
    for rel_path, needles in {
        "src/inverse_net.py": ["InverseLightSectionNet", "softplus", "A_h", "A_v"],
        "src/torch_forward.py": ["RestrictedStripeForward", "delta_q", "phase_offsets", "affine_residual", "blur_sigma_raw", "grid_coeff", "predict_modulation"],
        "scripts/train_inverse_lightsection_net.py": ["L_mod", "lambda_grid", "lambda_sec", "lambda_theta", "lambda_platform"],
        "scripts/evaluate_inverse_lightsection_net.py": ["grid_energy", "height_pred", "platform"],
    }.items():
        text = (ROOT / rel_path).read_text(encoding="utf-8") if exists(rel_path) else ""
        source_ok = source_ok and all(needle in text for needle in needles)
    status = metrics.get("status") if metrics else None
    eval_status = evaluation.get("status") if evaluation else None
    completed_outputs = all(
        exists(path)
        for path in [
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
    )
    graceful_missing = status == "missing_torch" and eval_status == "missing_predictions"
    ok = bool(status_from_files(files) == "satisfied" and source_ok and (completed_outputs or graceful_missing))
    return [
        item(
            "P18_INVERSE_NET",
            "Neural inverse process is implemented: six raw phase frames enter a tiny U-Net, A_h/A_v are predicted, constrained theta forward model reprojects Y_hat, and missing PyTorch is reported without fake outputs.",
            "satisfied" if ok else "missing",
            files,
            f"training_status={status}; evaluation_status={eval_status}; completed_outputs={completed_outputs}; graceful_missing={graceful_missing}",
        )
    ]


def audit_external_limits() -> list[dict[str, Any]]:
    assets = load_json("outputs/calibration/assets_inventory.json")
    forward = load_json("outputs/forward/t6_hv/model_compare.json")
    tiny = load_json("outputs/tiny_net/metrics.json")
    inverse = load_json("outputs/inverse_net/t6_hv/metrics.json")
    matched = bool(assets and assets.get("projection_sequence", {}).get("matched_t6_hv_frames_100_105"))
    mapping = assets.get("preferred_mapping", {}) if assets else {}
    measured_forward_ok = bool(
        forward
        and all(
            "level2_measured_dmd_calibrated_map" in forward.get("models", {}).get(group, {}).get("losses", {})
            for group in ("ossim_t6_h_3step", "ossim_t6_v_3step")
        )
    )
    return [
        item(
            "LIMIT_DMD_BITMAPS",
            "Measured DMD binary bitmap files are required for calibrated Level-2 claims.",
            "satisfied" if matched else "blocked_external",
            ["outputs/calibration/assets_inventory.json", "outputs/forward/t6_hv/model_compare.json", "outputs/final_report.md"],
            "Projection sequence `saomiao3_6.txt` maps raw frames 100-105 to measured T6 H/V 3-step DMD bitmaps."
            if matched
            else "Current implementation falls back to generated binary gratings from period/direction/phase count.",
        ),
        item(
            "LIMIT_CALIBRATION",
            "Real DMD-CCD affine/warp calibration is required before claiming mapped DMD model superiority.",
            "satisfied" if mapping and measured_forward_ok else "blocked_external",
            ["src/calibration.py", "outputs/calibration/assets_inventory.json", "outputs/forward/t6_hv/model_compare.json"],
            f"Using `{mapping.get('path')}` as a dense DMD-CCD map with coordinate_frame={mapping.get('coordinate_frame')}."
            if mapping and measured_forward_ok
            else "Affine/polynomial/B-spline warp interfaces exist; measured calibration has not been loaded into the forward comparison.",
        ),
        item(
            "LIMIT_HEIGHT_TRUTH",
            "Independent calibrated step-height or height truth is required for absolute metrology error.",
            "documented_limit",
            [
                "outputs/z_unit/evidence.json",
                "outputs/height_truth_assets/assets.json",
                "outputs/height_truth_assets/height_truth_assets_report.md",
                "outputs/height/platform_compare.json",
                "outputs/final_report.md",
            ],
            "Local search found derived morphology/point-cloud pseudo-reference assets, but no independent calibrated step-height truth is available.",
        ),
        item(
            "LIMIT_TORCH",
            "PyTorch is required only for optional tiny-net/autograd training beyond the NumPy/SciPy first pass.",
            "satisfied" if tiny and tiny.get("status") != "skipped_missing_torch" and inverse and inverse.get("status") == "completed" else "documented_limit",
            ["outputs/tiny_net/metrics.json", "outputs/inverse_net/t6_hv/metrics.json"],
            f"tiny_status={tiny.get('status') if tiny else None}; inverse_status={inverse.get('status') if inverse else None}",
        ),
    ]


def summarize_status(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    return counts


def write_report(path: Path, rows: list[dict[str, Any]], counts: dict[str, int]) -> None:
    lines = [
        "# OS-SIM Reproducibility Coverage Audit",
        "",
        "## Summary",
        "",
        f"- Status counts: `{counts}`",
        "- `satisfied` means current evidence directly supports the requirement.",
        "- `partial` means implementation exists but does not cover the full original scope.",
        "- `blocked_external` means the repo can still make progress, but a specific external input is needed for a required claim.",
        "- `documented_limit` means the framework is complete for the requested first pass, but a stronger optional or absolute-metrology claim is not supported.",
        "",
        "## Requirement Table",
        "",
        "| id | status | requirement | notes |",
        "|---|---|---|---|",
    ]
    for row in rows:
        notes = str(row.get("notes", "")).replace("|", "\\|").replace("\n", " ")
        req = row["requirement"].replace("|", "\\|")
        lines.append(f"| {row['id']} | {row['status']} | {req} | {notes} |")
    lines += [
        "",
        "## Evidence Paths",
        "",
    ]
    for row in rows:
        lines.append(f"### {row['id']}")
        lines.append("")
        for evidence in row.get("evidence", []):
            lines.append(f"- `{evidence}`")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit OS-SIM research artifacts against the explicit staged requirements.")
    parser.add_argument("--output", default="outputs/audit")
    args = parser.parse_args()

    out_dir = ROOT / args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    rows += audit_files()
    rows += audit_p0()
    rows += audit_p1()
    rows += audit_p2()
    rows += audit_p3()
    rows += audit_calibration_assets()
    rows += audit_complete_samples()
    rows += audit_p4_p5()
    rows += audit_p6()
    rows += audit_z_unit_evidence()
    rows += audit_p7_p8_p9()
    rows += audit_final_question_answers()
    rows += audit_experiment_contracts()
    rows += audit_height_truth_asset_search()
    rows += audit_cli_surface()
    rows += audit_requirement_traceability()
    rows += audit_inverse_net()
    rows += audit_external_limits()
    counts = summarize_status(rows)
    if counts.get("partial") or counts.get("missing") or counts.get("blocked_external"):
        overall = "active_not_complete"
    elif counts.get("documented_limit"):
        overall = "complete_with_documented_limits"
    else:
        overall = "complete"
    result = {
        "status_counts": counts,
        "overall": overall,
        "requirements": rows,
    }
    data_io.write_json(out_dir / "coverage.json", result)
    data_io.write_json(out_dir / "config.json", vars(args))
    write_report(out_dir / "coverage_report.md", rows, counts)
    data_io.append_report(
        ROOT / "outputs/nightly_report.md",
        "P10 Reproducibility Coverage Audit",
        [
            f"- Ran requirement coverage audit with counts `{counts}`.",
            f"- Overall audit status: `{result['overall']}`.",
            f"- Outputs saved under `{out_dir}`.",
        ],
    )
    print(f"Wrote audit outputs to {out_dir}")
    print(json.dumps({"overall": result["overall"], "status_counts": counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
