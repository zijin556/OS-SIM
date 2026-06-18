from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import data_io


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def first_existing(paths: list[str]) -> Path | None:
    for item in paths:
        path = ROOT / item
        if path.exists():
            return path
    return None


def metric_at(data: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return "`missing`" if value is None else str(value)


def scan_inverse_metrics() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(ROOT.glob("outputs/inverse_net*/**/metrics.json")):
        data = read_json(path)
        if not data:
            continue
        rows.append(
            {
                "path": str(path.relative_to(ROOT)),
                "status": data.get("status", "completed"),
                "architecture": data.get("architecture"),
                "select_best": data.get("select_best"),
                "lambda_platform": metric_at(data, ["args", "lambda_platform"]),
                "baseline_modulation_loss": data.get("baseline_modulation_loss"),
                "final_modulation_loss": data.get("final_modulation_loss"),
                "baseline_grid_energy": data.get("baseline_grid_energy"),
                "pred_grid_energy": data.get("pred_grid_energy"),
                "platform": data.get("platform", {}),
                "theta": data.get("theta", {}),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit v0/v1 OS-SIM code and science claims before continuing experiments.")
    parser.add_argument("--output", default="outputs/audit/model_code_science_audit.md")
    parser.add_argument("--json-output", default="outputs/audit/model_code_science_audit.json")
    args = parser.parse_args()

    readme = read_text(ROOT / "README.md")
    nightly = read_text(ROOT / "outputs/nightly_report.md")
    theory = read_text(ROOT / "outputs/theoretical_model_for_gpt_pro.md")
    train_code = read_text(ROOT / "scripts/train_inverse_lightsection_net.py")
    torch_forward = read_text(ROOT / "src/torch_forward.py")
    forward_models = read_text(ROOT / "src/forward_models.py")
    os_sim = read_text(ROOT / "src/os_sim.py")

    baseline = read_json(ROOT / "outputs/baseline/t6_hv/metrics.json")
    ablation = read_json(ROOT / "outputs/ablation/t6_hv/metrics.json")
    freq_compare = read_json(ROOT / "outputs/frequency/group_compare/t6_vs_t12_compare.json")
    forward = read_json(ROOT / "outputs/forward/t6_hv/model_compare.json")
    platform_compare = read_json(ROOT / "outputs/height/platform_compare.json")
    inverse_rows = scan_inverse_metrics()

    best_inverse = None
    completed_inverse = [r for r in inverse_rows if r.get("status") == "completed" and r.get("final_modulation_loss") is not None]
    if completed_inverse:
        best_inverse = min(completed_inverse, key=lambda row: float(row.get("final_modulation_loss") or 1e9))

    lambda_platform_defaults = re.findall(r"--lambda-platform\".*?default=([0-9.]+)", train_code, flags=re.S)
    has_select_best = "--select-best" in train_code and "selection_score" in train_code
    raw_mod_duplicate = "loss_raw = charbonnier_loss(yb - y_hat)" in train_code and "loss_mod = charbonnier_loss" in train_code
    d0_fixed_mean = "D0 = Y.mean(axis=2)" in train_code or "d0b = D0[idx]" in train_code
    invariant_grid_np = "out + grid" in forward_models and "normalize_m" in forward_models
    invariant_grid_torch = "out = out + grid[:, None]" in torch_forward
    phase_dependent_grid_present = "phase_dependent_grid_basis" in forward_models and "gmbhw->gmhw" in torch_forward
    phase_offsets_zero = "phase_offsets" in theory and ("about 1e-22" in theory or "almost zero" in theory)
    blur_tiny = "blur_sigma" in theory and "0.0064557" in theory
    os_sim_m3_correct = all(s in os_sim for s in ["(2.0 * i1 - i2 - i3) / 3.0", "(i3 - i2) / np.sqrt(3.0)", "np.linalg.pinv"])

    conclusions = [
        f"Traditional OS-SIM baseline is implemented with the requested M=3 formula and an M!=3 least-squares path: `{os_sim_m3_correct}`.",
        f"`Y = D0 + A*M_theta` modulation-domain forward consistency exists in NumPy/PyTorch code; measured calibrated DMD patterns are already available in v0 outputs.",
        f"Data policy is explicit: raw data are read-only, `台阶曝光1` is excluded, and main v0 sample is `细台阶`.",
    ]
    if baseline:
        conclusions.append(
            "Baseline T6 H/V output exists: fused height std "
            f"`{fmt(metric_at(baseline, ['fused', 'height_std']))}`, invalid fraction "
            f"`{fmt(metric_at(baseline, ['fused', 'invalid_fraction']))}`."
        )
    if freq_compare:
        conclusions.append(
            "A diagnostic T6/T12 comparison exists; current best by grid/sharpness score is "
            f"`{metric_at(freq_compare, ['ranking', 'best_by_grid_sharpness_score'], 'missing')}`."
        )

    insufficient = [
        "No independent calibrated height truth is present, so absolute metrology-error improvement is not established.",
        "Cross-sample and held-out-crop validation remain insufficient for network claims.",
        "`A_cls` is useful as a weak diagnostic baseline, but it is not clean ground truth for training or evaluation.",
        "The current evidence does not yet answer whether real ROI A(z) curves support continuous modulation response; v1 axial-response analysis is required.",
    ]
    if best_inverse:
        insufficient.append(
            "Best inverse-net metrics improve modulation/grid/platform diagnostics, but the run uses same-crop selection and must be compared against LS projection and simple filters before claiming network necessity."
        )

    possible_bugs = []
    if (invariant_grid_np or invariant_grid_torch) and phase_dependent_grid_present:
        possible_bugs.append(
            "Legacy phase-invariant grid branches still exist for backward compatibility, but v1 code now includes phase-dependent grid basis `G[j,m,y,x]`; only the phase-dependent path should be interpreted scientifically."
        )
    elif invariant_grid_np or invariant_grid_torch:
        possible_bugs.append(
            "Phase-invariant grid/Moire terms are added identically to every phase and then phase-normalized; such terms are mathematically canceled or made nearly unidentifiable. Use phase-dependent grid basis `G[j,m,y,x]`."
        )
    if raw_mod_duplicate and d0_fixed_mean:
        possible_bugs.append(
            "`L_raw` and `L_mod` are effectively redundant when `D0` is fixed to the phase mean; reporting both as independent evidence is misleading."
        )
    if blur_tiny:
        possible_bugs.append("Learned `blur_sigma` in the v0 report is around 0.006 px, too small to support a meaningful PSF/blur interpretation.")
    if phase_offsets_zero:
        possible_bugs.append("Reported `phase_offsets` are nearly zero, suggesting the phase parameters are not identifiable or not active in the current best run.")
    if not possible_bugs:
        possible_bugs.append("No high-confidence code bug found by static audit; still run v1 smoke tests.")

    risks = [
        "`lambda_platform` and same-crop `--select-best` can make a step sample look flatter without proving better morphology.",
        "Network output may be a smoothed/regularized version of `A_cls`; this must be tested against lowpass/TV/median/bilateral and LS projection baselines.",
        "`theta` parameters can trade off with `A` and `D0`; parameter values are not directly interpretable without ablations and residual FFT evidence.",
        "Simulation can support identifiability sanity checks only; real-data conclusions must come from raw stack diagnostics.",
    ]
    if lambda_platform_defaults:
        risks.append(f"Training code default `lambda_platform` is `{lambda_platform_defaults[-1]}`; v1 should set it to zero unless running a control.")
    if has_select_best:
        risks.append("The inverse-net script supports `--select-best`, which is useful diagnostically but is not an unbiased validation protocol.")

    required = [
        "Run ROI-wise A(z) axial-response curves on real T6/T12 data and report FWHM, peak count, entropy, and confidence.",
        "Run fixed-`M_theta` LS projection and compare it to `A_cls`, simple filters, latent optimization, and existing network outputs.",
        "Patch grid/Moire forward terms to be phase-dependent or mark the old phase-invariant term as unidentifiable.",
        "Run residual FFT diagnostics after every accepted A-stack method.",
        "Run latent optimization with platform loss disabled, plus a separate platform-loss control.",
        "Only report standard-piece error when an independent standard height or repeatability dataset is available.",
    ]

    audit_json = {
        "inputs": {
            "baseline_metrics": bool(baseline),
            "ablation_metrics": bool(ablation),
            "frequency_compare": bool(freq_compare),
            "forward_metrics": bool(forward),
            "platform_compare": bool(platform_compare),
            "inverse_metric_count": len(inverse_rows),
        },
        "static_checks": {
            "os_sim_m3_formula_and_ls_path": os_sim_m3_correct,
            "raw_and_mod_loss_redundancy_risk": raw_mod_duplicate and d0_fixed_mean,
            "phase_invariant_numpy_grid_risk": invariant_grid_np,
            "phase_invariant_torch_grid_risk": invariant_grid_torch,
            "phase_dependent_grid_present": phase_dependent_grid_present,
            "phase_offsets_near_zero_reported": phase_offsets_zero,
            "blur_sigma_tiny_reported": blur_tiny,
            "select_best_present": has_select_best,
        },
        "best_inverse_metric": best_inverse,
        "sections": {
            "conclusions_supported": conclusions,
            "evidence_insufficient": insufficient,
            "possible_bugs": possible_bugs,
            "interpretation_risks": risks,
            "required_experiments": required,
        },
    }

    lines = [
        "# OS-SIM v1 Model/Code/Science Audit",
        "",
        "Scope: static audit plus existing-output review before v1 continuous-modulation experiments.",
        "",
        "## 结论成立项",
        "",
        *[f"- {item}" for item in conclusions],
        "",
        "## 证据不足项",
        "",
        *[f"- {item}" for item in insufficient],
        "",
        "## 可能 bug",
        "",
        *[f"- {item}" for item in possible_bugs],
        "",
        "## 科学解释风险",
        "",
        *[f"- {item}" for item in risks],
        "",
        "## 必须补做实验",
        "",
        *[f"- {item}" for item in required],
        "",
        "## Audit Inputs",
        "",
        f"- Baseline metrics: `{bool(baseline)}`",
        f"- Ablation metrics: `{bool(ablation)}`",
        f"- Frequency T6/T12 compare: `{bool(freq_compare)}`",
        f"- Forward metrics: `{bool(forward)}`",
        f"- Inverse-net metric files scanned: `{len(inverse_rows)}`",
    ]

    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    data_io.write_json(ROOT / args.json_output, audit_json)
    data_io.append_report(
        ROOT / "outputs/nightly_report.md",
        "P0 v1 Model/Code/Science Audit",
        [
            f"- Wrote audit report to `{out_path}`.",
            "- Main risks: platform-loss dominance, raw/mod loss redundancy, phase-invariant grid cancellation, and missing independent height truth.",
        ],
    )
    print(f"Wrote audit report to {out_path}")


if __name__ == "__main__":
    main()
