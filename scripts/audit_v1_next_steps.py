from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import data_io


def repo_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def load_optional_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return data_io.read_json(path)
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the v1_next audit of metric definitions, frame order, and leakage risks.")
    parser.add_argument("--config", default="configs/ossim_dataset.yaml")
    parser.add_argument("--output", default="outputs/v1_next/audit_next_steps.md")
    args = parser.parse_args()

    config = data_io.load_config(args.config)
    groups = data_io.frame_groups_by_name(config)
    ls_metrics = load_optional_json(ROOT / "outputs/ls_projection/v1_t6_hv/metrics.json")
    forward_metrics = load_optional_json(ROOT / "outputs/forward/v1_t6_hv/model_compare.json")
    freq_metrics = load_optional_json(ROOT / "outputs/frequency/v1_t6_hv/t6_vs_t12_compare.json")

    projection_map = config.get("calibration_assets", {}).get("projection_frame_map", {})
    t6_h = groups.get("ossim_t6_h_3step", {})
    t6_v = groups.get("ossim_t6_v_3step", {})

    calibration_lines: list[str] = []
    if ls_metrics and ls_metrics.get("pattern_infos"):
        for info in ls_metrics["pattern_infos"]:
            calibration_lines.append(
                "- measured-calibrated pattern: frames `{}` from `{}`, calibration `{}`, camera shape `{}`, valid fraction `{}`.".format(
                    info.get("frame_ids"),
                    info.get("sequence_path"),
                    info.get("calibration_source"),
                    info.get("camera_shape"),
                    info.get("valid_fraction"),
                )
            )
    else:
        calibration_lines.append("- Existing LS metrics with measured-calibrated pattern provenance were not found.")

    frequency_answer = "not_available"
    if freq_metrics:
        frequency_answer = str(freq_metrics.get("ranking", {}).get("best_by_grid_sharpness_fwhm_score", "not_ranked"))

    forward_answer = "not_available"
    if forward_metrics:
        details = forward_metrics.get("models", {})
        parts = []
        for group_name, row in details.items():
            losses = row.get("losses", {})
            if losses:
                best = min(losses.items(), key=lambda item: item[1].get("mod_rmse", float("inf")))
                parts.append(f"`{group_name}` best mod-RMSE `{best[0]}` = `{best[1].get('mod_rmse'):.6g}`")
        forward_answer = "; ".join(parts) if parts else "not_ranked"

    report = [
        "# v1_next Audit and Next-Step Contract",
        "",
        "## 1. Current Result Vocabulary",
        "",
        "The old v1 report uses several related but non-identical `grid energy` numbers. They must not be compared unless the mask, signal, and normalization are the same.",
        "",
        "- `A_cls grid`: masked FFT energy measured on the traditional demodulated modulation stack `A_cls`. It answers whether the recovered modulation map itself contains grid/Moire structure.",
        "- `A_ls grid`: the same kind of masked FFT energy, but measured on the fixed-template LS estimate `A_ls`. It is comparable to `A_cls grid` only when the same FFT mask is reused.",
        "- `fused grid`: masked FFT energy after H/V fusion. This can change simply because fusion suppresses or mixes directional artifacts.",
        "- `residual grid`: masked FFT energy on `Y - A*M` after phase centering. This is the forward-model error, not the visible grid remaining in `A`.",
        "- `stripe leakage`: masked FFT energy at the raw carrier stripe frequency. It is a different nuisance channel from low-frequency grid/Moire peaks.",
        "",
        "## 2. Frame and Pattern Contract",
        "",
        f"- Configured T6 H group: `{t6_h.get('frames')}`, direction `{t6_h.get('direction')}`, period `{t6_h.get('period')}`.",
        f"- Configured T6 V group: `{t6_v.get('frames')}`, direction `{t6_v.get('direction')}`, period `{t6_v.get('period')}`.",
        f"- Projection frame map in config: `{projection_map}`.",
        "- The config frame IDs are internally consistent with the projection sequence names for T6 H/V, but Task 1 is still required because config consistency is not an optical phase-order proof.",
        "- T12 frame order is defined in the config, but no T12 projection-frame map is currently listed under `calibration_assets.projection_frame_map`; it should be treated as less verified than T6 until checked directly.",
        "",
        "## 3. D0 and Residual Domain",
        "",
        "- Traditional OS-SIM uses `D0 = mean_m(Y)` per pixel, z layer, and group. The fixed-M LS experiments also phase-center with `Y - mean_m(Y)` and estimate only modulation amplitude `A`.",
        "- `L_raw` and `L_mod` are not the same diagnostic. `L_raw` includes the DC/base image channel; `L_mod` tests whether the fixed pattern template explains the phase-varying modulation channel.",
        "- v1_next should rank inverse methods primarily by modulation-domain residuals and structured residual FFT, not by apparent height flatness.",
        "",
        "## 4. Measured-Calibrated `M_theta` Provenance",
        "",
        *calibration_lines,
        "- These patterns come from calibration/projection assets, not from a trained inverse network. They use the same crop geometry when the crop argument is the same.",
        "- They are still not independent height truth; they only define the forward-template hypothesis.",
        "",
        "## 5. Current Evidence Snapshot",
        "",
        f"- Prior T6/T12 diagnostic winner: `{frequency_answer}`.",
        f"- Prior forward-model snapshot: {forward_answer}.",
        "",
        "## 6. Data Leakage and Claim Risks",
        "",
        "- `A_cls` is a weak diagnostic baseline generated from the same raw frames; it is not clean ground truth.",
        "- Network outputs trained against `A_cls` can inherit `A_cls` artifacts or smooth them away without proving metrology improvement.",
        "- Fixed-template LS can reduce modulation loss while changing height statistics; without independent truth this is not an absolute accuracy claim.",
        "- Platform RMS/height std can improve by oversmoothing. It is only a diagnostic unless repeatability or a nominal standard supports it.",
        "- Calibration assets and raw sample are read-only, but using the same sample for both model selection and reporting still creates selection bias. v1_next reports must label such results as internal consistency.",
        "",
        "## 7. Required Follow-Up",
        "",
        "- Task 1 must verify phase order before blaming V direction on optical/calibration mismatch.",
        "- Task 2 must diagnose V forward-template mismatch with measured, phase-shifted, q-corrected, affine, binary, harmonic, and blur variants.",
        "- Tasks 3-7 must recompute all method comparisons with a single metric implementation and confidence mask.",
    ]

    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote {repo_path(out_path)}")


if __name__ == "__main__":
    main()
