from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src import data_io
from src.height import confidence_metrics
from src.os_sim import demodulate_multigroup, fuse_hv
from src.visualization import save_scalar_image

from analyze_axial_modulation_response import PAIR_DEFS, roi_dict, summarize_rois


def choose_sample(config: dict[str, Any], sample: str | None) -> str:
    if sample:
        return sample
    excluded = data_io.parse_excluded_samples(None)
    for row in reversed(config.get("samples", [])):
        name = row.get("name")
        if name and name not in excluded:
            return name
    raise ValueError("No usable sample found in config")


def gaussian(z: np.ndarray, base: float, amp: float, z0: float, sigma: float) -> np.ndarray:
    return base + amp * np.exp(-0.5 * ((z - z0) / (sigma + 1e-8)) ** 2)


def lorentzian(z: np.ndarray, base: float, amp: float, z0: float, gamma: float) -> np.ndarray:
    return base + amp / (1.0 + ((z - z0) / (gamma + 1e-8)) ** 2)


def fwhm_from_fit(kind: str, width: float) -> float:
    if kind == "gaussian":
        return float(2.0 * np.sqrt(2.0 * np.log(2.0)) * abs(width))
    if kind == "lorentzian":
        return float(2.0 * abs(width))
    return float("nan")


def fit_curve(z_values: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    z = np.asarray(z_values, dtype=np.float32)
    curve = np.asarray(y, dtype=np.float32)
    base0 = float(np.percentile(curve, 10.0))
    amp0 = float(np.max(curve) - base0)
    z0 = float(z[int(np.argmax(curve))])
    width0 = max(float(np.median(np.diff(z))) * 2.0, 1e-3) if z.size > 1 else 1.0
    out: dict[str, Any] = {}
    for name, fn in [("gaussian", gaussian), ("lorentzian", lorentzian)]:
        try:
            popt, _ = curve_fit(fn, z, curve, p0=[base0, amp0, z0, width0], maxfev=10000)
            pred = fn(z, *popt)
            rmse = float(np.sqrt(np.mean((pred - curve) ** 2)))
            out[name] = {
                "status": "ok",
                "params": [float(v) for v in popt],
                "rmse": rmse,
                "fwhm_um": fwhm_from_fit(name, float(popt[3])),
            }
        except Exception as exc:
            out[name] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    ok = {k: v for k, v in out.items() if v.get("status") == "ok"}
    out["best_fit"] = min(ok, key=lambda k: ok[k]["rmse"]) if ok else None
    return out


def curve_metrics_um(curve: np.ndarray, z_values: np.ndarray) -> dict[str, Any]:
    y = np.asarray(curve, dtype=np.float32)
    z = np.asarray(z_values, dtype=np.float32)
    peak_idx = int(np.argmax(y))
    peak = float(y[peak_idx])
    bg = float(np.percentile(y, 10.0))
    half = bg + 0.5 * (peak - bg)
    above = np.where(y >= half)[0]
    fwhm = float(z[above[-1]] - z[above[0]] + (np.median(np.diff(z)) if z.size > 1 else 1.0)) if above.size else 0.0
    return {
        "peak_z_um": float(z[peak_idx]),
        "peak_value": peak,
        "background": bg,
        "fwhm_um": fwhm,
        "fit": fit_curve(z, y),
    }


def load_pair_A(config: dict[str, Any], sample: str, groups: tuple[str, str], crop: data_io.CropSpec | None, excluded: set[str]) -> tuple[np.ndarray, np.ndarray]:
    bundle = data_io.load_os_sim_stack(config, sample, list(groups), crop, excluded_samples=excluded)
    demod = demodulate_multigroup(bundle["Y"].astype(np.float32))
    A = fuse_hv(demod["A"][:, 0], demod["A"][:, 1])
    return A.astype(np.float32), bundle["z_values"].astype(np.float32)


def theory_fwhm(args: argparse.Namespace) -> dict[str, Any]:
    if args.na is None or args.wavelength_um is None:
        return {"status": "not_computed", "reason": "NA and wavelength_um were not supplied"}
    na = float(args.na)
    wavelength = float(args.wavelength_um)
    if na <= 0 or na >= 1:
        return {"status": "failed", "reason": "NA must be in (0,1) for the simple diagnostic formula"}
    alpha = float(np.arcsin(na))
    widefield_axial = 0.88 * wavelength / (1.0 - np.cos(alpha) + 1e-8)
    result = {
        "status": "computed_diagnostic",
        "formula": "0.88*lambda/(1-cos(arcsin(NA)))",
        "widefield_axial_fwhm_um": float(widefield_axial),
    }
    if args.fringe_period_um is not None:
        period = float(args.fringe_period_um)
        result["fringe_period_um"] = period
        result["note"] = "A full Stokseth/SIM axial response requires optical geometry and illumination angle; period is recorded but not folded into a claimed theory curve."
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert axial modulation response metrics into physical z units and fit empirical H(z) curves.")
    parser.add_argument("--config", default="configs/ossim_dataset.yaml")
    parser.add_argument("--sample", default=None)
    parser.add_argument("--groups", default="ossim_t6_h_3step,ossim_t6_v_3step")
    parser.add_argument("--exclude-samples", default="鍙伴樁鏇濆厜1")
    parser.add_argument("--crop", default="center:256")
    parser.add_argument("--na", type=float, default=None)
    parser.add_argument("--wavelength-um", type=float, default=None)
    parser.add_argument("--fringe-period-um", type=float, default=None)
    parser.add_argument("--output", default="outputs/v1_next/axial_physical")
    args = parser.parse_args()

    config = data_io.load_config(args.config)
    sample = choose_sample(config, args.sample)
    excluded = data_io.parse_excluded_samples(args.exclude_samples)
    groups = data_io.parse_groups(args.groups, config)
    crop = data_io.infer_crop_for_sample(config, sample, args.crop)
    bundle = data_io.load_os_sim_stack(config, sample, groups, crop, excluded_samples=excluded)
    demod = demodulate_multigroup(bundle["Y"].astype(np.float32))
    A = fuse_hv(demod["A"][:, 0], demod["A"][:, 1] if demod["A"].shape[1] > 1 else demod["A"][:, 0])
    z_values = bundle["z_values"].astype(np.float32)
    z_step = float(config.get("z_axis", {}).get("observed_step", np.median(np.diff(z_values))))
    z_unit = str(config.get("z_axis", {}).get("unit", "unknown"))
    if z_unit.lower() not in {"micrometer", "um", "micron", "micrometers"}:
        raise ValueError(f"z_axis.unit is {z_unit!r}; supply/update metadata before physicalizing FWHM")

    conf = confidence_metrics(A)
    fwhm_um_map = conf["fwhm_layers"].astype(np.float32) * z_step
    rois, _ = roi_dict(A, z_values)
    roi_summary = summarize_rois(A, z_values, rois)
    roi_physical: dict[str, Any] = {}
    for name, row in roi_summary.items():
        curve = np.asarray(row["mean"], dtype=np.float32)
        roi_physical[name] = {
            "pixel_count": row["pixel_count"],
            "curve_metrics_um": curve_metrics_um(curve, z_values),
        }

    pair_rows: dict[str, Any] = {}
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for pair_name, pair_groups in PAIR_DEFS.items():
        try:
            A_pair, z_pair = load_pair_A(config, sample, pair_groups, crop, excluded)
            curve = np.mean(A_pair, axis=(1, 2))
            curve_norm = (curve - np.min(curve)) / (np.ptp(curve) + 1e-8)
            conf_pair = confidence_metrics(A_pair)
            pair_rows[pair_name] = {
                "status": "ok",
                "mean_fwhm_um": float(np.mean(conf_pair["fwhm_layers"]) * z_step),
                "mean_peak_sharpness": float(np.mean(conf_pair["sharpness"])),
                "mean_peak_to_background": float(np.mean(conf_pair["peak_to_background"])),
                "global_curve_metrics_um": curve_metrics_um(curve, z_pair),
            }
            ax.plot(z_pair, curve_norm, marker="o", ms=2, label=pair_name)
        except Exception as exc:
            pair_rows[pair_name] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    ax.set_title("T6/T12 normalized global H(z) comparison")
    ax.set_xlabel("z (um)")
    ax.set_ylabel("normalized mean A")
    ax.grid(True, alpha=0.3)
    ax.legend()

    out_dir = ROOT / args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / "T6_T12_H_compare.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for name, row in roi_summary.items():
        ax.plot(z_values, row["mean"], marker="o", ms=2, label=name)
    ax.set_title("ROI axial response in physical z")
    ax.set_xlabel("z (um)")
    ax.set_ylabel("A")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "axial_curves_physical.png", dpi=150)
    fig.savefig(out_dir / "empirical_H_curves.png", dpi=150)
    plt.close(fig)

    save_scalar_image(fwhm_um_map, out_dir / "fwhm_um_map.png", "FWHM (um)", cmap="magma")
    save_scalar_image(fwhm_um_map, out_dir / "FWHM_um_map.png", "FWHM (um)", cmap="magma")

    metrics = {
        "sample": sample,
        "groups": groups,
        "crop": crop.label if crop else "full",
        "z_step": z_step,
        "z_unit": z_unit,
        "global": {
            "mean_fwhm_um": float(np.mean(fwhm_um_map)),
            "median_fwhm_um": float(np.median(fwhm_um_map)),
            "p95_fwhm_um": float(np.percentile(fwhm_um_map, 95.0)),
            "mean_peak_sharpness": float(np.mean(conf["sharpness"])),
            "mean_peak_to_background": float(np.mean(conf["peak_to_background"])),
        },
        "roi_physical": roi_physical,
        "t6_t12_compare": pair_rows,
        "theory": theory_fwhm(args),
        "interpretation": "Physical z units come from configs/ossim_dataset.yaml z_axis metadata, not from an independent height standard.",
    }
    data_io.write_json(out_dir / "metrics.json", metrics)
    data_io.write_json(out_dir / "axial_physical_metrics.json", metrics)

    ok_pairs = {k: v for k, v in pair_rows.items() if v.get("status") == "ok"}
    best_pair = min(ok_pairs, key=lambda k: ok_pairs[k]["mean_fwhm_um"]) if ok_pairs else "n/a"
    report = [
        "# Physical Axial Response Report",
        "",
        f"- Sample: `{sample}`",
        f"- Crop: `{crop.label if crop else 'full'}`",
        f"- z step: `{z_step}` `{z_unit}`",
        "",
        "## Required Answers",
        "",
        f"- Mean fused FWHM: `{metrics['global']['mean_fwhm_um']:.6g}` um.",
        f"- Median fused FWHM: `{metrics['global']['median_fwhm_um']:.6g}` um.",
        f"- Narrowest T6/T12 global response by mean FWHM: `{best_pair}`.",
        f"- Theory status: `{metrics['theory']['status']}`.",
        "",
        "## T6/T12",
        "",
        "| pair | mean FWHM (um) | sharpness | peak/background | global curve FWHM (um) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in pair_rows.items():
        if row.get("status") != "ok":
            report.append(f"| {name} | failed | | | |")
            continue
        report.append(
            f"| {name} | {row['mean_fwhm_um']:.6g} | {row['mean_peak_sharpness']:.6g} | {row['mean_peak_to_background']:.6g} | {row['global_curve_metrics_um']['fwhm_um']:.6g} |"
        )
    report += [
        "",
        "## ROI Fits",
        "",
        "| ROI | pixels | empirical FWHM (um) | best fit | fit FWHM (um) |",
        "|---|---:|---:|---|---:|",
    ]
    for name, row in roi_physical.items():
        cm = row["curve_metrics_um"]
        best = cm["fit"].get("best_fit")
        fit_fwhm = cm["fit"].get(best, {}).get("fwhm_um") if best else None
        report.append(
            f"| {name} | {row['pixel_count']} | {cm['fwhm_um']:.6g} | {best or 'n/a'} | {'' if fit_fwhm is None else f'{fit_fwhm:.6g}'} |"
        )
    report += [
        "",
        "No absolute metrology claim is made here; this only physicalizes the scan-axis response width.",
    ]
    (out_dir / "axial_physical_report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote physical axial response outputs to {out_dir}")


if __name__ == "__main__":
    main()
