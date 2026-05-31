from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import data_io
from src.forward_models import estimate_q_from_frames, normalize_m, phases_for_m
from src.height import confidence_metrics, height_parabolic, summarize_height
from src.losses import charbonnier, tv_l1
from src.os_sim import demodulate_multigroup, fuse_hv
from src.visualization import save_loss_curve, save_montage, save_scalar_image


PARAMS_PER_GROUP = 6


def unpack_group_params(params: np.ndarray, group_index: int, phase_steps: int) -> dict[str, np.ndarray | float]:
    base = group_index * PARAMS_PER_GROUP
    dq_x, dq_y, dphi0, dphi1, a3, a5 = params[base : base + PARAMS_PER_GROUP]
    dphi = np.zeros(phase_steps, dtype=np.float32)
    if phase_steps >= 1:
        dphi[0] = dphi0
    if phase_steps >= 2:
        dphi[1] = dphi1
    if phase_steps >= 3:
        dphi[2] = -dphi0 - dphi1
    if phase_steps > 3:
        # Keep the first version low-dimensional: distribute residual zero-sum
        # correction over remaining phase frames.
        dphi[2:] = -np.sum(dphi[:2]) / max(phase_steps - 2, 1)
    return {"dq_x": float(dq_x), "dq_y": float(dq_y), "dphi": dphi, "a3": float(a3), "a5": float(a5)}


def build_patterns(shape: tuple[int, int], groups: list[dict], q_init: list[tuple[float, float]], params: np.ndarray) -> np.ndarray:
    h, w = shape
    yy, xx = np.indices(shape, dtype=np.float32)
    out = []
    for g, group in enumerate(groups):
        m = int(group.get("phase_steps", 3))
        p = unpack_group_params(params, g, m)
        qx = q_init[g][0] + float(p["dq_x"])
        qy = q_init[g][1] + float(p["dq_y"])
        dphi = p["dphi"]
        phi0 = qx * xx + qy * yy
        frames = []
        for phase, phase_err in zip(phases_for_m(m), dphi):
            phi = phi0 + phase + phase_err
            frame = np.cos(phi) + float(p["a3"]) * np.cos(3.0 * phi) + float(p["a5"]) * np.cos(5.0 * phi)
            frames.append(frame)
        out.append(normalize_m(np.stack(frames, axis=0)))
    return np.stack(out, axis=0).astype(np.float32)


def solve_A_closed_form(Y: np.ndarray, M: np.ndarray) -> np.ndarray:
    Y_tilde = Y - Y.mean(axis=2, keepdims=True)
    numerator = np.sum(Y_tilde * M[None, :, :, :, :], axis=2)
    denominator = np.sum(M[None, :, :, :, :] ** 2, axis=2) + 1e-6
    return np.maximum(numerator / denominator, 0.0).astype(np.float32)


def objective(
    params: np.ndarray,
    Y: np.ndarray,
    groups: list[dict],
    q_init: list[tuple[float, float]],
    lambda_q: float,
    lambda_phi: float,
    lambda_harm: float,
    lambda_sec: float,
    A_ref: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    M = build_patterns(Y.shape[-2:], groups, q_init, params)
    A = solve_A_closed_form(Y, M)
    Y_tilde = Y - Y.mean(axis=2, keepdims=True)
    residual = Y_tilde - A[:, :, None, :, :] * M[None, :, :, :, :]
    loss = charbonnier(residual)
    reg_q = 0.0
    reg_phi = 0.0
    reg_harm = 0.0
    for g, group in enumerate(groups):
        p = unpack_group_params(params, g, int(group.get("phase_steps", 3)))
        reg_q += float(p["dq_x"]) ** 2 + float(p["dq_y"]) ** 2
        reg_phi += float(np.sum(np.asarray(p["dphi"]) ** 2))
        reg_harm += float(p["a3"]) ** 2 + float(p["a5"]) ** 2
    if lambda_sec:
        loss += lambda_sec * float(np.mean(np.abs(A - A_ref)))
    loss += lambda_q * reg_q + lambda_phi * reg_phi + lambda_harm * reg_harm
    return float(loss), A, M


def coordinate_search(
    Y: np.ndarray,
    groups: list[dict],
    q_init: list[tuple[float, float]],
    A_ref: np.ndarray,
    passes: int,
    lambda_q: float,
    lambda_phi: float,
    lambda_harm: float,
    lambda_sec: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
    params = np.zeros(len(groups) * PARAMS_PER_GROUP, dtype=np.float32)
    steps_one = np.array([0.02, 0.02, 0.06, 0.06, 0.025, 0.015], dtype=np.float32)
    limits_one = np.array([0.18, 0.18, 0.50, 0.50, 0.18, 0.12], dtype=np.float32)
    steps = np.tile(steps_one, len(groups))
    limits = np.tile(limits_one, len(groups))
    best_loss, best_A, best_M = objective(params, Y, groups, q_init, lambda_q, lambda_phi, lambda_harm, lambda_sec, A_ref)
    curve = [best_loss]
    for _ in range(passes):
        improved = False
        for i in range(params.size):
            for sign in (1.0, -1.0):
                trial = params.copy()
                trial[i] = np.clip(trial[i] + sign * steps[i], -limits[i], limits[i])
                loss, A, M = objective(trial, Y, groups, q_init, lambda_q, lambda_phi, lambda_harm, lambda_sec, A_ref)
                if loss < best_loss:
                    params, best_loss, best_A, best_M = trial, loss, A, M
                    improved = True
            curve.append(best_loss)
        if not improved:
            steps *= 0.5
    return params, best_A, best_M, curve


def params_to_report(params: np.ndarray, groups: list[dict], q_init: list[tuple[float, float]]) -> list[dict]:
    rows = []
    for g, group in enumerate(groups):
        p = unpack_group_params(params, g, int(group.get("phase_steps", 3)))
        rows.append(
            {
                "group": group["name"],
                "q_init": [float(q_init[g][0]), float(q_init[g][1])],
                "dq": [float(p["dq_x"]), float(p["dq_y"])],
                "q_final": [float(q_init[g][0] + float(p["dq_x"])), float(q_init[g][1] + float(p["dq_y"]))],
                "dphi": np.asarray(p["dphi"]).astype(float).tolist(),
                "a3": float(p["a3"]),
                "a5": float(p["a5"]),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize low-dimensional q/phase/harmonic theta and closed-form A on a crop.")
    parser.add_argument("--config", default="configs/ossim_dataset.yaml")
    parser.add_argument("--sample", default="细台阶")
    parser.add_argument("--groups", default="ossim_t6_h_3step,ossim_t6_v_3step")
    parser.add_argument("--exclude-samples", default="台阶曝光1")
    parser.add_argument("--crop", default="center:128")
    parser.add_argument("--passes", type=int, default=4)
    parser.add_argument("--lambda-q", type=float, default=0.05)
    parser.add_argument("--lambda-phi", type=float, default=0.01)
    parser.add_argument("--lambda-harm", type=float, default=0.02)
    parser.add_argument("--lambda-sec", type=float, default=0.01)
    parser.add_argument("--output", default="outputs/theta_opt/t6_hv")
    args = parser.parse_args()

    config = data_io.load_config(args.config)
    excluded = data_io.parse_excluded_samples(args.exclude_samples)
    group_names = data_io.parse_groups(args.groups, config)
    crop = data_io.infer_crop_for_sample(config, args.sample, args.crop)
    bundle = data_io.load_os_sim_stack(config, args.sample, group_names, crop, excluded_samples=excluded)
    Y = bundle["Y"]
    z_values = bundle["z_values"]
    demod = demodulate_multigroup(Y)
    A_ref = demod["A"]
    q_init = []
    q_info = []
    for g, group in enumerate(bundle["groups"]):
        qx, qy, info = estimate_q_from_frames(Y[Y.shape[0] // 2, g], group)
        q_init.append((qx, qy))
        q_info.append(info)

    zero_params = np.zeros(len(group_names) * PARAMS_PER_GROUP, dtype=np.float32)
    initial_loss, initial_A, initial_M = objective(
        zero_params, Y, bundle["groups"], q_init, args.lambda_q, args.lambda_phi, args.lambda_harm, args.lambda_sec, A_ref
    )
    params, A_opt, M_opt, curve = coordinate_search(
        Y,
        bundle["groups"],
        q_init,
        A_ref,
        args.passes,
        args.lambda_q,
        args.lambda_phi,
        args.lambda_harm,
        args.lambda_sec,
    )
    final_loss, _, _ = objective(params, Y, bundle["groups"], q_init, args.lambda_q, args.lambda_phi, args.lambda_harm, args.lambda_sec, A_ref)
    A_base_fused = fuse_hv(A_ref[:, 0], A_ref[:, 1] if A_ref.shape[1] > 1 else A_ref[:, 0])
    A_fused = fuse_hv(A_opt[:, 0], A_opt[:, 1] if A_opt.shape[1] > 1 else A_opt[:, 0])
    height_base = height_parabolic(A_base_fused, z_values)
    height_opt = height_parabolic(A_fused, z_values)
    conf_base = confidence_metrics(A_base_fused)
    conf_opt = confidence_metrics(A_fused)

    out_dir = ROOT / args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "z_values.npy", z_values)
    np.save(out_dir / "A_h_stack.npy", A_opt[:, 0])
    np.save(out_dir / "A_v_stack.npy", A_opt[:, 1] if A_opt.shape[1] > 1 else A_opt[:, 0])
    np.save(out_dir / "A_fused_stack.npy", A_fused)
    np.save(out_dir / "height_theta_peakfit.npy", height_opt)
    np.save(out_dir / "height_baseline_peakfit.npy", height_base)
    mid = Y.shape[0] // 2
    D0 = demod["D0"]
    Y_hat = D0[:, :, None, :, :] + A_opt[:, :, None, :, :] * M_opt[None, :, :, :, :]
    save_loss_curve(curve, out_dir / "theta_loss_curve.png")
    save_montage(
        [A_base_fused[mid], A_fused[mid], height_base, height_opt, np.abs(height_opt - height_base)],
        out_dir / "theta_A_height_compare.png",
        ["A baseline fused", "A theta fused", "height baseline", "height theta", "height abs diff"],
        cols=3,
    )
    save_montage(
        [Y[mid, 0, 0], Y_hat[mid, 0, 0], np.abs(Y[mid, 0, 0] - Y_hat[mid, 0, 0]), Y[mid, 1, 0], Y_hat[mid, 1, 0], np.abs(Y[mid, 1, 0] - Y_hat[mid, 1, 0])],
        out_dir / "theta_reprojection_examples.png",
        ["real H", "hat H", "abs H", "real V", "hat V", "abs V"],
        cols=3,
    )
    save_scalar_image(height_opt, out_dir / "theta_height_map.png", "Theta-optimized fused height")

    metrics = {
        "sample": args.sample,
        "groups": group_names,
        "crop": crop.label if crop else "full",
        "passes": args.passes,
        "initial_objective": float(initial_loss),
        "final_objective": float(final_loss),
        "objective_delta": float(final_loss - initial_loss),
        "initial_A_tv": tv_l1(A_base_fused),
        "theta_A_tv": tv_l1(A_fused),
        "height_baseline": summarize_height(height_base, conf_base),
        "height_theta": summarize_height(height_opt, conf_opt),
        "theta": params_to_report(params, bundle["groups"], q_init),
        "q_initialization": q_info,
        "regularization": {
            "lambda_q": args.lambda_q,
            "lambda_phi": args.lambda_phi,
            "lambda_harm": args.lambda_harm,
            "lambda_sec": args.lambda_sec,
        },
    }
    data_io.write_json(out_dir / "metrics.json", metrics)
    data_io.write_json(out_dir / "config.json", vars(args))
    report = [
        "# Theta Optimization Report",
        "",
        f"- Sample: `{args.sample}`",
        f"- Groups: `{', '.join(group_names)}`",
        f"- Crop: `{crop.label if crop else 'full'}`",
        f"- Objective: `{initial_loss:.6g}` -> `{final_loss:.6g}`",
        "",
        "This stage optimizes low-dimensional q offsets, zero-sum global phase errors, and a3/a5 harmonics. A is solved in closed form for each theta candidate.",
    ]
    (out_dir / "theta_opt_report.md").write_text("\n".join(report), encoding="utf-8")
    data_io.append_report(
        ROOT / "outputs/nightly_report.md",
        "P5b Low-Dimensional Theta Optimization",
        [
            f"- Optimized q/phase/a3/a5 theta for `{args.sample}` on `{crop.label if crop else 'full'}`.",
            f"- Objective `{initial_loss:.6g}` -> `{final_loss:.6g}`.",
            f"- Outputs saved under `{out_dir}`.",
        ],
    )
    print(f"Wrote theta optimization outputs to {out_dir}")


if __name__ == "__main__":
    main()

