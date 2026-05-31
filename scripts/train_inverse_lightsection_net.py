from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src import data_io
from src.calibration import measured_camera_patterns_for_group
from src.fft_analysis import circular_peak_mask, detect_frequency_peaks
from src.forward_models import fixed_grid_basis
from src.height import confidence_metrics, height_parabolic, height_softargmax, platform_metrics_from_masks, summarize_height
from src.os_sim import demodulate_multigroup, fuse_hv
from src.visualization import save_montage, save_scalar_image

from optimize_modulation_latents import build_grid_masks, grid_energy_mean, modulation_loss


def torch_install_text() -> str:
    return "\n".join(
        [
            "Recommended: install PyTorch inside the project virtual environment, not into system Python.",
            "",
            "PowerShell example:",
            "  cd E:\\research\\AIResearch\\PhysicsDeepLearning_OSSIM",
            "  python -m venv .venv",
            "  .\\.venv\\Scripts\\Activate.ps1",
            "  python -m pip install --upgrade pip",
            "  python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu",
            "",
            "For GPU/CUDA, use the matching command from https://pytorch.org/get-started/locally/ inside the same .venv.",
            "",
            "Then run:",
            "  python scripts/train_inverse_lightsection_net.py --crop center:128 --epochs 30 --pattern-source measured_calibrated",
            "  python scripts/evaluate_inverse_lightsection_net.py --input outputs/inverse_net/t6_hv/A_fused_pred_stack.npy",
        ]
    )


def write_missing_torch(out_dir: Path, args: argparse.Namespace, error: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "status": "missing_torch",
        "stage": "inverse_lightsection_net_training",
        "reason": error,
        "outputs_not_generated": [
            "A_h_pred_stack.npy",
            "A_v_pred_stack.npy",
            "A_fused_pred_stack.npy",
            "height_pred.npy",
            "height_pred.png",
            "reprojection_examples.png",
            "A_comparison.png",
            "height_compare.png",
            "loss_curve.png",
        ],
        "install_in_virtualenv": True,
        "install_notes": torch_install_text(),
    }
    data_io.write_json(out_dir / "metrics.json", metrics)
    data_io.write_json(out_dir / "config.json", vars(args))
    report = [
        "# Inverse Light-Section Network Report",
        "",
        "Status: `missing_torch`",
        "",
        "The inverse network code and training entrypoint are present, but PyTorch is not installed in the current Python environment.",
        "",
        "This script intentionally did not create fake A-stack or height outputs.",
        "",
        "## Install PyTorch",
        "",
        "```powershell",
        torch_install_text(),
        "```",
    ]
    (out_dir / "inverse_net_report.md").write_text("\n".join(report), encoding="utf-8")
    data_io.append_report(
        ROOT / "outputs/nightly_report.md",
        "P18 Inverse Light-Section Network",
        [
            "- Inverse network training requested, but PyTorch is unavailable.",
            f"- Missing torch report saved under `{out_dir}`.",
        ],
    )


def load_training_bundle(args: argparse.Namespace) -> dict[str, Any]:
    config = data_io.load_config(args.config)
    excluded = data_io.parse_excluded_samples(args.exclude_samples)
    groups = data_io.parse_groups(args.groups, config)
    crop = data_io.infer_crop_for_sample(config, args.sample, args.crop)
    bundle = data_io.load_os_sim_stack(config, args.sample, groups, crop, excluded_samples=excluded)
    Y = bundle["Y"].astype(np.float32)
    demod = demodulate_multigroup(Y)
    A_cls = demod["A"].astype(np.float32)
    D0 = Y.mean(axis=2).astype(np.float32)
    x_input = Y.reshape(Y.shape[0], Y.shape[1] * Y.shape[2], Y.shape[3], Y.shape[4]).astype(np.float32)
    if args.pattern_source == "measured_calibrated":
        pattern_rows = []
        infos = []
        for group_name in groups:
            patterns, info = measured_camera_patterns_for_group(
                config,
                group_name,
                crop=crop,
                sequence_stem=args.projection_sequence,
                calibration_label_prefix=args.calibration_label,
                mat_name=args.calibration_mat,
            )
            pattern_rows.append(patterns)
            info["source"] = "measured_calibrated"
            infos.append(info)
        M = np.stack(pattern_rows, axis=0).astype(np.float32)
    else:
        from src.forward_models import cos_patterns

        pattern_rows = [cos_patterns(Y.shape[-2:], group) for group in bundle["groups"]]
        M = np.stack(pattern_rows, axis=0).astype(np.float32)
        infos = [{"source": "cos_fixed", "group": name} for name in groups]
    grid_masks, grid_peaks = build_grid_masks(A_cls)
    grid_basis = []
    for peaks in grid_peaks:
        grid_basis.append(fixed_grid_basis(Y.shape[-2:], peaks, max_basis=args.grid_basis_peaks))
    max_basis = max((basis.shape[0] for basis in grid_basis), default=0)
    padded_basis = []
    for basis in grid_basis:
        if basis.shape[0] < max_basis:
            pad = np.zeros((max_basis - basis.shape[0], *Y.shape[-2:]), dtype=np.float32)
            basis = np.concatenate([basis, pad], axis=0)
        padded_basis.append(basis)
    grid_basis_np = np.stack(padded_basis, axis=0).astype(np.float32) if max_basis else np.zeros((Y.shape[1], 0, *Y.shape[-2:]), dtype=np.float32)
    return {
        "config": config,
        "groups": groups,
        "crop": crop,
        "Y": Y,
        "D0": D0,
        "x_input": x_input,
        "A_cls": A_cls,
        "A_cls_fused": fuse_hv(A_cls[:, 0], A_cls[:, 1] if A_cls.shape[1] > 1 else A_cls[:, 0]),
        "z_values": bundle["z_values"].astype(np.float32),
        "patterns": M,
        "pattern_infos": infos,
        "grid_masks": grid_masks,
        "grid_peaks": grid_peaks,
        "grid_basis": grid_basis_np,
    }


def soft_height_torch(A_fused: Any, z_values: Any, beta: float = 20.0) -> Any:
    import torch

    amin = A_fused.min(dim=0, keepdim=True).values
    amax = A_fused.max(dim=0, keepdim=True).values
    norm = (A_fused - amin) / (amax - amin + 1e-8)
    logits = beta * (norm - norm.max(dim=0, keepdim=True).values)
    probs = torch.softmax(logits, dim=0)
    return torch.sum(probs * z_values[:, None, None], dim=0)


def platform_loss_torch(height: Any, low_mask: Any, high_mask: Any) -> Any:
    if low_mask is None or high_mask is None:
        return height.new_tensor(0.0)
    vals_low = height[low_mask]
    vals_high = height[high_mask]
    if vals_low.numel() < 2 or vals_high.numel() < 2:
        return height.new_tensor(0.0)
    return vals_low.std() + vals_high.std()


def save_loss_plot(history: list[dict[str, float]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot([row["total"] for row in history], label="total")
    ax.plot([row["mod"] for row in history], label="mod")
    ax.plot([row["grid"] for row in history], label="grid")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Inverse light-section training")
    ax.grid(True, alpha=0.3)
    ax.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train inverse net to recover ideal OS-SIM light-section maps with a constrained forward model.")
    parser.add_argument("--config", default="configs/ossim_dataset.yaml")
    parser.add_argument("--sample", default="细台阶")
    parser.add_argument("--groups", default="ossim_t6_h_3step,ossim_t6_v_3step")
    parser.add_argument("--exclude-samples", default="台阶曝光1")
    parser.add_argument("--crop", default="center:128")
    parser.add_argument("--output", default="outputs/inverse_net/t6_hv")
    parser.add_argument("--pattern-source", choices=["measured_calibrated", "cos_fixed"], default="measured_calibrated")
    parser.add_argument("--projection-sequence", default="saomiao3_6")
    parser.add_argument("--calibration-label", default="3-6")
    parser.add_argument("--calibration-mat", default="DMD_CCD_Calibration_Dict.mat")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=0, help="0 means use all z layers in one batch.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--architecture", choices=["tiny_unet", "freq_res_unet", "freq_res_unet_tx"], default="tiny_unet")
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--freq-bins", type=int, default=8)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-window", type=int, default=8)
    parser.add_argument("--lambda-raw", type=float, default=0.05)
    parser.add_argument("--lambda-tv", type=float, default=0.01)
    parser.add_argument("--lambda-grid", type=float, default=0.05)
    parser.add_argument("--lambda-sec", type=float, default=0.05)
    parser.add_argument("--lambda-theta", type=float, default=0.001)
    parser.add_argument("--lambda-platform", type=float, default=0.02)
    parser.add_argument("--select-best", action="store_true", help="Restore the best epoch by a posthoc mod/grid/platform score before saving outputs.")
    parser.add_argument("--score-grid-weight", type=float, default=1.0)
    parser.add_argument("--score-platform-weight", type=float, default=0.5)
    parser.add_argument("--score-height-weight", type=float, default=0.1)
    parser.add_argument("--lowpass-kernel", type=int, default=9)
    parser.add_argument("--grid-basis-peaks", type=int, default=4)
    parser.add_argument("--platform-low-mask", default="outputs/height/baseline_t6_hv/platform_low_mask.npy")
    parser.add_argument("--platform-high-mask", default="outputs/height/baseline_t6_hv/platform_high_mask.npy")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = ROOT / args.output
    try:
        import torch

        from src.inverse_net import build_inverse_lightsection_net, count_parameters
        from src.torch_forward import RestrictedStripeForward, charbonnier_loss, fft_grid_penalty, lowpass_avg, tv_l1_torch
    except Exception as exc:
        write_missing_torch(out_dir, args, f"{type(exc).__name__}: {exc}")
        print(f"PyTorch unavailable; wrote missing_torch report to {out_dir}")
        return

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    bundle = load_training_bundle(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_io.write_json(out_dir / "config.json", vars(args))
    np.save(out_dir / "z_values.npy", bundle["z_values"])

    Y_np = bundle["Y"]
    A_cls_np = bundle["A_cls"]
    A_cls_fused_np = bundle["A_cls_fused"]
    x = torch.from_numpy(bundle["x_input"]).to(device)
    Y = torch.from_numpy(Y_np).to(device)
    D0 = torch.from_numpy(bundle["D0"]).to(device)
    A_cls = torch.from_numpy(A_cls_np).to(device)
    M = torch.from_numpy(bundle["patterns"]).to(device)
    grid_masks = torch.from_numpy(bundle["grid_masks"].astype(np.float32)).to(device)
    grid_basis = torch.from_numpy(bundle["grid_basis"]).to(device)
    z_t = torch.from_numpy(bundle["z_values"]).to(device)

    low_mask = high_mask = None
    low_path = ROOT / args.platform_low_mask
    high_path = ROOT / args.platform_high_mask
    if low_path.exists() and high_path.exists():
        low_np = np.load(low_path).astype(bool)
        high_np = np.load(high_path).astype(bool)
        if low_np.shape == tuple(Y_np.shape[-2:]):
            low_mask = torch.from_numpy(low_np).to(device)
            high_mask = torch.from_numpy(high_np).to(device)

    net = build_inverse_lightsection_net(
        in_channels=6,
        out_channels=2,
        base_channels=args.base_channels,
        architecture=args.architecture,
        freq_bins=args.freq_bins,
        transformer_heads=args.transformer_heads,
        transformer_window=args.transformer_window,
    ).to(device)
    forward_model = RestrictedStripeForward(M, grid_basis=grid_basis).to(device)
    optimizer = torch.optim.Adam(list(net.parameters()) + list(forward_model.parameters()), lr=args.lr)
    batch_size = x.shape[0] if args.batch_size <= 0 else int(args.batch_size)
    history: list[dict[str, float]] = []
    baseline_grid_for_score = grid_energy_mean(A_cls_np, bundle["grid_masks"])
    baseline_height_for_score = height_parabolic(A_cls_fused_np, bundle["z_values"])
    baseline_conf_for_score = confidence_metrics(A_cls_fused_np)
    baseline_height_summary = summarize_height(baseline_height_for_score, baseline_conf_for_score)
    baseline_platform_for_score = {"status": "not_run"}
    if low_path.exists() and high_path.exists():
        low_np_score = np.load(low_path).astype(bool)
        high_np_score = np.load(high_path).astype(bool)
        if low_np_score.shape == baseline_height_for_score.shape:
            baseline_platform_for_score = platform_metrics_from_masks(baseline_height_for_score, low_np_score, high_np_score)
    baseline_platform_rms = float(baseline_platform_for_score.get("pooled_platform_rms", 1.0) or 1.0)
    baseline_height_std = float(baseline_height_summary.get("height_std", 1.0) or 1.0)
    best_selection_score = float("inf")
    best_epoch = 0
    best_net_state = None
    best_forward_state = None
    selection_history: list[dict[str, float | int]] = []

    # Research-plan loss names: L_mod, L_raw, L_tv, L_grid_fft, L_weak_Acls, L_theta_reg, L_platform.
    for epoch in range(1, args.epochs + 1):
        order = torch.randperm(x.shape[0], device=device)
        total_rows = []
        for start in range(0, x.shape[0], batch_size):
            idx = order[start : start + batch_size]
            xb = x[idx]
            yb = Y[idx]
            d0b = D0[idx]
            a_cls_b = A_cls[idx]
            pred = net(xb)["A"]
            y_mod_hat = forward_model.predict_modulation(pred)
            y_hat = d0b[:, :, None] + y_mod_hat
            y_tilde = yb - yb.mean(dim=2, keepdim=True)
            loss_mod = charbonnier_loss(y_tilde - y_mod_hat)
            loss_raw = charbonnier_loss(yb - y_hat)
            loss_tv = tv_l1_torch(pred)
            loss_grid = fft_grid_penalty(pred, grid_masks)
            loss_sec = charbonnier_loss(lowpass_avg(pred, args.lowpass_kernel) - lowpass_avg(a_cls_b, args.lowpass_kernel))
            loss_theta = forward_model.theta_regularization()
            loss_platform = pred.new_tensor(0.0)
            if args.lambda_platform and idx.numel() == x.shape[0]:
                A_fused = torch.sqrt((pred[:, 0] ** 2 + pred[:, 1] ** 2) * 0.5 + 1e-8)
                loss_platform = platform_loss_torch(soft_height_torch(A_fused, z_t[idx]), low_mask, high_mask)
            loss = (
                loss_mod
                + args.lambda_raw * loss_raw
                + args.lambda_tv * loss_tv
                + args.lambda_grid * loss_grid
                + args.lambda_sec * loss_sec
                + args.lambda_theta * loss_theta
                + args.lambda_platform * loss_platform
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_rows.append(
                {
                    "total": float(loss.detach().cpu()),
                    "mod": float(loss_mod.detach().cpu()),
                    "raw": float(loss_raw.detach().cpu()),
                    "tv": float(loss_tv.detach().cpu()),
                    "grid": float(loss_grid.detach().cpu()),
                    "weak_acls": float(loss_sec.detach().cpu()),
                    "theta": float(loss_theta.detach().cpu()),
                    "platform": float(loss_platform.detach().cpu()),
                }
            )
        epoch_row = {key: float(np.mean([row[key] for row in total_rows])) for key in total_rows[0]}
        epoch_row["epoch"] = epoch
        history.append(epoch_row)
        if args.select_best:
            with torch.no_grad():
                A_eval_t = net(x)["A"]
                y_mod_eval = forward_model.predict_modulation(A_eval_t)
                select_mod = float(charbonnier_loss((Y - Y.mean(dim=2, keepdim=True)) - y_mod_eval).detach().cpu())
                A_eval_np = A_eval_t.detach().cpu().numpy().astype(np.float32)
            A_eval_fused = fuse_hv(A_eval_np[:, 0], A_eval_np[:, 1] if A_eval_np.shape[1] > 1 else A_eval_np[:, 0])
            h_eval = height_parabolic(A_eval_fused, bundle["z_values"])
            conf_eval = confidence_metrics(A_eval_fused)
            h_summary = summarize_height(h_eval, conf_eval)
            platform_eval = {"status": "not_run", "pooled_platform_rms": baseline_platform_rms}
            if low_path.exists() and high_path.exists():
                low_np_eval = np.load(low_path).astype(bool)
                high_np_eval = np.load(high_path).astype(bool)
                if low_np_eval.shape == h_eval.shape:
                    platform_eval = platform_metrics_from_masks(h_eval, low_np_eval, high_np_eval)
            select_grid = grid_energy_mean(A_eval_np, bundle["grid_masks"])
            platform_rms = float(platform_eval.get("pooled_platform_rms", baseline_platform_rms) or baseline_platform_rms)
            height_std = float(h_summary.get("height_std", baseline_height_std) or baseline_height_std)
            selection_score = (
                select_mod / (history[0]["mod"] + 1e-8)
                + args.score_grid_weight * select_grid / (baseline_grid_for_score + 1e-8)
                + args.score_platform_weight * platform_rms / (baseline_platform_rms + 1e-8)
                + args.score_height_weight * height_std / (baseline_height_std + 1e-8)
            )
            selection_row = {
                "epoch": epoch,
                "score": float(selection_score),
                "mod": select_mod,
                "grid": float(select_grid),
                "platform_rms": platform_rms,
                "height_std": height_std,
            }
            selection_history.append(selection_row)
            if selection_score < best_selection_score:
                best_selection_score = float(selection_score)
                best_epoch = epoch
                best_net_state = {key: value.detach().cpu().clone() for key, value in net.state_dict().items()}
                best_forward_state = {key: value.detach().cpu().clone() for key, value in forward_model.state_dict().items()}
        print(f"epoch {epoch:03d} total={epoch_row['total']:.6g} mod={epoch_row['mod']:.6g} grid={epoch_row['grid']:.6g}")

    if args.select_best and best_net_state is not None and best_forward_state is not None:
        net.load_state_dict({key: value.to(device) for key, value in best_net_state.items()})
        forward_model.load_state_dict({key: value.to(device) for key, value in best_forward_state.items()})

    net.eval()
    forward_model.eval()
    with torch.no_grad():
        A_pred = net(x)["A"]
        y_mod_hat = forward_model.predict_modulation(A_pred)
        y_hat = D0[:, :, None] + y_mod_hat
        final_mod_loss = float(charbonnier_loss((Y - Y.mean(dim=2, keepdim=True)) - y_mod_hat).cpu())
        final_raw_loss = float(charbonnier_loss(Y - y_hat).cpu())
        A_pred_np = A_pred.detach().cpu().numpy().astype(np.float32)
        y_hat_np = y_hat.detach().cpu().numpy().astype(np.float32)

    A_h = A_pred_np[:, 0]
    A_v = A_pred_np[:, 1]
    A_fused = fuse_hv(A_h, A_v)
    height = height_parabolic(A_fused, bundle["z_values"])
    conf = confidence_metrics(A_fused)
    np.save(out_dir / "A_h_pred_stack.npy", A_h)
    np.save(out_dir / "A_v_pred_stack.npy", A_v)
    np.save(out_dir / "A_fused_pred_stack.npy", A_fused)
    np.save(out_dir / "height_pred.npy", height)
    np.save(out_dir / "Y_hat.npy", y_hat_np)
    torch.save({"model": net.state_dict(), "theta": forward_model.state_dict(), "args": vars(args)}, out_dir / "model.pt")

    baseline_mod_loss = modulation_loss(Y_np, A_cls_np, bundle["patterns"])
    pred_mod_loss = modulation_loss(Y_np, A_pred_np, bundle["patterns"])
    baseline_grid = grid_energy_mean(A_cls_np, bundle["grid_masks"])
    pred_grid = grid_energy_mean(A_pred_np, bundle["grid_masks"])
    initial_epoch_mod_loss = history[0]["mod"] if history else None
    modulation_loss_decreased = bool(initial_epoch_mod_loss is not None and final_mod_loss < initial_epoch_mod_loss)
    height_metrics = summarize_height(height, conf)
    platform = {"status": "not_run"}
    if low_path.exists() and high_path.exists():
        low_np = np.load(low_path).astype(bool)
        high_np = np.load(high_path).astype(bool)
        if low_np.shape == height.shape:
            platform = platform_metrics_from_masks(height, low_np, high_np)

    save_loss_plot(history, out_dir / "loss_curve.png")
    mid = A_fused.shape[0] // 2
    save_scalar_image(height, out_dir / "height_pred.png", "inverse_net height from A_fused")
    save_montage([A_cls_fused_np[mid], A_fused[mid], A_fused[mid] - A_cls_fused_np[mid]], out_dir / "A_comparison.png", ["A_cls fused", "A_pred fused", "A_pred - A_cls"], cols=3, cmap="viridis")
    baseline_height = height_parabolic(A_cls_fused_np, bundle["z_values"])
    save_montage([baseline_height, height, height - baseline_height], out_dir / "height_compare.png", ["baseline height", "inverse_net height", "delta"], cols=3, cmap="viridis")
    save_montage(
        [
            Y_np[mid, 0, 0],
            y_hat_np[mid, 0, 0],
            Y_np[mid, 0, 0] - y_hat_np[mid, 0, 0],
            Y_np[mid, 1, 0],
            y_hat_np[mid, 1, 0],
            Y_np[mid, 1, 0] - y_hat_np[mid, 1, 0],
        ],
        out_dir / "reprojection_examples.png",
        ["Y H phase0", "Yhat H phase0", "res H", "Y V phase0", "Yhat V phase0", "res V"],
        cols=3,
        cmap="gray",
    )

    metrics = {
        "status": "completed",
        "sample": args.sample,
        "groups": bundle["groups"],
        "crop": bundle["crop"].label if bundle["crop"] else "full",
        "model": net.__class__.__name__,
        "architecture": args.architecture,
        "parameter_count": count_parameters(net),
        "pattern_source": args.pattern_source,
        "pattern_infos": bundle["pattern_infos"],
        "device": str(device),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "epochs": args.epochs,
        "history": history,
        "select_best": bool(args.select_best),
        "selection_history": selection_history,
        "best_epoch": best_epoch if args.select_best else args.epochs,
        "best_selection_score": best_selection_score if args.select_best else None,
        "score_grid_weight": args.score_grid_weight,
        "score_platform_weight": args.score_platform_weight,
        "score_height_weight": args.score_height_weight,
        "initial_epoch_mod_loss": initial_epoch_mod_loss,
        "baseline_modulation_loss": baseline_mod_loss,
        "final_modulation_loss": final_mod_loss,
        "modulation_loss_decreased": modulation_loss_decreased,
        "final_raw_loss": final_raw_loss,
        "posthoc_modulation_loss": pred_mod_loss,
        "baseline_grid_energy": baseline_grid,
        "pred_grid_energy": pred_grid,
        "grid_energy_improved": bool(pred_grid < baseline_grid),
        "height": height_metrics,
        "platform": {k: v for k, v in platform.items() if k not in {"low_mask", "high_mask"}},
        "theta": forward_model.theta_summary(),
        "outputs": {
            "A_h_pred_stack": str(out_dir / "A_h_pred_stack.npy"),
            "A_v_pred_stack": str(out_dir / "A_v_pred_stack.npy"),
            "A_fused_pred_stack": str(out_dir / "A_fused_pred_stack.npy"),
            "height_pred": str(out_dir / "height_pred.npy"),
        },
    }
    data_io.write_json(out_dir / "metrics.json", metrics)
    report = [
        "# Inverse Light-Section Network Report",
        "",
        "Status: `completed`",
        "",
        f"- Sample: `{args.sample}`",
        f"- Crop: `{metrics['crop']}`",
        f"- Pattern source: `{args.pattern_source}`",
        f"- Architecture: `{args.architecture}`",
        f"- Parameter count: `{metrics['parameter_count']}`",
        f"- Best epoch: `{metrics['best_epoch']}`",
        f"- Baseline modulation loss: `{baseline_mod_loss:.6g}`",
        f"- Final modulation loss: `{final_mod_loss:.6g}`",
        f"- Baseline grid energy: `{baseline_grid:.6g}`",
        f"- Pred grid energy: `{pred_grid:.6g}`",
        f"- Height std: `{height_metrics['height_std']:.6g}`",
        f"- Platform status: `{platform.get('status')}`",
        "",
        "The network predicts A_h/A_v only. Height is read out afterward from A_fused; the network does not directly predict height.",
    ]
    (out_dir / "inverse_net_report.md").write_text("\n".join(report), encoding="utf-8")
    data_io.append_report(
        ROOT / "outputs/nightly_report.md",
        "P18 Inverse Light-Section Network",
        [
            f"- Trained inverse net for `{args.sample}` on `{metrics['crop']}`.",
            f"- Final modulation loss `{final_mod_loss:.6g}`; grid energy `{pred_grid:.6g}`.",
            f"- Outputs saved under `{out_dir}`.",
        ],
    )
    print(f"Wrote inverse network outputs to {out_dir}")


if __name__ == "__main__":
    main()
