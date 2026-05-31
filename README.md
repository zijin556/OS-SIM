# PhysicsDeepLearning OSSIM

OS-SIM reflective 3D morphology reconstruction research prototype.

This repository contains a v0 neural inverse-process framework for recovering H/V light-section modulation maps from real six-frame OS-SIM stripe scans, using a constrained differentiable forward model for low-dimensional degradation residuals.

## Current Research Goal

The current goal is not direct height prediction. The v0 pipeline is:

```text
real degraded six-frame stripe sequence
  -> neural inverse network
  -> A_h and A_v light-section modulation maps
  -> H/V fused A stack
  -> z-axis peak height readout
```

The forward consistency model is:

```text
Y_hat = D0 + A_pred * M_theta
```

where `M_theta` starts from measured/calibrated stripe patterns and only allows constrained low-dimensional residual terms.

## Data Policy

Raw data is not included in this repository.

The local dataset used for v0 experiments is:

```text
E:\research\AIResearch\程序\01_raw_data\dlp6500_sim_confocal\scans_raw\3-6扫描
```

Experiment policy:

- Exclude `台阶曝光1` because it is incomplete.
- First-round main sample: `细台阶`.
- Main frame groups:
  - H: frames `100,101,102`
  - V: frames `103,104,105`
- Complete samples reserved for follow-up validation:
  - `台阶曝光2`
  - `回型曝光20000`
  - `小孔`
  - `细台阶`

## v0 Model

Main architecture:

```text
freq_res_unet_tx
```

Components:

- Residual U-Net
- Frequency FiLM bottleneck conditioning
- Window transformer block
- Softplus nonnegative output for `A_h/A_v`

Core files:

```text
src/inverse_net.py
src/torch_forward.py
scripts/train_inverse_lightsection_net.py
scripts/evaluate_inverse_lightsection_net.py
```

## Degradation Parameters

The current `theta` is a constrained low-dimensional residual, not a complete physical calibration model.

Current terms:

- `delta_q`: stripe frequency/direction residual
- `phase_offsets`: 3-step phase residual
- `affine_residual`: small global residual affine warp after calibration
- `blur_sigma`: simplified global blur
- `grid_coeff`: fixed grid/Moiré basis coefficients

The v0 result should not be interpreted as complete optical aberration recovery or complete DMD-CCD installation-error recovery.

## v0 Best Result

Best current run:

```text
sample: 细台阶
crop: center:256
architecture: freq_res_unet_tx
lambda_platform: 0.15
lambda_sec: 0.12
lambda_grid: 0.03
select_best: True
device: CUDA
```

Compared with classical baseline `A_cls`:

| Metric | Baseline A_cls | v0 inverse net | Change |
|---|---:|---:|---:|
| grid energy | 0.888514 | 0.841115 | -5.33% |
| platform RMS | 0.308210 | 0.078964 | -74.38% |
| modulation loss | 0.061518 | 0.048277 | -21.52% |

## Current Limitations

- No independent calibrated height ground truth.
- No cross-sample validation yet.
- `theta` is not uniquely identifiable without additional constraints or validation.
- Moiré/grid energy is reduced but not eliminated.
- Current `--select-best` is selected on the same sample/crop, so a proper validation split is needed.

## Recommended v1 Work

1. Cross-sample validation.
2. Theta ablation.
3. Loss ablation.
4. z-aware 2.5D model.
5. B-spline/TPS residual DMD-CCD warp.
6. More physical aberration model, such as Zernike or spatially varying PSF.
7. Unified grid-energy metric for H, V, and fused outputs.

## Key Documents

```text
docs/new_conversation_handoff_v0.md
docs/research_v0_baseline.md
docs/github_upload_plan_v0.md
outputs/review_for_gpt_pro_portable.md
outputs/theoretical_model_for_gpt_pro_visual.md
```

## GitHub Upload Notes

Do not commit raw data, virtual environments, model weights, `.npy` arrays, or large archive files. See:

```text
docs/github_upload_plan_v0.md
```

Recommended v0 tag:

```text
v0-neural-inverse-prototype
```
