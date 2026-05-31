# Inverse-Net Architecture Comparison

## Setup

- Sample: `细台阶`
- Groups: `ossim_t6_h_3step`, `ossim_t6_v_3step`
- Pattern source: `measured_calibrated`
- GPU: `NVIDIA GeForce RTX 3070`
- Torch: `2.11.0+cu128`
- 128 smoke setting: `center:128`, `20` epochs, same seed and loss weights.
- 256 validation setting: `center:256`, `50` epochs.

## 128 Smoke Results

| architecture | params | final L_mod | eval grid energy | platform RMS | height std |
|---|---:|---:|---:|---:|---:|
| `tiny_unet` | 267,170 | 0.374168 | 0.670558 | 0.072107 | 3.218117 |
| `freq_res_unet` | 458,354 | 0.183290 | 0.624063 | 0.107011 | 2.898892 |
| `freq_res_unet_tx` | 533,138 | 0.057210 | 0.461612 | 0.063917 | 3.025939 |

Baseline 128 reference: grid energy `0.645377`, platform RMS `0.250023`.

## 256 Validation

| architecture | params | final L_mod | eval grid energy | platform RMS | height std |
|---|---:|---:|---:|---:|---:|
| previous `tiny_unet` CUDA | 267,170 | 0.270700 | 0.860322 | 0.035572 | 2.860825 |
| `freq_res_unet_tx` | 533,138 | 0.048258 | 0.854872 | 0.075568 | 2.839839 |
| `freq_res_unet_tx` + `--select-best` | 533,138 | 0.048277 | 0.841115 | 0.078964 | 2.842237 |

Baseline 256 reference: grid energy `0.888514`, platform RMS `0.308210`.
Measured platform-opt reference: grid energy `0.889704`, platform RMS `0.341849`.

## Recommendation

Use `freq_res_unet_tx` as the next main research architecture. It gives the best
inverse-process fit and grid suppression in the controlled 128 sweep, and it
generalizes to the 256 crop with much lower modulation loss than the previous
Tiny U-Net while keeping platform RMS far below the baseline. For the de-grid
priority, use the `--select-best` checkpoint result because it improves 256 grid
energy from `0.854872` to `0.841115` with only a small platform RMS increase
from `0.075568` to `0.078964`.

Keep `tiny_unet` as a conservative platform-flatness baseline because its 256
platform RMS is slightly lower, but it underfits the inverse reprojection task.

The follow-up loss sweep is recorded in
`outputs/inverse_net_sweep/sweep_summary.md`. The current recommended setting is
`lambda_platform=0.15`, `lambda_sec=0.12`, `lambda_grid=0.03`,
`--select-best`, and CUDA execution.
