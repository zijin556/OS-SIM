# OS-SIM v1_next GPT Evaluation Report

This GitHub-readable report is a compact review packet for evaluating the current OS-SIM v1_next research state. The full local version is at `outputs/gpt_evaluation/os_sim_v1_next_gpt_evaluation_report.md`.

## Reviewer Task

Please evaluate whether the reported OS-SIM claims are supported by the experiments. Focus on formula validity, phase-order evidence, forward-model diagnosis, LS/network comparison fairness, metric consistency, and whether metrology claims are overstated.

## Study Scope

- Dataset: `细台阶`, crop `center:256`
- Main frames: T6 H `100-102`, T6 V `103-105`
- z step: `0.5 micrometer`
- No new network training in this round
- Main objective: verify measured-calibrated `M_theta`, fixed-template `A_ls`, continuous axial modulation response, and confidence masks before any further network work

## Main Methods

1. Audit current code/report metric definitions.
2. Enumerate H/V phase permutations, contrast inversion, and phase sign flip.
3. Diagnose H/V forward models: ideal cos, measured pattern, phase offset, q correction, affine shift, binary DMD, harmonic terms, blur.
4. Recompute metrics with one definition in `src/metrics_v1.py`.
5. Build signed, nonnegative, and cos/sin LS baselines.
6. Convert FWHM from layers to micrometers.
7. Build confidence/invalid/reason maps.
8. Report metrology only as internal consistency because no nominal truth exists.

## Three-Step OS-SIM Formula

For ideal 3-step phases:

```text
I1 = D0 + A cos(phi)
I2 = D0 + A cos(phi + 2*pi/3)
I3 = D0 + A cos(phi + 4*pi/3)

D0 = (I1 + I2 + I3) / 3
Dc = (2*I1 - I2 - I3) / 3
Ds = (I3 - I2) / sqrt(3)
A  = sqrt(Dc^2 + Ds^2)
```

This is valid under the ideal sinusoidal/equal-phase assumptions. The measured-template LS path is more sensitive to phase order and template mismatch, so phase-order validation is still required.

## Key Results

| Method | Meaning | Modulation loss | Fused grid | Height std | H/V median diff |
|---|---|---:|---:|---:|---:|
| `A_cls_traditional_3step` | original OS-SIM baseline | 0.06152 | 0.88851 | 2.93635 | 0.97205 |
| `A_ls_nonnegative_measured_template` | default measured-template LS | 0.03354 | 0.62218 | 3.06324 | 2.76922 |
| `A_ls_nonnegative_phase_aligned_V` | phase-aligned V LS | **0.02719** | 0.83281 | 2.90407 | **0.73479** |
| `latent_no_platform` | old latent optimization | 0.03357 | 0.62320 | 3.05567 | 2.70951 |
| `existing_network_fused_only` | saved network output | n/a | 0.84112 | **2.84224** | n/a |

## Phase Order Result

- T6-H default `[0,1,2]` is best.
- T6-V default is not best.
- T6-V best raw permutation is `[2,0,1]`, equivalent to measured-template permutation `[1,2,0]`.
- H/V median height difference improves from `2.769` to `0.735`.

## Forward Model Result

| Group | Best model | Loss | Key parameter |
|---|---|---:|---|
| T6-H | measured + blur | 0.02186 | `sigma=1.2` |
| T6-V | measured + phase offset + affine | 0.02768 | `perm=[1,2,0], dy=1, dx=1` |

Interpretation: V is mainly a phase-alignment/phase-zero problem, with secondary mapping shift.

## LS Baseline Result

- Signed LS loss: `0.03018`
- Nonnegative LS loss: `0.03354`
- Signed LS negative amplitude fraction: `38.19%`
- Best current LS form: nonnegative LS with V phase alignment, loss `0.02719`

Signed LS is not physically clean because many pixels are negative.

## Axial Response Result

| Pair | Mean FWHM | Peak sharpness | Peak/background |
|---|---:|---:|---:|
| T6 3-step | **4.226 um** | **0.03123** | **3.0563** |
| T12 3-step | 6.772 um | 0.01683 | 2.1449 |
| T12 6-step | 6.809 um | 0.01571 | 2.1439 |

T6 is narrower and sharper, supporting continuous axial modulation response. Quantitative Stokseth agreement is not claimed because NA/wavelength/illumination geometry are missing.

## Confidence Mask Result

- Valid single peak: `52.43%`
- Invalid fraction: `47.57%`
- Main invalid reasons: broad peak `10.68%`, double peak `12.23%`, H/V inconsistent `13.14%`, high residual `9.70%`, saturated/invalid raw `1.66%`

## Metrology Constraint

No nominal step height or independent standard was found. Therefore:

- Absolute height error is not reported.
- Network lower height std is not proof of better metrology.
- `A_ls` improves modulation fit, not proven absolute accuracy.

## Supported Claims

1. Real data support continuous axial modulation response.
2. T6 gives narrower/sharper response than T12 on this crop.
3. Fixed measured-calibrated `M_theta` LS is a strong non-network baseline.
4. V direction requires phase alignment.
5. Confidence masks are necessary and nontrivial.

## Unsupported Claims

1. Absolute metrology improvement.
2. Network superiority over `A_ls`.
3. Stokseth quantitative agreement.
4. Complete forward-model explanation of residual FFT structure.
5. Generalization beyond this crop/sample.

## Reviewer Questions

1. Is the phase-order conclusion robust enough without held-out crop validation?
2. Does the high signed-LS negative fraction imply deeper physical mismatch?
3. Should residual-grid increase under lower modulation loss be treated as a model failure?
4. Is phase-aligned `A_ls` already the correct main baseline?
5. What independent metrology evidence is required before publishing accuracy claims?

## Next Experiments

1. Re-run phase-aligned `A_ls` on held-out crops and other complete samples.
2. Independently calibrate V phase offset and affine shift.
3. Acquire standard step height or repeat scans.
4. Only resume network training if it beats `A_ls` and simple filters on held-out residual FFT and confidence-masked metrics without oversmoothing.
