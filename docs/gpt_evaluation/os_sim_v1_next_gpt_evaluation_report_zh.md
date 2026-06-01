# OS-SIM v1_next GPT 评审报告（中文版）

这份报告用于交给外部 GPT 或技术评审者，评价当前 OS-SIM v1_next 研究是否严谨、结果是否支持结论。

## 评审重点

请重点审查：

1. 三步 OS-SIM 解调公式是否在假设下成立。
2. H/V 相位顺序验证是否充分。
3. V 方向 forward model 异常是否已定位。
4. `A_cls`、`A_ls`、latent、simple filters、existing network 的比较是否公平。
5. grid/residual/height 指标是否统一。
6. 在没有独立真值时，是否避免了绝对计量精度声明。

## 研究设置

- 样品：`细台阶`
- crop：`center:256`
- 主 frame groups：T6 H `100-102`，T6 V `103-105`
- z step：`0.5 micrometer`
- 本轮没有训练新网络
- 研究目标：验证 measured-calibrated `M_theta`、固定模板 `A_ls`、连续轴向调制度响应、confidence mask，而不是继续优化“更平”的高度图

## 三步 OS-SIM 解调公式

理想模型：

```text
I1 = D0 + A cos(phi)
I2 = D0 + A cos(phi + 2*pi/3)
I3 = D0 + A cos(phi + 4*pi/3)

D0 = (I1 + I2 + I3) / 3
Dc = (2*I1 - I2 - I3) / 3
Ds = (I3 - I2) / sqrt(3)
A  = sqrt(Dc^2 + Ds^2)
```

该公式在三帧相位确实为 0/120/240 度、顺序正确、近似正弦、曝光稳定时成立。measured-template LS 对相位顺序更敏感，因此仍必须做 phase-order 验证。

## 核心方法和结果

| 方法 | 含义 | Modulation loss | Fused grid | Height std | H/V median diff |
|---|---|---:|---:|---:|---:|
| `A_cls_traditional_3step` | 原始 OS-SIM baseline | 0.06152 | 0.88851 | 2.93635 | 0.97205 |
| `A_ls_nonnegative_measured_template` | 默认 measured-template LS | 0.03354 | 0.62218 | 3.06324 | 2.76922 |
| `A_ls_nonnegative_phase_aligned_V` | V 相位修正后的 LS | **0.02719** | 0.83281 | 2.90407 | **0.73479** |
| `latent_no_platform` | 旧 latent 优化 | 0.03357 | 0.62320 | 3.05567 | 2.70951 |
| `existing_network_fused_only` | 已保存网络输出 | n/a | 0.84112 | **2.84224** | n/a |

解释：

- `A_cls` 是原始算法结果，不是真值。
- `A_ls_nonnegative_phase_aligned_V` 是当前最强 modulation-domain baseline。
- existing network 只有 fused output，不能公平比较 modulation residual；更低 height std 可能只是平滑。

## Phase Order 结果

- T6-H 默认 `[0,1,2]` 是最佳。
- T6-V 默认不是最佳。
- T6-V 最佳 raw permutation 是 `[2,0,1]`，等价 measured-template permutation `[1,2,0]`。
- H/V median height difference 从 `2.769` 降到 `0.735`。

结论：V 方向异常首先是 phase alignment / phase zero 问题。

## Forward Model 结果

| Group | 最佳模型 | Loss | 参数 |
|---|---|---:|---|
| T6-H | measured + blur | 0.02186 | `sigma=1.2` |
| T6-V | measured + phase offset + affine | 0.02768 | `perm=[1,2,0], dy=1, dx=1` |

结论：H 方向 measured pattern 基本可用；V 方向需要相位偏移和小映射修正。

## LS Baseline 结果

- signed LS loss：`0.03018`
- nonnegative LS loss：`0.03354`
- signed LS 负幅值比例：`38.19%`
- 当前最佳形式：V 相位修正后的 nonnegative LS，loss `0.02719`

signed LS 负值过多，不能直接作为物理调制度图；nonnegative LS 更适合作正式 baseline。

## 轴向响应结果

| Pair | Mean FWHM | Peak sharpness | Peak/background |
|---|---:|---:|---:|
| T6 3-step | **4.226 um** | **0.03123** | **3.0563** |
| T12 3-step | 6.772 um | 0.01683 | 2.1449 |
| T12 6-step | 6.809 um | 0.01571 | 2.1439 |

T6 更窄、更尖，支持连续轴向调制度响应解释。由于缺少 NA、wavelength、照明几何，不能声称与 Stokseth 理论定量吻合。

## Confidence Mask 结果

- valid single peak：`52.43%`
- invalid fraction：`47.57%`

主要无效原因：

| Reason | Fraction |
|---|---:|
| broad peak | 10.68% |
| double peak | 12.23% |
| H/V inconsistent | 13.14% |
| high residual | 9.70% |
| saturated/invalid raw | 1.66% |

结论：confidence mask 已有区分度，不能再把 invalid fraction 当作 0。

## Metrology 限制

没有 nominal step height 或独立标准件真值。因此：

- 不能报告 absolute error。
- 不能声明绝对计量精度提升。
- network 更低 height std 不能证明更准。
- `A_ls` 改善的是 modulation-domain fit，不是已证明的绝对高度精度。

confidence-masked height std：

| Method | Full height std | Masked height std |
|---|---:|---:|
| `A_cls` | 2.93635 | 2.84982 |
| `A_ls_nonnegative` | 3.06324 | 2.87374 |
| `A_ls_signed` | 3.10816 | 2.88756 |
| `latent_no_platform` | 3.05567 | 2.87237 |
| `existing_network` | 2.84224 | 2.78542 |

## 已支持的结论

1. 真实数据支持连续轴向调制度响应。
2. 当前 crop 上 T6 比 T12 的轴向响应更窄、更尖。
3. 固定 measured-calibrated `M_theta` 的 LS 是强 baseline。
4. V 方向默认相位顺序不可靠，需要相位对齐。
5. confidence/invalid mask 必须进入后续分析。

## 尚未支持的结论

1. 绝对高度精度提升。
2. network 优于 `A_ls`。
3. 更低 height std 等于更好 metrology。
4. 与 Stokseth 理论定量吻合。
5. forward model 已完全解释 residual FFT 结构。
6. 结果已泛化到其他 crop 或其他样品。

## 建议 GPT 重点质疑

1. phase-order 结论是否需要 held-out crop 验证。
2. signed LS 高负值比例是否说明模板/相位仍有深层 mismatch。
3. modulation loss 降低但 residual-grid 增加时，如何解释。
4. existing network 是否只是 smoothing。
5. 没有独立真值时，哪些 metrology 指标可以写，哪些不能写。

## 下一阶段关键实验

1. 在 held-out crop 和其他完整样品上重跑 phase-aligned `A_ls`。
2. 独立校准 V phase offset 和 affine shift。
3. 获取标准件高度或重复扫描数据。
4. 只有当 `A_ls` 和简单滤波在 held-out 验证上不够用，且网络真正降低 residual FFT 并不过度平滑时，才恢复网络训练。

## 相关文件

- 本地完整中文版：`outputs/gpt_evaluation/os_sim_v1_next_gpt_evaluation_report_zh.md`
- 英文评审版：`outputs/gpt_evaluation/os_sim_v1_next_gpt_evaluation_report.md`
- 主结果报告：`outputs/v1_next/v1_next_report.md`
- GitHub 图文报告：`docs/os_sim_v1_next_report.md`

最终判断：当前主线应从网络训练转向 phase-aligned、fixed-template、nonnegative LS projection。最强证据是 modulation-domain residual 下降，而不是高度图变平。
