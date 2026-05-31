# OS-SIM 神经逆向过程自包含审查报告（给 GPT Pro）

这是一份可直接粘贴或上传给 GPT Pro 的自包含报告。报告中的本地路径只作为项目来源说明，不要求 GPT Pro 访问本地磁盘。若需要审查代码和图片，请同时上传同目录下的 `gpt_pro_review_bundle.zip`。

生成时间：2026-05-31  
研究任务：OS-SIM 反射式三维形貌重建中的神经逆向过程、理想光切片恢复和退化参数估计。

---

## 1. 我们要解决的问题

目标不是只做传统三步解调，而是建立一个神经逆向过程：

```text
真实退化条纹帧 Y_real
  -> 神经网络恢复理想化光切片调制度 A_h, A_v
  -> H/V 融合得到 A_fused
  -> 沿 z 方向峰值读出高度 height
```

同时希望估计退化参数，例如条纹频率/方向残差、相位偏移、DMD-camera 安装/映射残差、模糊或像差、残余网格/Moiré 项。

当前实现不是直接输出高度，而是输出 H/V 两个方向的光切片调制度：

```text
input:  Y_real[z, 6, H, W]
output: A_h_pred[z, H, W], A_v_pred[z, H, W]

A_fused = sqrt((A_h_pred^2 + A_v_pred^2) / 2)
height  = arg/peak readout along z from A_fused_stack
```

---

## 2. 数据和约束

使用样本策略：

- 排除 `台阶曝光1`，因为该样本缺帧。
- 第一轮主样本使用 `细台阶`。
- 后续可完整对比样本：`台阶曝光2`、`回型曝光20000`、`小孔`、`细台阶`。

主线帧组：

```text
T6 H/V 3-step
水平 H：frame 100, 101, 102
竖直 V：frame 103, 104, 105
```

当前主实验：

```text
sample: 细台阶
z layers: 31
crop: center 256x256
input channels per z: 6
output channels per z: 2
```

当前尚未做跨样本泛化验证。

---

## 3. 当前实现的核心思想

### 3.1 神经逆向网络

网络学习：

```text
Y_real[z, 6, H, W] -> A_pred[z, 2, H, W]
```

其中输出两通道：

```text
A_pred[:, 0] = A_h
A_pred[:, 1] = A_v
```

输出使用 `softplus`，使调制度非负。

### 3.2 可微 forward model

为了避免网络只生成看起来平滑但不物理的结果，引入受限 forward model：

```text
Y_hat = D0 + A_pred * M_theta
```

其中：

- `D0` 是每组三步条纹的平均背景。
- `A_pred` 是网络预测的光切片调制度。
- `M_theta` 是从实测/标定条纹模式出发，加低维可学习残差得到的退化调制模式。

这意味着网络输出必须能通过 `M_theta` 重投影回真实 raw frames。

---

## 4. 网络结构

共实现 3 个结构。

| architecture | parameters | description |
|---|---:|---|
| `tiny_unet` | 267,170 | 初版小 U-Net |
| `freq_res_unet` | 458,354 | Residual U-Net + frequency FiLM |
| `freq_res_unet_tx` | 533,138 | Residual U-Net + frequency FiLM + window transformer |

当前推荐使用：

```text
freq_res_unet_tx
```

### 4.1 tiny_unet

结构概要：

```text
6-channel input
  -> ConvBlock encoder
  -> downsample
  -> ConvBlock
  -> bottleneck
  -> upsample + skip
  -> upsample + skip
  -> 2-channel output
  -> softplus
```

问题：表达力不足，调制重投影 loss 明显偏高。

### 4.2 freq_res_unet

改进：

- 使用 residual blocks。
- bottleneck 使用 dilation residual block。
- 加入 `FrequencyFiLM`。

`FrequencyFiLM` 做法：

```text
raw input -> FFT -> radial frequency energies + H/V high-frequency energies
           -> MLP -> gamma/beta
           -> modulate bottleneck feature
```

目的：让网络对条纹/Moiré 残余的频域结构更敏感。

### 4.3 freq_res_unet_tx

进一步加入 window transformer：

```text
6-channel input
  -> residual encoder
  -> residual bottleneck
  -> FrequencyFiLM
  -> WindowTransformerBlock
  -> residual decoder
  -> softplus output A_h/A_v
```

动机：

- 卷积适合局部纹理。
- Moiré 和网格残余具有周期性和较长范围相关。
- window attention 可以补充局部窗口内非卷积关系。

---

## 5. 退化参数 theta

当前 forward model 中的 `theta` 是低维、受限残差参数，不是完整任意退化场。

包含：

| parameter | meaning | current limitation |
|---|---|---|
| `delta_q` | 条纹空间频率/方向残差 | 只能表达小的全局频率/方向变化 |
| `phase_offsets` | 三步相位偏移 | 当前学到值接近 0，可能不可辨识 |
| `affine_residual` | DMD-camera 标定后的全局小仿射残差 | 不能表达非线性安装误差或镜头畸变 |
| `blur_sigma` | 全局简化 blur | 不能代表真实复杂像差 |
| `grid_coeff` | 固定 grid/Moiré basis 系数 | 当前系数很小，作用需消融验证 |

当前最佳 run 的 theta 摘要：

```text
delta_q:
  H: [ 0.0004836913, 0.0006069596]
  V: [-0.0006686280, 0.0005879356]

phase_offsets:
  H/V: about 1e-22 to 1e-21, almost zero

affine_residual before scale:
  H:
    [-0.017351,  0.004581,  0.034887]
    [-0.033200,  0.017243, -0.014514]
  V:
    [-0.034662,  0.022851,  0.038467]
    [ 0.010508,  0.009356,  0.028067]

affine_residual_scale = 0.02

blur_sigma:
  H: 0.0064557
  V: 0.0064557

grid_coeff:
  mostly around 1e-6 to 1e-8
```

严格解释：

```text
当前可以说已经估计出低维退化残差 theta。
不能说已经完整恢复了真实像差或完整 DMD-CCD 安装误差。
```

如果研究主张要包括像差和安装误差，后续必须加入更强物理模型，例如 Zernike、空间变化 PSF、B-spline/TPS residual warp、分块 affine 或 field-dependent phase/frequency。

---

## 6. 训练目标

训练 loss：

```text
L_total =
  L_mod
+ lambda_raw      * L_raw
+ lambda_tv       * L_tv
+ lambda_grid     * L_grid_fft
+ lambda_sec      * L_weak_Acls
+ lambda_theta    * L_theta_reg
+ lambda_platform * L_platform
```

各项含义：

- `L_mod`：真实条纹去均值后，与 `A_pred * M_theta` 的调制项匹配。
- `L_raw`：完整 raw frame 重投影误差。
- `L_tv`：光切片图 TV 平滑。
- `L_grid_fft`：抑制已检测网格/Moiré 频段能量。
- `L_weak_Acls`：低通后的传统三步解调结果作为弱先验。
- `L_theta_reg`：约束退化参数不要发散。
- `L_platform`：使用台阶平台 mask，让同一平台高度更平坦。

当前推荐权重：

```text
lambda_platform = 0.15
lambda_sec      = 0.12
lambda_grid     = 0.03
select_best     = True
```

---

## 7. 当前最佳命令

训练：

```powershell
python scripts/train_inverse_lightsection_net.py `
  --sample 细台阶 `
  --groups ossim_t6_h_3step,ossim_t6_v_3step `
  --crop center:256 `
  --epochs 50 `
  --architecture freq_res_unet_tx `
  --pattern-source measured_calibrated `
  --lambda-platform 0.15 `
  --lambda-sec 0.12 `
  --lambda-grid 0.03 `
  --select-best `
  --device cuda
```

评估：

```powershell
python scripts/evaluate_inverse_lightsection_net.py `
  --input A_fused_pred_stack.npy `
  --a-h A_h_pred_stack.npy `
  --a-v A_v_pred_stack.npy `
  --z-values z_values.npy `
  --baseline baseline_A_fused_stack.npy `
  --latent latent_platform_opt_A_fused_stack.npy `
  --platform-low-mask platform_low_mask.npy `
  --platform-high-mask platform_high_mask.npy
```

---

## 8. 输出结果

当前最佳结果文件包括：

```text
A_h_pred_stack.npy
A_v_pred_stack.npy
A_fused_pred_stack.npy
height_pred.npy
height_pred.png
height_compare.png
A_comparison.png
reprojection_examples.png
loss_curve.png
metrics.json
evaluation_metrics.json
model.pt
z_values.npy
```

形状：

```text
A_h_pred_stack:     (31, 256, 256)
A_v_pred_stack:     (31, 256, 256)
A_fused_pred_stack: (31, 256, 256)
height_pred:        (256, 256)
```

---

## 9. 架构搜索结果

128 crop 架构搜索：

| architecture | params | final L_mod | grid energy | platform RMS | height std |
|---|---:|---:|---:|---:|---:|
| `tiny_unet` | 267,170 | 0.374168 | 0.670558 | 0.072107 | 3.218117 |
| `freq_res_unet` | 458,354 | 0.183290 | 0.624063 | 0.107011 | 2.898892 |
| `freq_res_unet_tx` | 533,138 | 0.057210 | 0.461612 | 0.063917 | 3.025939 |

128 baseline：

```text
grid energy:  0.645377
platform RMS: 0.250023
```

结论：

```text
freq_res_unet_tx 在 128 crop 上显著优于 tiny_unet 和 freq_res_unet。
```

256 crop 验证：

| architecture | final L_mod | grid energy | platform RMS | height std |
|---|---:|---:|---:|---:|
| previous tiny_unet CUDA | 0.270700 | 0.860322 | 0.035572 | 2.860825 |
| freq_res_unet_tx final epoch | 0.048258 | 0.854872 | 0.075568 | 2.839839 |
| freq_res_unet_tx select-best | 0.048277 | 0.841115 | 0.078964 | 2.842237 |

256 baseline：

```text
baseline A_cls:
  grid energy:  0.888514
  platform RMS: 0.308210

previous measured platform-opt:
  grid energy:  0.889704
  platform RMS: 0.341849
```

---

## 10. loss 权重扫描

128 crop：

| run | lambda_platform | lambda_sec | lambda_grid | final L_mod | grid | platform RMS | height std |
|---|---:|---:|---:|---:|---:|---:|---:|
| p0.15/s0.12/g0.03 | 0.15 | 0.12 | 0.03 | 0.057210 | 0.461612 | 0.063917 | 3.025939 |
| p0.30/s0.12/g0.03 | 0.30 | 0.12 | 0.03 | 0.080654 | 0.465998 | 0.054048 | 2.903469 |
| p0.30/s0.08/g0.06 | 0.30 | 0.08 | 0.06 | 0.086344 | 0.500146 | 0.053245 | 2.890767 |
| p0.20/s0.08/g0.06 | 0.20 | 0.08 | 0.06 | 0.072327 | 0.551695 | 0.061505 | 3.143843 |
| p0.20/s0.05/g0.10 | 0.20 | 0.05 | 0.10 | 0.084841 | 0.566359 | 0.057882 | 2.950377 |
| p0.50/s0.12/g0.03 | 0.50 | 0.12 | 0.03 | 0.138065 | 0.597286 | 0.051630 | 3.023480 |

256 crop：

| run | lambda_platform | lambda_sec | lambda_grid | final L_mod | grid energy | platform RMS | height std |
|---|---:|---:|---:|---:|---:|---:|---:|
| p0.15/s0.12/g0.03 | 0.15 | 0.12 | 0.03 | 0.048258 | 0.854872 | 0.075568 | 2.839839 |
| p0.30/s0.12/g0.03 | 0.30 | 0.12 | 0.03 | 0.050783 | 0.886686 | 0.040429 | 2.855970 |
| p0.15/s0.12/g0.03 + select-best | 0.15 | 0.12 | 0.03 | 0.048277 | 0.841115 | 0.078964 | 2.842237 |

结论：

- `lambda_platform=0.30` 能让平台 RMS 更低，但 grid energy 几乎回到 baseline。
- `lambda_platform=0.15` 加 `select-best` 是当前去网格和平坦度的折中。

---

## 11. 当前最佳量化效果

最佳配置：

```text
architecture: freq_res_unet_tx
crop: center 256x256
epochs: 50
best_epoch: 49
device: CUDA, NVIDIA RTX 3070
torch: 2.11.0+cu128
```

对比：

| method | final L_mod | grid energy | platform RMS | height std |
|---|---:|---:|---:|---:|
| baseline A_cls | n/a | 0.888514 | 0.308210 | 2.936347 |
| measured platform-opt | n/a | 0.889704 | 0.341849 | 2.969203 |
| previous tiny_unet CUDA | 0.270700 | 0.860322 | 0.035572 | 2.860825 |
| freq_res_unet_tx final epoch | 0.048258 | 0.854872 | 0.075568 | 2.839839 |
| freq_res_unet_tx select-best | 0.048277 | 0.841115 | 0.078964 | 2.842237 |

相对 baseline A_cls：

```text
grid energy: 0.888514 -> 0.841115
relative decrease: about 5.33%

platform RMS: 0.308210 -> 0.078964
relative decrease: about 74.38%

baseline modulation loss: 0.061518 -> 0.048277
relative decrease: about 21.52%
```

高度读出：

```text
height_min:  30.790279
height_max:  36.973938
height_mean: 33.683075
height_std:  2.842237
step_height: 5.679253
invalid_fraction: 0
```

平台统计：

```text
low platform:
  mean: 30.984734
  rms:  0.063906

high platform:
  mean: 36.663986
  rms:  0.091578

pooled_platform_rms: 0.078964
```

---

## 12. 目前能成立的结论

当前可以说：

1. 已经实现 PyTorch/CUDA 神经逆向网络。
2. 已经使用 T6 H/V 六帧输入恢复 `A_h/A_v/A_fused`。
3. 已经实现受限 forward model，让网络输出必须能解释 raw frames。
4. 已经估计一组低维退化残差 `theta`。
5. 当前最佳结果相比传统 baseline，平台 RMS 明显降低。
6. 当前最佳结果相比传统 baseline，grid/Moiré 频带能量有所降低。
7. `freq_res_unet_tx` 在当前架构搜索中表现最好。

---

## 13. 目前不能过度宣称的结论

不能说：

1. 已经恢复绝对真实高度。因为没有独立高度真值。
2. 已经完全消除 Moiré。当前只是 grid energy 下降约 5.33%。
3. 已经完整恢复像差。当前只有一个过于简化的 global blur。
4. 已经完整恢复 DMD-CCD 安装误差。当前只有小仿射残差，不能表达非线性 warp。
5. 模型已经泛化。当前没有跨样本验证。
6. theta 各项都有唯一物理意义。`A_pred` 和 `M_theta` 可能互相补偿。

---

## 14. 主要科学风险

### 14.1 select-best 可能是测试集调参

当前 best epoch 是在同一个 `细台阶 center:256` 上选的。这对工程探索可以接受，但论文中可能被认为是测试样本调参。

建议：

```text
用一个完整样本做 validation，另一个完整样本做 held-out test。
```

### 14.2 A_cls 弱约束可能带回传统解调 bias

`L_weak_Acls` 用低通后的传统三步解调作为弱先验。它防止网络漂移，但可能限制网络摆脱 classical baseline。

建议做 `lambda_sec` 消融。

### 14.3 theta 可辨识性不足

可能存在：

```text
A_pred changes + M_theta opposite changes -> similar reprojection loss
```

建议对 theta 各项做打开/关闭消融。

### 14.4 grid energy 口径需要统一

训练中有 H/V grid energy，评估中有 fused grid energy。应指定 primary metric，并同时报告 H、V、fused。

### 14.5 当前没有 z-context

当前是：

```text
Y[z] -> A[z]
```

可改为：

```text
Y[z-1], Y[z], Y[z+1] -> A[z]
```

---

## 15. 建议 GPT Pro 重点审查的问题

请作为计算成像、结构光/OS-SIM 和深度学习重建方向的审稿人，审查这个方案：

1. `Y_real -> A_h/A_v -> A_fused -> height` 的路线是否合理？
2. `Y_hat = D0 + A_pred * M_theta` 这个 forward model 是否足够物理？
3. 当前 theta 中的 `delta_q`、`phase_offsets`、`affine_residual`、`blur_sigma`、`grid_coeff` 哪些能解释为退化参数，哪些不能过度解释？
4. 当前是否能说恢复了像差和 DMD-CCD 安装误差？如果不能，需要哪些模型和实验？
5. loss 设计是否会导致伪结果，例如过平滑或复制 A_cls？
6. grid energy、platform RMS、height std、modulation loss 是否足够支持当前结论？
7. `--select-best` 在同一样本上选择 checkpoint 是否科学？
8. `freq_res_unet_tx` 是否适合该任务？是否应该换成更强的频域网络、U-Net Transformer、2.5D/3D 网络，或 physics-informed 网络？
9. 下一步最关键的消融和验证是什么？

请明确输出：

```text
可以成立的结论；
证据不足的结论；
可能的实现 bug；
科学解释风险；
下一步实验优先级。
```

---

## 16. 建议下一步

按优先级：

1. 跨样本验证：`细台阶`、`台阶曝光2`、`回型曝光20000`、`小孔` 之间交叉验证。
2. theta 消融：关闭/打开 `delta_q`、`phase_offsets`、`affine_residual`、`blur_sigma`、`grid_coeff`。
3. loss 消融：关闭/打开 `L_grid_fft`、`L_weak_Acls`、`L_platform`。
4. z-aware 网络：从 6-channel 单 z 改成 18-channel 三 z。
5. 加 residual warp：B-spline/TPS 或分块 affine。
6. 加物理像差：Zernike 或空间变化 PSF。
7. 统一 grid energy 指标。
8. 做更完整可视化：raw frames、A_h/A_v/A_fused、FFT residual、height compare。

---

## 17. 一句话总结

当前已经实现了一个 OS-SIM 神经逆向恢复原型：它能从 T6 H/V 六帧真实条纹中恢复 H/V 光切片调制度，并用受限 forward model 同时估计低维退化残差。当前结果相对传统 baseline 明显改善平台平坦度，并降低部分网格/Moiré 频谱能量；但尚未证明绝对高度准确性、跨样本泛化、完整像差恢复或完整 DMD-CCD 安装误差恢复。
