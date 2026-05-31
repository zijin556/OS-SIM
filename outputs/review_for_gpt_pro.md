# OS-SIM 神经逆向过程阶段性报告（供 GPT Pro 审查）

生成时间：2026-05-31  
项目目录：`E:\research\AIResearch\PhysicsDeepLearning_OSSIM`  
原始数据目录：`E:\research\AIResearch\程序\01_raw_data\dlp6500_sim_confocal\scans_raw\3-6扫描`

本文档用于把目前已经完成的工作、实现步骤、实验结果、已知不足和需要外部审查的问题整理清楚。报告重点不是证明结论已经完全成立，而是让 GPT Pro 或其他审查者能快速判断当前方案是否合理、哪里有潜在漏洞、下一步应该怎么补强。

---

## 1. 任务目标和我对目标的理解

你提出的核心目标是：

1. 不再只做传统三步解调或简单高度恢复。
2. 建立神经网络来学习 OS-SIM 的逆向过程。
3. 从真实退化条纹帧中恢复更接近“理想光切片”的调制度图。
4. 同时学习或估计退化参数。
5. 再基于恢复出的理想光切片栈进行三维高度重建。

我当前实现的主线是：

```text
真实六帧条纹 Y_real[z, 6, H, W]
        ↓ 神经逆向网络
A_h_pred[z, H, W], A_v_pred[z, H, W]
        ↓ H/V 融合
A_fused_pred = sqrt((A_h_pred^2 + A_v_pred^2) / 2)
        ↓ 沿 z 方向峰值读出
height_pred[H, W]
```

同时有一个受限可微 forward model：

```text
Y_hat = D0 + A_pred * M_theta
```

其中：

- `D0` 是每组三步条纹的平均背景项。
- `A_pred` 是网络预测的 H/V 理想光切片调制度。
- `M_theta` 是从实测/标定 DMD-camera 条纹图案出发，再加入少量可学习退化残差后的条纹调制模式。

需要特别说明：现在不是直接让网络输出高度，也不是完全自由学习一个任意退化场。当前设计是让网络预测 `A_h/A_v`，让可微 forward model 用低维、受限的 `theta` 去解释真实条纹退化。

---

## 2. 数据使用策略

### 2.1 样本选择

按照你的要求：

- 明确排除 `台阶曝光1`，因为它缺帧。
- 第一轮主样本使用 `细台阶`。
- 后续可对比的完整样本包括：
  - `台阶曝光2`
  - `回型曝光20000`
  - `小孔`
  - `细台阶`

当前神经网络主结果只在 `细台阶` 上完成训练、选择和评估，尚未完成跨样本验证。

### 2.2 帧组

使用 T6 双方向三步条纹：

```text
水平 H：100, 101, 102
竖直 V：103, 104, 105
```

当前主实验使用 `center:256` crop。每个 z 层输入 6 张图，输出 2 张光切片调制度图：

```text
input:  [H100, H101, H102, V103, V104, V105]
output: [A_h, A_v]
```

### 2.3 已知数据问题

部分旧的 JSON/Markdown 输出里，中文样本名和路径存在编码错乱，例如 `细台阶` 在历史 `metrics.json` 里可能显示成乱码。这不影响已经保存的数值数组和图像结果，但会影响报告可读性和可复现性记录。

建议后续单独修复：

- 项目内部统一使用 UTF-8 写 JSON/Markdown。
- 原始 DMD 序列文件如果是 GBK，只在读取该文件时局部指定 GBK。
- 输出报告时不要把 GBK 字符串直接当 UTF-8 写出。

---

## 3. 环境和 PyTorch 安装状态

当前 `.venv` 中已经安装 CUDA 版 PyTorch：

```text
torch: 2.11.0+cu128
CUDA runtime: 12.8
GPU: NVIDIA GeForce RTX 3070
```

已经做过 GPU smoke test，训练脚本使用的是 `--device cuda`。CUDA 训练已经能跑通 128 和 256 crop。

---

## 4. 已实现的主要代码文件

### 4.1 神经网络结构

文件：

```text
E:\research\AIResearch\PhysicsDeepLearning_OSSIM\src\inverse_net.py
```

关键实现位置：

```text
FrequencyFiLM:                  src\inverse_net.py:75
WindowTransformerBlock:         src\inverse_net.py:126
InverseLightSectionNet:         src\inverse_net.py:158
FreqResInverseLightSectionNet:  src\inverse_net.py:198
```

已经实现 3 种结构：

| 结构名 | 参数量 | 说明 |
|---|---:|---|
| `tiny_unet` | 267,170 | 初版小 U-Net，作为保守基线 |
| `freq_res_unet` | 458,354 | 残差 U-Net，加频域 FiLM 条件 |
| `freq_res_unet_tx` | 533,138 | 残差 U-Net + 频域 FiLM + window transformer |

当前推荐结构是 `freq_res_unet_tx`。

### 4.2 受限可微 forward model

文件：

```text
E:\research\AIResearch\PhysicsDeepLearning_OSSIM\src\torch_forward.py
```

关键实现位置：

```text
RestrictedStripeForward: src\torch_forward.py:122
```

该 forward model 从实测/标定投影条纹模式出发，只允许小范围退化残差：

- `delta_q`：条纹空间频率/方向的小残差。
- `phase_offsets`：三步相位的小残差。
- `affine_residual`：DMD-camera 映射后的全局小仿射残差。
- `blur_sigma`：全局简化模糊参数。
- `grid_coeff`：固定残余网格/Moiré basis 上的小系数。

它不是一个自由 per-pixel 退化网络，因此可解释性比完全自由网络强一些，但表达能力也有限。

### 4.3 训练脚本

文件：

```text
E:\research\AIResearch\PhysicsDeepLearning_OSSIM\scripts\train_inverse_lightsection_net.py
```

主要能力：

- 支持 `--architecture tiny_unet|freq_res_unet|freq_res_unet_tx`
- 支持 `--pattern-source measured_calibrated`
- 支持 `--lambda-grid`
- 支持 `--lambda-sec`
- 支持 `--lambda-platform`
- 支持 `--select-best`
- 保存网络输出、loss curve、theta、环境信息和高度图。

核心 loss 组合：

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

含义：

- `L_mod`：真实条纹去均值后，与 `A_pred * M_theta` 的调制项匹配。
- `L_raw`：完整 raw frame 重投影匹配。
- `L_tv`：抑制调制度图过度噪声。
- `L_grid_fft`：惩罚已检测网格/Moiré 频带能量。
- `L_weak_Acls`：只用低通后的经典三步解调 `A_cls` 做弱约束，避免网络完全漂移。
- `L_theta_reg`：限制退化参数不要无约束发散。
- `L_platform`：利用台阶平台 mask，使恢复高度在同一平台内更平坦。

### 4.4 评估脚本

文件：

```text
E:\research\AIResearch\PhysicsDeepLearning_OSSIM\scripts\evaluate_inverse_lightsection_net.py
```

主要输出：

- `evaluation_metrics.json`
- `evaluation_report.md`
- `height_compare.png`
- `A_comparison.png`
- `reprojection_examples.png`

评估包含：

- 神经网络结果。
- 传统 baseline `A_cls`。
- 之前的 platform-constrained latent optimization 结果。

---

## 5. 网络结构细节

### 5.1 初版 `tiny_unet`

初版网络是一个小型 2D U-Net：

```text
input 6 channels
  ↓ ConvBlock
  ↓ downsample
  ↓ ConvBlock
  ↓ downsample
  ↓ bottleneck
  ↓ upsample + skip
  ↓ upsample + skip
output 2 channels: A_h, A_v
```

输出层后使用 `softplus`，保证调制度非负。

这个网络可以跑通，但表达力不足，`L_mod` 较高；它有时能给出较平坦的平台，但更像过平滑结果，不能很好解释真实条纹调制。

### 5.2 改进版 `freq_res_unet`

改进点：

- 用 residual block 替换普通卷积块。
- bottleneck 加 dilation residual block。
- 加 `FrequencyFiLM`：从输入 raw frames 的 FFT 中提取全局频谱描述，调制 bottleneck feature。

`FrequencyFiLM` 不是直接在频域生成图像，而是用频谱能量描述来影响空间网络的特征表达，目的是让网络对残余条纹/Moiré 更敏感。

### 5.3 当前推荐版 `freq_res_unet_tx`

在 `freq_res_unet` 基础上加入 window transformer：

```text
raw 6-frame input
  ↓ residual encoder
  ↓ residual bottleneck
  ↓ FrequencyFiLM
  ↓ WindowTransformerBlock
  ↓ residual decoder
  ↓ softplus
A_h, A_v
```

加入 transformer 的理由：

- U-Net 的卷积主要建模局部纹理。
- Moiré / 网格残余往往有较长距离周期结构。
- window attention 可以在局部窗口内建模非卷积相关性，改善条纹残差识别。

当前实验证据显示 `freq_res_unet_tx` 在 128 crop 架构对比中明显优于另外两个网络，在 256 crop 上也优于旧 `tiny_unet`。

---

## 6. “理想光切片参数”目前恢复到了什么程度

当前已经得到的光切片相关输出：

```text
E:\research\AIResearch\PhysicsDeepLearning_OSSIM\outputs\inverse_net_sweep\tx_p015_s012_g003_256_selectbest\A_h_pred_stack.npy
E:\research\AIResearch\PhysicsDeepLearning_OSSIM\outputs\inverse_net_sweep\tx_p015_s012_g003_256_selectbest\A_v_pred_stack.npy
E:\research\AIResearch\PhysicsDeepLearning_OSSIM\outputs\inverse_net_sweep\tx_p015_s012_g003_256_selectbest\A_fused_pred_stack.npy
```

shape：

```text
A_h_pred_stack:     (31, 256, 256)
A_v_pred_stack:     (31, 256, 256)
A_fused_pred_stack: (31, 256, 256)
```

这里的 `A_h/A_v/A_fused` 可以理解为“网络恢复的理想化光切片调制度栈”，但它不是有独立真值监督的绝对理想光切片。它是通过以下约束共同推出来的：

1. 能通过 forward model 重投影回真实六帧条纹。
2. 不含过强的已检测网格/Moiré 频带。
3. 低频结构不要偏离经典三步解调太多。
4. 高度读出后同一平台内尽量平坦。
5. H/V 两方向融合后能形成稳定高度栈。

所以更严格的说法应该是：

```text
当前得到的是“受物理 forward model 和先验约束约束的神经估计光切片调制度”，
不是经过独立标定验证的绝对理想光切片真值。
```

---

## 7. “退化参数”目前恢复到了什么程度

当前最佳模型保存了 `theta`，位置：

```text
E:\research\AIResearch\PhysicsDeepLearning_OSSIM\outputs\inverse_net_sweep\tx_p015_s012_g003_256_selectbest\metrics.json
```

主要 theta 值如下。

### 7.1 `delta_q`

```text
H: [ 0.0004836913, 0.0006069596]
V: [-0.0006686280, 0.0005879356]
```

解释：表示 H/V 两组条纹的空间频率或方向小残差。注意实际作用还有 scale：

```text
delta_q_scale = 0.15
delta_q_pattern_scale = 0.05
```

### 7.2 `phase_offsets`

当前接近 0：

```text
H: 约 e-22
V: 约 e-21
```

解释：这说明当前模型没有明显学到三步相位偏移，可能原因包括：

- 实测/标定条纹模式已经吸收了大部分相位误差。
- 该参数梯度较弱。
- 该误差被 `A_pred`、`delta_q` 或 `affine_residual` 间接吸收。

这需要后续审查和消融实验确认。

### 7.3 `affine_residual`

原始 learned residual：

```text
H:
[-0.017351,  0.004581,  0.034887]
[-0.033200,  0.017243, -0.014514]

V:
[-0.034662,  0.022851,  0.038467]
[ 0.010508,  0.009356,  0.028067]
```

实际进入仿射变换前还乘以：

```text
affine_residual_scale = 0.02
```

解释：它只能表示标定后的全局小仿射残差，不能表示完整 DMD-CCD 安装误差、镜头畸变或非线性 warp。

### 7.4 `blur_sigma`

```text
H: 0.0064557
V: 0.0064557
```

解释：当前 blur 很小，说明简化模糊项基本没有起到强作用。它不能代表真实复杂像差。

### 7.5 `grid_coeff`

`grid_coeff` 数值大多在 `1e-6` 到 `1e-8` 量级。

解释：当前残余网格 basis 系数很小，可能说明：

- 网格问题主要由网络输出和 loss 约束处理。
- 当前 basis 表达不够好。
- 该项在 loss 中辨识度不足。

### 7.6 对“像差”和“DMD-CCD 安装误差”的准确回答

目前不能说已经完整恢复了真实物理像差和 DMD-CCD 安装误差。

当前可以说：

```text
已经估计出一组受限低维退化残差 theta：
delta_q、phase_offsets、affine_residual、blur_sigma、grid_coeff。
```

但不能说：

```text
已经恢复了完整像差参数；
已经恢复了完整 DMD-CCD 安装误差；
已经得到可直接用于物理标定的参数。
```

如果研究目标要明确包含“像差”和“DMD-CCD 安装误差”，后续需要加入更物理可解释的模型，例如：

- Zernike 像差参数。
- 空间变化 PSF。
- 各向异性 blur。
- B-spline/TPS 非线性残余 warp。
- field-dependent phase/frequency 参数。
- 独立标定板或仿真真值验证。

---

## 8. 高度重建和结果位置

当前神经网络最佳结果目录：

```text
E:\research\AIResearch\PhysicsDeepLearning_OSSIM\outputs\inverse_net_sweep\tx_p015_s012_g003_256_selectbest
```

主要结果文件：

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

最重要的对比图：

```text
E:\research\AIResearch\PhysicsDeepLearning_OSSIM\outputs\inverse_net_sweep\tx_p015_s012_g003_256_selectbest\height_compare.png
```

单独高度图：

```text
E:\research\AIResearch\PhysicsDeepLearning_OSSIM\outputs\inverse_net_sweep\tx_p015_s012_g003_256_selectbest\height_pred.png
```

光切片对比图：

```text
E:\research\AIResearch\PhysicsDeepLearning_OSSIM\outputs\inverse_net_sweep\tx_p015_s012_g003_256_selectbest\A_comparison.png
```

重投影示例：

```text
E:\research\AIResearch\PhysicsDeepLearning_OSSIM\outputs\inverse_net_sweep\tx_p015_s012_g003_256_selectbest\reprojection_examples.png
```

你之前指出的两个图：

```text
E:\research\AIResearch\PhysicsDeepLearning_OSSIM\outputs\platform_opt\t6_hv_measured_smoke128\height_best_map.png
E:\research\AIResearch\PhysicsDeepLearning_OSSIM\outputs\platform_opt\t6_hv_measured\height_best_map.png
```

它们是早期 platform-constrained latent optimization 的恢复结果，不是当前最终神经网络 `freq_res_unet_tx` 的最佳结果。你说它们表面仍有 Moiré 条纹，这个判断是对的；后续神经网络版本就是为了进一步压制这类残余网格/条纹。

---

## 9. 当前最佳训练命令

```powershell
.\.venv\Scripts\python.exe scripts/train_inverse_lightsection_net.py `
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
  --output outputs/inverse_net_sweep/tx_p015_s012_g003_256_selectbest `
  --device cuda
```

当前最佳评估命令：

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_inverse_lightsection_net.py `
  --input outputs/inverse_net_sweep/tx_p015_s012_g003_256_selectbest/A_fused_pred_stack.npy `
  --a-h outputs/inverse_net_sweep/tx_p015_s012_g003_256_selectbest/A_h_pred_stack.npy `
  --a-v outputs/inverse_net_sweep/tx_p015_s012_g003_256_selectbest/A_v_pred_stack.npy `
  --z-values outputs/inverse_net_sweep/tx_p015_s012_g003_256_selectbest/z_values.npy `
  --baseline outputs/baseline/t6_hv/A_fused_stack.npy `
  --latent outputs/platform_opt/t6_hv_measured/A_fused_best_stack.npy `
  --platform-low-mask outputs/height/baseline_t6_hv/platform_low_mask.npy `
  --platform-high-mask outputs/height/baseline_t6_hv/platform_high_mask.npy `
  --output outputs/inverse_net_sweep/tx_p015_s012_g003_256_selectbest
```

---

## 10. 架构对比结果

128 crop 架构搜索：

| architecture | params | final L_mod | grid energy | platform RMS | height std |
|---|---:|---:|---:|---:|---:|
| `tiny_unet` | 267,170 | 0.374168 | 0.670558 | 0.072107 | 3.218117 |
| `freq_res_unet` | 458,354 | 0.183290 | 0.624063 | 0.107011 | 2.898892 |
| `freq_res_unet_tx` | 533,138 | 0.057210 | 0.461612 | 0.063917 | 3.025939 |

128 baseline reference：

```text
grid energy:  0.645377
platform RMS: 0.250023
```

结论：

- `freq_res_unet_tx` 的调制重投影误差最低。
- `freq_res_unet_tx` 的 128 crop grid energy 也最低。
- `tiny_unet` 虽然某些平台 RMS 表现保守，但明显欠拟合逆向调制过程。

256 crop 结果：

| architecture | final L_mod | grid energy | platform RMS | height std |
|---|---:|---:|---:|---:|
| previous `tiny_unet` CUDA | 0.270700 | 0.860322 | 0.035572 | 2.860825 |
| `freq_res_unet_tx` final epoch | 0.048258 | 0.854872 | 0.075568 | 2.839839 |
| `freq_res_unet_tx` + `--select-best` | 0.048277 | 0.841115 | 0.078964 | 2.842237 |

256 baseline reference：

```text
baseline A_cls:
  grid energy:  0.888514
  platform RMS: 0.308210

measured platform-opt:
  grid energy:  0.889704
  platform RMS: 0.341849
```

---

## 11. loss 权重扫描结果

128 crop 中对 `freq_res_unet_tx` 做了若干 loss 权重组合：

| run | lambda_platform | lambda_sec | lambda_grid | final L_mod | grid | platform RMS | height std |
|---|---:|---:|---:|---:|---:|---:|---:|
| `tx_p015_s012_g003_128` | 0.15 | 0.12 | 0.03 | 0.057210 | 0.461612 | 0.063917 | 3.025939 |
| `tx_p030_s012_g003_128` | 0.30 | 0.12 | 0.03 | 0.080654 | 0.465998 | 0.054048 | 2.903469 |
| `tx_p030_s008_g006_128` | 0.30 | 0.08 | 0.06 | 0.086344 | 0.500146 | 0.053245 | 2.890767 |
| `tx_p020_s008_g006_128` | 0.20 | 0.08 | 0.06 | 0.072327 | 0.551695 | 0.061505 | 3.143843 |
| `tx_p020_s005_g010_128` | 0.20 | 0.05 | 0.10 | 0.084841 | 0.566359 | 0.057882 | 2.950377 |
| `tx_p050_s012_g003_128` | 0.50 | 0.12 | 0.03 | 0.138065 | 0.597286 | 0.051630 | 3.023480 |

256 crop 验证：

| run | lambda_platform | lambda_sec | lambda_grid | final L_mod | grid energy | platform RMS | height std |
|---|---:|---:|---:|---:|---:|---:|---:|
| `tx_p015_s012_g003_256` | 0.15 | 0.12 | 0.03 | 0.048258 | 0.854872 | 0.075568 | 2.839839 |
| `tx_p030_s012_g003_256` | 0.30 | 0.12 | 0.03 | 0.050783 | 0.886686 | 0.040429 | 2.855970 |
| `tx_p015_s012_g003_256_selectbest` | 0.15 | 0.12 | 0.03 | 0.048277 | 0.841115 | 0.078964 | 2.842237 |

结论：

- 如果优先压制 Moiré/grid，推荐 `lambda_platform=0.15, lambda_sec=0.12, lambda_grid=0.03, --select-best`。
- 如果优先追求平台极平坦，可以提高 `lambda_platform` 到 0.30，但会几乎失去 grid energy 改善。
- 当前推荐是偏向去网格和整体可解释性的折中，而不是单纯追求平台 RMS 最低。

---

## 12. 当前最佳结果的量化效果

当前最佳结果：

```text
outputs\inverse_net_sweep\tx_p015_s012_g003_256_selectbest
architecture: freq_res_unet_tx
crop: center:256
epochs: 50
best_epoch: 49
device: cuda
```

主要指标：

| method | final L_mod | grid energy | platform RMS | height std |
|---|---:|---:|---:|---:|
| baseline A_cls | n/a | 0.888514 | 0.308210 | 2.936347 |
| measured platform-opt | n/a | 0.889704 | 0.341849 | 2.969203 |
| previous tiny_unet CUDA | 0.270700 | 0.860322 | 0.035572 | 2.860825 |
| freq_res_unet_tx final epoch | 0.048258 | 0.854872 | 0.075568 | 2.839839 |
| freq_res_unet_tx select-best | 0.048277 | 0.841115 | 0.078964 | 2.842237 |

相对 baseline A_cls：

```text
grid energy: 0.888514 -> 0.841115，降低约 5.33%
platform RMS: 0.308210 -> 0.078964，降低约 74.38%
baseline_modulation_loss: 0.061518 -> 0.048277，降低约 21.52%
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

解释：

- 平台平坦度相比传统 baseline 明显改善。
- Moiré/grid energy 有改善，但幅度不算巨大，说明表面可能仍能看到残余弱网格。
- 这可以作为阶段性神经逆向恢复结果，但还不能宣称完全消除了 Moiré。

---

## 13. 已完成的验证和审计

已有输出：

```text
E:\research\AIResearch\PhysicsDeepLearning_OSSIM\outputs\cli_surface\cli_surface.json
E:\research\AIResearch\PhysicsDeepLearning_OSSIM\outputs\audit\coverage.json
E:\research\AIResearch\PhysicsDeepLearning_OSSIM\outputs\traceability\requirements_traceability.json
E:\research\AIResearch\PhysicsDeepLearning_OSSIM\outputs\reproducibility\contracts.json
```

当前状态摘要：

```text
cli_surface: satisfied
coverage: complete_with_documented_limits
traceability: complete_with_documented_limits
reproducibility: satisfied
```

主要 documented limit：

```text
没有独立标定高度真值，因此目前只能评价相对指标、平台平坦度、grid energy 和重投影误差。
```

---

## 14. 当前方案的科学解释边界

### 14.1 可以说已经完成的

1. 已经排除缺帧样本 `台阶曝光1`。
2. 已经使用 `细台阶` 的 T6 H/V 六帧数据。
3. 已经实现 PyTorch CUDA 神经逆向网络。
4. 已经实现 `A_h/A_v/A_fused` 光切片栈恢复。
5. 已经实现受限可微 forward model。
6. 已经保存低维退化残差 `theta`。
7. 已经用 `A_fused_pred_stack` 读出高度图。
8. 已经和 baseline A_cls、早期 platform-opt 结果做了量化对比。
9. 已经做过网络结构对比和 loss 权重扫描。

### 14.2 不能过度宣称的

1. 不能说已经恢复了真实绝对高度，因为没有独立高度真值。
2. 不能说已经完全消除 Moiré，只能说当前指标有下降。
3. 不能说已经完整估计像差，当前 `blur_sigma` 太简单。
4. 不能说已经完整估计 DMD-CCD 安装误差，当前只有小仿射残差。
5. 不能说模型已经泛化，因为还没做跨样本验证。
6. 不能说 `theta` 每一项都有唯一物理意义，因为 `A_pred` 和 `M_theta` 之间可能存在补偿。

---

## 15. 我认为最重要的潜在问题

### 15.1 `--select-best` 的选择标准可能有测试集调参问题

当前 best epoch 是在同一个 `细台阶 center:256` crop 上按综合 score 选择的。这对工程上找好结果可以接受，但如果要写论文，容易被质疑为在测试样本上选择模型。

改进方式：

- 用 `台阶曝光2` 或另一个完整样本做 validation。
- 在 validation 上选择 epoch 和 loss 权重。
- 把 `细台阶` 作为 held-out 或反过来做交叉验证。

### 15.2 grid energy 指标需要统一

训练时有的指标在 H/V 单方向 `A_pred` 上算，评估时主要在 fused `A_fused` 上算。两者都有意义，但论文或正式报告里最好指定一个 primary metric，否则容易被质疑指标口径不一致。

### 15.3 `A_cls` 弱约束是否太强需要审查

`L_weak_Acls` 用低通后的经典三步解调作为弱先验。优点是防止网络漂移，缺点是可能把传统解调的 bias 带回网络。

建议做消融：

```text
lambda_sec = 0
lambda_sec = 0.05
lambda_sec = 0.12
lambda_sec = 0.20
```

并比较：

- 重投影误差。
- 平台 RMS。
- grid energy。
- 视觉细节是否过度平滑。

### 15.4 `theta` 的可辨识性不足

`A_pred` 和 `M_theta` 都参与解释真实条纹，因此可能存在等价补偿：

```text
A_pred 变一点，M_theta 反向变一点，也能重投影得差不多。
```

这意味着当前 `theta` 可以作为受限残差参数，但还不能直接解释为真实物理误差。后续需要：

- 固定部分 theta 做消融。
- 单独打开/关闭 `delta_q`、`affine_residual`、`blur_sigma`、`grid_coeff`。
- 看每一项对重投影和高度的贡献。

### 15.5 当前模型没有 z 上下文

当前是逐层 z 独立预测：

```text
Y[z] -> A[z]
```

但真实 confocal/OS-SIM 光切片沿 z 应该有连续性。后续可以改为：

```text
Y[z-1], Y[z], Y[z+1] -> A[z]
```

或者用轻量 3D/2.5D 网络，可能改善高度峰值稳定性。

### 15.6 当前像差模型太弱

如果研究重点要强调像差，当前 `blur_sigma` 只是全局 isotropic blur，不够。建议加入：

- Zernike 系数。
- 空间变化 PSF。
- 各向异性 blur。
- field-dependent phase。

### 15.7 当前 DMD-CCD 安装误差模型太弱

当前 `affine_residual` 只能表示标定后小范围全局仿射误差，不能表示：

- 透镜畸变。
- 局部非线性 warp。
- 投影平面与成像平面复杂不共面误差。

建议加入：

- B-spline residual warp。
- TPS residual warp。
- 分块 affine。
- 用标定板/仿真数据验证 warp 参数。

---

## 16. 建议 GPT Pro 重点审查的问题

可以直接把下面这段作为给 GPT Pro 的审查提示词。

```text
请作为计算成像、结构光/OS-SIM 和深度学习重建方向的审稿人，审查这份 OS-SIM 神经逆向过程阶段性报告。

请重点判断：

1. 当前把真实六帧条纹 Y_real 映射到 A_h/A_v 理想光切片调制度，再由 A_fused 沿 z 读出高度的研究路线是否合理？
2. 当前受限 forward model Y_hat = D0 + A_pred * M_theta 是否足够物理合理？
3. theta 中的 delta_q、phase_offsets、affine_residual、blur_sigma、grid_coeff 是否可以被称为退化参数？哪些可以解释，哪些不能过度解释？
4. 当前是否能说恢复了像差和 DMD-CCD 安装误差？如果不能，需要补哪些模型或实验？
5. loss 组合 L_mod、L_raw、L_grid_fft、L_weak_Acls、L_theta_reg、L_platform 是否存在相互矛盾或导致伪结果的风险？
6. 使用 A_cls 的低通结果作为弱监督是否会让网络只是复制传统三步解调？
7. 当前 grid energy、platform RMS、height std、modulation loss 的指标是否足够支撑“去 Moiré”和“改善高度恢复”的结论？
8. --select-best 在同一样本上选择 epoch 是否构成测试集调参？应该如何设计验证集？
9. 当前架构 freq_res_unet_tx（Residual U-Net + FrequencyFiLM + window transformer）是否适合这个任务？有没有更合理的网络结构建议？
10. 下一步最小但关键的实验是什么？请按优先级给出。

请明确指出：

- 哪些结论目前可以成立；
- 哪些结论证据不足；
- 哪些实现可能有 bug 或科学解释风险；
- 如果要写成论文或课题阶段汇报，应该补哪些图、表、消融和验证。
```

---

## 17. 建议下一步实验

按优先级排序：

1. 跨样本验证  
   用 `细台阶` 训练/调参，在 `台阶曝光2`、`回型曝光20000`、`小孔` 上评估。或者反过来做交叉验证。

2. theta 消融  
   分别关闭 `delta_q`、`phase_offsets`、`affine_residual`、`blur_sigma`、`grid_coeff`，看每个退化参数是否真的有贡献。

3. loss 消融  
   分别去掉 `L_grid_fft`、`L_weak_Acls`、`L_platform`，检查结果是否依赖某个强先验。

4. z-aware 网络  
   从单层 6 channel 改成三层 18 channel：

   ```text
   Y[z-1], Y[z], Y[z+1] -> A[z]
   ```

5. 更物理的 DMD-CCD residual warp  
   在 `affine_residual` 后加入 B-spline/TPS residual warp，并用强正则限制幅度。

6. 更物理的像差项  
   加入 Zernike 或空间变化 PSF，而不是只用全局 `blur_sigma`。

7. 指标统一  
   统一 grid energy 的 primary definition，并同时报告 H、V、fused 三种结果。

8. 可视化增强  
   对同一个 z 层展示：

   ```text
   raw H/V frames
   baseline A_h/A_v/A_fused
   network A_h/A_v/A_fused
   residual FFT
   height compare
   ```

---

## 18. 总结结论

当前已经从“传统 baseline 和 SciPy/latent 优化”推进到了“CUDA PyTorch 神经逆向网络 + 受限物理 forward model + H/V 理想光切片恢复 + 高度读出”的阶段。

最佳当前结果是：

```text
freq_res_unet_tx
lambda_platform = 0.15
lambda_sec = 0.12
lambda_grid = 0.03
select_best = True
center:256
```

它的主要效果是：

```text
grid energy 比 baseline 下降约 5.33%
platform RMS 比 baseline 下降约 74.38%
modulation loss 比 classical A_cls forward baseline 下降约 21.52%
```

但目前最重要的限制是：

```text
没有独立高度真值；
没有跨样本验证；
theta 只是受限退化残差，不是完整像差或完整 DMD-CCD 安装误差；
Moiré/grid 只是降低，还没有完全消除。
```

因此，最稳妥的阶段性表述是：

```text
已经实现并验证了一个 OS-SIM 神经逆向恢复原型。
该原型能从 T6 H/V 六帧真实条纹中恢复 H/V 光切片调制度，
并通过受限 forward model 同时估计低维退化残差。
当前结果相对传统 baseline 改善了平台平坦度，并降低了部分网格/Moiré 频谱能量。
但它还不是完整物理标定系统，也尚未证明跨样本泛化和绝对高度准确性。
```
