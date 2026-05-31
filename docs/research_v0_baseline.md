# OS-SIM 神经逆向恢复研究基线 v0

版本：v0  
日期：2026-05-31  
状态：阶段性原型完成，等待 GPT Pro 审查和 v1 研究计划合并。

## 1. 研究问题

传统 OS-SIM 三步解调在反射式三维形貌恢复中容易残留条纹、网格和 Moiré 伪影。当前研究尝试把 OS-SIM 重建表述为一个受限神经逆问题：

```text
真实退化六帧条纹 -> H/V 理想化光切片调制度 -> H/V 融合 -> z 向高度读出
```

同时通过可微 forward model 估计低维退化残差，以提升物理可解释性。

## 2. v0 方法定义

### 2.1 输入和输出

输入：

```text
Y_real[z, 6, H, W]
```

通道顺序：

```text
H100, H101, H102, V103, V104, V105
```

输出：

```text
A_h_pred[z,H,W]
A_v_pred[z,H,W]
A_fused_pred[z,H,W]
height_pred[H,W]
```

### 2.2 H/V 融合

```text
A_fused = sqrt((A_h^2 + A_v^2) / 2)
```

### 2.3 Forward Model

```text
Y_hat = D0 + A_pred * M_theta
```

其中 `M_theta` 从实测/标定条纹模式出发，只允许低维残差修正：

- `delta_q`
- `phase_offsets`
- `affine_residual`
- `blur_sigma`
- `grid_coeff`

### 2.4 Loss

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

当前推荐：

```text
lambda_platform = 0.15
lambda_sec = 0.12
lambda_grid = 0.03
select_best = True
```

## 3. v0 网络结构

当前比较过 3 个网络：

| architecture | params | 说明 |
|---|---:|---|
| `tiny_unet` | 267,170 | 小型 U-Net 基线 |
| `freq_res_unet` | 458,354 | Residual U-Net + FrequencyFiLM |
| `freq_res_unet_tx` | 533,138 | Residual U-Net + FrequencyFiLM + Window Transformer |

v0 推荐：

```text
freq_res_unet_tx
```

## 4. v0 实验设置

```text
sample: 细台阶
groups: ossim_t6_h_3step, ossim_t6_v_3step
frames: 100-105
crop: center:256
z layers: 31
device: CUDA
torch: 2.11.0+cu128
GPU: NVIDIA GeForce RTX 3070
```

数据策略：

- `台阶曝光1` 缺帧，默认排除。
- `细台阶` 是第一轮主样本。
- `台阶曝光2`、`回型曝光20000`、`小孔` 留作后续验证。

## 5. v0 主要结果

当前最佳输出：

```text
outputs/inverse_net_sweep/tx_p015_s012_g003_256_selectbest/
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
grid energy 下降约 5.33%
platform RMS 下降约 74.38%
modulation loss 下降约 21.52%
```

## 6. v0 产物

代码：

```text
src/inverse_net.py
src/torch_forward.py
scripts/train_inverse_lightsection_net.py
scripts/evaluate_inverse_lightsection_net.py
```

审查包：

```text
outputs/gpt_pro_review_bundle.zip
```

主要说明文档：

```text
outputs/review_for_gpt_pro_portable.md
outputs/theoretical_model_for_gpt_pro_visual.md
docs/new_conversation_handoff_v0.md
docs/research_v0_baseline.md
```

## 7. v0 科学边界

当前可以成立：

1. 已经实现神经逆向恢复原型。
2. 已经从 T6 H/V 六帧恢复 `A_h/A_v/A_fused`。
3. 已经实现受限可微 forward model。
4. 已经得到低维退化残差 `theta`。
5. 当前结果相对 baseline 改善平台 RMS，并降低部分 grid/Moiré 能量。

当前不能过度宣称：

1. 不能说已经恢复绝对高度真值。
2. 不能说已经完整恢复像差。
3. 不能说已经完整恢复 DMD-CCD 安装误差。
4. 不能说已经证明跨样本泛化。
5. 不能说 Moiré 已完全消除。

## 8. v1 研究优先级

v1 应优先补齐：

1. 跨样本验证。
2. theta 消融。
3. loss 消融。
4. z-aware 网络。
5. 更物理的 residual warp。
6. 更物理的像差模型。
7. 统一 grid energy 指标。
8. 论文/课题汇报级图表规范。

## 9. GitHub 版本建议

建议 GitHub v0 只上传：

- `configs/`
- `docs/`
- `scripts/`
- `src/`
- `README.md`
- `requirements.txt`
- `.gitignore`
- 轻量级 `outputs/**/*.md/json/png/html`

不要上传：

- `.venv/`
- 原始数据
- `.npy/.npz`
- 模型权重 `.pt/.pth`
- 大型压缩包
- 原始 DMD bitmaps 或 `.mat` 标定大文件

## 10. v0 Tag 建议

建议第一个 Git tag：

```text
v0-neural-inverse-prototype
```

建议 commit message：

```text
Add v0 OS-SIM neural inverse prototype baseline
```
