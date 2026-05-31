# 新对话接续说明 v0

更新时间：2026-05-31  
项目：OS-SIM 反射式三维形貌神经逆向恢复  
当前工作目录：`E:\research\AIResearch\PhysicsDeepLearning_OSSIM`

这份文档用于在新的 Codex/GPT 对话中快速接续当前研究，不依赖上一轮聊天记录。

## 1. 研究目标

当前研究目标是建立一个 OS-SIM 神经逆向恢复框架：

1. 从真实退化的 T6 H/V 六帧条纹中恢复 H/V 两方向理想化光切片调制度。
2. 用受限可微 forward model 约束网络输出，使其能重投影解释 raw frames。
3. 同时估计低维退化残差参数 `theta`。
4. 由 `A_fused` 光切片栈沿 z 方向读出高度。
5. 后续补充跨样本验证、theta 消融、loss 消融和更物理的像差/DMD-CCD 误差模型。

当前不能过度宣称已经恢复了完整像差、完整 DMD-CCD 安装误差或绝对高度真值。

## 2. 数据约束

原始数据位于本机：

```text
E:\research\AIResearch\程序\01_raw_data\dlp6500_sim_confocal\scans_raw\3-6扫描
```

实验策略：

- 排除 `台阶曝光1`，因为缺帧。
- 第一轮主样本：`细台阶`。
- 后续完整对比样本：`台阶曝光2`、`回型曝光20000`、`小孔`、`细台阶`。
- 主帧组：T6 H/V 3-step。
  - H: frames `100,101,102`
  - V: frames `103,104,105`

## 3. 当前理论模型

神经逆向网络：

```text
Y_real[z, 6, H, W] -> A_h_pred[z,H,W], A_v_pred[z,H,W]
```

H/V 融合：

```text
A_fused = sqrt((A_h^2 + A_v^2) / 2)
```

高度读出：

```text
height = peak/parabolic readout along z from A_fused_stack
```

受限 forward model：

```text
Y_hat = D0 + A_pred * M_theta
```

`theta` 包含：

- `delta_q`：条纹频率/方向残差。
- `phase_offsets`：三步相位残差。
- `affine_residual`：标定后小仿射残差。
- `blur_sigma`：简化全局模糊。
- `grid_coeff`：固定 grid/Moiré basis 系数。

## 4. 当前实现文件

核心代码：

```text
src/inverse_net.py
src/torch_forward.py
scripts/train_inverse_lightsection_net.py
scripts/evaluate_inverse_lightsection_net.py
```

关键文档：

```text
outputs/review_for_gpt_pro_portable.md
outputs/theoretical_model_for_gpt_pro_visual.md
outputs/gpt_pro_review_bundle.zip
docs/research_v0_baseline.md
```

## 5. 当前推荐网络

当前推荐结构：

```text
freq_res_unet_tx
```

结构：

```text
Residual U-Net + FrequencyFiLM + WindowTransformerBlock
```

参数量：

```text
533,138
```

该网络输出 `A_h/A_v`，不直接输出高度，不允许自由生成 `M_theta`。

## 6. 当前最佳配置

```text
sample: 细台阶
groups: ossim_t6_h_3step, ossim_t6_v_3step
crop: center:256
epochs: 50
architecture: freq_res_unet_tx
pattern_source: measured_calibrated
lambda_platform: 0.15
lambda_sec: 0.12
lambda_grid: 0.03
select_best: True
device: cuda
```

输出目录：

```text
outputs/inverse_net_sweep/tx_p015_s012_g003_256_selectbest/
```

## 7. 当前最佳结果

相对 baseline A_cls：

| 指标 | baseline A_cls | 当前 inverse net | 相对变化 |
|---|---:|---:|---:|
| grid energy | 0.888514 | 0.841115 | 下降约 5.33% |
| platform RMS | 0.308210 | 0.078964 | 下降约 74.38% |
| modulation loss | 0.061518 | 0.048277 | 下降约 21.52% |

高度统计：

```text
height_min: 30.790279
height_max: 36.973938
height_mean: 33.683075
height_std: 2.842237
step_height: 5.679253
invalid_fraction: 0
```

## 8. 当前主要限制

1. 没有独立高度真值，不能证明绝对高度准确性。
2. 没有跨样本验证，不能证明泛化。
3. `theta` 是低维退化残差，不等同于完整像差或完整 DMD-CCD 安装误差。
4. Moiré/grid 只是降低，不是完全消除。
5. `--select-best` 在同一样本上选 epoch，存在测试集调参风险。
6. `A_cls` 低通弱约束可能带入传统解调 bias。

## 9. 下一轮优先任务

建议新对话从这里开始：

1. 合并 GPT Pro 新研究计划，形成 `docs/research_plan_v1.md`。
2. 做跨样本验证：
   - train/tune: `细台阶`
   - eval: `台阶曝光2`、`回型曝光20000`、`小孔`
3. 做 theta 消融：
   - fixed theta
   - only delta_q
   - only affine_residual
   - only blur_sigma
   - all theta
4. 做 loss 消融：
   - remove `L_grid_fft`
   - remove `L_weak_Acls`
   - remove `L_platform`
5. 加 z-aware 模型：
   - `Y[z-1],Y[z],Y[z+1] -> A[z]`
6. 设计更物理的 DMD-CCD residual warp：
   - B-spline/TPS residual warp
7. 设计更物理的像差项：
   - Zernike 或空间变化 PSF。

## 10. 新对话建议提示词

可以复制以下内容给新对话：

```text
我正在研究 OS-SIM 反射式三维形貌重建。请先阅读 docs/new_conversation_handoff_v0.md、docs/research_v0_baseline.md、outputs/theoretical_model_for_gpt_pro_visual.md 和 outputs/review_for_gpt_pro_portable.md。当前项目已经实现了 freq_res_unet_tx 神经逆向网络，从 T6 H/V 六帧 raw 条纹恢复 A_h/A_v/A_fused，并用受限 forward model Y_hat = D0 + A_pred * M_theta 估计低维退化残差。当前最佳结果在 细台阶 center:256 上 grid energy 相对 baseline 下降约 5.33%，platform RMS 下降约 74.38%，但尚无独立高度真值和跨样本验证。请基于这些上下文，继续执行 GPT Pro 新生成的研究计划，优先做跨样本验证、theta 消融、loss 消融和更物理的 DMD-CCD/像差模型。
```

## 11. 新对话不要重复踩的坑

- 不要使用缺帧样本 `台阶曝光1` 做训练或结论。
- 不要只考虑水平帧 `100-102`，必须同时考虑竖直帧 `103-105`。
- 不要把早期 `platform_opt` 高度图当成当前神经网络最终结果。
- 不要宣称已经完整恢复像差和 DMD-CCD 安装误差。
- 不要把 `outputs/` 里的大体积 `.npy/.pt/.zip` 全部上传 GitHub。
