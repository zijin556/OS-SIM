# OS-SIM 神经逆向恢复理论模型说明（供 GPT Pro 审查）

生成时间：2026-05-31  
用途：说明当前 OS-SIM 神经逆向恢复项目背后的物理模型、数学假设、优化目标和解释边界。

这份文档重点回答：

1. 当前模型到底在学什么？
2. “理想光切片图”和“退化参数”在数学上如何定义？
3. 现有实现和更完整物理模型之间还有哪些差距？
4. 哪些结论可以作为阶段性研究结论，哪些还不能过度宣称？

---

## 1. 符号定义

设：

```text
z = 1, ..., Z               扫描层索引
g ∈ {H, V}                  条纹方向，水平/竖直两组
m ∈ {0,1,2}                 三步相移索引
(x,y)                       相机像素坐标
Y_{z,g,m}(x,y)              真实采集到的 raw 条纹图
D_{z,g}(x,y)                背景/直流项
A_{z,g}(x,y)                理想光切片调制度，也就是希望恢复的主变量
M_{g,m,θ}(x,y)              退化后的条纹调制模式
θ                           低维退化参数
```

当前输入为 T6 双方向三步：

```text
Y_z = [Y_{z,H,0}, Y_{z,H,1}, Y_{z,H,2}, Y_{z,V,0}, Y_{z,V,1}, Y_{z,V,2}]
```

网络输出：

```text
f_φ(Y_z) = [Â_{z,H}, Â_{z,V}]
```

其中 `φ` 是神经网络参数。

---

## 2. 理想三步 OS-SIM 调制模型

在理想情况下，每个方向的三步相移条纹可以写成：

```math
Y_{z,g,m}(x,y)
= D_{z,g}(x,y)
+ A_{z,g}(x,y)\cos(q_g^\top r + \varphi_m)
+ \epsilon_{z,g,m}(x,y)
```

其中：

```text
r = (x,y)^T
q_g                 条纹空间频率向量
φ_m = 2πm/3          三步相位
ε                   噪声和未建模误差
```

传统三步解调通常估计：

```math
D_{z,g}(x,y) = \frac{1}{3}\sum_{m=0}^{2}Y_{z,g,m}(x,y)
```

```math
A^{cls}_{z,g}(x,y)
= \frac{2}{3}
\sqrt{
  (Y_{z,g,0}-Y_{z,g,1})^2
+ (Y_{z,g,1}-Y_{z,g,2})^2
+ (Y_{z,g,2}-Y_{z,g,0})^2
}
```

传统方法的问题是：真实系统中存在条纹畸变、相位误差、投影-成像映射误差、Moiré/grid 残余、像差、噪声、非均匀照明等，导致 `A_cls` 中仍残留条纹和网格伪影。

---

## 3. 当前实现的受限 forward model

当前实现使用如下可微 forward model：

```math
\hat{Y}_{z,g,m}(x,y)
= D_{z,g}(x,y)
+ \hat{A}_{z,g}(x,y) M_{g,m,\theta}(x,y)
```

其中：

```math
\hat{A}_{z,H}, \hat{A}_{z,V} = f_\phi(Y_z)
```

`M_{g,m,θ}` 不是自由生成的 per-pixel 图，而是从实测/标定条纹模式 `M^0_{g,m}` 出发，经过一组受限低维残差得到：

```math
M_{g,m,\theta}
= \mathcal{N}\{
  \mathcal{B}_{\sigma_g}
  [
    \mathcal{W}_{\alpha_g}
    (
      \mathcal{P}_{\delta \psi_{g,m}}
      (
        M^0_{g,m} + R^{grid}_{g,m}(\beta_g) + R^{q}_{g,m}(\Delta q_g)
      )
    )
  ]
\}
```

这不是代码中的逐字操作顺序的论文式最终模型，而是对实现意图的理论概括。对应实现中的参数：

| 理论项 | 代码参数 | 含义 |
|---|---|---|
| `Δq_g` | `delta_q` | 条纹频率/方向残差 |
| `δψ_{g,m}` | `phase_offsets` | 三步相位残差 |
| `W_{α_g}` | `affine_residual` | 小仿射安装/映射残差 |
| `B_{σ_g}` | `blur_sigma` | 简化全局模糊 |
| `R^{grid}` | `grid_coeff` | 固定 grid/Moiré basis 残差 |
| `N{}` | `normalize_phase_patterns` | 归一化条纹调制模式 |

因此当前模型的核心约束是：

```text
网络不能随便输出一个看着平滑的 A；
它输出的 A 必须能和受限 M_theta 一起重投影回 raw frames。
```

---

## 4. 神经逆向模型

神经网络目标是近似一个逆算子：

```math
f_\phi:
\{Y_{z,H,0},Y_{z,H,1},Y_{z,H,2},Y_{z,V,0},Y_{z,V,1},Y_{z,V,2}\}
\rightarrow
\{\hat{A}_{z,H},\hat{A}_{z,V}\}
```

当前推荐网络是：

```text
freq_res_unet_tx
= Residual U-Net
 + frequency FiLM bottleneck conditioning
 + window transformer
```

理论动机：

1. U-Net 负责局部空间结构恢复。
2. residual blocks 改善训练稳定性和表达力。
3. frequency FiLM 用 raw frame 的 FFT 能量描述调制瓶颈特征，使网络对周期性伪影更敏感。
4. window transformer 在局部窗口内捕捉非卷积相关性，用于建模 Moiré/grid 残余的局部周期关系。

输出非负约束：

```math
\hat{A}_{z,g}(x,y) = softplus(s_{z,g}(x,y))
```

因为调制度物理上应为非负。

---

## 5. H/V 融合模型

恢复 H/V 两个方向的光切片调制度后，当前用 RMS 融合：

```math
\hat{A}^{fused}_{z}(x,y)
= \sqrt{
\frac{
  \hat{A}_{z,H}(x,y)^2
+ \hat{A}_{z,V}(x,y)^2
}{2}
}
```

动机：

- H 和 V 两个方向对方向性伪影的敏感性不同。
- RMS 融合可以降低单方向条纹残余带来的偏差。
- 也可以保留 H/V 单方向结果用于诊断方向性误差。

限制：

- 当前融合是固定公式，不是学习式融合。
- 如果 H/V 两方向退化程度差异很大，固定 RMS 可能不是最优。

可扩展方向：

```math
A^{fused}_z
= w_H(x,y,z)\hat{A}_{z,H}
+ w_V(x,y,z)\hat{A}_{z,V}
```

其中 `w_H,w_V` 可由置信度或小网络预测，并加 `w_H+w_V=1` 约束。

---

## 6. 高度读出模型

从光切片栈恢复高度：

```math
\hat{h}(x,y)
= \operatorname{PeakZ}
\left(
  \{\hat{A}^{fused}_{z}(x,y)\}_{z=1}^{Z}
\right)
```

当前实现使用沿 z 的峰值/抛物线细化读出，可理解为：

```math
z^*(x,y) = \arg\max_z \hat{A}^{fused}_{z}(x,y)
```

再在峰值附近做 parabolic refinement。

限制：

- 当前没有学习式高度解码器。
- 高度准确性依赖 z-stack 峰形是否稳定。
- 没有独立高度真值时，只能用平台 RMS、step height、confidence 和视觉检查做间接评估。

---

## 7. 优化目标

训练时优化：

```math
\min_{\phi,\theta}
\mathcal{L}_{mod}
+ \lambda_{raw}\mathcal{L}_{raw}
+ \lambda_{tv}\mathcal{L}_{tv}
+ \lambda_{grid}\mathcal{L}_{grid}
+ \lambda_{sec}\mathcal{L}_{sec}
+ \lambda_{\theta}\mathcal{L}_{\theta}
+ \lambda_{platform}\mathcal{L}_{platform}
```

### 7.1 调制重投影损失

先去掉 raw frame 的组内均值：

```math
\tilde{Y}_{z,g,m}
= Y_{z,g,m} - \frac{1}{3}\sum_{m'}Y_{z,g,m'}
```

然后：

```math
\mathcal{L}_{mod}
= \rho
\left(
\tilde{Y}_{z,g,m}
- \hat{A}_{z,g}M_{g,m,\theta}
\right)
```

其中 `ρ` 是 Charbonnier loss。

意义：只约束条纹调制项，避免背景强度影响主任务。

### 7.2 raw frame 重投影损失

```math
\mathcal{L}_{raw}
= \rho
\left(
Y_{z,g,m}
- \hat{Y}_{z,g,m}
\right)
```

意义：保证完整 raw frame 一致性。

### 7.3 TV 正则

```math
\mathcal{L}_{tv}
= \|\nabla_x \hat{A}\|_1
+ \|\nabla_y \hat{A}\|_1
```

意义：抑制噪声，但过强会损失微结构。

### 7.4 FFT grid/Moiré 惩罚

对 `A_pred` 做 FFT，在已检测网格/Moiré 频带 mask 上计算能量比：

```math
\mathcal{L}_{grid}
=
\frac{
\| \mathcal{F}(\hat{A}) \odot \Omega_{grid} \|_2
}{
\| \mathcal{F}(\hat{A}) \|_2 + \epsilon
}
```

意义：直接压制特定残余网格频率。

风险：如果 mask 不准确，可能压制真实结构。

### 7.5 传统解调弱约束

使用传统解调 `A^{cls}` 的低通版本：

```math
\mathcal{L}_{sec}
= \rho
\left(
LP(\hat{A}) - LP(A^{cls})
\right)
```

意义：约束低频形貌不漂移。

风险：可能继承传统解调 bias。

### 7.6 theta 正则

```math
\mathcal{L}_{\theta}
=
\|\Delta q\|^2
+ \|\delta\psi\|^2
+ \|\alpha\|^2
+ \|\sigma\|^2
+ \|\beta\|^2
```

意义：防止退化模型过度吸收解释能力。

### 7.7 平台平坦度约束

使用台阶平台 mask：

```math
\mathcal{L}_{platform}
=
\operatorname{Var}(\hat{h}(x,y)\mid (x,y)\in \Omega_{low})
+
\operatorname{Var}(\hat{h}(x,y)\mid (x,y)\in \Omega_{high})
```

实现中使用可微 soft-height 近似读出高度。

意义：利用台阶样本的先验，要求同一平台高度应平坦。

风险：如果权重过强，会过度平滑真实结构。

---

## 8. 当前理论模型和实现结果的对应关系

当前最佳设置：

```text
architecture: freq_res_unet_tx
input: T6 H/V six frames
crop: 256x256
epochs: 50
best_epoch: 49
pattern_source: measured_calibrated
lambda_platform: 0.15
lambda_sec: 0.12
lambda_grid: 0.03
```

主要结果：

```text
grid energy:
  baseline A_cls: 0.888514
  current inverse net: 0.841115
  relative decrease: about 5.33%

platform RMS:
  baseline A_cls: 0.308210
  current inverse net: 0.078964
  relative decrease: about 74.38%

modulation loss:
  baseline modulation loss: 0.061518
  current inverse net: 0.048277
  relative decrease: about 21.52%
```

这说明：

```text
forward consistency、平台平坦度和网格频率压制三方面都有改善。
```

但也说明：

```text
Moiré/grid 只是降低，不是完全消除。
```

---

## 9. 可辨识性问题

当前模型存在典型 blind inverse problem 风险：

```math
Y \approx D + A M_{\theta}
```

如果 `A` 和 `M_θ` 都可变，则可能出现补偿：

```math
A' = A + \Delta A
```

```math
M'_\theta = M_\theta + \Delta M
```

仍然让：

```math
A'M'_\theta \approx AM_\theta
```

因此 `theta` 的每个数值不一定有唯一物理解释。

当前缓解方式：

1. `M_theta` 从实测/标定 pattern 出发。
2. `theta` 是低维小残差。
3. 对 `theta` 加正则。
4. 对 `A` 加低通 classical prior、grid penalty 和平台约束。

但这还不足以证明 theta 是真实物理参数。

需要补充：

- theta 消融实验。
- 仿真数据真值验证。
- 独立标定板验证。
- 跨样本共享 theta 或部分共享 theta。

---

## 10. 当前模型对像差和 DMD-CCD 安装误差的覆盖程度

### 10.1 像差

当前只包含：

```text
global blur_sigma
```

这最多是非常粗略的 blur 近似，不能代表完整像差。

完整像差模型可能需要：

```math
PSF(x,y; a_1,...,a_K)
```

其中 `a_k` 可为 Zernike 系数：

```math
W(\rho,\theta)
= \sum_k a_k Z_k(\rho,\theta)
```

成像退化可写成：

```math
I_{obs}(x,y)
= [I_{ideal} * PSF_{a}(x,y)] + noise
```

更复杂时 PSF 还应空间变化：

```math
PSF = PSF(x,y; a(x,y))
```

当前实现没有达到这个程度。

### 10.2 DMD-CCD 安装误差

当前只包含：

```text
small global affine_residual after calibration
```

理论上 DMD 到 CCD 映射可以写成：

```math
r_{ccd}
= T(r_{dmd})
```

其中更完整的模型应该是：

```math
T(r)
= T_{calib}(r)
+ W_{residual}(r)
```

`W_residual` 可为：

- affine residual
- polynomial distortion
- B-spline warp
- TPS warp
- piecewise affine

当前只有最弱的 global affine residual，不能代表完整安装误差。

---

## 11. 更完整的推荐理论模型

如果后续要写论文或形成更强科研结论，建议把 forward model 扩展为：

```math
\hat{Y}_{z,g,m}
=
\mathcal{C}
\left[
  D_{z,g}
  + \hat{A}_{z,g}
    \cdot
    \mathcal{M}
    (
      P^0_{g,m};
      \Delta q_g,
      \delta\psi_{g,m},
      W_{res},
      a_{aberr},
      \beta_{moire}
    )
\right]
+ \eta
```

其中：

| 项 | 含义 |
|---|---|
| `P^0_{g,m}` | DMD 原始投影图案 |
| `W_res` | DMD-CCD residual warp |
| `a_aberr` | 像差/PSF 参数 |
| `β_moire` | Moiré/grid 残差参数 |
| `C[]` | 相机响应、裁剪、归一化 |
| `η` | 噪声 |

逆向网络：

```math
[\hat{A}_{z,H},\hat{A}_{z,V}, c_z]
= f_\phi(Y_{z-k:z+k})
```

其中 `c_z` 是置信度，可用于 H/V 和 z 方向融合。

高度读出：

```math
\hat{h}(x,y)
=
\frac{
\sum_z z \cdot softmax(\gamma \hat{A}^{fused}_{z}(x,y))
}{
\sum_z softmax(\gamma \hat{A}^{fused}_{z}(x,y))
}
```

或使用 peak + parabolic refinement，并报告置信度。

---

## 12. 建议审查者重点判断

请 GPT Pro 重点检查：

1. 当前 `Y -> A_h/A_v -> A_fused -> height` 是否是合理的 inverse formulation。
2. 当前 `Y_hat = D0 + A_pred * M_theta` 是否足以约束网络，避免伪平滑结果。
3. `M_theta` 的低维参数是否过弱，是否无法解释真实 DMD-CCD 或像差问题。
4. `L_weak_Acls` 是否使网络过度依赖 classical demodulation。
5. `L_platform` 是否让台阶样本结果过于平坦，牺牲真实微结构。
6. `L_grid_fft` 是否可能误伤真实周期结构。
7. `theta` 是否可辨识，是否需要共享 theta、固定 theta 或仿真真值验证。
8. 当前没有跨样本验证时，哪些结论不能写进论文。

---

## 13. 当前阶段最稳妥的理论表述

建议使用如下表述：

```text
We formulate OS-SIM reconstruction as a constrained neural inverse problem.
The network estimates H/V light-section modulation maps from six measured
phase-shifted raw frames. A differentiable, low-dimensional degradation model
projects the estimated modulation maps back to the measured frames, enforcing
raw-frame consistency while restricting the degradation parameters to small
residual corrections around measured/calibrated stripe patterns.
```

中文：

```text
我们将 OS-SIM 重建表述为一个受限神经逆问题。网络从六帧实测相移条纹中估计 H/V 光切片调制度图；一个低维可微退化模型将估计的调制度图重投影回实测帧，以保证 raw-frame 一致性，同时把退化参数限制为围绕实测/标定条纹模式的小残差修正。
```

谨慎表述：

```text
当前 theta 是低维退化残差，不等同于完整像差或完整 DMD-CCD 安装误差。
当前高度结果是相对扫描坐标下的峰值读出结果，尚未由独立高度真值标定。
```

---

## 14. 总结

当前理论模型的核心是：

```text
神经网络负责恢复 A_h/A_v；
受限 forward model 负责约束这些 A 必须能解释 raw frames；
H/V 融合和 z 峰值读出负责生成高度；
theta 负责吸收小范围退化残差。
```

该理论框架适合作为阶段性科研原型，但若要支撑“完整像差恢复”或“完整 DMD-CCD 安装误差估计”的强主张，还需要更完整的物理退化模型、theta 消融、跨样本验证和独立真值验证。
