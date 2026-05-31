# OS-SIM 神经逆向恢复理论模型说明（公式可视版）

生成时间：2026-05-31  
用途：给人阅读和给 GPT Pro 审查。这个版本避免把公式放在灰色代码块里，便于 VS Code 的 Markdown Preview Enhanced 渲染。

---

## 1. 符号定义

| 符号 | 含义 |
|---|---|
| $z=1,\ldots,Z$ | 扫描层索引 |
| $g\in\{H,V\}$ | 条纹方向，水平/竖直两组 |
| $m\in\{0,1,2\}$ | 三步相移索引 |
| $(x,y)$ | 相机像素坐标 |
| $Y_{z,g,m}(x,y)$ | 真实采集到的 raw 条纹图 |
| $D_{z,g}(x,y)$ | 背景/直流项 |
| $A_{z,g}(x,y)$ | 理想光切片调制度，也是希望恢复的主变量 |
| $M_{g,m,\theta}(x,y)$ | 退化后的条纹调制模式 |
| $\theta$ | 低维退化参数 |
| $\phi$ | 神经网络参数 |

当前输入为 T6 双方向三步：

$$
Y_z=
\left[
Y_{z,H,0},
Y_{z,H,1},
Y_{z,H,2},
Y_{z,V,0},
Y_{z,V,1},
Y_{z,V,2}
\right].
$$

网络输出：

$$
f_{\phi}(Y_z)
=
\left[
\hat{A}_{z,H},
\hat{A}_{z,V}
\right].
$$

---

## 2. 理想三步 OS-SIM 调制模型

理想情况下，每个方向的三步相移条纹可写为：

$$
Y_{z,g,m}(x,y)
=
D_{z,g}(x,y)
+
A_{z,g}(x,y)
\cos\left(q_g^\top r+\varphi_m\right)
+
\epsilon_{z,g,m}(x,y).
$$

其中：

| 符号 | 含义 |
|---|---|
| $r=(x,y)^\top$ | 像素坐标向量 |
| $q_g$ | 条纹空间频率向量 |
| $\varphi_m=2\pi m/3$ | 三步相移相位 |
| $\epsilon_{z,g,m}$ | 噪声和未建模误差 |

传统三步解调会估计直流项：

$$
D_{z,g}(x,y)
=
\frac{1}{3}
\sum_{m=0}^{2}
Y_{z,g,m}(x,y).
$$

以及调制度：

$$
A^{cls}_{z,g}(x,y)
=
\frac{2}{3}
\sqrt{
\left(Y_{z,g,0}-Y_{z,g,1}\right)^2
+
\left(Y_{z,g,1}-Y_{z,g,2}\right)^2
+
\left(Y_{z,g,2}-Y_{z,g,0}\right)^2
}.
$$

传统方法的问题是，真实系统存在条纹畸变、相位误差、投影-成像映射误差、Moiré/grid 残余、像差、噪声和非均匀照明，导致 $A^{cls}$ 中仍残留伪影。

---

## 3. 当前实现的受限 Forward Model

当前实现使用的可微 forward model 为：

$$
\hat{Y}_{z,g,m}(x,y)
=
D_{z,g}(x,y)
+
\hat{A}_{z,g}(x,y)
M_{g,m,\theta}(x,y).
$$

网络负责估计：

$$
\hat{A}_{z,H},\hat{A}_{z,V}
=
f_{\phi}(Y_z).
$$

$M_{g,m,\theta}$ 不是自由生成的 per-pixel 图，而是从实测/标定条纹模式 $M^0_{g,m}$ 出发，经过一组受限低维残差得到。可概括为：

$$
M_{g,m,\theta}
=
\mathcal{N}
\left\{
\mathcal{B}_{\sigma_g}
\left[
\mathcal{W}_{\alpha_g}
\left(
\mathcal{P}_{\delta\psi_{g,m}}
\left(
M^0_{g,m}
+
R^{grid}_{g,m}(\beta_g)
+
R^{q}_{g,m}(\Delta q_g)
\right)
\right)
\right]
\right\}.
$$

该式是理论概括，不是代码逐行顺序。实现中的参数对应关系：

| 理论项 | 代码参数 | 含义 |
|---|---|---|
| $\Delta q_g$ | `delta_q` | 条纹频率/方向残差 |
| $\delta\psi_{g,m}$ | `phase_offsets` | 三步相位残差 |
| $\mathcal{W}_{\alpha_g}$ | `affine_residual` | 小仿射安装/映射残差 |
| $\mathcal{B}_{\sigma_g}$ | `blur_sigma` | 简化全局模糊 |
| $R^{grid}$ | `grid_coeff` | 固定 grid/Moiré basis 残差 |
| $\mathcal{N}\{\cdot\}$ | `normalize_phase_patterns` | 归一化条纹调制模式 |

核心约束是：

$$
\text{网络输出的 } \hat{A}
\text{ 必须能和受限 } M_{\theta}
\text{ 一起重投影回 raw frames。}
$$

---

## 4. 神经逆向模型

神经网络近似一个逆算子：

$$
f_{\phi}:
\left\{
Y_{z,H,0},
Y_{z,H,1},
Y_{z,H,2},
Y_{z,V,0},
Y_{z,V,1},
Y_{z,V,2}
\right\}
\rightarrow
\left\{
\hat{A}_{z,H},
\hat{A}_{z,V}
\right\}.
$$

当前推荐网络：

$$
\texttt{freq\_res\_unet\_tx}
=
\text{Residual U-Net}
+
\text{Frequency FiLM}
+
\text{Window Transformer}.
$$

理论动机：

| 模块 | 作用 |
|---|---|
| Residual U-Net | 恢复局部空间结构 |
| Frequency FiLM | 用 raw frame 的 FFT 能量描述调制 bottleneck 特征 |
| Window Transformer | 捕捉局部窗口内非卷积周期关系 |

输出非负约束：

$$
\hat{A}_{z,g}(x,y)
=
\operatorname{softplus}
\left(
s_{z,g}(x,y)
\right).
$$

---

## 5. H/V 融合模型

恢复 H/V 两方向光切片调制度后，当前使用 RMS 融合：

$$
\hat{A}^{fused}_{z}(x,y)
=
\sqrt{
\frac{
\hat{A}_{z,H}(x,y)^2
+
\hat{A}_{z,V}(x,y)^2
}{2}
}.
$$

这样做的理由：

1. H 和 V 对方向性伪影的敏感性不同。
2. RMS 融合可降低单方向条纹残余带来的偏差。
3. H/V 单方向结果仍可保留用于诊断方向性误差。

可扩展成置信度加权融合：

$$
A^{fused}_z(x,y)
=
w_H(x,y,z)\hat{A}_{z,H}(x,y)
+
w_V(x,y,z)\hat{A}_{z,V}(x,y),
$$

并加约束：

$$
w_H(x,y,z)+w_V(x,y,z)=1.
$$

---

## 6. 高度读出模型

从融合光切片栈恢复高度：

$$
\hat{h}(x,y)
=
\operatorname{PeakZ}
\left(
\left\{
\hat{A}^{fused}_{z}(x,y)
\right\}_{z=1}^{Z}
\right).
$$

最基本的峰值位置：

$$
z^*(x,y)
=
\arg\max_z
\hat{A}^{fused}_{z}(x,y).
$$

当前实现还会在峰值附近做 parabolic refinement。限制是：没有独立高度真值时，高度准确性只能通过平台 RMS、step height、置信度和视觉检查间接评估。

---

## 7. 优化目标

训练优化：

$$
\min_{\phi,\theta}
\mathcal{L}_{mod}
+
\lambda_{raw}\mathcal{L}_{raw}
+
\lambda_{tv}\mathcal{L}_{tv}
+
\lambda_{grid}\mathcal{L}_{grid}
+
\lambda_{sec}\mathcal{L}_{sec}
+
\lambda_{\theta}\mathcal{L}_{\theta}
+
\lambda_{platform}\mathcal{L}_{platform}.
$$

### 7.1 调制重投影损失

先去掉 raw frame 的组内均值：

$$
\tilde{Y}_{z,g,m}
=
Y_{z,g,m}
-
\frac{1}{3}
\sum_{m'=0}^{2}
Y_{z,g,m'}.
$$

调制损失：

$$
\mathcal{L}_{mod}
=
\rho
\left(
\tilde{Y}_{z,g,m}
-
\hat{A}_{z,g}
M_{g,m,\theta}
\right),
$$

其中 $\rho$ 是 Charbonnier loss。

### 7.2 Raw Frame 重投影损失

$$
\mathcal{L}_{raw}
=
\rho
\left(
Y_{z,g,m}
-
\hat{Y}_{z,g,m}
\right).
$$

### 7.3 TV 正则

$$
\mathcal{L}_{tv}
=
\left\|
\nabla_x\hat{A}
\right\|_1
+
\left\|
\nabla_y\hat{A}
\right\|_1.
$$

### 7.4 FFT Grid/Moiré 惩罚

$$
\mathcal{L}_{grid}
=
\frac{
\left\|
\mathcal{F}(\hat{A})
\odot
\Omega_{grid}
\right\|_2
}{
\left\|
\mathcal{F}(\hat{A})
\right\|_2
+
\epsilon
}.
$$

其中 $\Omega_{grid}$ 是已检测到的残余网格/Moiré 频带 mask。

### 7.5 传统解调弱约束

$$
\mathcal{L}_{sec}
=
\rho
\left(
LP(\hat{A})
-
LP(A^{cls})
\right).
$$

这里 $LP(\cdot)$ 是低通滤波。它能防止网络漂移，但也可能带回传统解调的 bias。

### 7.6 Theta 正则

$$
\mathcal{L}_{\theta}
=
\left\|\Delta q\right\|^2
+
\left\|\delta\psi\right\|^2
+
\left\|\alpha\right\|^2
+
\left\|\sigma\right\|^2
+
\left\|\beta\right\|^2.
$$

### 7.7 平台平坦度约束

对于低平台区域 $\Omega_{low}$ 和高平台区域 $\Omega_{high}$：

$$
\mathcal{L}_{platform}
=
\operatorname{Var}
\left(
\hat{h}(x,y)
\mid
(x,y)\in\Omega_{low}
\right)
+
\operatorname{Var}
\left(
\hat{h}(x,y)
\mid
(x,y)\in\Omega_{high}
\right).
$$

实现中用可微 soft-height 近似高度读出。

---

## 8. 当前结果和理论模型的对应

当前最佳配置：

| 项 | 值 |
|---|---|
| architecture | `freq_res_unet_tx` |
| input | T6 H/V six frames |
| crop | 256×256 |
| epochs | 50 |
| best_epoch | 49 |
| pattern_source | measured_calibrated |
| $\lambda_{platform}$ | 0.15 |
| $\lambda_{sec}$ | 0.12 |
| $\lambda_{grid}$ | 0.03 |

主要结果：

| 指标 | baseline A_cls | 当前 inverse net | 相对变化 |
|---|---:|---:|---:|
| grid energy | 0.888514 | 0.841115 | 下降约 5.33% |
| platform RMS | 0.308210 | 0.078964 | 下降约 74.38% |
| modulation loss | 0.061518 | 0.048277 | 下降约 21.52% |

解释：

1. forward consistency 有改善。
2. 平台平坦度明显改善。
3. 网格/Moiré 频率能量有所下降，但未完全消除。

---

## 9. 可辨识性问题

当前模型本质上是 blind inverse problem：

$$
Y
\approx
D
+
A M_{\theta}.
$$

如果 $A$ 和 $M_{\theta}$ 都可变，可能存在补偿：

$$
A' = A+\Delta A,
$$

$$
M'_{\theta} = M_{\theta}+\Delta M,
$$

并且仍然满足：

$$
A'M'_{\theta}
\approx
A M_{\theta}.
$$

因此，$\theta$ 的每个数值不一定有唯一物理解释。

当前缓解方式：

1. $M_{\theta}$ 从实测/标定 pattern 出发。
2. $\theta$ 是低维小残差。
3. 对 $\theta$ 加正则。
4. 对 $\hat{A}$ 加低通 classical prior、grid penalty 和平台约束。

但这还不足以证明 $\theta$ 是真实物理参数。

---

## 10. 像差和 DMD-CCD 安装误差的覆盖程度

### 10.1 像差

当前只包含全局简化模糊：

$$
\sigma_g = \texttt{blur\_sigma}.
$$

这不能代表完整像差。更完整的模型可能需要空间变化 PSF：

$$
I_{obs}(x,y)
=
\left[
I_{ideal}
*
PSF_a(x,y)
\right]
+
noise.
$$

也可以用 Zernike 表示波前像差：

$$
W(\rho,\theta)
=
\sum_k
a_k
Z_k(\rho,\theta).
$$

当前实现没有达到这个程度。

### 10.2 DMD-CCD 安装误差

理论上 DMD 到 CCD 映射可写成：

$$
r_{ccd}
=
T(r_{dmd}).
$$

更完整模型应该是：

$$
T(r)
=
T_{calib}(r)
+
W_{residual}(r).
$$

其中 $W_{residual}$ 可以是：

| 模型 | 能表达的误差 |
|---|---|
| affine residual | 全局平移、旋转、缩放、剪切 |
| polynomial distortion | 低阶镜头畸变 |
| B-spline warp | 平滑非线性残余形变 |
| TPS warp | 控制点驱动的非线性形变 |
| piecewise affine | 分块局部安装误差 |

当前只有小的 global affine residual，不能代表完整安装误差。

---

## 11. 更完整的推荐理论模型

后续可以把 forward model 扩展为：

$$
\hat{Y}_{z,g,m}
=
\mathcal{C}
\left[
D_{z,g}
+
\hat{A}_{z,g}
\cdot
\mathcal{M}
\left(
P^0_{g,m};
\Delta q_g,
\delta\psi_{g,m},
W_{res},
a_{aberr},
\beta_{moire}
\right)
\right]
+
\eta.
$$

其中：

| 符号 | 含义 |
|---|---|
| $P^0_{g,m}$ | DMD 原始投影图案 |
| $W_{res}$ | DMD-CCD residual warp |
| $a_{aberr}$ | 像差/PSF 参数 |
| $\beta_{moire}$ | Moiré/grid 残差参数 |
| $\mathcal{C}[\cdot]$ | 相机响应、裁剪、归一化 |
| $\eta$ | 噪声 |

逆向网络也可以加入 z 上下文：

$$
\left[
\hat{A}_{z,H},
\hat{A}_{z,V},
c_z
\right]
=
f_{\phi}
\left(
Y_{z-k:z+k}
\right).
$$

高度也可以使用 soft-argmax：

$$
\hat{h}(x,y)
=
\frac{
\sum_z
z
\cdot
\operatorname{softmax}
\left(
\gamma
\hat{A}^{fused}_{z}(x,y)
\right)
}{
\sum_z
\operatorname{softmax}
\left(
\gamma
\hat{A}^{fused}_{z}(x,y)
\right)
}.
$$

---

## 12. 建议 GPT Pro 重点审查

1. $Y\rightarrow A_H,A_V\rightarrow A^{fused}\rightarrow h$ 是否是合理的 inverse formulation。
2. $\hat{Y}=D+\hat{A}M_{\theta}$ 是否足以约束网络，避免伪平滑结果。
3. 当前 $M_{\theta}$ 的低维参数是否太弱，是否无法解释真实 DMD-CCD 或像差问题。
4. $\mathcal{L}_{sec}$ 是否会让网络过度依赖 classical demodulation。
5. $\mathcal{L}_{platform}$ 是否会过度平滑台阶样本。
6. $\mathcal{L}_{grid}$ 是否可能误伤真实周期结构。
7. $\theta$ 是否可辨识，是否需要仿真真值或独立标定验证。
8. 没有跨样本验证时，哪些结论不能写进论文。

---

## 13. 当前最稳妥的理论表述

英文表述：

> We formulate OS-SIM reconstruction as a constrained neural inverse problem. The network estimates H/V light-section modulation maps from six measured phase-shifted raw frames. A differentiable, low-dimensional degradation model projects the estimated modulation maps back to the measured frames, enforcing raw-frame consistency while restricting the degradation parameters to small residual corrections around measured/calibrated stripe patterns.

中文表述：

> 我们将 OS-SIM 重建表述为一个受限神经逆问题。网络从六帧实测相移条纹中估计 H/V 光切片调制度图；一个低维可微退化模型将估计的调制度图重投影回实测帧，以保证 raw-frame 一致性，同时把退化参数限制为围绕实测/标定条纹模式的小残差修正。

谨慎表述：

> 当前 $\theta$ 是低维退化残差，不等同于完整像差或完整 DMD-CCD 安装误差。当前高度结果是相对扫描坐标下的峰值读出结果，尚未由独立高度真值标定。

---

## 14. 总结

当前理论模型的核心是：

1. 神经网络负责恢复 $\hat{A}_{z,H}$ 和 $\hat{A}_{z,V}$。
2. 受限 forward model 约束这些 $A$ 必须能解释 raw frames。
3. H/V 融合和 z 峰值读出生成高度。
4. $\theta$ 只吸收小范围退化残差。

这个理论框架适合作为阶段性科研原型。但如果要支撑“完整像差恢复”或“完整 DMD-CCD 安装误差估计”的强主张，还需要更完整的物理退化模型、theta 消融、跨样本验证和独立真值验证。
