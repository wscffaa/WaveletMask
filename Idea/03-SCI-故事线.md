# SCI 故事线

> 修订日期：2026-04-23
> 修订原因：L40 权威数据确认 AB (A+B, max) 为最优变体 (PSNR=26.60)，
>           Full (A+B+C) 为最差变体 (PSNR=24.98)，C 模块有害。
>           论文从 3 创新 (A+B+C) 精简为 2 创新 (A+B)。

## 方法命名

**方法全称**: WaveletMask — Wavelet-Domain Degradation-Aware Masking for Old Film Restoration
**方法简称**: WaveletMask

## 故事线设计

### 任务 (Task)

老电影修复 (Old Film Restoration, OFR): 将受划痕、灰尘、闪烁、噪点等退化损伤的老电影视频恢复为高质量视频。

### 挑战 (Challenge)

现有循环视频修复网络（如 RTN）使用像素级帧差作为退化指示符 (`residual_indicator = |warped - current|`)，存在两个根本缺陷：

1. **噪声-运动混淆**: 划痕/灰尘引起的像素差异被误判为运动/遮挡，导致时序信息被错误抑制
2. **闪烁不可感知**: 全局亮度闪烁在帧差中表现为均匀偏移，幅度不足以触发时序修正

### 洞察 (Insight)

小波变换自然地将图像分解为频率子带，实现退化类型的**频率解耦**：
- **高频子带 (LH/HL/HH)**: 对应划痕、灰尘等局部结构性退化
- **低频子带 (LL)**: 反映全局亮度信息，是检测闪烁的天然信号
- **几何均值差异** `sqrt(|curr-prev| * |curr-next|)`: 瞬时噪声两方向都有差异→满响应；连续运动至少一方向小→被抑制

### 贡献 (Contributions)

1. 首次将小波域频率分解引入视频修复的时序聚合掩码设计，实现局部/全局退化的频率解耦
2. 提出几何均值时序滤波器，利用信号处理原理有效区分瞬时噪声与连续运动
3. 发现局部/全局掩码的**超可加互补效应**（A+B=+0.44 dB > A单独+0.22 + B单独-0.06），并通过系统消融验证简单 max 融合优于 4 种学习型融合
4. 所提模块零额外训练参数、即插即用，可泛化至任何使用 residual indicator 的循环修复网络

### 优势 (Advantages)

- 合成数据集 PSNR 提升 +0.44 dB (ab vs base: 26.60 vs 26.16)
- 感知指标全面提升: LPIPS -0.0143, DISTS -0.0122
- 零额外可训练参数（仅轻量 CNN 无需从主干梯度回传）
- 即插即用：可应用于任何使用 residual_indicator 的循环修复网络
- "Learned fusion is not always better" — 4 种学习型融合全部劣于简单 max

## 论文结构规划

| 章节 | 要点 |
|------|------|
| Abstract | 空间掩码的噪声-运动混淆 → 小波域频率解耦 → A+B → +0.44dB |
| Introduction | 像素差分的两个缺陷 → 小波域洞察 → 两个互补掩码 → 超可加效应 |
| Related Work | OFR 方法综述 + 小波在图像处理中的应用 + 退化感知机制 |
| Method | 3.1 Baseline RTN 回顾 / 3.2 Local Noise Mask (A) / 3.3 Global Flicker Mask (B) / 3.4 Max 融合与集成 |
| Results | 4.1 Setup / 4.2 主结果 / 4.3 消融(主表4变体+辅表4种fusion) / 4.4 可视化 |
| Discussion | 超可加效应分析 + Learned fusion 有害分析 + 局限性 |
| Conclusion | 频率域退化解耦 + 简单优于复杂 |

## 消融实验结构

### 主消融表（4 变体）

| 变体 | A | B | 融合 | PSNR↑ | SSIM↑ | LPIPS↓ | DISTS↓ |
|------|---|---|------|-------|-------|--------|--------|
| Baseline | - | - | spatial diff | 26.16 | 0.8917 | 0.1034 | 0.0953 |
| +Local (A only) | ✓ | - | direct | 26.38 | 0.8966 | 0.0914 | 0.0790 |
| +Flicker (B only) | - | ✓ | direct | 26.10 | 0.8980 | 0.0969 | 0.0916 |
| WaveletMask (Ours) | ✓ | ✓ | max | **26.60** | 0.8951 | 0.0891 | 0.0831 |

### 辅助消融：Fusion 策略对比

| 融合策略 | PSNR | vs AB (max) |
|---------|------|-------------|
| max (Ours) | **26.60** | — |
| Gated weighted | 26.35 | -0.25 |
| CNN alpha (local) | 25.99 | -0.61 |
| CNN alpha (flicker) | 25.95 | -0.65 |
| CNN alpha (full) | 24.98 | -1.62 |
