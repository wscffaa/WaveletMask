# Step 00 Motivation Anchor

## 源文件检查

| 产物 | 状态 | 说明 |
|------|------|------|
| `Idea/03-SCI-故事线.md` | present | 4 贡献点：频率解耦、几何均值时序滤波、超可加互补效应、零参数即插即用 |
| `Idea/00-创新分析.md` | present | 创新 A（局部高频掩码）+ B（全局闪烁掩码）+ max 融合；C（像素级融合）已淘汰 |
| `Idea/01-baseline-decision.md` | present | Baseline = RTN (Swin Spatial)；未选 MambaOFR（DCN 耦合风险高） |
| `Idea/diff_plan.json` | present | S3_DegPrompt slot 替换 residual_indicator → WaveletNoiseMask |
| `Idea/ablation_matrix.json` | present | 8 变体 + 1 baseline = 9 总；主消融 4 变体（Base/A/B/AB） |

## 候选 Motivation

### M1. 频率解耦（Frequency Decoupling）

- Motivation: 像素域 residual indicator 混淆运动纹理与真实损伤，小波域频率分解天然分离局部高频损伤和全局低频闪烁
- Suitable figure archetypes: `frequency_spectrum`, `degradation_showcase`
- Required evidence: 小波子带（LL/LH/HL/HH）可视化对比 + 像素域 indicator 在运动场景下的假阳性案例
- Reviewer risk: "为什么不用其他频域变换（FFT/DCT）而是小波？" → 小波提供空间-频率联合定位

### M2. 退化模式分离（Degradation Pattern Separation）

- Motivation: 老电影有两类退化模式——局部瞬时损伤（划痕/灰尘）和全局低频闪烁，现有方法用统一 indicator 无法区分
- Suitable figure archetypes: `method_limitation_grid`, `degradation_taxonomy`
- Required evidence: 局部 vs 全局退化的对比帧 + 掩码分离效果
- Reviewer risk: "两类退化的交互/耦合情况怎么处理？"

### M3. 超可加互补效应（Super-additive Complementarity）

- Motivation: 消融发现 WaveletMask = +0.44dB，超过 +LHFM only (+0.22) 和 +GFM only (-0.06) 的线性叠加，且 4 种学习型融合均不如简单 max
- Suitable figure archetypes: `ablation_table_visual`, `bar_comparison`
- Required evidence: 8 变体消融完整数据
- Reviewer risk: "是否只是这个数据集上的偶然？"

### M4. 即插即用泛化性（Plug-and-Play Generalizability）

- Motivation: WaveletMask 是零参数模块，可泛化到任何使用 residual indicator 的循环修复网络
- Suitable figure archetypes: `pipeline_overview`, `scatter_pareto`
- Required evidence: 多方法对比表
- Reviewer risk: "零参数声明的边界条件是什么？"

## 选定 Motivation

- Motivation ID: M1
- 一句话论文声明: 小波域频率分解为视频修复提供了物理驱动的退化感知能力，天然分离像素域 indicator 无法区分的局部高频损伤和全局低频闪烁
- 为什么选这个故事: 频率解耦是最底层的 insight——解释了为什么要把问题搬到小波域（M2 退化分离是它的推论，M3 超可加是它的实验验证，M4 泛化性是它的工程属性）。Figure 1 已有手绘成品展示此 motivation。
- 本故事不得声明的边界: 不声称 WaveletMask 在所有退化类型上都优于像素域方法（仅在老电影特有的局部+闪烁退化组合上验证）；不声称小波变换在所有频域方法中最优（只说它在空间-频率联合定位上的优势）

## 5-Slide Talk 大纲

| Slide | 目的 | 内容 | Figure 锚点 |
|-------|---------|---------|---------------|
| 1 | 问题 | 老电影退化严重影响文化遗产保存；退化模式复杂（划痕/灰尘/闪烁同时存在） | none |
| 2 | 已有方法失败 | 像素域 residual indicator 在运动区域假阳性、在静态闪烁区域假阴性，掩码不准 | Fig. 1 motivation |
| 3 | 核心洞察 | 小波子带天然分离局部高频损伤（LH/HL/HH）和全局低频闪烁（LL 时序差异） | Fig. 1 motivation |
| 4 | 方法 | WaveletMask：Haar DWT → LHFM + GFM → max fusion → 注入 RTN 门控聚合 | Fig. 2 pipeline, Fig. 3 module |
| 5 | 证据 | +0.75 dB vs RTN, +0.61 dB vs MambaOFR；LHFM/GFM 互补效应；max > learned fusion | Fig. 4/5 qualitative |

## 故事弧线

1. 问题: 老电影同时存在局部瞬时损伤（划痕/灰尘，高频）和全局低频闪烁（亮度漂移，低频），两者需要不同的检测策略
2. 已有方法失败: 现有视频修复方法（RTN, RVRT 等）用像素域帧间差作为 residual indicator，但像素域无法区分运动纹理（高频非损伤）和真实噪声（高频是损伤），也捕获不到低频闪烁
3. 本文洞察: Haar 小波变换将信号分解为频率子带——LH/HL/HH 子带捕获空间高频变化（局部损伤的自然载体），LL 子带的时序差异反映全局亮度漂移（闪烁的自然载体），分离是物理驱动的
4. 方法: WaveletMask 在 Haar DWT 域分别构建 LHFM（几何均值时序滤波）和 GFM（低频时序响应），通过 element-wise max 融合替换 RTN 的 pixel-domain indicator
5. 证据: WaveletMask 在 REDS-val4 达 26.60 dB PSNR；LHFM/GFM 展现互补行为（+0.44 dB 相对 re-trained RTN baseline）；4 种学习型融合均劣于 max

## 叙事锚点

### 锚点 A1: 频域退化分离

- 支撑的声明: 小波子带可视化证明高频子带捕获局部损伤、低频时序差异捕获闪烁，而像素域 indicator 两者混淆
- 所需 figure 角色: motivation
- 候选 archetype: `frequency_spectrum`, `degradation_showcase`
- 证据路径: `Paper/applsci/figures/Figure1.pdf`（已手绘完成，archetype=frequency_spectrum）
- Reviewer 风险: "FFT/DCT 也能做频率分解，为什么选 Haar 小波？" → 空间-频率联合定位 + 正交分解 + 计算效率

### 锚点 A2: 双掩码融合架构

- 支撑的声明: WaveletMask 将独立的局部/闪烁掩码通过 max 融合注入门控聚合，零额外参数
- 所需 figure 角色: method
- 候选 archetype: `pipeline_overview`, `module_detail`
- 证据路径: `archs/components/wavelet_mask.py` → WaveletNoiseMask 类（代码已验证）
- Reviewer 风险: "max 融合是否丢失了掩码间的交互信息？" → 消融证明 learned fusion 反而更差

### 锚点 A3: 超可加互补效应

- 支撑的声明: WaveletMask 的 PSNR 增益（+0.44dB）超过 +LHFM only (+0.22) 与 +GFM only (-0.06) 的线性叠加，且 max 优于 4 种 learned fusion
- 所需 figure 角色: results / discussion
- 候选 archetype: `bar_comparison`, `ablation_table_visual`
- 证据路径: `Exp/paper_data/main_results.yaml` + `Exp/ablation/ablation_results.yaml`（数据已完成）
- Reviewer 风险: "仅在一个数据集上验证的超可加性是否可推广？"

## Step 1 交接清单

- 所需 figure 数量: 8
- 必须的 figure 角色:
  - motivation visualization（Figure 1 — 已有手绘成品）
  - pipeline overview（Figure 2 — PPT 手绘）
  - LHFM module detail（Figure 3 — PPT 手绘）
  - GFM module detail（Figure 4 — PPT 手绘）
  - qualitative comparison - synthetic（Figure 5 — PPT 手绘）
  - qualitative comparison - real-world（Figure 6 — 条件：需可追溯的 real-world 输出）
- 可选 figure 角色:
  - ablation visual（Figure 7 — 可选加分）
  - error map diagnostic（Figure 8 — 条件：synthetic ground truth 支撑）
- 锁定术语:
  - 方法名: WaveletMask（`Idea/method_name.json`）
  - 模块名: WaveletNoiseMask, HaarDWT2d, high_freq_conv, low_freq_conv（`Idea/module_names.json`）
  - 主要数据集: REDS-val4 (synthetic), real-world old films (if traceable)
- 门控决策: user confirmed
