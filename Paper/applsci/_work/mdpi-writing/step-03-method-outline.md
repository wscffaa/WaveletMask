# Step 03 — Method 审查

审查日期: 2026-05-13

## Method 结构

| § | 内容 | 行 | Figure 消费 |
|---|------|----|-------------|
| 3.1 | Pipeline Overview | 5-28 | ❌ 未引用 Fig2 |
| 3.2 | Local High-Frequency Mask | 64-82 | ✅ `\ref{fig:module_detail}` (行66) |
| 3.3 | Global Flicker Mask | 84-108 | 间接消费 Fig3 |
| 3.4 | Max Fusion and Integration | 110-116 | — |
| 3.5 | Training Objective | 118-124 | — |
| 3.6 | Evaluation Metrics | 126-136 | — |

## Figure 消费核对

| Contract | Required Ref | Method 实际 | 匹配？ |
|----------|-------------|------------|--------|
| Fig2 (METHOD_OVERVIEW) | `\ref{fig:pipeline}` in Method | 未引用 | ❌ 缺失 |
| Fig3 (MODULE_DETAIL_WaveletMask) | `\ref{fig:module_detail}` in Method | §3.2 行66 | ✅ |

## 代码一致性

| 代码类名 | Method 文本 | 一致？ |
|---------|------------|--------|
| WaveletNoiseMask | WaveletNoiseMask (行17) | ✅ |
| HaarDWT2d | HaarDWT (行69) | ⚠️ 缺少"2d"后缀 |
| high_freq_conv | "lightweight CNN" (行79) | ✅ 描述性 |
| low_freq_conv | "separate lightweight CNN" (行98) | ✅ 描述性 |
| PixelWiseMaskFusion | PixelWiseMaskFusion (行116) | ✅ |

## Method 三元素检查（彭思达）

| 模块 | Design | Motivation | Technical Advantage |
|------|--------|------------|-------------------|
| Local HF Mask | ✅ Eq.7-8 | ✅ 行66 | ✅ 几何均值抑制运动 |
| Flicker Mask | ✅ Eq.11-12 | ✅ 行86 | ✅ 最小值拒绝渐变 |
| Max Fusion | ✅ Eq.10 | ✅ 行108/112 | ✅ 互补互补零参数 |

## 发现问题

### P1 — Fig2 未在 Method 中引用
- Method Overview (§3.1) 描述了 pipeline 但没有 `\ref{fig:pipeline}`
- 影响：读者看 Method 时无法定位架构总图
- 修复：§3.1 首段或次段加一句 "The overall pipeline is illustrated in Figure~\ref{fig:pipeline}."

### P2 — HaarDWT vs HaarDWT2d 命名不一致
- 代码类名 `HaarDWT2d`，Method 文本写 `HaarDWT`
- 影响：reviewer 检查代码复现时可能困惑
- 修复：全文统一为 `HaarDWT2d` 或在首次出现时注明 "HaarDWT (implemented as HaarDWT2d)"

### 自检通过项
- [x] METHOD_OVERVIEW contract 存在（Fig2）
- [x] MODULE_DETAIL_WaveletMask contract 存在（Fig3）
- [x] 每个子节有 module motivation 在公式之前
- [x] 所有公式变量已定义
- [x] 模块名与 module_names.json 一致（除 HaarDWT 后缀）
- [x] Algorithm 1 伪代码与文本描述一致
- [x] 训练 loss 与 RTN baseline 兼容（行120-124）

## 实验状态

实验已完成（ablation_matrix.json 含 L40 指标，main_results.yaml 存在），跳过训练启动。直接进 Step 4。

## 下一步

Step 4: 改 Introduction + Method，修复 Fig2 引用缺失和 HaarDWT 命名
