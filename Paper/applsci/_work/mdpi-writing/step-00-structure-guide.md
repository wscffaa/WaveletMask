# 结构指南 — 007 WaveletMask

## 模型信息

- 项目: `007-waveletmask`
- 方法名: `WaveletMask`
- 最终架构: `WaveletMaskABNet`
- 任务: 老电影修复 (Old Film Restoration)
- 目标期刊: MDPI Applied Sciences
- 证据策略: synthetic 声明已冻结；RealWorld 排名声明因过期日志仍被阻塞。

## Introduction 结构

| 段落 | 功能 | 必须内容 | Figure 锚点 |
|------|------|----------|-------------|
| 1 | 问题价值 | 老电影退化损害修复和保存。 | 无 |
| 2 | Baseline 局限 | 像素域 residual indicator 混淆运动/噪声，遗漏闪烁。 | `Figure1` |
| 3 | 核心思路 | 小波子带分离局部高频损伤和低频闪烁。 | `Figure1` |
| 4 | 方法预览 | WaveletNoiseMask 计算局部/闪烁掩码并 max 融合注入 RTN 门控聚合。 | `Figure2` |
| 5 | 证据预览 | `waveletmask_ab` 是 synthetic 最佳提交模型；RealWorld 声明为临时状态。 | `Figure6`（仅在可追溯输出存在后） |
| 6 | 贡献列表 | 频域掩码、几何均值/局部-闪烁分离、max 融合优于 learned fusion。 | `Figure3`, `Figure5` |
| 7 | 论文概览 | 可选，保持简短。 | 无 |

## Method 模块清单

| 模块 | 代码名 | 角色 |
|------|--------|------|
| WaveletNoiseMask | `WaveletNoiseMask` | 最终 LHFM+GFM 掩码生成器（max 融合）。 |
| HaarDWT2d | `HaarDWT2d` | 纯 PyTorch Haar DWT 分解。 |
| LHFM | `high_freq_conv` | 从 LH/HL/HH 细节子带检测划痕/灰尘证据。 |
| GFM | `low_freq_conv` | 从 LL 响应检测低频亮度闪烁。 |
| PixelWiseMaskFusion | `PixelWiseMaskFusion` | 已归档的 learned fusion 变体；不是提交方法。 |

## 消融矩阵

| 变体 | 含义 | 论文角色 |
|------|------|----------|
| `waveletmask_base` | RTN baseline（像素 residual indicator） | baseline |
| `waveletmask_local` | +LHFM only | 消融 |
| `waveletmask_flicker` | +GFM only | 消融 |
| `waveletmask_ab` | WaveletMask（LHFM+GFM max fusion） | 最终提交方法 |
| `waveletmask_local_fusion` | +LHFM + learned fusion | 负消融 |
| `waveletmask_flicker_fusion` | +GFM + learned fusion | 负消融 |
| `waveletmask_full` | WaveletMask + learned fusion | 负消融 |
| `waveletmask_gated_flicker` | WaveletMask + gated learned fusion | 负消融 |

## Results 规则

- Synthetic 对比可声明 `waveletmask_ab` 为提交的 synthetic 最佳模型。
- RealWorld 表格仅能描述为临时审计快照，直到刷新日志存在。
- Figure caption 必须区分 motivation 素材和定量证据。
- 不得将 PixelWiseMaskFusion 描述为核心提交模块。

## Conclusion 结构

1. 总结 WaveletMask 为小波域退化感知掩码方案，服务 RTN 聚合。
2. 陈述证据边界：synthetic 增益已冻结；RealWorld 排名待刷新日志确认。
3. 以开放方向结束：稳健的真实世界视觉验证、保守融合设计探索。
