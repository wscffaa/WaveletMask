# Step 02 — Introduction 写作思路 + 实验清单

审查日期: 2026-05-13

## Introduction 大纲（基于已完成的 Fig1 + Fig2 重审）

### ¶1-2: 任务定义 + 应用背景
- 老电影退化：局部损伤（划痕/灰尘）+ 全局闪烁 + 噪声
- 退化跨频域 → 需要多频域检测机制
- ✅ 消费 Fig1：无直接引用，铺垫场景

### ¶3-4: 已有方法 + 共同局限
- RTN/RRTN/MambaOFR 使用像素域 frame difference 做 degradation indicator
- 局限1：稀疏高频损伤弱响应（被周围像素淹没）
- 局限2：全局闪烁与场景亮度变化无法区分
- ✅ 消费 Fig1：`\ref{fig:wavelet_motivation}`（行7），图嵌入行11-15

### ¶5: 本文方法概述（wavelet insight + WaveletMask）
- Haar DWT → LL/LH/HL/HH 子带
- 高频子带分离局部损伤，低频子带分离闪烁
- WaveletMask = A (local HF mask) + B (flicker mask) + max fusion
- ✅ 消费 Fig1：行17，Fig2：`\ref{fig:pipeline}`（行17），图嵌入行19-23

### ¶6: 贡献列表（4 条）
1. 小波域退化分析（频域解耦）
2. Local HF Mask（几何均值时序滤波）
3. Global Flicker Mask（最小值策略）
4. Max-fusion 框架 + 超可加互补效应 + learned fusion 系统消融

## Figure 消费核对

| Figure | Contract Label | Introduction 实际 Label | 匹配？ |
|--------|---------------|----------------------|--------|
| Fig1 | `fig:motivation` | `fig:wavelet_motivation` | ⚠️ 不一致 |
| Fig2 | `fig:pipeline` | `fig:pipeline` | ✅ |

> Fig1 label 不一致 → 记录到 Step 6 drift 检查，Step 2 不阻塞。

## 实验清单

### Comparison Experiments (SOTA)
| 数据集 | 对比方法 | 指标 |
|--------|---------|------|
| REDS-val4 (synthetic) | RTN, RVRT, MambaOFR, + others | PSNR, SSIM, LPIPS, DISTS |
| SRWOV (real-world) | RTN, RVRT, MambaOFR, + others | NIQE, BRISQUE, CLIPIQA |

### Ablation Studies
| 对比 | 变体 | 证明什么 |
|------|------|---------|
| RTN baseline vs +LHFM only | LHFM 单独 | LHFM 贡献 +0.22 dB |
| RTN baseline vs +GFM only | GFM 单独 | GFM 贡献 SSIM +0.006 |
| RTN baseline vs WaveletMask | LHFM+GFM | 总增益 +0.44 dB |
| +LHFM only vs +LHFM + learned fusion | LHFM vs learned fusion | learned fusion 对 LHFM 有害 -0.39 |
| +GFM only vs +GFM + learned fusion | GFM vs learned fusion | learned fusion 对 GFM 有害 -0.15 |
| WaveletMask vs WaveletMask + learned fusion | max vs CNN fusion | learned fusion 对所有变体有害 -1.62 |
| WaveletMask vs WaveletMask + gated learned fusion | max vs gated+CNN | gated learned fusion 仍不如 max -0.25 |

### 论文主展示（4变体）
RTN baseline → +LHFM only → +GFM only → WaveletMask

## 自检

- [x] 4 段式逻辑自洽（问题→局限→方法→贡献）
- [x] 贡献列表每条在实验清单中有对应验证
- [x] Fig1 已在 ¶3 消费（`\ref{fig:wavelet_motivation}`）
- [x] Fig2 已在 ¶4 消费（`\ref{fig:pipeline}`）
- [x] 引用的是已完成的真实图，非空引
- [ ] Fig1 label 不一致 → 记录，Step 6 处理

## 下一步

Introduction 初稿质量好，无需重写。推进到 Step 3：Method 审查。
