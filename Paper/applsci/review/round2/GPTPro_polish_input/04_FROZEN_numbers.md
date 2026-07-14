# 冻结数字白名单（FROZEN — GPTPro 润色绝不可改）

> 用途：GPTPro 语言润色后，逐项核对下列数字是否原样保留。任何偏差 = 润色失败，打回重做。
> 口径：ab 口径主稿（unified 主表 = main_results waveletmask_ab）。上次翻车点见文末。

## 1. 头条数字（Abstract / Conclusion / Results 正文）

| 项 | 冻结值 | 出处 |
|---|---|---|
| WaveletMask synth PSNR（unified 主表） | **26.6017**（正文简写 26.60） | results.tex tab:literature_context |
| PSNR 增益 over strongest competitor | **+0.61 dB**（over MambaOFR） | abstract / results.tex:48 |
| paired-protocol PSNR 增益 over RTN | **+0.45 dB** | results.tex:48 / appendix_a |
| paired PSNR 显著性 | **p = 5×10⁻⁵** | appendix_a 表 + 回复信（2026-06-12：摘要/结论/Results 正文的精确 p 值已删；正文保留 "statistically significant" 与 p≤1.5×10⁻³，精确值仅留附录A表与回复信） |
| 主观研究偏好率 | **45.17%** | supplementary only（2026-06-12 方案一：摘要+正文零提及，S1–S2/S4 保留） |
| 主观研究显著性 | **p ≈ 1.1×10⁻²¹**（one-sided binomial） | supplementary only（同上） |
| 新增检测器参数 | **898** | abstract/conclusion/appendix_d/e |

## 2. Synth 主表 WaveletMask 行（tab:literature_context，ab 口径）

`26.6017 / 0.8951 / 0.0891 / 0.0831`（PSNR / SSIM / LPIPS / DISTS）
- bold/underline：PSNR=\textbf、LPIPS=\underline、SSIM/DISTS 无高亮。**不可改成四项全 bold**。

## 3. Real 主表 WaveletMask 行（tab:realworld_literature）

`5.2288 / 24.2023 / 0.4356`（NIQE / BRISQUE / CLIPIQA+），三项均 \underline（次优）。
- DegradedInput 锚点：`5.7745 / 43.1050 / 0.3873`（不参与排名）。

## 4. paired-protocol 快照（appendix_a/d，与 unified 不同，勿混）

- paired PSNR：**26.6125**（≠ unified 26.6017，两套快照故意不同，勿统一）
- 三seed：WaveletMask **26.2183±0.3725** vs RTN **24.5570±1.6667**
- paired 四指标：26.6125/0.8954/0.0886/0.0827

## 5. 探针/消融关键值（appendix_b/c/d）

- scratch stratum ΔPSNR **+0.64**；dust +0.45；flicker +0.47；mixed +0.34
- GFM 探针：scene cut **2.35×**、flash **10.41×**
- persistent scratch 抑制：L=2 **46%**、L≥3 **65%**
- 小波族：Haar1=26.6125、Haar2=26.5175、DTCWT=26.4506、Coiflet=25.8533、Symlet=24.2087、db2=17.9208
- module_overhead：Haar DWT=0、CNN=898、PixelWiseMaskFusion=14,497

## 5a. 复杂度/Runtime（Table 6, BasicOFR 统一框架 L40, 2026-06-12 全 5 法重测）

> 协议：L40, FP32, batch1, 1×7×3×256×256, warmup3+runs10, FlopCounterMode。全 5 法同框架同协议。产物 `review/round2/artifacts/table6_runtime/*.json`。
> 注：旧 Antigravity 框架值（188.7ms/2445.7G/6.1783M）已弃用，整表切 BasicOFR（与 accuracy 同源）。

| 方法 | Params(M) | FLOPs(G) | Latency(ms) | fps | Mem(GB) |
|---|---|---|---|---|---|
| DeepRemaster | 9.8781 | 914.0 | 17.9 | 391.6 | 0.35 |
| RTN | 5.7351 | 2184.7 | 169.5 | 41.3 | 0.24 |
| RRTN | 7.0503 | 2782.2 | 298.4 | 23.5 | 0.75 |
| MambaOFR | 8.2984 | 2832.8 | 404.8 | 17.3 | 0.60 |
| WaveletMask | 5.7360 | 2184.9 | 171.4 | 40.8 | 0.24 |

- WaveletMask−RTN：**+898 参数、+0.1494G FLOPs、+1.9ms、同显存(0.24GB)**。
- 额外设置（RTN/WaveletMask）：batch4-256 = 114.2/113.1 ms、61.3/61.9 fps、0.85GB；720p-b1 = 5379.1/5419.1 ms、9.92GB。

## ⚠️ 上次 GPTPro 翻车点（务必不要重犯）

1. Abstract 写成 26.61/+0.62 → 正确是 **26.60/+0.61**
2. Conclusion 把 p=5e-5 安到 34 人主观研究上 → 主观研究是 **二项检验 p=1.1e-21**；p=5e-5 只属 **clip 级 PSNR 配对检验**
3. 不得把 unified 主表 WaveletMask 改回"四项全胜"——ab 口径下只有 PSNR 第一
