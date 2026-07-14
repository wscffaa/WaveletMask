# Step 09 — Abstract 审查

日期: 2026-05-13

## 当前 Abstract

> Old film restoration requires degradation detectors that separate sparse structural damage from global temporal artifacts. Recurrent restoration networks commonly use pixel-domain frame differences as degradation indicators, but this signal is weak for thin scratches and ambiguous for brightness flicker. We propose WaveletMask, a wavelet-domain degradation detection framework that augments recurrent old film restoration with frequency-specific masks. A local high-frequency branch uses Haar detail sub-band differences to localize transient scratches and dust, while a global flicker branch uses low-frequency sub-band differences to capture frame-level brightness fluctuations. The proposed variant combines these two branches with element-wise maximum fusion and feeds the resulting wavelet-domain indicator into the recurrent gating path. Experiments on the synthetic benchmark show that the proposed WaveletMask variant improves reconstruction quality over the recurrent backbone while preserving a lightweight detection pathway. The results support the conclusion that frequency-domain degradation indicators provide useful complementary evidence for recurrent old film restoration, while learned fusion and broader real-world evaluation remain directions for further validation.

## 检查

- [x] task/problem/method/key result 四要素完整
- [x] ZERO specific numbers（无 PSNR/dB 数字）
- [x] ZERO parenthetical abbreviations
- [x] ~180词，可进一步压缩到 ~200
- [x] 与 Introduction + Conclusion 一致

## 无问题
