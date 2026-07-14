# Reviewer 3 — Simulated Round-2 Assessment

> 模拟视角：Round 1 提出 4 条结构性意见、最关注"贡献是否真正 wavelet-specific"的技术审稿人。
> 审查对象：2026-06-12 04:40 编译态主稿（35 页）+ 回复信。

## Overall Recommendation: **Accept**

All four of my concerns were addressed head-on, including the frequency-operator ablation I requested verbatim. I particularly appreciate that the authors allowed the diagnostic to *weaken* their own claim and rewrote the conclusion accordingly — this is how revision should work.

## Point-by-Point Verification

| R3 意见 | 状态 | 证据锚点 |
|---|---|---|
| C1 经验验证不足（单 run、无配对检验） | **Resolved** | Appendix A：配对 bootstrap CI + permutation 四指标全显著；三 seed 复测均值差 +1.66 dB 且逐 seed 全胜（26.61/26.17/25.87 vs 26.16/24.67/22.84）；正式检验选 baseline 最强的 seed-2021，保守性论证成立 |
| C2 对比仅部分受控、表格视觉上夸大 | **Resolved** | 全部 10 法统一重训消除了"published numbers 混排"问题；严格配对证据与 unified benchmark 角色二分（§5.2 明示"controlled paired comparison remains the primary evidence"）；SSIM 中游、DISTS 持平等不利结果如实呈现且正文点名 |
| C3 检测器无运动补偿的技术局限 | **Resolved** | 正文不再暗示 motion-invariance；motion/occlusion binning 显示增益在高运动 tercile 反而最大；FC pilot 失败的机理（DWT 放大流误差）+ camera-motion dominated（29/30）与 tercile 相对性两条 caveat 均已写入 |
| C4 频域算子对照表（Haar vs Sobel/Laplacian/Gaussian/Fourier/learned） | **Resolved（完全按建议执行）** | tab:cross_domain_frequency_ablation 六算子对照；结果显示 Fourier-temporal（0.3885）与 learned shallow（0.4265）的 AUPRC 高于 Haar fused（0.3166），作者据此把声明收窄为"the most stable *tested* wavelet for end-to-end restoration gating rather than a universally optimal frequency detector"（§5.1） |

## Remaining Note（一条，可选）

**N1.** The cross-domain diagnostic shows Fourier-temporal and proxy-supervised operators outperform frozen Haar on residual-proxy alignment, and the authors honestly state these "have not yet been retrained end-to-end inside the gating pipeline" (§5.3). This asymmetry — Haar evaluated end-to-end, alternatives evaluated frozen — is disclosed, which is sufficient for the present scope. I encourage (not require) the authors to add the strongest alternative (Fourier-temporal) as an end-to-end gating variant in future work; it is already listed in the limitations, so no change is mandatory.

The manuscript is now technically sound, statistically supported, and honestly scoped. I recommend acceptance.
