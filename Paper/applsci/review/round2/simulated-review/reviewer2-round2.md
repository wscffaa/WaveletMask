# Reviewer 2 — Simulated Round-2 Assessment

> 模拟视角：Round 1 提出 10 条具体技术意见的审稿人，关注边界条件、运动鲁棒性与部署指标。
> 审查对象：2026-06-12 04:40 编译态主稿（35 页）+ 回复信 + Supplementary。

## Overall Recommendation: **Accept**

The authors responded to every one of my ten comments with either a dedicated experiment or an explicit scope statement. The revision is substantially more rigorous than the original submission. My remaining notes are editorial.

## Point-by-Point Verification

| R2 意见 | 状态 | 证据锚点 |
|---|---|---|
| C1 持久划痕 | **Resolved** | tab:persistent_scratch_probe：fixed L=2 抑制 46%、L≥3 抑制 65%（饱和）；drift 模式不受抑制；正文 §3.6 + Discussion 以 boundary condition 定性 |
| C2 运动感知分析 | **Resolved** | tab:motion_binning：增益随 motion/occlusion tercile 递增（+0.085→+0.769 dB，Spearman 0.58）；LHFM clean-FP 与运动几乎无关（−0.04）；FC pilot 三 seed 平均 −2.6 dB 且不稳定，机理解释（流误差经 DWT 放大）合理 |
| C3 小波族消融 | **Resolved** | Haar1/Haar2/DTCWT/coif1/sym2/db2 全测；结论收窄为 scoped default |
| C4 场景切换评估 | **Resolved** | tab:gfm_failure_probe：cut 2.35×、flash 10.41×、ramp 1.52×；luminance guard 抑制 61%/95%/2.5%；guard 未并入模型的理由（轻量化定位）成立 |
| C5 响应可视化与相关性 | **Resolved** | tab:cue_complementarity：local/flicker Spearman −0.34、top-5% overlap 0.0024；S3/S6 可视化齐备 |
| C6 更强融合替代 | **Resolved** | attention/residual/scalar-gate/normalization/bias-init 六种全测，全部劣于 max（最近 −0.11 dB）；结论限定在 tested recipe 内；未实现的 variance-head 已列为 future work 且方向性证据充分 |
| C7 补充引用 | **Resolved** | bernardini2025driveby 已引入 Introduction 小波段落 + references.bib |
| C8 独立真实数据集 | **Resolved（限界内）** | DAVIS 50 clips 跨内容迁移正向；"无公开成对老电影基准"声明成立；cross-degradation 留作 open，正文已明示 |
| C9 部署效率指标 | **Resolved** | tab:complexity 合并表：Latency 183.3 vs 188.7 ms、Throughput 38.2 vs 37.1 fps、Peak Mem 0.33 GB；720p/batch-4 协议下沉 Appendix D；外部方法不实测的理由（协议失配）可接受 |
| C10 掩码可解释性 | **Resolved** | tab:mask_interpretability：fused AUPRC 0.494、top-5% hit 0.691 vs 0.291 随机基线；LHFM 与 residual proxy 弱对齐被诚实承认并定位为 gating prior |

## Remaining Editorial Notes

**E1.** C10 的残留张力值得一句话强化：LHFM 的 top-5% hit（0.041）低于其随机基线（0.082），即 local 分支在该 proxy 下呈"反定位"，但端到端仍 +0.21 dB。目前正文解释（proxy 偏向低频残差、LHFM 是 gating prior 而非分割器）可以接受，但若在 Appendix C 中补一句"高通 residual proxy 本身无法区分真划痕与纹理边缘"会更完备。

**E2.** Appendix D 运行时协议中 "batch size 4 at 720p exceeded the available memory for both models in this session" — "in this session" 略口语化，建议改为 "under the 48 GB L40 configuration used here"，顺带给出显存上限数字。

**E3.** Table 4 中 RTN（unified 重训快照）BRISQUE 19.00 与 Appendix D Table（paired 快照）29.31 差异已被披露并反向支持了我对 NR 指标敏感性的关注——处理得当。

无需进一步实验，以上编辑性修改后即可接受。
