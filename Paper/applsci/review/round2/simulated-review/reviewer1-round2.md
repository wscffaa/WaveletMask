# Reviewer 1 — Simulated Round-2 Assessment

> 模拟视角：Round 1 中最严苛的方法论审稿人，逐条核对其 10 条意见在 round2 重投稿中的落实情况。
> 审查对象：2026-06-12 04:40 编译态主稿（35 页）+ 3 封回复信 + Supplementary（8 页）。

## Overall Recommendation: **Accept with Minor Revisions**

The authors have made a serious and largely successful effort to address my previous concerns. The statistical validation, the re-built comparison protocol, and the failure-mode quantification transform the empirical basis of the paper. A small number of residual issues remain, none of which requires new experiments.

## Point-by-Point Verification

| R1 意见 | 状态 | 证据锚点 | 残留问题 |
|---|---|---|---|
| C1 统计验证缺失 | **Resolved** | Appendix A：seed-2021 配对 bootstrap CI + permutation（PSNR +0.4477, p=5e-5）；三 seed 复测 26.22±0.37 vs 24.56±1.67 | 3 seeds 仍属 modest，但 Limitations 已自曝，可接受 |
| C2 比较协议不公平 | **Resolved（超额完成）** | 全部 10 法统一重训（tab:literature_context caption 明示）；严格配对证据独立放 Appendix A | 见新意见 N1：重训方法的各自训练配置需可查 |
| C3 必须加主观评价 | **Partially Resolved** | 34 人 ×14 clips×3 维度研究完成，统计显著（binomial p=1.1e-21），但**全部内容仅在 Supplementary（Tables S1–S2, Figure S4），正文零提及** | 见 N2：我要求"must include"，正文不可见有再质疑风险 |
| C4 LHFM/GFM 理论依据弱 | **Resolved** | §3.6 Response Analysis 以边界条件方式重述（不再声称最优算子）；persistent/cut 两个 probe 量化支撑 | 无 |
| C5 无 artifact 分型评估 | **Resolved** | Appendix B tab:artifact_breakdown，四 strata 全正向（+0.34~+0.64 dB），并诚实声明是 residual-derived 而非 generator 标注 | 无 |
| C6 小波选择未论证 | **Resolved** | tab:ablation_core panel (c) + Appendix D 全 sweep（Haar2/DTCWT/coif1/sym2/db2）；结论已收窄为 scoped empirical choice | db2 17.92 dB 的崩溃归因（padding/支撑长度）已预答 reflective padding，可接受 |
| C7 失败模式无量化 | **Resolved** | tab:gfm_failure_probe（cut 2.35×、flash 10.41×、guard 抑制 61%/95%）；tab:persistent_scratch_probe（L≥3 抑制 65%）；FC pilot −2.6 dB | 无 |
| C8 消融不完整 | **Resolved** | 可解释性 probe、互补性 probe、6 种 fusion 变体、跨域频域算子诊断全数补齐 | 无 |
| C9 Table 2 协议混杂 | **Resolved** | 同 C2，表已拆分且 caption 声明出处 | 无 |
| C10 NR 指标互相矛盾 | **Resolved** | §4.4.2+§5.2 给出三指标先验差异解释；WaveletMask 唯一全部 top-2 | 见 N3：CLIPIQA+ 第二名领先 MambaOFR 仅 0.0005 |

## Remaining Minor Comments（无需新实验）

**N1（协议透明度）.** All ten methods were re-trained by the authors — commendable. But re-training raises a symmetric fairness question: were the external methods (e.g., MambaOFR, RVRT, VRT) trained with their officially recommended configurations and budgets? One appendix sentence (or a pointer to the per-method configs in the released evidence package) stating that official configurations/iteration budgets were used would close this loop.

**N2（主观研究可见性）.** My round-1 comment asked the authors to "include" subjective assessment. The study was completed and is statistically convincing, but the main text now contains zero pointer to it; it survives only as back-matter Supplementary declarations. I accept the journal-convention argument (MambaOFR likewise relegates its user study), but a single anchor sentence in Section 4.4.2 ("An expert-level subjective study with 34 trained participants, reported in Supplementary Tables S1–S2, corroborates this ranking") would cost two lines and remove any impression that the requested evidence was buried.（对应作者侧已知回退预案"方案二"。）

**N3（脆弱的 top-two 声明）.** In Table 4, WaveletMask's second place on CLIPIQA+ (0.4356) leads MambaOFR (0.4351) by 0.0005. The conclusion's "the only method placing within the top two on all three no-reference metrics" is technically true but hinges on a margin far below metric noise. Suggest qualifying with "(narrowly on CLIPIQA+)" or rounding-consistent wording.

**N4（基线快照披露不对称）.** The caption of Appendix Table A1 discloses that WaveletMask's 26.6125 (paired) vs 26.6017 (unified) differ across protocols. The same discrepancy is larger for RTN — 26.1648 (paired/ablation, Table 5) vs 25.8514 (unified, Table 3) — yet is only implicitly covered. Add one explicit sentence to prevent a careless reader from reading Table 3's RTN row as the paired baseline.

## Reject-First Check（自检式）

我尝试为"应拒"立论：唯一可立论点是"主增益 +0.45 dB 偏小且依赖单一退化管线"。但该立论被三处证据反驳：(i) 三 seed 全胜且方差缩小 4.5×；(ii) 四 artifact strata 全正向；(iii) DAVIS 跨内容迁移仍正向。+0.45 dB 在 898 参数代价下的 cost-effectiveness 立得住。拒稿理由不成立。
