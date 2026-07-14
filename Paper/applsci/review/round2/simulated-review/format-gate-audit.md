# 格式严格期刊审稿专家 — MDPI Applied Sciences 合规审计（Round 2 重投前）

> 模式：ofr-paper-mdpi-review Mode A（全文审查 + 投稿硬门）。
> 审计快照：2026-06-12 04:40 编译态（35 页主稿 / 45 页 diff / 8 页 supplementary / 40 文件源码包）。
> 注意：审计期间检测到 sections/results.tex、appendix_d.tex 于 04:38–04:39 被并行会话改动措辞（"test session"→"evaluation run" 等）并于 04:40 重编译主稿，**晚于 04:22 的提交包三件套**。

## 总体判定：**CONDITIONAL-GO**

无数据硬错、无编译错、无声明矛盾；但存在 1 项投稿时序阻断（IRB 编号）+ 1 项资产包时效失配 + 若干回复信一致性项，修完指定项再投。

## 硬门核查结果

| 硬门 | 结果 | 证据 |
|---|---|---|
| 编译门 | **PASS** | 35 页，0 LaTeX Warning / 0 undefined / 0 `{??}`，bbl 53 条，182 标签全解析 |
| 数值 provenance | **PASS** | 摘要 26.60/+0.61/+0.45/p=5e-5/898 全部可溯源至 tab:literature_context、tab:same_protocol_main、tab:module_overhead；冻结清单核销 |
| 跨源一致性 | **PASS（含 1 条 P1 披露建议）** | 26.6017 vs 26.6125 双快照已在两处 caption 披露；RTN 双快照（25.8514/26.1648）仅隐式覆盖 → P1-4 |
| 统计声明 | **PASS** | 不显著项无夸大；SSIM 中游、DISTS 持平如实写出；三 seed 逐 seed 数据透明 |
| 引用门禁 | **PASS（1 P2）** | cited 53 / bib 54，0 missing；1 个 unused key `radford2021clip` → P2-1 |
| 资产可重编译 | **PASS（时效 P1）** | `\includegraphics` 14 处全部解析；zip 40 文件无 png/pptx/临时物；但见 P1-1 时效失配 |
| Data Availability | **PASS（既有 P1 表述已缓解）** | REDS/SRWOV 公开链接齐；repo 为 "being prepared" 属 release promise，但已给出"提交包即权威复现记录"的缓解句 |
| MDPI 必填结构 | **PASS** | Title/Author/ORCID/corres/abstract(194词≤200)/keywords(7)/正文五段式/back-matter 9 项声明/Abbreviations/Appendix A–E/References 全齐 |

## 问题清单（按严重度）

### P0（投稿时序阻断，非稿件缺陷）

- **P0-1 IRB 批准号未回填。** `\institutionalreview` 现为 "under application ... will be supplied upon issuance"。涉人主观研究（34 人）虽属匿名无创偏好调查，但 MDPI 编辑部对 "approval pending" 表述通常会发 query。**动作**：等批准号签发后回填 protocol code + 日期再投（台账既定遗留项），或投稿前与编辑部书面确认豁免路径。Reject & Resubmit 无时限，等待无成本。

### P1（重投前必修，半小时内可完成）

- **P1-1 提交包时效失配（新发现）。** 主稿 tex/PDF 于 04:38–04:40 被改动重编译（"evaluation run" 等措辞润色），而 `review/round2/submission/` 三件套停留在 04:22。clean PDF 与当前主稿 md5 必不一致。**动作**：待并行润色收敛后，按五步铁律整体重刷（clean PDF → latexdiff → zip → 干净目录复验 → docx 重导出）。
- **P1-2 三封回复信总注"five tables and seven figures"已陈旧。** §4.6 成本表合并后 Results 实为 **4 表** 7 图。三封信开头总注均需同步改为 "four tables and seven figures"（cover letter 已改为模糊表述，无需动）。
- **P1-3 R1 回复信 Response 10 与现稿不符。** 信中称 NR 评估 "external-method and variant panels are now combined in a single clearly separated table"，但现稿已将变体面板下沉至 Appendix D（tab:realworld_variants），主文仅保留外部方法表。**动作**：改写该句为"split into the unified main-text table and a paired-protocol variant table in Appendix D"。
- **P1-4 RTN 双快照披露不对称。** Table 3（25.8514）与 Table 5（26.1648）的 RTN PSNR 差 0.31 dB，目前仅靠 unified/paired 协议总声明隐式覆盖；WaveletMask 的对应差异（0.011）反而有两处显式披露。**动作**：在 §4.5 引表段或 tab:ablation_core caption 补一句 RTN 快照归属。
- **P1-5 Supplementary S4 / Tables S1–S2 正文零引用。** 方案一剥离后，用户研究材料仅存在于 back-matter `\supplementary` 声明中，正文无任何指针；MDPI 惯例要求 supplementary 条目在正文被引用。S1–S3/S5/S6 均有正文引用，唯独 S4+S1–S2 表悬空。**动作**：按既定"方案二"回退预案在 §4.4.2 加一句锚点（推荐），或保持现状并接受编辑部可能的 query（知情风险已记录于 revision-log D8）。

### P2（润色级，不阻断）

- **P2-1** 清理 unused bib key `radford2021clip`（或在 CLIPIQA+ 首次出现处补引）。
- **P2-2** 11 处 Overfull/Underfull hbox（纯排版瑕疵）。
- **P2-3** 源码包内 4 个补充图文件名残留旧主文编号（`Figure8.pdf`→S5、`Figure9_*`→S6、`Figure10/11_*`→S4a/b），建议改名为 FigureS4a/S4b/S5/S6 以免编辑部重排版混淆；改名需同步 supplementary.tex 与 zip。
- **P2-4** 摘要中 PSNR/RTN 已展开，但 LPIPS 未展开；MDPI 摘要缩写规范上从严可补 "Learned Perceptual Image Patch Similarity (LPIPS)"——若超 200 词上限则放弃（现 194 词，余量 6 词，恰好不够，可接受现状）。
- **P2-5** BibTeX `nov/jul/may` 月份宏警告（plain 风格常见，cosmetic）。

## 图表专项抽查（caption 动词 / 三线表 / 方向箭头）

- 全部 caption 使用解释性/中性动词（compares/illustrates/reports/summarizes），0 个 "shows/presents/displays" 违例。
- 全部表为 booktabs 三线表；指标列均带 ↑/↓ 方向标注；消融变体使用描述式命名（+LHFM only / Branch-level maximum），无内部代号残留。
- 主文图 7 张、表 6 张（Related Work 2 + Results 4），证据覆盖完整（动机/管线/双模块/定量/定性×2/消融/成本），数量叙事合理。

## 结论

数据与声明层面本稿已达投稿态；阻断仅剩 **IRB 编号回填（P0-1）** 与 **提交包重刷（P1-1）**。P1-2~P1-5 为回复信/披露一致性微修，与 P1-1 同批完成即可将判定升为 **GO**。
