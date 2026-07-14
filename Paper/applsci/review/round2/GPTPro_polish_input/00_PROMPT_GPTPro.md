# GPTPro 整体润色指令（round2 投稿前最后一步）

你是计算机视觉领域资深学术写作专家，专攻 old film restoration 方向 SCI 论文润色，有 CVPR/ECCV/TOG 审稿经验。目标期刊 MDPI Applied Sciences。这是 round2 重投前的整体语言润色。

## 输入文件

- `01_manuscript_current_full.tex` — **当前主稿全文（SSOT，唯一真源）**：abstract + 6 正文章节 + 5 附录（A–E）。
- `02_STYLE_GUIDE.md` — 全局润色规则（杀 hedging、去 AI 味、因果归因、disclaimer 控制）。**先读它。**
- `03_reviewer_comments_all.md` — 三审意见（R1×10 / R2×10 / R3×4），用于确认润色不削弱任何回应。
- `04_FROZEN_numbers.md` — **冻结数字白名单（最高优先级红线）**。

## 绝对红线（违反即作废）

1. **数字全冻结**：`04_FROZEN_numbers.md` 列出的所有数值、p 值、百分比、参数量、表格单元格一律原样保留。**严禁**任何改动（含"顺手纠正"）。尤其注意上次翻车的三点（见白名单文末）。
2. **不碰 LaTeX 骨架**：所有 `\label{}` / `\cite{}` / `\ref{}` / `\begin{table}` / `\toprule` / `\midrule` / `\textbf` / `\underline` / 公式 / 表格列结构原样保留。只改自然语言句子。
3. **两套快照不混**：unified 主表 PSNR=26.6017 与 paired-protocol PSNR=26.6125 是故意不同的两套测量，**不得统一**。unified 主表 WaveletMask 只在 PSNR 第一，**不得写成"四项全胜/best on all four metrics"**（那是错的）。
4. **审稿回应不削弱**：为回应审稿人新增的实验描述、表格引用、探针结论，可改善表达，但结论与数据必须保留。

## 润色任务

按 `02_STYLE_GUIDE.md` 执行，重点：
1. 杀死病态 hedging（"bounded"/"should be read as"/"does not by itself"等，见 STYLE_GUIDE 原则 1）。
2. 去 AI 味词汇（leveraging/robust/comprehensive/novel 等，见原则 2）。
3. 术语全文一致（old film restoration / degradation mask / same-protocol vs unified benchmark）。
4. Claim 强度三处一致：Introduction ≈ Results 证据 ≈ Conclusion，不过强不过弱。
5. Disclaimer 密度：Results 中 per-table/figure 的 "diagnostic not statistical" 类提醒压到 ≤3 处。
6. 强证据突出：45.17% 主观偏好（Intro 前向引用 + Conclusion 高亮）、p=5×10⁻⁵、cross-content +0.15 dB 正面呈现。

## 输出（两个文件）

1. `manuscript_polished.tex` — 润色后全文，结构与输入逐节对应，关键修改处加 `%% POLISHED:` 注释说明改动理由（一句话）。
2. `polish_report.md` — 问题清单（critical/minor）+ 逐项 before/after + 一节专门列出"我确认未改动的冻结数字清单（逐条对照白名单打勾）"。

## 自检（输出前必做）

逐条核对 `04_FROZEN_numbers.md`：把润色稿里每个冻结数字与白名单比对，全部一致才输出。任何不一致先回退。
