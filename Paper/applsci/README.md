# Applied Sciences (MDPI) — applsci-4348902

WaveletMask 在 Applied Sciences 的期刊工作区。

## 结构

- `waveletmask.tex` / `supplementary.tex` / `sections/` / `figures/` / `Definitions/` / `references.bib`
  — 当前 canonical 可编译主稿工作区（**勿移动**，编译入口在此）
- `waveletmask_diff.tex` — latexdiff 高亮稿
- `review/` — 审稿往来证据链（round1/round2），稳定锚点，物理不进 lifecycle（在 `MANIFEST.yaml` 中逻辑索引）
- `lifecycle/` — 按 `YYYY-MM-DD_<stage>/` 归档的提交包 / 伦理包 / proof 包快照
- `assets/ppt/` — 论文配图可编辑源（Figure1~4.pptx）
- `_work/` — 写作与画图工作产物（mdpi-writing、fig9）
- `_supp_build/` — supplementary 构建产物
- `MANIFEST.yaml` — 期刊生命周期清单（lifecycle + review 逻辑索引 + assets/work 指针）

## Lifecycle

| 日期 | 阶段 | 路径 |
|---|---|---|
| 2026-06-08 | round1 revision 提交包 | `lifecycle/2026-06-08_round1-revision-submission` |
| 2026-06-22 | 伦理不适用独立包 | `lifecycle/2026-06-22_ethics-not-required-package` |
| 2026-06-23 | final proof / SuSy author proofreading | `lifecycle/2026-06-23_final-proof-applsci-4348902` |
