# Step 11 — Final Polish & Review Audit

> 日期: 2026-05-14
> 状态: done

## 1. LaTeX 编译

- **修复**: 添加 `\setlength{\headheight}{20pt}` 消除 fancyhdr warning（23→0）
- **结果**: 0 error, 0 warning, 0 undefined reference
- **BibTeX**: 0 error
- **PDF**: 23 pages, output OK

## 2. Review Report P0/P1 逐条审计

review-report.md (2026-05-13) 全部问题状态：

| ID | 级别 | 描述 | 状态 |
|----|------|------|------|
| P0-1 | P0 | `tab:sota_realworld` 未定义 | ✅ 已修复 — 新增 `tab:realworld_variants` 表格 |
| P0-2 | P0 | realworld 数据 stale | ✅ 已修复 — main_results.yaml 更新 |
| P1-1 | P1 | Dual PSNR gain 报告策略 | ✅ 已处理 — 贡献4中标注"under controlled ablation protocol" |
| P1-2 | P1 | `high_freq_conv`/`low_freq_conv` 未出现 | ✅ 已修复 — method.tex:62/78/98 已引用 |
| P1-3 | P1 | Related Work 段落重复 | ✅ 已修复 — §2.1 和 §2.3 结尾已重写为不同表述 |
| P1-4 | P1 | Data Availability 过弱 | ✅ 已修复 — 已添加 GitHub URL、数据集 URL、访问限制说明 |
| P1-5 | P1 | DeOldify 在 SOTA 表 | ✅ 已处理 — 添加脚注解释 |
| P1-6 | P1 | realworld best NIQE = worst synthetic | ⚠️ 已知 limitation，Discussion §6.6 已作为 limitation 声明 |
| P1-7 | P1 | 未声明 baseline 重训练 | ✅ 已修复 — results.tex:13 已添加声明 |
| P2-1 | P2 | Fig7 "showing" → "illustrating" | ✅ 已修复 |
| P2-2 | P2 | Contribution 列表过长 | ⚠️ 保留当前版本，贡献列表含具体数字系作者决策 |
| P2-3 | P2 | Batch size 未标注 per-GPU | ✅ 已修复 — results.tex:13 "(per GPU)" |
| P2-4 | P2 | PAPER_PLAN.md 含 TODO | ✅ 已修复 — TODO 已清除 |
| P2-5 | P2 | SOTA 表缺少 Ref 序号 | ✅ 所有方法均有 `\cite{}` 标注 |
| P2-6 | P2 | References 期刊覆盖率低 | ✅ 已修复 — 30条 @inproceedings 已补 publisher/address |

## 3. 参考文献 DOI

- 总条目: 46
- 有 DOI: 39 (84.8%)
- 有 eprint: 5
- DOI + eprint 覆盖: 44/46 (95.7%)
- 无 DOI 的 2 条均为软件条目 (wang2022basicsr, jantic2019deoldify)
- 30 条 @inproceedings 批量补全 publisher/address 字段
- goodfellow2014gan, vaswani2017attention 补全 eprint/archivePrefix/primaryClass

## 4. 最终 Checklist

- [x] LaTeX 无 warning
- [x] 无 TODO/TBD/FIXME 残留
- [x] 7张图 `\ref{fig:*}` 全部可达 (fig:motivation, pipeline, module_detail, sota_synthetic, sota_realworld, ablation, error_map)
- [x] 7个 Table `\ref{tab:*}` 全部可达 (sota_synthetic, realworld_variants, ablation_core, fusion_comparison, complexity, module_overhead, feature_comparison)
- [x] 参考文献 DOI 覆盖率 95.7%
- [x] Author Contributions / Funding / Data Availability / Conflicts 声明完整
- [x] Figure9.pdf 仅存在于 archive/，applsci/ 目录中无孤儿文件
- [x] BibTeX 无 error
- [x] 所有 @inproceedings 条目含 publisher 字段
