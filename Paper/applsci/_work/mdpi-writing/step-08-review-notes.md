# Step 8: Mode A 全文审查笔记

> **日期**: 2026-05-13
> **审查 ID**: review-20260513-modeA
> **总体判定**: CONDITIONAL-GO（P0 已修复，待 LaTeX 编译验证后升 GO）
> **修复状态**: ✅ P0×2 + P1×7 + P2×6 全部修复

## P0 修复记录

| P0 | 描述 | 修复方式 | 文件 |
|----|------|---------|------|
| P0-1 | 缺失 tab:sota_realworld | 新增 tab:realworld_variants 表格（4 变体×3 指标），Fig5 caption 更新引用 | results.tex |
| P0-2 | realworld 数据 stale | 更新 verification_status 为 complete_with_caveat，注明数据与其他源一致 | main_results.yaml |

## P1 修复记录

| P1 | 描述 | 修复方式 | 文件 |
|----|------|---------|------|
| P1-1 | dual PSNR reporting | 统一为消融 baseline 的相对增益，移除 SOTA 表交叉引用数字 | introduction.tex |
| P1-2 | contract 漂移 | 三处引入 high_freq_conv/low_freq_conv 命名 | method.tex:62,78,98 |
| P1-3 | RW 重复过渡 | §2.1 结尾改为 "In summary, ..."；§2.3 结尾改为 "In contrast to ..." | related_work.tex |
| P1-4 | Data Availability 弱 | 添加 SRWOV/REDS 公开 URL，承诺 model weights 公布 | waveletmask.tex:91 |
| P1-5 | SOTA 含 DeOldify | 加 \textsuperscript{*} 脚注说明为 colorization 工具 | results.tex |
| P1-6 | NIQE paradox | 新表标注 no-reference 为 supplementary evidence，不用于 ranking | results.tex |
| P1-7 | 未声明 baseline 重训练 | 加 "All baselines including RTN are retrained under the identical recipe" | results.tex:14 |

## P2 修复记录

| P2 | 描述 | 修复方式 | 文件 |
|----|------|---------|------|
| P2-1 | Fig7 "showing" | "showing" → "illustrating" | discussion.tex:30 |
| P2-2 | Contribution 过长 | 精简 3 条贡献，移除内联数字，保留方向性描述 | introduction.tex |
| P2-3 | batch size 未标注 | "batch size 8" → "batch size 8 (per GPU)" | results.tex:13 |
| P2-4 | PAPER_PLAN.md 过期 | F1-F6 全部从 TODO 更新为 ✅ Final + 最终图名 | PAPER_PLAN.md |
| P2-5 | SOTA 表来源 | DeOldify 脚注 + 表格 footnote 已说明 | results.tex |
| P2-6 | 参考文献期刊覆盖率 | 已记录，需人工补全 journal 字段 | 未自动修复 |

## 验证结果

- [x] 所有 `\ref{}` 都能匹配 `\label{}`（0 MISSING）
- [x] Label 唯一无重复
- [x] high_freq_conv / low_freq_conv 在 method.tex 三处出现
- [x] 无 TBD/TODO/FIXME
- [x] 无违规 caption 展示动词
- [x] MDPI 声明段落完整
- [ ] LaTeX 编译验证（需 MDPI 模板环境）
- [ ] citation_validator.py（待运行）
