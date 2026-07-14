# Round-2 提交材料核查与修复方案(2026-06-12)

> 来源:2026-06-12 05:08–05:30 全量核查(qtwx-mcp-2 会话)。
> 判定:CONDITIONAL-GO —— 回复信内容质量过关,但提交包 7/9 件过期;修完本方案各项即可升 GO(IRB 批准号除外,属投稿时机问题)。

## 一、核查结论摘要

### 通过项
- 24/24 审稿意见全覆盖:R1 10 条、R2 10 条、R3 4 条,均为 MDPI「Comments N → Response N → Revision made」三段式,意见原文完整引述、顺序一致。
- 回复信引用的 33 个 label(20 表 + 13 小节)在 `waveletmask.aux` 全部存在。
- 抽验约 90 个关键数字(统计 CI/p 值、三种子、分层、DAVIS、运动分箱、持久划痕、GFM 探针、互补性、掩码可解释性、跨域诊断、融合替代、运行时)与正文/附录全部一致。
- 「four tables and seven figures」与现稿一致;`tab:same_protocol_main` 在附录 A、`tab:literature_context` 在正文,与 R3 信声明吻合。
- 上轮审计遗留已修:RTN 双快照披露(§4.2)、radford2021clip 删除(bib 53 条)、补充图改名 FigureS4a/S4b/S5/S6。
- 主稿编译干净:34 页、0 Error/undefined/`{??}`;页空白扫描仅末页豁免项。
- Cover letter 必备项齐全(原稿 ID applsci-4348902、resubmission 关联、两条法定声明),docx 与 md 同步。
- 人类评价已按方案一从摘要/正文剥离(0 残留),back-matter 三声明保留。

### 问题清单
| 级别 | 问题 | 证据 |
|---|---|---|
| P0-1 | 提交包 7/9 件过期(clean/diff/supp PDF、zip、3 封回复信 docx) | 主稿 04:36–04:55 改约 110 行(`[H]→[t]`、融合表扩四指标、术语),35→34 页;submission/ 停在 04:22;docx 仍写 "five tables",R1-C10 docx 为旧表述 |
| P0-2 | IRB 批准号未回填(既定遗留,投稿时机) | `\institutionalreview` 为 "under application…" |
| P1-1 | Supplementary S4 / Tables S1–S2 正文零引用 | 正文仅引 S5;R1-C3 为 must 级意见,正文无锚点 |
| P1-2 | 回复信用原始 LaTeX label 指位,审稿人不可见 | R1 docx 10 处、R2 11 处、R3 6 处 "(label tab:…)" |
| P2 | 台账页数 35 过期;GPTPro_polish_input 旧快照;少量 Overfull/月份宏警告 | 非阻断 |

## 二、修复执行步骤(本次执行 1–8;IRB 为唯一例外项)

1. **正文锚点(P1-1,方案二)**:`sections/results.tex` §4.4 NR 段尾加一句 Supplementary Tables S1–S2 / Figure S4 锚点;重编译验证页数与空白扫描。
2. **回复信 label → 可见编号(P1-2)**:按新编译 aux 的编号映射,把三封 md 中 "(label `tab:xxx`)" 全部替换为可见编号(如 "Table A2 in Appendix A");同步复核 "four tables" 等措辞不回退。
3. **主稿干净重建**:`latexmk -C && latexmk -pdf waveletmask.tex`,0 warning/undefined。
4. **重生成 diff 稿**:git 基线 `fd18e0ce` + latexpand 展平 + latexdiff(PICTUREENV 整表替换),编译通过。
5. **重编译 supplementary.pdf**(8 页,0 undefined)。
6. **重建源码 zip**:当前 tex/bib/bbl/Definitions/全部引用图 + supplementary.tex;干净临时目录解压复验独立编译(主稿+补充材料 EXIT=0)。
7. **docx 重导出**:三封回复信 md → docx;检索验证 `five tables`、`label tab:` 0 命中。
8. **覆盖入 `review/round2/submission/` 并终验**:clean PDF 与主稿 md5 一致;页空白扫描仅末页豁免;更新 `submission-checklist.md`(页数/时间戳)与 `revision-log.md`。
9. **(例外)IRB 批准号**:签发后回填 `\institutionalreview` 的 protocol code + 日期,重跑步骤 3–8 再投;或先与编辑部书面确认豁免路径。

## 三、SuSy 提交清单(重投时核对)
clean PDF、highlighted PDF、LaTeX 源码 zip、supplementary PDF、Cover-Letter.docx、Response-to-reviewer1/2/3.docx 共 7 件;原稿 ID 填系统关联字段;建议/排除审稿人只填系统不写入 cover letter。
