# Round 2 Resubmission Checklist — MDPI Applied Sciences

> WaveletMask (007). Round 1 决定：Reject & Resubmit (major revision)。
> 本文件追踪从「round1 返修证据」到「round2 重投提交」的收尾门禁。
> 最后更新：2026-06-12 07:30（修复 supplementary.pdf 布局/空隙：float+[H]+图尺寸适配+去 clearpage，8→6 页；supp PDF + zip 刷新）。

## 1. 集成态编译门禁（P1，已通过）

| 检查项 | 结果 | 证据 |
|---|---|---|
| 全文干净重建 | PASS | `latexmk -C` 后 `latexmk -pdf` EXIT=0 |
| 页数 / 体积 | **34 页** / 11.0 MB（2026-06-12 07:16，删除 Appendix E 训练伪代码后；此前 35 页含 Appendix E） | `waveletmask.log` `Output written ... (34 pages)` |
| Undefined references | 0 | log 审计 |
| Undefined citations | 0 | log 审计 |
| Multiply-defined labels | 0 | log 审计 |
| Fatal / Emergency / LaTeX Error | 0 | log 审计 |
| LaTeX Warning 总数 | 0 | python re 审计 |
| `??` 未解析标记 (aux) | 0 | `grep -F "{??}" waveletmask.aux` |
| 参考文献条目 (bbl) | 53 | `waveletmask.bbl` |
| 交叉引用标签 (newlabel) | 182（全解析） | `waveletmask.aux` |
| 正文占位符 / TODO / 草稿标记 | 0 | `rg TODO\|FIXME\|placeholder\|\todo` sections |
| MDPI 必填字段 | 全齐 | Title/Author/corres/abstract/keyword/funding/dataavailability/conflictsofinterest/institutionalreview/informedconsent/authorcontributions/acknowledgments/reftitle/bibliography |

非阻断遗留（可选优化，不影响投稿）：
- 13 个 Overfull \hbox + 2 个 Underfull（排版微瑕）。
- BibTeX `nov/jul/may` 月份宏未定义警告（plain 风格常见，纯 cosmetic）。

## 2. 三审意见覆盖（round1 返修，已闭环）

| 审稿人 | 条数 | 回复信 | 关键证据任务 |
|---|---|---|---|
| Reviewer 1 | 10 | `review/round1/response-letter-reviewer1.md` | PB/T2/T5/T7/T8/T9/T11/T17/P0 |
| Reviewer 2 | 10 | `review/round1/response-letter-reviewer2.md` | T2/T4/T6/T7/T8/T14 |
| Reviewer 3 | 4 | `review/round1/response-letter-reviewer3.md` | 统计验证(P0/三seed) / 对比受控(表拆分) / 运动补偿(T6/T16) / 频域算子消融(T20) |

实验返修任务全量见 `review/round1/execution-log.md`（PB,T2,T4,T5,T6,T7,T8,T9,T11,T14,T15,T17,T20,P0 全部 done）。

> **网页逐条核对（2026-06-12，agent-browser-cli 读取 SuSy 三份 Review Report）**：R1=report 56538831（10 条）/ R2=56630588（10 条）/ R3=56629847（4 条），与本地 `reviewer{1,2,3}.md` 逐字一致，**24/24 全覆盖、零遗漏**，回复信 Comments 编号与网页段落顺序一一对应。详见 `revision-log.md` V1。

## 3. 待办（P2 / P3）

- [x] **P2 AI润色轨道处置**（已执行 C-light）：旧 GPTPro 润色稿已归档（commit `f67b83d7`），abstract 开篇句按冻结数值轻润色（commit `f947937c`），未发生数据回灌回退。
- [x] **P3 round2 提交包组装**（产物在 `review/round2/submission/`）：
  - [x] Response letters：采用**三份独立 docx**（Response-to-reviewer1/2/3.docx，MDPI 系统按审稿人分别回复可接受；与原计划"合并单文件"不同，2026-06-12 拍板）
  - [x] Cover letter（cover-letter.md + Cover-Letter.docx）：2026-06-12 03:18 补齐 MDPI 必备项——原稿 ID **applsci-4348902** + resubmission 关联请求 + 两条法定声明（不在他刊审理 / 全体作者批准），docx 已同步重导出
  - [x] Highlighted / diff PDF（WaveletMask-highlighted-changes.pdf，latexdiff，45 页 — 见 §3c 终态）
  - [x] 最终 clean PDF（WaveletMask-clean.pdf，与主稿 `waveletmask.pdf` md5 一致（9c308adf…），35 页 — 见 §3c 终态）
  - [x] 补充材料 PDF（WaveletMask-supplementary.pdf，8 页，独立编译 0 undefined，含 Tables S1–S2 / Figures S1–S6）
  - [x] LaTeX 源码包（WaveletMask-latex-source.zip，40 文件：tex/bib/bbl/Definitions/14 张引用图；已在干净临时目录解压验证可独立编译出 35+8 页 — 见 §3c 终态）
  - [x] **§4.6 Computational Complexity 精简对齐 04119（2026-06-12 03:46，用户指示）**：双面板表合并为单一 Cost 表（Params/FLOPs/Latency/Throughput/Peak Mem，外部方法 runtime 列 "---"），OOM 行与正文 OOM 表述全部移除；batch-4 与 720p 测量值连同协议移入 Appendix D `subsec:runtime_protocol`；720p batch-1 成本以一句话保留在正文；R2 回复信第 9 条同步改写并重生成 `Response-to-reviewer2.docx`；diff 稿改用 git 基线 `fd18e0ce` + latexdiff PICTUREENV 整表替换模式重新生成（47 页编译通过）；主稿 37→36 页
  - [x] **§4 长表注精简（2026-06-12 04:04，用户拍板"方案一"）**：三个最长表注压缩 `tab:ablation_core` 141→55 词、`tab:literature_context` 102→60 词、`tab:realworld_literature` 89→55 词；P1-1/P1-2 的披露内容未删除而是迁入正文（跨会话 LPIPS/DISTS 漂移 <0.002 → §4.5 引表段；26.6017 vs 26.6125 快照差异 → §4.4.1 正文；NR 指标方向句与附录指针句为表头箭头/正文的字面重复，直接删除）；35 页，0 undefined
  - [x] **删除正文前向伪代码（2026-06-12 03:57，用户指示，控制篇幅）**：移除 §3.2 `alg:waveletmask`（Algorithm "WaveletMask forward pass for a $T$-frame clip"），衔接句改为 "The recurrent aggregation step can be written as"；Appendix E 训练伪代码 `alg:training` 保留（自动改编号为 Algorithm 1），其引言句去掉对 `alg:waveletmask` 的引用；全文无残留 `\ref{alg:waveletmask}`；主稿 36→35 页
  - [x] **ethics 声明与主观实验对齐（2026-06-12 03:20，作者拍板：IRB 申请中 + 参与者已知情同意）**：`\institutionalreview` 改为 Helsinki 合规 + 匿名无创偏好调查 + 伦理审查申请中（按作者拍板泛指 "the ethics committee of the authors' institution"，批准号签发后回填具体名称）；`\informedconsent` 改为 "Informed consent was obtained from all subjects involved in the study."；主稿与 diff 稿（`\DIFadd` 高亮）均已重编译（37/47 页，主稿 0 undefined），提交包三件套（clean PDF / highlighted PDF / 源码 zip）已刷新，源码 zip 重新解压验证可独立编译。**遗留**：IRB 批准号下来后需回填 protocol code 与日期再投，或与编辑部沟通后补

  - [x] **人类评价完全剥离出摘要与正文（2026-06-12 04:25，用户拍板"方案一"）**：对齐 OFR 论文惯例（RTN/RRTN/MambaOFR/GLMA 正文均无 user study），摘要+正文 10 处提及全删，实验完整保留于 Supplementary（Tables S1–S2 / Figure S4）。正文改动：abstract 换 A′ 修订版（主观句删除，real-world 定性句补位）；introduction 贡献③删 45.17%/3.5×；results.tex 节首导览/数据集/指标三处指针删除（数据集句改 "no-reference metrics and qualitative comparisons"），§4.4.2 主观研究大段删除（段尾补 "NR 分数与定性比较联合解读" 一句）；discussion 两处引用删除；conclusion 主观整句改为 NR top-two + 定性证据句。back-matter 三字段（`\institutionalreview`/`\informedconsent`/`\supplementary`）保留不动（Supplementary 属稿件，涉人声明必须保留）。回复信同步：R1 总注 + R1-C3 + R1-C10 改为 Supplementary 全量指向（删除 "a summary sentence remains in Section 4.2.2" 承诺句），R2 总注同步限定 "appendices" 范围，cover letter Principal Revision 3 补 Supplementary 指向；Response-to-reviewer1/2.docx 与 Cover-Letter.docx 已重导出（R3 无涉及不动）。冻结清单 `04_FROZEN_numbers.md` 两行出处改 "supplementary only"。编译：主稿 35 页 0 warning 0 undefined；diff 稿（git 基线 `fd18e0ce`，latexpand+latexdiff PICTUREENV）45 页编译通过；clean PDF / highlighted PDF / 源码 zip 三件套刷新，zip 干净目录复验独立编译 35+8 页 EXIT=0。**遗留**：`GPTPro_polish_input/01_manuscript_current_full.tex` 为旧态快照（仍含主观句），下轮润色重建输入包时覆盖即可

## 3a. 章节名统一 + 复杂度小节精简（2026-06-12，作者拍板）

- [x] 结果章名 `Experiment` → `Experiments`（与已录用 applsci-16-04119 一致）：`sections/results.tex` + `waveletmask_diff.tex` 同步修改。正文/回复信/cover letter 均无 "Experiment section" 文字引用，无连带改动。
- [x] **Computational Complexity 小节精简**（对齐 applsci-16-04119 的 transparency 风格）：
  - 删除 runtime 表 (b) 中 720p batch-4 两行全 OOM 行（无区分度信息）
  - caption 从 ~110 词压到 ~55 词，删除 shared-GPU/OOM 免责解释
  - 介绍段与分析段精简，加入「transparency rather than minimum-latency claim」定位句
  - 测量协议细节（PyTorch 2.7.0/CUDA 12.6/warm-up/timed runs/720p batch-4 超显存说明）下沉到 Appendix D 新增小节 `subsec:runtime_protocol`
  - **回复信承诺核验**：R2-C9 承诺的 `tab:runtime_benchmark` 标签与双 panel 表保留；正文仍明确陈述 720p batch-4 超显存事实，回复信「We also state that both models exceed available memory…」仍成立
  - 重编译：主稿 36 页（精简后 -1 页）0 警告 0 错误；diff 稿 47 页；三件套刷新；zip 独立编译复验 36 页 EXIT=0

## 3b. SuSy 提交要点（Reject & Resubmit 规范，2026-06-12 网络核实）

- 以**新投稿**入口提交，系统分配新 Manuscript ID；在系统字段填原稿 ID `applsci-4348902` 关联旧稿（SuSy 自动 link，原审稿人将被再次邀请）。
- 建议/排除审稿人填在投稿系统里，**不写**进 cover letter。
- 无 5/10 天时限（区别于 Major Revision），可在 IRB 批准号落地后再投。
- 上传清单（7 件）：clean PDF、highlighted PDF、LaTeX 源码 zip、supplementary PDF、Cover Letter docx、Response-to-reviewer1/2/3 docx。

## 3c. submission-fix-plan 执行收尾（2026-06-12 05:57，步骤 1–8 全绿）

执行 `submission-fix-plan-20260612.md` 修复 CONDITIONAL-GO 阻断项（P0-2 IRB 除外）：

| 步骤 | 内容 | 结果 / 证据 |
|---|---|---|
| 1 P1-1 锚点 | `results.tex` §4.4.2 NR 段尾加一句中性指针（不含 45.17% 等冻结数字）："an expert-level subjective evaluation is reported in the Supplementary Materials (Tables~S1 and~S2 and Figure~S4)" | S1–S2/S4 不再正文零引用；R1-C3 有正文锚点（采 revision-log D8 回退预案=方案二） |
| 3 主稿重建 | `latexmk -C && latexmk -pdf` | 35 页；undefined 0 / multiply-defined 0 / `{??}` 0 / Fatal 0 / LaTeX Warning 0；bbl 53 条；fitz 页空白扫描=NONE |
| 2 P1-2 label | 三封 round1 回复信 `(label tab:/subsec:…)` → 可见编号（依新 aux：正文表 3/4/5/6、附录表 A1–A14、Section 3.6/4.6/5.1/5.3、Appendix B.2/B.3/C.1–C.4/D.1）。R1 24 处 / R2 27 处 / R3 13 处 | 三文件 `(label`/`` `tab: ``/`` `subsec: `` 残留=0；`four tables` 保留、`five tables`=0；bibtex 键 `bernardini2025driveby`/`references.bib` 未误伤 |
| 5 补充材料 | `supplementary.tex` 重编译 | 8 页；undefined 0 / Warning 0；页空白=NONE |
| 4 diff 高亮 | latexpand 两侧 + latexdiff（基线 `fd18e0ce` + `PICTUREENV=(?:figure\|table\|table*\|sidewaystable\|algorithm\|tabularx)…`） | 45 页编译通过；0 fatal；页空白=NONE |
| 6 源码 zip | 40 文件（4 根 + 12 sections + 10 Definitions + 14 图：Figure1–7 + FigureS1/S2/S3/S4a/S4b/S5/S6），剔除已废弃 Figure8–11 | 干净临时目录解压独立编译：主稿 35 页 + 补充 8 页，均 EXIT=0 / undefined 0 |
| 7 docx 重导出 | pandoc 三封信 → `Response-to-reviewer1/2/3.docx` | docx 文本 `five tables`=0 / `label tab`=0；`four tables`=1；含可见 Table A.. / Appendix C.. |
| 8 覆盖终验 | 7 件刷新入 `submission/` | `WaveletMask-clean.pdf` md5 == 主稿 `waveletmask.pdf`（9c308adf…）；`WaveletMask-supplementary.pdf` md5 == 就地 supp（6f233707…）；三 PDF 页空白扫描均 NONE |

> 注：修复方案 P0-1 写"35→34 页"不准确；干净编译稳定为 **35 页**（与本台账 §1 一致）。
> **唯一遗留 = P0-2 IRB 批准号**（`\institutionalreview` 仍"under application"），属投稿时机问题，签发后回填 protocol code + 日期再投，或与编辑部书面确认豁免路径。

## 3d. 对标录用/OFR 论文的语言润色（2026-06-12 06:08）

依据 `papers/Refs/`（已录用 ApplSci：LightVSR/CMDDNet/GLMA/HVIFormer + OFR：DeepRemaster/RTN/RRTN/MambaOFR）与 `PaperDesign/` 双维度规范盘点，执行用户拍板"全部"中的零风险纯润色项：

| 项 | 内容 | 对照 | 证据 |
|---|---|---|---|
| #2 指标公式 | §4.3 补 PSNR、SSIM 两条核心公式（NR 维持引用） | LightVSR/HVIFormer 给指标公式（ApplSci 跨领域读者惯例） | 新增 2 个无 label 编号公式，不影响 method 公式引用 |
| #3 Discussion 段标题 | §5.1 加 3 个 run-in `\paragraph`（最大融合/频域线索/小波族）打破 6 段长墙 | HVIFormer/GLMA 用更细子节 | **不改小节编号**（仍 §5.1），回复信引用不失配 |
| #4 结论 Limitations 呼应 | §6 第2段前补一句呼应 §5.3 边界（persistent scratch/突变/大缺失） | 04119/ApplSci Conclusion 末段=Limitations+Future | — |
| ~~#1b 时域证据声明~~（已撤除） | 曾在 §5.2 加 warping-error 声明，**用户拍板不需要 E_warp（轮次17）→ 已删除还原**；全文不再提该指标 | — | 时域稳定性仅由 §4.4.3 定性 + Appendix B 运动分箱隐性支撑 |
| #6 主表 Year 列 | `tab:literature_context` 加回 Year 列（2019–2026），**只加列、指标值与 bold/underline 一字未动** | GLMA/HVIFormer 主表带 Venue/Year（用 Year 而非 Venue，避免重新暗示文献报告值与 R1-C2 冲突） | 数据完整性保持 |
| #7 应用前景 | §6 补一句落地场景（档案馆/流媒体 remastering，898 参数 drop-in） | HVIFormer 有 Application Prospects 小节 | 冻结数字 898 一致 |

验证：主稿 latexmk -C && -pdf **35 页（未增页）**，undefined/multiply-defined/`{??}`/Fatal/Warning 全 0，bbl 53，无空白页；**全部 table/subsection 编号经 aux 比对未变**（回复信硬编码可见编号仍有效，docx 无需重导出）；diff 45 页、zip 40 文件干净目录复验独立编译（主稿 35+补充 8 EXIT=0）；`WaveletMask-clean.pdf` md5==主稿（e3c7cd85…），三 PDF 页空白扫描 NONE。

**本轮刷新**：clean PDF / highlighted diff / 源码 zip（06:08）。supplementary PDF 与三封 docx 因未触相关源/编号未变而保持（仍有效）。

**未执行（需决策）**：
- **#1(a) 补算 E_warp/tOF：用户拍板不需要（轮次17），不做**；连带撤除 #1b 的 warping-error 文本声明，全文不再出现该指标。
- **#5 表格红蓝标记**：现 bold/underline 与 2026 录用样本 04119 同款、journal 合规；全局改红蓝属横向变更且触碰 frozen 高亮结构，**建议保持**，未改。

## 3d. 摘要/正文精确 p 值收口 + 提交包重刷（2026-06-12 06:2x，作者拍板 D9）

- [x] 删除 abstract（waveletmask.tex:45）、conclusion、results §4.2 的精确 `p=5×10⁻⁵`；结论/正文保留定性 "statistically significant" 与回复信承诺的 `p≤1.5×10⁻³`；**附录 A 配对统计表（CI+p 值）保留不动**（R1-C1/R3-C1 回复信 "Revision made: Appendix A" 的字面交付物）；三封回复信不动。
- [x] 主稿干净重建 35 页 0 undefined / 0 warning / 53 bbl；diff（latexpand + latexdiff 基线 `fd18e0ce`，与上轮 diff 仅差 3 处 p 值）45 页；supp 8 页未变。
- [x] zip 重建（40 文件），干净临时目录解压复验独立编译主稿 35 页 + 补充 8 页，均 EXIT=0 / 0 undefined。
- [x] submission/ 刷新：`WaveletMask-clean.pdf`/`WaveletMask-highlighted-changes.pdf`/`WaveletMask-latex-source.zip` 三件覆盖；clean PDF md5 == 主稿、highlighted md5 == diff 稿、supplementary md5 == 就地 supp，三件全同步✓。
- [x] 冻结台账 `04_FROZEN_numbers.md` 第 13 行出处更新为"附录A表 + 回复信"（数值 p=5×10⁻⁵ 不变，仅出处变化）。
- 备注：clean PDF 仍 35 页（删 3 处短括号不改页数）；`Cover-Letter.pdf`（pandoc 生成，2 页）已在包内供 SuSy "Cover Letter (PDF)" 字段。

## 3e. 表格标记改红/蓝（2026-06-12 06:30，证据驱动）

用户要求核实期刊规范后整体改红/蓝。**一手证据**（Refs 录用同刊 Applied Sciences）：

| 论文 | 表注原文 | 标记 |
|---|---|---|
| GLMA (15-10931) | "The best result is in red color while the second best result is in blue color" | 红/蓝 |
| HVIFormer (16-02450) | "Best results are in red and second best in blue"（Table 3–6） | 红/蓝 |
| CNN-Mamba (15-03173) | 既往拆解确认 | 红/蓝 |
| LightVSR (15-01506) | 不标注 | — |

结论：**3/4 红蓝、0 篇 bold/underline**（联网搜索当时不可用，以一手证据为准）。

执行：
- 全表最优值 `\textbf{数值}`→`\textcolor{red}`、次优 `\underline{数值}`→`\textcolor{blue}`（results.tex Table 3/4/5、appendix_a A1/A2、appendix_b A5、appendix_d A11/A12/A14、supplementary S1/S2）；**仅改数值单元格**，表头/方法名/`WaveletMask (ours)`/appendix_a 的 `\textbf{95\% CI...}` 表头均不动（用"去 \pm/\%/$ 后无字母残留"判定数值）。
- 表注措辞 "highlighted in bold and second-best underlined" → "in red and second-best in blue"；同步给 supplementary 补 `\usepackage{xcolor}`。
- 同时清除 appendix_b 残留的 "a dedicated warping-error metric is left for future work"（彻底落实上轮 E_warp 撤除）。

验证：主稿 35 页、补充 8 页、diff 45 页，全门禁 0、无空白页；**table/subsection 编号 aux 比对未变**（回复信可见编号/docx 仍有效）；PDF span 颜色核验最优=红(16711680)/次优=蓝(255) 渲染正确；zip 40 文件干净目录复验独立编译；`WaveletMask-clean.pdf` md5==主稿、`WaveletMask-supplementary.pdf` md5==就地。提交包 clean/diff/supp/zip 全刷新（06:30）。

规范回写（skills 同步）：`PaperDesign/论文整体架构设计.md` §5.3/§10、`007论文整体架构规划.md` △6、`.claude|.codex/skills/ofr-paper-mdpi-review` 审查项、`.claude/skills/ofr-paper-mdpi-writing` 回填脚本说明——红/蓝定为 MDPI Applied Sciences 默认表格标记。

## 3f. 回复信 ASCII 数学记法→正规排版（2026-06-12 06:42，网络核实规范）

用户指出回复信 md 的 ASCII 数字风格不专业。**联网核实**（Editage / AIP / STEM editing guide）：科学计数法用 `1.23 × 10⁻⁵`（× = U+00D7、指数真上标 + 真减号 U+2212），χ²（非 chi-square），±（非 +-），≥/≤（非 >=/<=），E 记法（1.1e-21）须改标准幂。

执行（三封 round1 回复信，**纯记法、数值不变**）：
- `N x 10^-M` / `Ne-M` → `N × 10^−M^`（pandoc 上标 → Word 真上标）；`+-`→`±`；`chi-square`→`χ²`；`>=/<=`→`≥/≤`；`Spearman rho`→`Spearman ρ`；`Delta PSNR`→`ΔPSNR`；`Nx`/`N x N`(分辨率)→`N×`；一元负号→真减号 `−`。
- R1 30 改 / R2 28 改 / R3 12 改；ASCII 残留全 0。
- **避坑**：digit-范围 en-dash 规则会破坏 DOI URL `s40534-025-00393-5`，已弃用该规则（`46-65%`/`A1-A2` 连字符本就 publisher-acceptable）；负号规则要求前置空格/(/|，不碰 URL 内数字连字符。

验证：docx 重导出后 × / ± / χ / ρ / Δ / 真减号 + 真上标 run 均渲染正确（R1 8 上标、R3 5 上标、R2 0 上标符合各信内容）；ASCII 残留 0；冻结值 45.17% / four tables 保留、five tables=0。**仅三封 docx 刷新**（回复信非主稿构建件，clean/diff/supp/zip 不变）。规范回写 `ofr-paper-rebuttal` skill。

## 3g. §3.6 run-in 段标题规范化（2026-06-12 06:52，网络核实）

用户指 §3.6 标题不规范。**联网核实**（APA / 技术写作 heading 规范，多源一致）：run-in（三级）标题须为**描述性短语**、句首大写、加粗、句点结尾、正文紧随；**避免裸缩写做标题**（除非已定义且仍保持描述性）；下级标题须用具体术语清楚传达内容。同刊录用稿（HVIFormer/CNN-Mamba）的模块 run-in 也用 Motivation/Design/Advantage 等描述性词。

执行（`method.tex` §3.6 Response Analysis）：
- `\paragraph{LHFM.}` → `\paragraph{LHFM response to transient and persistent defects.}`
- `\paragraph{GFM.}` → `\paragraph{GFM response to flicker and abrupt transitions.}`
- 子节名 "Response Analysis" 本身为合规的简洁下级标题，**保留不动**（故回复信 "Section 3.6"/"'Response Analysis' subsection" 引用仍有效）。

验证：主稿 35 页、0 warning/undefined、无空白；`subsec:response_analysis` 仍 §3.6（编号未变）；diff 45 页；zip 40 文件干净复验；clean md5==主稿。**本轮刷新 clean/diff/zip**（supp/docx/cover letter 未触及，保持有效）。规范回写 `ofr-paper-mdpi-review` skill。

## 3h. 对齐 04119 录用模板（2026-06-12 07:02）

用户指明以 `papers/Others/Paper3/applsci-16-04119-manuscript-v2`（HDR，2026 录用，含 MDPI 编辑往返）为格式权威。核查 04119 实证：
- **run-in 子标题**：用 `\textbf{{简洁描述词.}}` **真行内**（如 `\textbf{{Linearization.}}`/`\textbf{{Channel packing.}}`，正文接同一行），**不用 `\paragraph{}`**。
- **表格标记**：**bold(最优)/underline(次优)**，表注明写"Bold values indicate the best result in each column, and underlined values indicate the second-best."；MDPI 编辑曾质疑加粗/下划线、作者答复"保留并已在表注加说明"——即 **bold/underline 经 MDPI 编辑审查确认合规**。

已执行：
- run-in 改造：`method.tex` §3.6 两处 + `discussion.tex` §5.1 三处 `\paragraph{}` → `\textbf{}` 行内；§3.6 标签精简为 "Local high-frequency response."/"Global flicker response."（04119 式简洁描述）。主稿 35 页 0 warning、run-in 真行内渲染、编号未变、diff 45p、zip 复验、clean md5==主稿；clean/diff/zip 刷新。

**表格标记决策（轮次23 已定）**：04119=bold/underline，GLMA/HVIFormer/CNN-Mamba=红/蓝，两种 MDPI 均合规。**用户拍板：保留红/蓝**（与 GLMA/HVIFormer 一致）。故表格不动（维持轮次19 的红/蓝），**无需重建**——当前提交包即终态（clean/diff/zip 07:02 含 run-in 行内化 + 红/蓝表）。规范更正回写：红/蓝与 bold/underline 均合规、择一统一，本项目用红/蓝（见 PaperDesign §5.3/§10 + mdpi skill）。

## 3e. 基线术语统一为 "RTN" + 提交包重刷（2026-06-12 06:4x，作者拍板 D10）

- [x] 按 Refs OFR 录用稿规范（RRTN/MambaOFR 用 "re-trained X"/"RTN"，无 "X-style baseline"；"-style" 仅架构形容词），把基线模型名全部统一为 **RTN**：正文 59 + supplementary 2 + 3 封回复信 docx（含术语）共计处理；**护栏句保留**（"trained with the same recipe"、"re-trained and re-evaluated by us"，公平性根基）；**架构形容词保留**（RTN-style propagation/gate/restoration、RTN spatial indicator、RTN backbone）。
- [x] 重打包：主稿 35 页 / diff 45 页 / supp 8 页，0 undefined；clean/highlighted/supplementary PDF + zip 刷新，md5 全同步；zip 干净目录复验独立编译 35+8 页 EXIT=0；3 docx 重导出（RTN baseline=0、four tables 保留、无 label）。全仓基线名变体残留=0。

## 3i. 删除 Appendix E 训练伪代码（2026-06-12 07:16，用户指示）

用户指"Algorithm A1 One training iteration of WaveletMask. 这个不要，直接删除"。`appendix_e.tex` 整体即该训练伪代码（section + Algorithm + 前后说明段），故整附录移除：
- 删 `sections/appendix_e.tex`；移除 `waveletmask.tex` 的 `\input{sections/appendix_e}`；删 `results.tex` §4.2 指向 `app:training_procedure` 的一句（训练 recipe 已在 §4.2 同段完整描述：Adam/lr/schedule/200K iters/7 frames/256×256 crops）。
- 回复信**无任何** Appendix E / 训练伪代码引用（已 grep 确认），docx 不受影响。kingma2015adam 引用仍被 §4.2 使用，bbl 不变。

验证：主稿 **35→34 页**，undefined/multiply/`{??}`/Warning/Fatal 全 0、无悬空 `\ref{app:training_procedure}`/`\ref{alg:training}`（残留 0）、无空白页；**全部 table/subsection 编号 aux 比对未变**（Appendix E 为末附录、无表，删除不连带renumber → 回复信可见编号/docx 仍有效）；diff 44 页、源码 zip **39 文件**（少 appendix_e）干净目录复验独立编译主稿 34 页 EXIT=0；`WaveletMask-clean.pdf` md5==主稿。clean/diff/zip 刷新（07:16）；supp/docx/cover letter 未触及保持有效。

## 3j. supplementary.pdf 布局/空隙修复（2026-06-12 07:30，用户指示）

用户指 supplementary.pdf 格式乱、大量空隙。fitz 逐页诊断：根因 = `[htbp]` 浮动漂移（如 §S1 Figure S1 浮走到 p2，p1 半空 418pt 隙）+ `\clearpage` 强制断页 + 大图独占页。修复：
- `\usepackage{float}` + 全部 figure/table `[htbp]`→`[H]`（精确定位，止漂移、保顺序）。
- 去掉 2 处 `\clearpage`（让正文回填图后空隙）。
- §S1 三张方形诊断网格 `\textwidth`→`0.75\textwidth`（使 S1 能与 §S1 引言同页，避免 p1 半空；注意：单图独占页时图越小底隙越大，0.75 是"能进 p1"与"单图页不过空"的折中）。
- Figure S4（柱状图双 minipage）`0.48/0.50`→`0.45\textwidth`（变矮 ~317pt，接在 §S2 分析文后落入 p4，消除 p4 的 402pt 大隙）。

结果：**8 页 → 6 页**；逐页 botGap：p1=1 / p3=4 / p4=11 / p5=56（均填满），仅余 p2=267（S2 网格独占页，§S1 三连图无穿插文本，固有）+ p6=245（Figure S6 末页尾部留白，正常）。0 warning/undefined/overfull，无空白页；图序与引用一致。supp PDF + 源码 zip（39 文件，含 `float`）刷新，干净目录复验主稿 34 + 补充 6 EXIT=0，md5(supp)==就地。主稿未触及，clean/diff 不变。

## 4. 未纳入本轮提交固化的工作区改动（待用户确认归属）

- `projects/007-waveletmask/Exp/paper_data/main_results.yaml`（Exp 数据小改，非 Paper 边界）
- `memory/core/datasets/SRWOV.yaml`（共享 SSOT）
- 其他项目（010/013/028）、`.claude/skills/nature-figure`、`.cursor/`、`papers/Refs/PaperDesign/` 等均不属于 007 round1 返修，未触碰。
