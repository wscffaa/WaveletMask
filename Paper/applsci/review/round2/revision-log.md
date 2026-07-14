# 007-WaveletMask · Applied Sciences Round-2 修订复盘日志

> 目的：固化 round1（Reject & Resubmit）→ round2 重投全过程的「拍板 → 改动 → 证据」链，供后期复盘与方法论沉淀。
> 信息源：git 提交历史 + 2026-06-12 多通道 Cursor 对话（QingTian qtwx-mcp-2/3/4/5）+ 台账。
> 分工：`submission-checklist.md` = 台账（门禁 + 核销，投稿前看它）；本文件 = 复盘日志（决策过程与教训，复盘时看它）。
> 最后更新：2026-06-18（伦理口径切换=路径 A｜Article 32 豁免，新增 D12；独立自洽提交包 `Paper/submission_applsci_waived/` 落地）。

## 0. 基线与分支

| 项 | 值 |
|---|---|
| round1 投稿态基线 commit | `fd18e0ce`（latexdiff 高亮稿统一基线，2026-06-10） |
| 返修工作分支 | `round2-restructure` |
| 当前主稿状态 | 34 页，0 LaTeX Warning / 0 undefined / 0 `{??}`，bbl 53 条 |
| diff 高亮稿 | 34 页（latexpand 展平 + latexdiff PICTUREENV 配置；D12 伦理口径切换后基线重生成，与主稿同页） |
| 提交包 | `review/round2/submission/` 9 件（clean/diff/supplementary PDF + 源码 zip + Cover Letter + 3 封回复信 docx + cover-letter.md） |

## 1. 背景（06-09 ～ 06-11）

- `3457ec26`（06-06）～ `ba36a2e6`（06-09）：review 变体与训练配置整理。
- `a7607c99` / `a84a034d` / `fd18e0ce`（06-10）：R1 返修正文修订、round1 回复信与证据、实验证据 + AI 润色工作区入库。**fd18e0ce 即 round1 投稿态基线。**
- 06-11：`实验部分审查-20260611.md`、`实验部分单表重构方案-20260611.md` 完成实验叙事审查；round1 执行日志中实验返修任务（PB,T2,T4–T9,T11,T14,T15,T17,T20,P0）全部 done。

## 2. 06-12 修订时间线（git + 对话整合）

| # | 时间 | 事件 / 拍板 | 关键改动 | 证据 / 提交 |
|---|---|---|---|---|
| 1 | 00:21–00:46 | 盘点未完成任务 → P0 分批提交固化 round1 返修 | 正文 + 5 附录集成、references.bib、64 个 review 证据文件分 3 批入库；建 round2 checklist | `41c05592` `2d538768` `ee527f58` `485efd06` |
| 2 | 00:41 | P1 集成态编译门禁 | latexmk 干净重建全绿（当时 37 页）；MDPI 必填字段核查全齐 | 台账 §1 |
| 3 | 00:23–01:00（并行窗口） | **拍板 B1**：SRWOV.yaml 中 WaveletMask 数字被未提交手动改动（SSIM 0.9151→0.9251 等）且无重训证据 → 回退 ab 口径实测真值 | 回退 `SRWOV.yaml`/`main_results.yaml`；abstract/conclusion 的 "best PSNR and LPIPS" 收窄为只保 PSNR 第一；主表与 §4.2.1 叙事同步 | 工作区改动（待入库）；台账 §4 |
| 4 | ~00:46 | **拍板 C-light** 润色策略：发现 GPTPro 旧润色稿含统计错误（把 p=5e-5 安到 34 人主观研究头上）且整体旧于主稿 → 不回灌，仅按冻结数值微调 abstract 开篇句，旧轨道归档 | abstract 开篇句微调；`AI润色/` 目录归档删除 | `f947937c` `f67b83d7`；冻结清单"翻车点"§ |
| 5 | 00:33–03:14 | 24/24 审稿意见闭环审计 → 发现提交包缺 supplementary PDF | 独立编译 `supplementary.tex`（8 页）入包；构建 40 文件源码 zip 并在干净目录复验独立编译 | 台账 §3 P3 |
| 6 | 03:18 | Cover letter 补 MDPI 强制项 | 原稿 ID applsci-4348902 关联 + 两条法定声明（不在他刊审理 / 全体作者批准） | 台账 §3 P3 |
| 7 | 03:20 | **拍板 IRB 口径**：34 人主观研究 vs 伦理声明缺失矛盾 → 作者确认"IRB 申请中 + 参与者已知情同意" | `\institutionalreview` 改 Helsinki 合规 + 申请中表述；`\informedconsent` 改标准句；三件套刷新 | 台账 §3 P3；**遗留：批准号回填** |
| 8 | 03:46 | **拍板对齐已录用稿 04119**：§4.6 复杂度精简 | 双面板表合并单 Cost 表、OOM 行删除、测量协议下沉 Appendix D `subsec:runtime_protocol`；R2-C9 回复信同步；diff 稿基线锁 `fd18e0ce` + PICTUREENV 解 `\noalign` 崩 | 台账 §3/§3a；37→36 页 |
| 9 | 03:57 | **拍板删正文前向伪代码**（控篇幅） | 删 §3.2 `alg:waveletmask`；Appendix E 训练伪代码保留自动改 Algorithm 1 | 台账 §3；36→35 页 |
| 10 | 04:04 | **拍板表注精简"方案一"** | 三长表注压缩（141→55 / 102→60 / 89→55 词）；披露内容迁入正文不删除 | 台账 §3 |
| 11 | ~04:00 | 章节名统一 `Experiment`→`Experiments`（对齐 04119） | results.tex + diff 稿同步 | 台账 §3a |
| 12 | 04:09–04:25 | **拍板"方案一"：人类评价完全剥离摘要与正文**（对齐 RTN/RRTN/MambaOFR/GLMA 正文惯例；知情风险：R1-C3 must 级要求改为仅靠 Supplementary 支撑） | 摘要 + 正文 10 处全删（abstract A′ / intro 贡献③ / results×4 / discussion×2 / conclusion 补位句）；back-matter 三字段保留；R1 总注 + C3 + C10、R2 总注、cover letter 第 3 条同步；冻结清单两行出处改 supplementary only；三件套 + 3 docx 刷新；diff 47→45 页 | 台账 §3；本日志 §3-D8 |
| 13 | 05:08–05:57 | 全量核查（qtwx-mcp-2）判 CONDITIONAL-GO（提交包 7/9 过期）→ 执行 `submission-fix-plan-20260612.md` 步骤 1–8 | ①P1-1：`results.tex` §4.4.2 NR 段尾加 S1–S2/S4 中性锚点（执行 D8 回退预案=方案二）；②P1-2：三封 round1 回复信 `(label …)`→可见编号（R1 24/R2 27/R3 13 处，残留 0，four tables 保留）；③主稿 35 页 0 warning/undefined 无空白；④diff 45 页（基线 fd18e0ce+PICTUREENV）；⑤supp 8 页；⑥zip 40 文件干净目录复验独立编译；⑦三 docx 重导出；⑧7 件覆盖入 submission/，clean PDF md5==主稿 | 台账 §3c |
| 14 | 05:46–06:08 | 对标 `papers/Refs/` 录用(LightVSR/CMDDNet/GLMA/HVIFormer)+OFR(DeepRemaster/RTN/RRTN/MambaOFR) 与 `PaperDesign/` 规范找不足 → 用户拍板"全部" → 执行零风险纯润色 6 项 | #2 §4.3 补 PSNR/SSIM 公式；#3 §5.1 加 3 run-in 段标题(不改编号)；#4 结论补 Limitations 呼应；~~#1b §5.2 时域证据声明~~(轮次17 用户拍板不需要 E_warp→已撤除还原)；#6 主表加 Year 列(仅加列)；#7 §6 补应用前景。主稿仍 35 页 0 warning/undefined；table/subsec 编号 aux 比对全未变(回复信/docx 不动仍有效)；diff 45p、zip 40 文件复验、clean md5==主稿 | 台账 §3d |

| 15 | 06:20–06:30 | 用户质疑"保持 bold/underline"→要求读 Refs 录用同刊+联网核实表格规范 | 一手证据：GLMA/HVIFormer/CNN-Mamba 表注均红/蓝、LightVSR 不标、**0 篇 bold/underline**（联网当时不可用）→ 用户拍板"整体改红蓝"。全表 `\textbf{数值}`→红/`\underline{数值}`→蓝（仅数值单元格，"95\% CI"等数字开头表头用无字母残留判定排除），表注改 best in red/second in blue，supplementary 补 xcolor，顺带清 appendix_b 残留 warping-error 句。编号未变；主稿35/补充8/diff45 全门禁0；PDF 颜色 span 核验红=16711680/蓝=255；提交包刷新。**规范回写**：PaperDesign §5.3/§10+△6、`ofr-paper-mdpi-review/writing` skill | 台账 §3e |

| 16 | 06:35–06:42 | 用户指出回复信 ASCII 数字风格不专业→网络核实规范 | Editage/AIP/STEM guide 一致：科学计数法 `× 10⁻ⁿ`(真上标+真减号)、χ²、±、≥/≤，禁 `x/e-n/+-/chi-square`。三封信 `N x 10^-M`/`Ne-M`→`× 10^−M^`、`+-`→±、`chi-square`→χ²、`>=/<=`→≥/≤、`Spearman rho`→ρ、`Delta`→Δ、`Nx`/分辨率→×、一元负号→真减号；R1 30/R2 28/R3 12 改，数值不变。**避坑**：range en-dash 规则破坏 DOI URL→弃用。docx 重导出渲染正确（真上标 run）。规范回写 ofr-paper-rebuttal skill | 台账 §3f |

| 17 | 06:45–06:52 | 用户指 §3.6 标题不规范→网络调研 | APA/技术写作规范：run-in 标题须描述性短语、非裸缩写。`method.tex` §3.6 `\paragraph{LHFM.}`/`\paragraph{GFM.}` → 描述性（"LHFM response to transient and persistent defects."/"GFM response to flicker and abrupt transitions."）；子节名 Response Analysis 合规保留（回复信引用不受影响）。主稿 35 页 0 warning、§3.6 编号未变、diff 45p、zip 复验、clean md5==主稿。规范回写 ofr-paper-mdpi-review skill | 台账 §3g |

| 18 | 06:55–07:02 | 用户指明以 04119 录用稿为格式权威（papers/Others/Paper3） | 核查 04119：run-in 用 `\textbf{简洁描述.}` 行内（非 \paragraph）；**表格用 bold/underline 且经 MDPI 编辑往返确认合规**（表注写 Bold=best/underlined=second-best）。已把 §3.6+§5.1 五处 run-in 改 `\textbf{}` 行内、§3.6 标签精简；主稿 35p 0 warning、编号不变、diff 45p、clean md5==主稿、提交包刷新。**表格红蓝-vs-bold/underline 冲突待用户决策**（轮次19 拍板红蓝 vs 04119 bold/underline，均合规） | 台账 §3h |

| 19 | 07:10–07:16 | 用户指示删除训练伪代码 Algorithm A1 | `appendix_e.tex` 整体即该算法 → 删整附录 E + `\input` + results.tex §4.2 指向句（训练 recipe 已在 §4.2 同段）。回复信无引用、docx 不受影响；kingma2015adam 仍被 §4.2 用、bbl 不变。主稿 35→34 页、0 warning/undefined、无悬空 ref、编号未变（E 为末附录无表）、diff 44p、zip 39 文件复验、clean md5==主稿 | 台账 §3i |

| 20 | 07:20–07:30 | 用户指 supplementary.pdf 格式乱+大量空隙 | fitz 诊断根因(htbp 漂移+clearpage+大图独占)→ 加 float 包、[htbp]→[H]、去 clearpage、§S1 三网格→0.75\textwidth(进 p1)、Figure S4→0.45 minipage(进 p4)。8→6 页，botGap p1=1/p3=4/p4=11/p5=56 填满，仅余 p2(S2 独占,固有)+p6(末页)；0 warning/overfull、无空白；supp+zip 刷新复验。教训：单图独占页时图越小底隙越大→填页要"够大"，但要进同页又需"够小"，按页预算折中 | 台账 §3j |

## 3. 关键拍板记录（Decision Log）

- **D1 C-light（00:46）**：旧 GPTPro 润色稿整体落后于主稿且含 p 值张冠李戴 → 弃用回灌，只做冻结数值下的开篇句微调。教训：润色稿延迟合并必然腐化，回灌前先做新旧事实比对。
- **D2 B1 数据回退（00:23 窗口）**：无证据的"更好数字"一律回退实测真值；论文接受支柱不依赖单项第一。教训：SSOT YAML 出现未提交手改 = 红色警报，先溯源再写稿。
- **D3 IRB 口径（03:20）**：涉人研究声明按真实状态写（申请中 + 已知情同意），不虚构批准号。
- **D4 04119 对齐（03:46）**：复杂度呈现以"transparency 而非 minimum-latency claim"定位，向同刊已录用稿的单表风格收敛。
- **D5 删前向伪代码（03:57）**：正文只留贡献必需信息，可推导的过程细节进附录。
- **D6 表注精简（04:04）**：表注 1–3 句，披露性内容迁正文而非删除。
- **D7 章节名（~04:00）**：`Experiments` 复数，与目标期刊已录用稿一致。
- **D8 人类评价方案一（04:25）**：摘要+正文零提及主观实验，完整研究保留在 Supplementary（Tables S1–S2 / Figure S4）；back-matter 涉人声明保留。**回退预案**：若 R1 二轮坚持正文呈现，加回 results §4.4.2 一句锚点（方案二），5 分钟可完成。
- **D9 表格标记红/蓝（06:30，证据驱动纠偏）**：Agent 初判"bold/underline 亦合规（依据 04119，不在 Refs）"被用户质疑；改读 Refs 一手证据——同刊录用稿 3/4 红/蓝、0 bold/underline——据此拍板整体改红/蓝并回写规范。教训：**期刊呈现规范以"可核实的同刊录用一手证据"为准，不以不可核实的二手记忆为据**；规范结论必须回写 SSOT(PaperDesign)+skill 形成闭环。
- **D10 表格标记终裁=红/蓝（07:08）**：用户指出 04119 原稿在 `papers/Others/Paper3/`，核验确认 04119 **确用 bold/underline 且经 MDPI 编辑往返验证合规**——D9 当时"04119 不在 Refs 无法核实"的前提其实有原稿。结论纠正为：**红/蓝（GLMA/HVIFormer/CNN-Mamba）与 bold/underline（04119）在 Applied Sciences 均合规，择一且全篇统一即可**。用户拍板本项目**保留红/蓝**。教训补充：二手记忆（D9 的 04119=bold/underline）这次反而是对的，但当时无原稿佐证——一旦原稿可得（Paper3）应立即一手核验；规范应表述为"多种合规方案+择一统一"而非"唯一默认"。
- **D9 精确 p 值收口（2026-06-12 06:2x，作者拍板）**：摘要里精确 `p=5×10⁻⁵` 偏重、不符 CV 惯例 → 从 abstract（waveletmask.tex:45）、conclusion、results §4.2 删除精确值；结论/正文保留定性 "statistically significant" 与回复信承诺的 `p≤1.5×10⁻³`；**附录 A 配对统计表（CI+p 值）保留不动**（它是三封回复信 R1-C1/R3-C1 "Revision made: added paired clip-level statistics table in Appendix A" 的字面交付物，删了即与回复信自相矛盾）。回复信不动。重打包：latexpand 重展平（与编辑前差异精确=3 处 p 值）+ latexdiff（基线 fd18e0ce，与上轮 diff 仅差 3 处 p 值）→ 主稿 35 页 / diff 45 页 / supp 8 页，0 undefined（diff 的 7 处 undefined 全在 \DIFdel 删除块，属固有正常）；clean/highlighted PDF + zip 三件刷新，md5 同步，zip 干净目录复验独立编译 35+8 页 EXIT=0。冻结台账出处行同步更新。
- **V1 网页逐条核对（2026-06-12 06:2x，agent-browser-cli 读取 SuSy）**：直接读取 SuSy 三份 Review Report（R1 report=56538831 / R2=56630588 / R3=56629847），与本地 `reviewer1/2/3.md` 逐字比对：R1 10 条（8 弱点 + Table 2 + Table 3）、R2 10 条、R3 4 条，**24/24 全覆盖、内容逐字吻合、零遗漏**；三封 `response-letter-reviewer{1,2,3}.md` Comments 编号与网页段落顺序一一对应。注：网页上 R1/R2/R3 原文均**无编号**（R2/R3 为段落式），"Comments N" 编号为我方自加且顺序一致；网页摘要为 Round-1 旧态（+0.44 dB）属正常。
- **D10 基线术语统一（2026-06-12 06:4x，作者拍板）**：核对 Refs 下 OFR 已录用稿规范——RRTN/MambaOFR 用 "re-trained X"/直呼 "RTN"、GLMA 用 "baseline"，无一篇写 "X-style baseline"；"-style" 仅作架构族形容词（HVIFormer "CIDNet-style"）。据此把基线模型名**全部统一为 "RTN"**：正文 `RTN baseline`/`re-trained RTN baseline`/`shared-backbone RTN baseline`/`RTN (re-trained)`/`independently re-trained RTN`/`re-trained RTN output·checkpoint·row` + supplementary `RTN-style baseline` + 回复信 `RTN baseline`/`RTN-style baseline` → 共 **59（sections+主稿）+ 2（supp）+ 16（3 封 docx，含此前 reviewer2 的 5 处）** 处。**护栏保留**：实验设置确立"同协议重训"的句子（"RTN and all WaveletMask variants are trained with the same recipe"×1、"re-trained and re-evaluated by us"×3）不删（R1-C2/R3-C2 公平性根基）；**架构形容词保留**：`RTN-style propagation/gate/restoration`(3)、`RTN spatial indicator`、`RTN backbone` 不动。重打包：主稿 35 页 / diff 45 页 / supp 8 页，0 undefined；clean/highlighted/supplementary PDF + zip 刷新，md5 全同步，zip 干净目录复验独立编译 35+8 页 EXIT=0；3 docx 重导出（RTN baseline=0、four tables 保留、无 label）。全仓基线名变体残留=0。
- **D11 Table 6 复杂度表补全（2026-06-12 07:xx，作者拍板方案 A）**：原表三外部法（DeepRemaster/RRTN/MambaOFR）的 latency/throughput/peak-mem 为 "---"，且 Params/FLOPs 口径不可比（RTN/WaveletMask 含 RAFT、外部为论文引用值）。**在 L40 服务器 BasicOFR 统一框架（= accuracy 统一基准来源）用同协议（FP32/batch1/1×7×3×256×256/warmup3+runs10/FlopCounterMode）4 GPU 并行重测全 5 法**，填满全表并使 Params/FLOPs 可比。新值（Params/FLOPs/Lat/fps/Mem）：DeepRemaster 9.8781/914.0/17.9/391.6/0.35、RTN 5.7351/2184.7/169.5/41.3/0.24、RRTN 7.0503/2782.2/298.4/23.5/0.75、MambaOFR 8.2984/2832.8/404.8/17.3/0.60、WaveletMask 5.7360/2184.9/171.4/40.8/0.24。WaveletMask−RTN=+898 参数/+0.1494G FLOPs/+1.9ms/同显存（与旧框架一致）。整表从旧 Antigravity 框架（188.7/2445.7/6.1783）切到 BasicOFR；同步改 caption（去"external 未重测"）、§4.6 正文（latency 171.4 vs 169.5、720p 5.4s/9.9GB）、appendix_d runtime protocol（batch4 114.2/113.1、720p 5379/5419）、R2-C9 回复信（runtime 表扩为全 5 法 + 改"只测 RTN"为"全测"）、冻结台账 §5a。产物 `review/round2/artifacts/table6_runtime/*.json`。重打包：主稿 **34 页**（Table 6 填实+caption 精简回流，35→34）、diff 44 页、supp 8 页，0 undefined；clean/highlighted PDF + zip + reviewer2.docx 刷新，md5 全同步，zip 干净目录复验独立编译 EXIT=0；旧 runtime 值全仓 0 残留。遗留：p21 有 26% 页尾空白（P1 微瑕，回流所致）。
- **D12 伦理口径切换=路径 A｜Article 32 豁免（2026-06-18，作者拍板）**：撤销 D3 / §5 路径 C（先投后补·under application）。34 人专家主观研究为匿名、无创、非干预性视觉偏好调查，不对人体伤害、不涉及敏感个人信息或商业利益、仅记录匿名汇总分数 → 据《涉及人的生命科学和医学研究伦理审查办法》（2023，国家卫健委/教育部/科技部/国家中医药局四部门联合印发）**第 32 条**「使用匿名化信息数据 / 通过观察且不干扰公共行为产生的数据」情形**自我判定豁免**伦理审查。主稿 `\institutionalreview` 由 "under application" 改为 "waived in accordance with Article 32"（waveletmask.tex 仅 1 行变更），`\informedconsent` 维持 "obtained"；`waveletmask_diff.tex` latexdiff 全量重生成（旧"申请中"删除线 + 新"豁免"高亮，并清旧 `Not applicable.` 残留）。落地为独立自洽提交包 `Paper/submission_applsci_waived/`：clean/diff/supplementary 三件套本机 latexmk 重编 EXIT=0（34/34/6 页）+ 55M 源码 zip 干净目录复验独立编译 + `MANIFEST.md` 决策留痕；与旧 `submission_applsci/source_zip`（`Not applicable.` 旧快照、口径混乱）并存不沿用。第 32 条官方英文译名 / 四部门 / 2023 / 豁免情形经联网核对（gov.cn 原文 + HHHA 英译）一致。`cover-letter.md` 增 **Ethics Statement** 段主动声明自判豁免依据并表态可补机构伦理委正式确认（应对 MDPI self-determined waiver query）。**教训**：法规第 32 条原意是「向机构伦理委申请免除」，论文写 self-determined waiver 是 MDPI 通行写法但**必须** cover letter 主动说明 + 备妥正式确认，否则易触发编辑 query。
- **D13 Cover letter docx/pdf 同步豁免口径（2026-06-19，作者批 AB 一起推进）**：`review/round2/submission/Cover-Letter.docx`(06-13) / `.pdf`(06-12) 此前停留在 D12 之前的旧口径——只有 Principal Revisions + Declarations，**缺整个 Ethics Statement 段**，与 `cover-letter.md`(06-18 豁免口径) 不一致。本次用 officecli 在 docx 的 closing 段与 Declarations 之间插入 Ethics Statement（`Heading2` 标题 + `FirstParagraph` 正文，法规名 `italic=true`），与 md 对齐；PDF 经 officecli `view html` → headless Chrome `--print-to-pdf` 重新导出（2 页，`p=5×10⁻⁵` 上标正确，无分页杂质；pandoc+xelatex 因 Times New Roman 缺 U+207B/U+2075 上标字形被否决）。产物：docx 14057B / pdf 93427B，均 2026-06-19 刷新。**避坑**：officecli auto-resident 60s 空闲超时会丢未落盘改动 → 多步编辑须在单次 shell 调用内 `open → edits → view → close`，且连续两次 `add` 会分配相同 paraId 致碰撞覆盖，单会话内可避免。

## 4. 未提交工作区（截至 04:40，待入库归属确认）

- 论文侧：`sections/`（results/discussion/conclusion/introduction/method/appendix_d/e）、`waveletmask.tex`、`waveletmask_diff.tex`、round1 两封回复信 md、round2 checklist、cover-letter.md、冻结清单、本日志 —— 均属 round2 返修，建议合并为一批提交。
- 数据侧（D2 产物）：`Exp/paper_data/main_results.yaml`、`memory/core/datasets/SRWOV.yaml` —— B1 回退结果，跨 Paper/Exp 边界，提交时单独成批。

## 5. 遗留与风险跟踪

1. **IRB 批准号回填**（D3 遗留）：批准号签发后回填 protocol code 与日期再投，或与编辑部沟通后补。**2026-06-12 06:03 作者拍板「路径 C｜先投后补」**：先与编辑部确认「伦理审查申请中可先投、protocol code 签发后补」，确认后用现稿（`\institutionalreview` 维持 "under application"）重投。06:05 作者已直接与编辑部线下沟通，AI 起草的询问函（`editor-irb-inquiry.md`）作为多余产物已删除。待办：现稿按 SuSy「新投稿」入口重投（关联原稿 ID applsci-4348902）；IRB 批准号签发后回填 `\institutionalreview` 的 protocol code + 日期并重跑五步刷包链。**2026-06-18 更新：作者改拍板路径 A（Article 32 豁免），本遗留项关闭** —— 不再「先投后补 / 等批准号」；主稿与新独立包 `submission_applsci_waived/` 统一为豁免口径（详见 §3 D12），投稿入口改用新包，停用旧 `submission_applsci/source_zip`。
2. **R1-C3 支撑强度**（D8 知情风险，已缓解）：2026-06-12 05:57 已按 D8 回退预案执行方案二——`results.tex` §4.4.2 NR 段尾补一句中性锚点指向 Supplementary（Tables S1–S2 / Figure S4，不含 45.17% 等冻结数字），正文不再零锚点；完整专家主观研究仍仅在 Supplementary。
3. `GPTPro_polish_input/01_manuscript_current_full.tex` 为旧态快照（仍含主观句与旧摘要），下轮润色重建输入包时自动覆盖。
4. 13 Overfull + 2 Underfull hbox、BibTeX 月份宏警告：纯排版瑕疵，非阻断。
5. **E_warp/时域一致性指标：用户拍板不需要（2026-06-12 轮次17）**，已撤除 §5.2 的 warping-error 文本声明，全文不再提该指标；时域稳定性仅由 §4.4.3 定性与 Appendix B 运动分箱隐性支撑，不显式点名 warping-error。

## 6. 复盘要点（已沉淀至 ofr-paper-gptpro skill，原名 ofr-paper-polish）

1. **双文件制**：台账（checklist，门禁+核销）与复盘日志（revision-log，拍板+过程）分开维护，互相引用不互相替代。
2. **每次拍板必须落条目**：时间戳 + 拍板人 + 方案对比 + 知情风险 + 回退预案。口头拍板不落账 = 复盘断链。
3. **叙事级改动的一致性半径**：改正文一个主张，必须同步检查 摘要/结论/贡献列表/回复信/cover letter/冻结清单 六处；本轮方案一即按此半径执行（10 处正文 + 6 处信件/清单）。
4. **冻结数字白名单要带"出处"字段**：数值不变但出处变化（如移入 supplementary）也必须更新，否则 number gate 会误判。
5. **提交包刷新铁律**：clean PDF → latexdiff（固定基线 commit + PICTUREENV）→ zip 重建 → 干净目录解压复验独立编译 → docx 重导出，五步缺一不可；diff 稿与正文必须同日同源。
6. **无证据数字零容忍**：SSOT 数据文件的任何"变好"都要 git diff 溯源 + 找重训证据，找不到就回退。
