# WaveletMask Writing Revision Log (R5 Pipeline)

## Round 0 (2026-05-13)

### 修改内容
- 旧 state/contracts/figures 归档到 `archive/v1-pre-r5/`
- R5 新管线从 Step 0 重新开始
- Step 0 motivation 确定：频率解耦（像素域 indicator 局限 → 小波域分离）
- Figure 1 已有手绘成品（frequency_spectrum archetype），可直接复用

### Review 发现的问题
- 无（首次初始化）

### 仍未解决
- Step 1 figure lifecycle 待执行

## Round 1 — Final Polish (2026-05-14)

### 修改内容
- LaTeX: 添加 `\setlength{\headheight}{20pt}` 消除 23 个 fancyhdr warning
- Bib: 30 条 @inproceedings 批量补全 publisher/address 字段
- Bib: goodfellow2014gan, vaswani2017attention 补全 eprint/archivePrefix/primaryClass
- Review P0-1~P2-6 全部已修复或已处理（详见 step-11-final-review-notes.md）

### Review 发现的问题
- P0-1 (tab:sota_realworld undefined): 已在 Round 0 修复，LaTeX 重新编译确认 ✅
- P0-2 (realworld data stale): 已修复 ✅
- P1-1~P1-7: 全部修复 ✅
- P2-1~P2-6: 全部修复 ✅

### 仍未解决
- P1-6 (realworld best NIQE = worst synthetic): 已知 limitation，Discussion §6.6 已声明
- P2-2 (Contribution 列表过长): 保留当前版本，系作者决策

### 最终状态
- LaTeX: 0 error, 0 warning, 0 undefined reference
- BibTeX: 0 error
- DOI 覆盖率: 44/46 (95.7%)
- 7 图 7 表全部交叉引用可达
- 投稿就绪: ✅

## Round 2 — Architecture Figure Sync (2026-05-14)

### 修改内容
- Method 主线更新为 3 张架构图：Figure 2 完整 pipeline，Figure 3 LHFM，Figure 4 GFM
- Figure 3 从合并模块图改为 LHFM 专属说明；Figure 4 新增 GFM 专属说明
- 结果图仅按 LaTeX 显示编号顺延：synthetic/real-world/ablation/error-map 分别为 Figure 5-8，PDF 文件名不重命名
- `state.json`、figure plan、contracts、Step 05/06 notes 同步到 8 图版本

### Review 发现的问题
- 旧 Discussion 静态写法仍引用 “Figure 7 selection record”，已改为 error-map figure 条件记录
- Figure 1 caption 存在展示性动词 `shows`，已改为 `reports`

### 仍未解决
- 无新增阻塞项

### 最终状态
- LaTeX: `latexmk -g -pdf -interaction=nonstopmode -halt-on-error waveletmask.tex` 退出码 0
- Figure consistency: 8 图的 tex label、contract label、PDF 路径一致
- 投稿图数状态: 8 图 7 表

## Round 3 — Ablation Naming Sync (2026-05-14)

### 修改内容
- Results 消融表格公开展示名从 A/B/local/flicker 同步为 LHFM/GFM、+LHFM only、+GFM only、WaveletMask
- Figure 7 caption 改为使用 +LHFM only、+GFM only、WaveletMask
- Fig7 contract、structure guide、motivation anchor、intro outline、Step 06 notes 同步消融命名

### Review 发现的问题
- 旧表格 caption 仍使用 A/B 过渡映射
- Figure 7 caption 仍使用旧 A/B 组合表达

### 仍未解决
- 无新增阻塞项

### 最终状态
- LaTeX: `latexmk -g -pdf -interaction=nonstopmode -halt-on-error waveletmask.tex` 退出码 0
- Naming scan: targeted public-facing A/B residue scan under `Paper/applsci` 无命中
- Diff hygiene: `git diff --check` 通过
