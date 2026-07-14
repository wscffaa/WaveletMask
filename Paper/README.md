# WaveletMask — Paper Root

本目录是 007-waveletmask 项目的论文入口。顶层只保留项目级索引；具体期刊的源码、提交包、proof、审稿、工作产物按期刊归入对应子目录。

- 当前期刊：Applied Sciences（MDPI）→ `applsci/`
- 进度总览：见 `PROGRESS.yaml`

## 结构

- `applsci/` — Applied Sciences 期刊根（canonical 主稿 + lifecycle + review + assets + 工作区）
- `PROGRESS.yaml` — 项目论文进度（期刊、稿号、阶段、更新时间）

## 约定

- 顶层不再堆放散落的提交包/proof/PPT/工作目录；新增期刊请新建 `Paper/<journal>/`。
- 历史提交包、proof 包按 `applsci/lifecycle/YYYY-MM-DD_<stage>/` 归档，不删除、不合并。
