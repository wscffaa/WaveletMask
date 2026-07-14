# Step 06 — Figure-Results 一致性审查

日期: 2026-05-13

## 审查结果

### \ref{fig:*} vs Contract 对齐

| Figure | Contract Label | tex Label | 文件 | 匹配 |
|--------|---------------|-----------|------|------|
| Fig1 | fig:motivation | fig:motivation | Figure1.pdf | ✅ |
| Fig2 | fig:pipeline | fig:pipeline | Figure2.pdf | ✅ |
| Fig3 | fig:lhfm | fig:lhfm | Figure3.pdf | ✅ |
| Fig4 | fig:gfm | fig:gfm | Figure4.pdf | ✅ |
| Fig5 | fig:sota_synthetic | fig:sota_synthetic | Figure5.pdf | ✅ |
| Fig6 | fig:sota_realworld | fig:sota_realworld | Figure6.pdf | ✅ |
| Fig7 | fig:ablation | fig:ablation | Figure7.pdf | ✅ |
| Fig8 | fig:error_map | fig:error_map | Figure8.pdf | ✅ |

### Caption vs Contract 对齐

| Figure | Contract core_conclusion | Caption 匹配 |
|--------|-------------------------|-------------|
| Fig1 | Pixel-domain confuses motion/damage; wavelet separates | ✅ |
| Fig2 | WaveletMask dual-mask replaces pixel indicator | ✅ |
| Fig3 | LHFM: geometric-mean detail-band responses → high_freq_conv → local mask | ✅ |
| Fig4 | GFM: minimum LL temporal response → low_freq_conv → flicker mask | ✅ |
| Fig5 | WaveletMask removes damage more completely, avoids false positives | ✅ |
| Fig6 | Generalizes to real-world, suppresses flicker | ✅ |
| Fig7 | LHFM fixes scratches, GFM reduces flicker, WaveletMask combines both | ✅ |
| Fig8 | Error maps compare baseline and WaveletMask under synthetic ground truth | ✅ |

### Drift 判定

无 drift。所有 figure caption 与 contract `core_conclusion` 一致。Figure3/Figure4 已按新架构图拆分为 LHFM/GFM，结果图仅显示编号顺延，PDF 文件名不改。
所有 `\includegraphics` 路径指向存在的文件。

### 修复记录

| 修复 | 文件 | 操作 |
|------|------|------|
| fig:wavelet_motivation → fig:motivation | introduction.tex | 2处 `\ref` + `\label` 统一为 contract label |
| Figure3 合并模块图 → LHFM | method.tex / contracts | 旧合并模块标签改为 `fig:lhfm`，caption 聚焦 LHFM |
| Figure4 GFM 新增 | method.tex / contracts | 新增 `fig:gfm`，caption 聚焦 GFM |

### Gate

- [x] 所有 `caption_synced=true`
- [x] 无 drift-suspect 残留
- [x] 所有 `\ref{fig:*}` 与实际文件名一致
- [x] Step 6 done → advance to Step 7
