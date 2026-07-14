# Step 05 — Results + Discussion 审查

审查日期: 2026-05-13

## Results 结构

| § | 内容 | Figure 消费 | 匹配？ |
|---|------|------------|--------|
| 4.1 | Experimental Setup | — | — |
| 4.2 | SOTA Comparison | Fig5 (sota_synthetic) | ✅ 修复后 |
| 4.2 | SOTA Comparison | Fig6 (sota_realworld) | ✅ 修复后 |
| 4.3 | Ablation Study | Fig7 (ablation) | ✅ 修复后 |
| 4.4 | Complexity | Table only | — |

## Discussion 结构

| § | 内容 | Figure 消费 | 匹配？ |
|---|------|------------|--------|
| 5.1 | Why max > learned | — | — |
| 5.2 | Freq complement | — | — |
| 5.3 | Error-map | Fig8 (error_map) | ✅ 修复后 |
| 5.4 | Wavelet choice | — | — |
| 5.5 | Transferability | — | — |
| 5.6 | Application | — | — |
| 5.7 | Limitations | — | — |

## 修复记录（4 处 Figure 漂移）

| 文件 | 行 | 旧 | 新 |
|------|----|----|-----|
| results.tex | 49-55 | `Figure4.pdf` + `fig:degradation_examples` | `Figure5.pdf` + `fig:sota_synthetic` |
| results.tex | 57-63 | `Figure6.pdf` + `fig:realworld_qualitative` | `Figure6.pdf` + `fig:sota_realworld` |
| results.tex | 119-125 | `Figure5.pdf` + `fig:mask_interpretation` | `Figure7.pdf` + `fig:ablation` |
| discussion.tex | 25-31 | `Figure9.pdf` + `fig:error_map_analysis` | `Figure8.pdf` + `fig:error_map` |

## 数据一致性

- Table 1 (sota_synthetic): 数据与 main_results.yaml 一致
- Table 2 (ablation_core): 数据与 ablation_matrix.json 一致（base/local/flicker/ab）
- Table 3 (fusion_comparison): 8 变体全量，数据与 ablation_matrix.json 一致
- Table 4 (complexity): Params/FLOPs 与 RTN baseline 一致

## 自检

- [x] 固定实验图谱无缺口（SOTA_SYNTHETIC/SOTA_REALWORLD/ABLATION 三个 contract 全部消费）
- [x] 所有数据可追溯（ablation_matrix.json / main_results.yaml）
- [x] Results/Discussion 边界清晰
- [x] 消融命名基于 ablation_matrix.json 变体名
- [x] Fig5/6/7/8 描述基于真实完成的图
- [x] 无 DRIFT-SUSPECT 标记（4 处漂移已全部修复）
- [ ] `\ref{tab:sota_realworld}` 引用需确认表存在（Fig5 caption 中提到）

## 下一步

Step 6: Figure-Results 一致性审查（轻量 drift 检查 + caption 同步）
