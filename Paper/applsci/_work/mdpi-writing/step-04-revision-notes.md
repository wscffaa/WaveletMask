# Step 04 — Revision Notes

日期: 2026-05-13

## 修改内容

### Fix 1 (P1): Fig2 引用缺失
- 文件: `sections/method.tex:7`
- 修改: §3.1 Pipeline Overview 首句加入 `Figure~\ref{fig:pipeline}`
- Before: "The complete WaveletMask-ab pipeline operates in two phases..."
- After: "The complete WaveletMask-ab pipeline, illustrated in Figure~\ref{fig:pipeline}, operates in two phases..."

### Fix 2 (P2): HaarDWT → HaarDWT2d
- 文件: `sections/method.tex:44,68`
- 修改: 2 处 `HaarDWT` → `HaarDWT2d`（代码类名一致）
- Algorithm 1 行44 + Eq.7 行68

## Contract 同步

无 contract 漂移。Fig2 contract 的 `required_refs` 已满足。

## 实验状态

实验全部完成（L40 指标在 ablation_matrix.json），无需等待。直接跳过等待点 B，进 Step 5。

## Revisit 决策

无需回 Step 1 — 2 个修复都是文本级，不改变 figure 内容。
