# WaveletMask Round 1 Review — 最终交叉分析

> 本文档合并四方信息：审稿人原话（R1 共 10 条 / R2 共 10 条）、GPTPro 返修方案（T1-T16）、Codex 审查（3 FAIL + 3 WARN）、Bundle 1 执行方案（Scheme A' v2 + final-revision-plan）。

## 0. 结论摘要

**Verdict**：两位审稿人均认可选题价值和论文写作质量，但一致认为证据链不足——统计验证缺失、比较协议不公平、failure case 未量化、真实世界验证仅靠 NR metrics。返修核心是把"诚实的 heuristic module 说明"升级为"有统计支撑、failure case 已量化、有人评、mask 可解释、部署开销可测的轻量频域门控方案"。

- **总任务数**：15（T1-T14 + T17，T15 已放弃，T16 条件触发）
- **必做数（P0）**：11 项（T1-T10 + T17），其中 2 项需新训练、6 项需新评估/分析、3 项为文本/表格修改
- **强烈建议数（P1）**：3 项（T11-T13），其中 2 项需新训练/新实现
- **可选数（P2）**：1 项（T14）+ T16 条件触发

---

## 1. 需求全景矩阵（合并 R1 + R2 + GPTPro T1-T16）

| # | 需求 | R1 | R2 | GPTPro 任务 | 需重训？ | 能 claim shrink？ | 优先级 |
|---|------|-----|-----|-------------|---------|------------------|--------|
| 1 | 多随机种子 + 统计显著性 | — | R2-1 "most serious weakness" | T1 | **是**（**已决策：3 seeds** x 2 模型 = 4 新 runs） | 绝对不能 shrink，这是接收门槛 | P0 |
| 2 | 比较协议重构（拆表 + 收缩 claim） | — | R2-2, R2-9 Table 2 | T2 | 否（文本/表格） | N/A | P0 |
| 3 | 真实世界主观评测 + 专家评审 | R1-8 | R2-3 "must include" | T3 | 否（评估） | 绝对不能 shrink | P0 |
| 4 | Artifact-type breakdown | — | R2-5 | T4 | 否（分析） | 绝对不能 shrink | P0 |
| 5 | Persistent scratches 专项 | R1-1 | R2-7 | T5 | 否（新合成参数 + 评估） | 绝对不能 shrink | P0 |
| 6 | Motion/occlusion 分层评测 | R1-2 | R2-7 | T6 | 否（分析，但依赖 T13 FC 变体） | 绝对不能 shrink | P0 |
| 7 | Scene cuts/flashes/illumination 评测 | R1-4 "dedicated evaluation" | R2-7 "dedicated experiments" | T7 | 否（新合成子集 + 评估） | 绝对不能 shrink | P0 |
| 8 | Mask interpretability + overlap | R1-10 | R2-8 (ablation) | T8 | 否（分析） | 绝对不能 shrink | P0 |
| 9 | Runtime/latency/memory/throughput | R1-9 | — | T9 | 否（benchmark） | 绝对不能 shrink | P0 |
| 10 | Wavelet 家族与 level ablation | R1-3 | R2-6 "Coiflets and multi-level" | T10 | **是**（新 wavelet 实现 + 训练） | 有条件可 shrink（风险高，双审稿人点名） | **P0**（Bundle 1 升级） |
| 11 | Complementarity 定量分析 | R1-5 | — | T11 | 否（分析 + 少量 frozen-cue 训练） | 有条件可 shrink | P1 |
| 12 | 更强 fusion 对比 | R1-6 | R2-8 | T12 | **是**（新 fusion 实现 + 训练） | 有条件可 shrink | P1 |
| 13 | Motion-aware/cut-aware 变体 | R1-2, R1-4 | R2-7 | T13 | **是**（新 FC 代码，可能升级主模型） | 有条件可 shrink | P1 |
| 14 | 外部公共数据/跨数据泛化 | R1-8 | — | T14 | 否（外部数据评估） | 有条件可 shrink | P2 |
| 15 | ~~额外强 baseline 同协议重训~~ | — | R2-2 | T15 | ~~**是**~~ | **已放弃**，靠拆表+收缩 claim 解决 | ~~P2~~ |
| 16 | 辅助 mask supervision | R1-10 | R2-8 | T16 | **是**（条件触发） | 有条件可 shrink | P2 |
| 17 | 补引 Springer 文献 | R1-7 | — | T17 | 否 | 直接做 | P0 |

### Bundle 1 关键决策（已纳入）

| 决策 | 结论 |
|------|------|
| T10 wavelet family | 从 P1 升级为 **P0**（双审稿人点名） |
| T15 额外 baseline 重训 | **放弃**，靠拆表+收缩 claim 解决公平性 |
| 方法 contract | **修论文匹配代码**（wavelet-only gate），不改代码 |
| 基础设施 | 用 **OFR Skill**（`ofr_cli.py train fleet`），不用手写脚本 |

### 与旧版的关键差异（Codex 审查修正）

| Codex 指出的问题 | 旧版状态 | 本版修正 |
|-----------------|---------|---------|
| scene cut 被标"部分"但 R2 用了 "dedicated evaluation" | 部分回应 | 改为 P0 必做，R1-4 明确 "dedicated evaluation"，R2-7 明确 "dedicated experiments" |
| 主观评测/独立数据集没进执行清单 | 未列入 | T3 主观评测 + T14 独立数据集均列入 |
| "必须做新实验"漏了 4 项强制评估 | 仅 5 项 | 扩展为完整矩阵，主观评测/artifact breakdown/scene cut/failure case 定量全部纳入 |
| "可以直接做"含实验/分析项 | runtime/mask overlap 归入"直接做" | 重新分类：T9 runtime / T8 mask overlap / T11 complementarity 归入"必须新增评估/分析" |
| 训练量估算漏了 Coiflet + DTCWT | 仅 db2/sym2/2-level | T10 增强版含 coif1/DTCWT 选项 |
| seed 可复用性未确认 | 未说明 | §3.2 详细标注 |
| 待确认项仅 3 项 | 3 项 | §5 扩展为 12 项决策点 |

---

## 2. 按执行方式分类

### 2.1 必须新增训练（不能 claim shrink）

#### T1. 多随机种子与统计显著性

**审稿人原话**：
> R2-1: "The most serious weakness of the manuscript is the absence of statistical validation. The authors explicitly state that all reported results are based on single-run experiments...it is impossible to determine whether the reported gain is statistically meaningful or simply due to training randomness."

**GPTPro 建议**：5 seeds（最低 3 seeds）。**已决策：选 3 seeds**，省时间，MDPI Applied Sciences 不需要顶会级别统计强度。

**训练量**（已确定）：3 seeds x 2 模型（RTN base + WaveletMask ab）= 6 runs，减去已有 seed 2021 = **4 新 runs**（base_seed2022, base_seed2023, ab_seed2022, ab_seed2023）。ablation 变体（+LHFM, +GFM 等）保留单 seed 结果，不额外重训。

**关键约束**：如果显著性不够，**已决策：主动收缩 claim**——"consistent trend with low parameter overhead" 比 "statistically dominant replacement" 更安全。不硬撑。

---

### 2.2 必须新增评估/分析（不重训但必须做）

#### T2. 比较协议重构

**审稿人原话**：
> R2-2: "Table 2 combines retrained RTN and retrained WaveletMask with results from other methods reported in the literature. This introduces a substantial protocol mismatch...claims that WaveletMask outperforms methods such as MambaOFR, RVRT, or RRTN are not scientifically justified."
> R2-9: "Results originate from different experimental protocols. Direct ranking is therefore questionable."

**GPTPro 建议**：拆成两张表——表 A（严格同协议比较，承载主结论）+ 表 B（文献结果上下文表，不做排名）。收缩所有跨协议 superiority claim。文本改法：从 "Relative to the reported MambaOFR number, the margin is +0.61 dB" 改为 "Relative to literature-reported numbers, WaveletMask is competitive, but those comparisons are contextual only and should not be interpreted as a fair same-protocol ranking."

**工作量**：纯文本/表格重写，0 GPU。1-2 天。

---

#### T3. 真实世界主观评测 + 专家评审

**审稿人原话**：
> R2-3: "The real-world evaluation relies entirely on NIQE, BRISQUE, CLIPIQA and qualitative figures. The authors correctly note that these metrics are content-sensitive and not conclusive. The authors **must** include subjective visual quality assessment, user preference study and expert restoration evaluation."
> R1-8: "additional experiments on independent real-world datasets or publicly available restoration benchmarks would be important"

**GPTPro 建议**：双层设计——(1) 普通观察者 pairwise preference（覆盖 scratch/flicker/强运动/cut 附近，4 维度：overall / scratch-dust removal / flicker-temporal stability / detail preservation）；(2) 专家评审（参考 ITU-R BT.2095 protocol，维度：artifact suppression / temporal coherence / fine-detail preservation / archival plausibility / over-restoration risk）。将 Table 3 NR metrics 降级为 diagnostic，主结论让给人评。

**工作量**：准备 stimuli + 招募被试/专家 + 数据分析。**1-3 周**（主要瓶颈是人员招募）。

---

#### T4. Artifact-type breakdown

**审稿人原话**：
> R2-5: "The paper claims improvements for scratches, dust, and flicker, yet no separate evaluation is reported for each degradation category."

**GPTPro 建议**：在 synthetic pipeline 里构建 scratch-only / dust-only / flicker-only / mixed 四个子集，分别报告 restoration metrics（PSNR/SSIM/LPIPS/DISTS）+ detection metrics（IoU/Precision/Recall/F1/AUPRC）。此实验同时回应 R1-5（互补性）、R1-10（mask 对应退化位置）、R2-4（算子在目标 artifact 上是否合理）。

**工作量**：修改 synthetic generator 标签逻辑 + 分子集评估脚本。0 GPU 训练。**2-3 天**。

---

#### T5. Persistent scratches 专项实验

**审稿人原话**：
> R1-1: "The method assumes that scratches and dust are transient artifacts. However, many real archival films contain persistent scratches spanning multiple consecutive frames. Please include a **dedicated experiment** quantifying performance on persistent damage and analyze whether the geometric-mean criterion in LHFM suppresses such defects."

**GPTPro 建议**：新增 persistent-scratch benchmark（persistence length L ∈ {1,2,3,5}，width/opacity/orientation 参数化，image-plane fixed + content-following 两种模式），比较 LHFM / fused mask / final gate / FC-LHFM 的响应随 persistence 变化曲线。在方法节明确承认 geometric-mean 对 persistent damage 的抑制机制："if S_{t-1} = S_t = S_{t+1}, then d_t^S → 0"。

**工作量**：修改 synthetic generator + 评估脚本 + 新图。**3-4 天**。

---

#### T6. Motion/occlusion 分层评测

**审稿人原话**：
> R1-2: "The wavelet-domain differences are computed without explicit motion compensation, whereas the RTN spatial indicator relies on optical-flow alignment. Please investigate the impact of camera motion, object motion, and occlusions on LHFM and GFM responses, and provide a motion-aware variant or an ablation using flow-compensated wavelet differences."

**GPTPro 建议**：用 RAFT flow 按 motion magnitude（low/medium/high）/ occlusion ratio / camera-vs-object motion 分层，每层报告 restoration metrics + mask false positive rate + clean region activation ratio。依赖 T13 FC 变体作为对比。

**工作量**：flow 计算 + 分层分析脚本。**3-5 天**（含 T13 依赖）。

---

#### T7. Scene cuts / flashes / illumination 评测

**审稿人原话**：
> R1-4: "The GFM branch is intended to capture brightness flicker, but the proposed minimum-difference criterion may also respond strongly to scene cuts, flashes, and abrupt illumination changes. Please provide a **dedicated evaluation** on clips containing scene transitions and discuss potential mitigation strategies such as cut detection or temporal consistency constraints."
> R2-7: "The manuscript repeatedly mentions that scene cuts remain problematic...However, **no dedicated experiments** quantify these failure modes."

**GPTPro 建议**：构建三个专项子集（scene-cut: 不连续 clip 拼接 / flash: 全局曝光脉冲 / illumination-ramp: gamma/exposure 漂移），评估 GFM activation strength + false positive rate + restoration quality + temporal artifacts。补 cut-aware GFM 缓解（histogram distance + low-freq SSIM drop + flow reliability collapse 的阈值规则组合）。最低可接受版本：至少做 cut-aware ablation 证明能 suppress false responses。

**工作量**：新合成子集 + cut detection 代码 + 评估。**3-4 天**。

---

#### T8. Mask interpretability 与 overlap 分析

**审稿人原话**：
> R1-10: "The detector masks are trained only through restoration losses and receive no explicit supervision. As a result, it is unclear whether the generated masks actually correspond to degradation locations. Please provide qualitative mask visualizations, overlap analyses with synthetic degradation annotations, or other interpretability studies to validate that the learned masks behave as intended."

**GPTPro 建议**：在 synthetic 上比较 m_local / m_flicker / m_wavelet / final gate 与 GT mask 的 IoU/Precision/Recall/F1/AUPRC；补 inside-vs-outside activation ratio（退化区域平均响应 / 非退化区域平均响应）+ top-k localization（top 5% 像素落在 GT 退化区域的比例）。real-world 上手工标 50-100 帧做 qualitative 验证。

**工作量**：分析脚本 + 少量手工标注。**3-4 天**。

---

#### T9. Runtime / latency / memory / throughput

**审稿人原话**：
> R1-9: "The computational analysis reports parameters and FLOPs but omits runtime, latency, memory consumption, and throughput. Since one motivation of the method is its lightweight design, practical deployment metrics on representative hardware should be reported and compared against RTN and other baselines."

**GPTPro 建议**：在同一硬件（L40，可选补 3090）上报告 latency（ms/clip 或 ms/frame）/ throughput（fps 或 clips/s）/ peak GPU memory / params / FLOPs。batch size ∈ {1, 4}，分辨率 ∈ {256x256, 720p}。warm-up 多轮 + `torch.cuda.synchronize()` + 平均。比较 RTN vs WaveletMask（可选加 RRTN / MambaOFR）。

**工作量**：benchmark 脚本 + 跑 inference。**1-2 天**。

---

#### T17. 补引 Springer 文献

**审稿人原话**：
> R1-7: "When introducing wavelets, please refer to https://link.springer.com/article/10.1007/s40534-025-00393-5"

**做法**：在 introduction 或 related work 的 wavelet 段落补引该文献。放在 wavelet 信号分析泛化层，不需要深度讨论。

**工作量**：5 分钟。

---

### 2.3 论文文本/表格修改（直接做）

以下修改贯穿所有任务，不单独编号，但必须完成：

1. **Abstract**：收缩 claim，不再写 "outperforms all baselines"，改为 "provides statistically verified, artifact-aware enhancement to RTN with minimal overhead"
2. **Introduction**：contribution 4 从 "max fusion is preferable" 改为 "under the tested training recipe, max fusion offers the most reliable trade-off"
3. **Method**：新增 "Response Analysis of LHFM and GFM" canonical-case subsection；GFM 衏述改为 "sensitive to brightness flicker, but can also respond to shot boundaries, flashes, or abrupt illumination changes"
4. **Results**：重构为 10 个小节（同协议比较 / 文献上下文 / 多 seed 统计 / artifact breakdown / failure-mode analysis / wavelet+fusion ablation / mask interpretability / real-world subjective / runtime）
5. **Discussion**：从 "limitations + future work" 改为 "已量化 failure modes / 已缓解 failure modes / 仍为边界条件"
6. **Conclusion**：删除 broad generalization，改为修订版导向的保守结论
7. **Table 2**：拆为 tab:same_protocol_main + tab:literature_context
8. **Table 3**：保留但降级为 diagnostic NR metrics，主结论让给人评
9. **文献**：补引 Bernardini et al. 2025（审稿人指定）+ Ivanova 2023 / Liu 2024 / Liu 2026 / ITU-R BT.500/BT.2095

---

### 2.4 有 claim shrink 空间的条目

#### 条目 A：Fusion alternatives 结论（对应 T12 / R1-6 / R2-8）

| | shrink 前 | shrink 后 |
|---|---|---|
| Claim | "Max fusion is broadly preferable to learned fusion" | "Under the tested CNN training recipe, max fusion offers the most reliable trade-off; stronger learned alternatives remain under exploration" |
| 条件 | 不跑 T12 新实验即可 shrink | 若跑 T12 且 max 仍赢，可恢复更强 claim |

**审稿人原话**：
> R1-6: "Since the conclusion that max fusion is preferable depends heavily on this design choice, please evaluate stronger alternatives...before drawing **broader** conclusions."
> R2-8: "The ablation study only examines LHFM, GFM and fusion strategies."

**判定**：有条件可 shrink。如果选择不做 T12（新 fusion 训练），必须把结论从 "broadly preferable" 收缩到 "under current CNN recipe"。如果做 T12 且某个 learned fusion 反超 max，应直接升级主模型。

---

#### 条目 B：Wavelet family 选择（对应 T10 / R1-3 / R2-6）

| | shrink 前 | shrink 后 |
|---|---|---|
| Claim | "Haar wavelet is optimal for this task" | "Haar offers the simplest efficient default; other families may provide accuracy gains at higher cost" |
| 条件 | 不跑 T10 即可 shrink | 但两位审稿人都明确要求此 ablation，shrink 风险高 |

**审稿人原话**：
> R1-3: "The paper does not demonstrate that Haar is the most suitable decomposition for old-film artifacts."
> R2-6: "No comparison is provided against Daubechies wavelets, Symlets, **Coiflets** and **multi-level wavelet decompositions**."

**判定**：理论上可 shrink，但风险高——两位审稿人都 explicit 要求此实验。建议做 T10（至少 Haar + db2 + sym2 + 2-level Haar）。

---

#### 条目 C：General applicability claim（贯穿全文）

| | shrink 前 | shrink 后 |
|---|---|---|
| Claim | "WaveletMask demonstrates general applicability for old-film restoration" | "WaveletMask provides an RTN-specific, frequency-aware gating enhancement validated under the SRWOV protocol with evidence of cross-scenario robustness" |
| 条件 | 无条件必须 shrink | 不管做不做 T14 |

**审稿人原话**：
> R2 总评: "the experimental evidence is not yet sufficiently strong to support the claimed effectiveness and **general applicability**"

**判定**：绝对 shrink 项，与是否做新实验无关。

---

#### 条目 D：Real-world superiority（对应 T3 / R2-3 / R2-10）

| | shrink 前 | shrink 后 |
|---|---|---|
| Claim | "WaveletMask achieves superior real-world performance" | "NR metrics provide diagnostic evidence; human evaluation confirms preference in targeted artifact categories" |
| 条件 | 不做 T3 人评则必须 shrink | 做了 T3 人评后可根据结果决定 |

**审稿人原话**：
> R2-10: "The no-reference metrics are inconsistent — LHFM achieves best BRISQUE, GFM achieves best NIQE and WaveletMask is not best on any metric. This weakens claims regarding real-world superiority."

**判定**：不做 T3 必须 shrink；做了 T3 后以人评为主证据。

---

## 3. 训练量估算（区间制）

### 3.1 训练 runs 汇总

| 任务 | 最低 | 中等 | 最高 | 说明 |
|------|------|------|------|------|
| T1 多 seed | **4 新 runs** | — | — | **已决策：3 seeds x 2 模型（base+ab）**，已有 seed 2021 |
| T10 Wavelet 家族 | 3 新 runs | 7 新 runs | 15 新 runs | Haar 已有；db2/sym2/2-level-Haar 必做；可选 db4/sym4/coif1/DTCWT |
| T12 更强 fusion | 2 新 runs | 4 新 runs | 6 新 runs | 2-4 种 fusion x 1 seed（若做多 seed 则翻倍） |
| T13 FC-WaveletMask | 0（ablation only） | 3 新 runs | 5 新 runs | **已决策：等 T13 数据出来再定是否升级主模型** |
| ~~T15 重训 baseline~~ | **已放弃** | — | — | 靠拆表+收缩 claim 解决 |
| T16 mask supervision | 0（条件触发） | 1 新 run | 3 新 runs | 仅当 T8 显示 mask 对齐很差时 |
| **合计** | **9 新 runs** | **19 新 runs** | **33 新 runs** | T1 已固定 4 runs，其余取决于 T10/T12/T13 决策 |

### 3.2 Seed 可复用性

| 来源 | 可复用到 | 条件 |
|------|---------|------|
| 已有单 seed RTN run | T1 的 seed-1 | 同配置即可，无需重跑 |
| 已有单 seed WaveletMask run | T1 的 seed-1 | 同配置即可 |
| T1 的 seeds | T12（fusion ablation）| 若 fusion 变体使用相同 backbone，可共享 seed 编号但需独立训练 |
| T1 的 seeds | T10（wavelet ablation）| **不可直接复用**——不同 wavelet 改变了 DWT 层滤波器权重，必须独立训练 |

**注意**：T10 的 wavelet 变体（db2/sym2/coif1/DTCWT 等）改变了 DWT 层的滤波器权重，与 Haar 不共享 checkpoint，每个 wavelet 变体需独立训练。Coiflet 和 DTCWT 在旧版估算中被遗漏，本版已补入 T10 增强版选项。

### 3.3 需要新实现代码的任务

| 任务 | 新代码内容 | 复杂度 |
|------|-----------|--------|
| T4 | Synthetic generator 退化标签输出 + 分子集脚本 | 低 |
| T5 | Persistent scratch 合成参数扩展（L, width, opacity, orientation, 两种模式） | 中 |
| T6 | RAFT flow 分层脚本（motion magnitude / occlusion ratio / camera-vs-object） | 中 |
| T7 | Scene-cut/flash/illumination 合成子集 + cut detection（histogram + SSIM + flow） | 中 |
| T8 | Mask overlap 计算 + inside-vs-outside activation + top-k localization | 低 |
| T10 | db2/sym2/coif1/DTCWT 的 DWT 实现替换 | 中-高（DTCWT 尤其） |
| T11 | Pearson/Spearman correlation / top-k overlap / frozen-cue logistic regression | 中 |
| T12 | Channel-attention / residual / uncertainty-aware / normalized fusion 模块 | 中 |
| T13 | Flow-compensated DWT（warp 邻帧 + 在对齐帧上做 DWT + LHFM/GFM） | 高 |

---

## 4. 执行依赖图

```
                    ┌──────────────────────────────────────┐
                    │           独立任务（可并行）            │
                    ├──────────────────────────────────────┤
                    │  T1 (多 seed 训练) ← 最高优先          │
                    │  T2 (拆表+收缩 claim，纯文本)          │
                    │  T7 (scene cut 合成子集)               │
                    │  T9 (runtime benchmark)               │
                    │  T10 (wavelet 家族训练)                │
                    └──────────┬───────────────────────────┘
                               │
              ┌────────────────┼────────────────────┐
              ▼                ▼                    ▼
    ┌─────────────┐  ┌──────────────┐   ┌──────────────────┐
    │ T4          │  │ T8           │   │ T12              │
    │ artifact    │  │ mask overlap │   │ 更强 fusion 训练  │
    │ breakdown   │  │ (依赖 T4 结构)│   │ (依赖 T1 seed)   │
    └──────┬──────┘  └──────┬───────┘   └────────┬─────────┘
           │                │                     │
           ▼                ▼                     ▼
    ┌─────────────┐  ┌──────────────┐   ┌──────────────────┐
    │ T5          │  │ T11          │   │ T13              │
    │ persistent  │  │ complement-  │   │ FC-WaveletMask   │
    │ scratches   │  │ arity (依赖T8)│   │ (新代码+训练)     │
    └──────┬──────┘  └──────────────┘   └────────┬─────────┘
           │                                      │
           │         ┌──────────────────────┐     │
           │         │ T6 motion/occlusion  │◄────┘
           │         │ 分层 (依赖 T13 FC)    │
           │         └──────────┬───────────┘
           │                    │
           ▼                    ▼
    ┌──────────────────────────────────────┐
    │              T3                       │
    │  主观评测 + 专家评审                    │
    │  (需要所有模型输出 + stimuli 准备)      │
    └──────────────────┬───────────────────┘
                       │
                       ▼
    ┌──────────────────────────────────────┐
    │  T14 (外部数据泛化，可选)              │
    │  T15 (重训 baseline，可选)            │
    │  T16 (mask supervision，条件触发)     │
    └──────────────────────────────────────┘
```

**关键路径**：T1 → T4 → T5 → T3（最长串行链）
**并行窗口**：T1 / T7 / T9 / T10 可完全并行启动

---

## 5. 需用户确认的决策点

1. ~~**Seed 数量**~~：**已决策：3 seeds**（11 新 runs）。省时间，MDPI 不需要顶会级别统计强度。
2. ~~**是否升级 FC-WaveletMask 为主模型**~~：**已决策：等 T13 数据出来再定**。FC 版在 motion/persistent 上明显更好则升级，否则降为 ablation。
3. ~~**是否重训外部 baseline（T15）**~~：**已决策：放弃**。靠拆表+收缩 claim 解决公平性问题。
4. **主观评测范围（T3）**：是否做专家评审（需要找档案修复/数字修复相关人员），还是仅做普通观察者 pairwise preference？
5. **Wavelet 家族范围（T10）**：最低版（Haar + db2 + sym2 + 2-level Haar）还是增强版（加 db4/sym4/coif1/DTCWT）？DTCWT 实现复杂度较高。
6. **Fusion 范围（T12）**：测几种 stronger alternatives？GPTPro 建议 4 种（normalized weighted sum / channel-attention / residual / uncertainty-aware），还是精简到 2 种？
7. **是否做 T14 外部数据泛化**：SRWOV 是目前唯一的公开 paired old-film video benchmark，外部数据只能用 Ivanova 2023 的 frame-based analogue film damage dataset 做辅助验证，是否值得投入？
8. ~~**如果 multi-seed 结果不显著怎么办**~~：**已决策：接受 claim 收缩**为 "consistent trend with low parameter overhead"。不硬撑，不加 seed。
9. **如果 wavelet family 中 Haar 不是最优怎么办**：接受 "best efficiency-accuracy trade-off" 结论，还是切换主 wavelet 并重跑所有实验？
10. **如果 T12 中某个 learned fusion 超过 max 怎么办**：升级主模型（需重跑 T1 多 seed），还是保持 max 但收缩 claim？
11. **审稿人指定文献（Bernardini 2025）的处理策略**：策略 A（稳妥引用，放在 wavelet 信号分析泛化层）还是策略 B（礼貌说明不直接相关，补引更贴题文献）？
12. **论文最终定位**：如果所有结果都偏弱，是否接受从 "SOTA improvement paper" 降级为 "lightweight interpretable gating enhancement study"？

---

## 6. 风险与缓解

### 6.1 如果 multi-seed 不显著

**风险**：+0.44 dB PSNR 在 3 seeds 下 confidence intervals 重叠，paired test p > 0.05。

**已决策缓解方案**：
- 主动收缩 claim 为 "consistent trend with low parameter overhead"
- 将论文定位为 "轻量可解释门控增强研究" 而非 "SOTA improvement paper"
- 用 effect size（Cliff's delta）而非仅 p-value 论证实际意义
- 不加 seed，不硬撑

### 6.2 如果 wavelet family Haar 不是最优

**风险**：db4 或 sym4 在某些 metric 上超过 Haar。

**缓解**：
- 将结论改为 "Haar offers the best accuracy-efficiency trade-off under tested configurations"
- 如果替代 wavelet 全面碾压 Haar，切换主 wavelet 并重跑 T1 多 seed（训练量 +3-5 runs）
- 审稿人要的是 "justify the choice"，不是一定要求 Haar 第一

### 6.3 如果 fusion alternatives 超过 max

**风险**：某个 learned fusion（如 channel-attention）在多个 metric 上稳定超过 max fusion。

**缓解**：
- **直接升级主模型**——把该 learned fusion 作为 revised proposed method，max 退为 lightweight baseline
- 不要在 rebuttal 里硬解释为什么 max 还是 proposed
- 升级后需重跑 T1 多 seed（训练量 +3-5 runs）

### 6.4 如果 FC-WaveletMask 收益不明显

**风险**：flow-compensated 版本在 motion 子集上改善有限，或 persistent scratch 改善不显著。

**已决策缓解方案**：
- **等 T13 数据出来再定**：FC 版明显更好则升级主模型，否则降为 ablation
- 至少正面回答了审稿人关于 motion compensation 的质疑
- 在论文中明确写出 "flow-compensated differencing substantially reduces false responses in strong-motion cases" 即使绝对数值改善有限

### 6.5 如果主观评测招募困难

**风险**：找不到足够的普通被试或专家评审人员。

**缓解**：
- 普通观察者：用 Prolific / MTurk 等在线平台，最低 15-20 人
- 专家评审：找影像修复/视频后期从业者或文化遗产数字化方向研究者，最低 3-5 人
- 如果专家实在找不到，至少完成普通观察者 pairwise preference + 在 limitation 中说明
- GPTPro 原话："如果你找不到真正的 film-restoration expert，至少找影像修复/视频后期从业者"

### 6.6 如果 NR metrics 持续矛盾

**风险**：即使补了人评，BRISQUE/NIQE/CLIPIQA 仍然互相冲突。

**缓解**：
- 以人评/专家评审为主证据，NR metrics 降级为 diagnostic
- 在 discussion 中解释：BRISQUE 偏爱平滑输出、NIQE 对 contrast/grain 敏感、CLIPIQA 偏语义感知——老胶片 grain/对比度/纹理保留会让不同 NR 指标给出相反判断
- 这段解释必须写，否则 R2-10 的 "Table 3 contradiction" 会回来
