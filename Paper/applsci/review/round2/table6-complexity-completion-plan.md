# Table 6（Computational Complexity）补全方案

> 触发：表内 9 格空白（DeepRemaster/RRTN/MambaOFR × Latency/Throughput/Peak Mem 全为 "---"），观感弱、且 R2-C9 审稿人本就要求"与其它 baseline 比 runtime"。
> 位置：`sections/results.tex` §4.6 `tab:complexity`（= `tab:runtime_benchmark`）。

## 一、现状诊断

| Method | Params(M) | FLOPs(G) | Latency | Throughput | Peak Mem |
|---|---|---|---|---|---|
| DeepRemaster | 9.88 | 65.3 | --- | --- | --- |
| RTN | 6.1783 | 2445.7 | 188.7 | 37.1 | 0.33 |
| RRTN | 7.05 | 198.7 | --- | --- | --- |
| MambaOFR | 8.30 | 202.3 | --- | --- | --- |
| WaveletMask | 6.1792 | 2445.8 | 183.3 | 38.2 | 0.33 |

**两层问题**：
1. **9 格 runtime 空白**：仅 RTN/WaveletMask 在 L40 实测；三外部法 "---"。caption 已声明"引用值未重测"，合规但观感弱，且未正面回应 R2-C9 "compare against other baselines"。
2. **Params/FLOPs 也不可比（更隐蔽）**：RTN/WaveletMask 的 FLOPs=2445.7/2445.8G **含 RAFT 光流**；外部法 65.3/198.7/202.3G 为论文引用值、**口径不同**（多半不含 flow、输入尺寸不同）。caption 虽注"different counting scopes"，但同列混不同口径，审稿人易误读"WaveletMask 比别人重 12×"。

**关键事实**：正文已声明"all ten methods were re-trained and re-evaluated by us under one unified benchmark pipeline"——即**我们已有三外部法的可运行代码/权重**，具备在同硬件同协议下重测的条件。

## 二、方案对比

### 方案 A（推荐）：统一协议重测三外部法，填满全表
在 **L40、FP32、batch=1、输入 1×7×3×256×256、3 warmup + 10 timed runs**（与现有 RTN/WaveletMask 协议完全一致）下，对 DeepRemaster/RRTN/MambaOFR 用我们 unified pipeline 的推理图测 Latency/Throughput/Peak Mem，并**在同一口径重算 Params/FLOPs**（统一是否计入 RAFT：建议给"含 flow 的端到端"口径，外部法无 flow 则按其自身端到端计，并在 caption 一句话说明各自是否含 flow）。
- **产出**：9 格全部填实测值；Params/FLOPs 变为同口径可比。
- **优点**：正面满足 R2-C9；消除空白与不可比；强化"轻量 gate"卖点（用真实 wall-clock 对比）。
- **连带改动（一致性必做）**：
  1. `tab:complexity` 填值 + caption 去掉"external...not re-benchmarked (---)"，改为"all rows measured by us on L40 under one protocol; FLOPs counting scope (incl./excl. RAFT) noted per row"。
  2. §4.6 复杂度正文改写（不再说"external not re-benchmarked"）。
  3. **R2-C9 回复信改写**：从"we restrict wall-clock to RTN... keep cross-method at published level"→"we measured all listed methods on L40 under one protocol (Table 6)"；同步 `Response-to-reviewer2.docx` 重导。
  4. 冻结台账 `04_FROZEN_numbers.md` 增列新 runtime 数；revision-log 记账。
  5. 五步重打包（主稿→diff→zip→clean/highlighted PDF）+ md5 复验。
- **前置条件 / 风险**：
  - 需 GPU（L40 或等效；若换卡须全表同卡重测，不可混卡）+ 三外部法可跑的 checkpoint/代码（应已存在于 unified pipeline）。
  - 公平性：不同架构（Mamba/Transformer/CNN）wall-clock 受实现成熟度影响，caption 须一句话提示"measured on identical hardware/protocol; absolute latency also reflects implementation maturity"。
  - 若某法 OOM 或无法在 256×256/7-frame 跑，按实际记 "OOM" 并在 caption 说明（不留空白）。

### 方案 B（备选，GPU/代码暂不可得时）：保留 "---" 但强化 rationale + 提升可比性
- 保留三外部法 runtime "---"，但 caption + 正文明确："we report wall-clock only for the shared-backbone pair (RTN vs WaveletMask) to isolate the gate's overhead; cross-method wall-clock under heterogeneous implementations is out of scope"。
- **同时修 Params/FLOPs 不可比**：要么把外部法 FLOPs 也按我们口径重算（轻量，仅前向计数，不需训练），要么在 caption 显式标注每行 FLOPs 是否含 flow。
- 合规但仍偏弱；只建议在 A 不可行时用。

### 方案 C（轻量重构，可与 A/B 叠加）：拆分逻辑分组
把表分两块：上半"Params/FLOPs（全方法，统一口径）"、下半"Measured runtime（实测子集）"，用 `\multicolumn` 分组标题，使"未测 runtime"从"空格"变成"该指标仅对实测子集报告"，视觉上消除残白感。

## 三、推荐路径
- **首选 A**（若 L40 + 三外部法代码就绪）：一次把表做实、做可比，并把 R2-C9 从"回避"升级为"正面满足"，对 round-2 加分最大。
- **A 不可行 → B + C**：先把 Params/FLOPs 口径统一（B 的轻量部分）+ 分组重构（C）消除观感问题，runtime 保留同骨干对比并强化 rationale。

## 四、执行后验证门（无论 A/B/C）
- [ ] 表内无裸空白；"---"/"OOM" 均在 caption 显式定义。
- [ ] 每个数字有 provenance（实测 log / 论文引用并标注）。
- [ ] caption 与表数据、§4.6 正文、R2-C9 回复信三处一致（跑 `references/internal-consistency-audit.md`）。
- [ ] 冻结台账更新；五步重打包 + md5 复验；diff/zip 同步。
- [ ] 主稿 0 undefined、页空白扫描通过。

## 五、待你拍板
1. 选 A / B / C（或 A+C）？
2. 若选 A：确认 GPU（L40?）与三外部法 checkpoint/推理脚本是否就绪、由谁跑（实验侧 vs 我协助拼脚本）。
