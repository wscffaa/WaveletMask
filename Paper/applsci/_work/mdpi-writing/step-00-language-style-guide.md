# 语言风格指南 — 轨道 A

来源: `memory/knowledge/know-067-paper-prompt-system.md`。

## 时态与语态

| 场景 | 时态 | 语态 | 规则 |
|------|------|------|------|
| 声明/贡献 | 一般现在时 | 主动 | 直接陈述声明。 |
| 方法描述 | 过去时或现在时 | 操作描述偏向被动 | 先描述模块功能，再给出公式。 |
| 实验设置 | 过去时 | 被动 | 将训练和评估条件作为已完成动作报告。 |
| 结果 | 一般现在时 | 主动 | 直接陈述观察到的对比，不使用对冲语气。 |
| 相关工作 | 一般现在时 | 主动 | 每段一个方法家族。 |

## 缩写规则

- Abstract：避免括号内缩写。
- Introduction：每个缩写定义一次。
- Caption：保持自包含性。
- 正文定义后：统一使用缩写。

## 禁用短语

除非引用来源要求，否则删除以下表述：
- to the best of our knowledge
- we observe that
- the key insight is that
- it is worth noting that
- an important observation is that
- the purpose of X is to
- this result deserves careful interpretation
- delve
- tapestry / landscape / realm
- underscore / crucial / pivotal
- harness / leverage（`use` 足够时不用这些词）

## 句式节奏

- 目标分布：30% 短句，50% 中长句，20% 长句。
- 密集技术句后跟一个短句。
- 不允许连续三句长度几乎相同。

## 数据完整性

- 不得将过期的 RealWorld 日志与已冻结的 synthetic 声明混用。
- 007 项目的 synthetic 声明必须使用 `Exp/paper_data/main_results.yaml` 和 `Exp/ablation/ablation_results.yaml`。
- RealWorld 声明在刷新冻结日志替换过期快照前保持临时状态。
- Bold 和次优标记必须按指标方向逐列检查。
