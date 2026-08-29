# 实验协议（BasicOFR 1.0.0）

本仓的实验口径由仓根 `PAPER.yaml:basicofr_version` 与 `VERSION` 共同锁定为
`1.0.0`。BasicOFR 1.1.0 引入不同的灰度评测口径；未经单独协议决策，不得把
进行中的 1.0.0 实验直接升级并与历史数字混用。

## 必过门禁

1. 配置只从 `Exp/options/{train,test}/` 选取，并在 `Exp/registry.yaml`
   登记；运行前执行 `python3 scripts/verify_configs.py --check` 做结构检查。
   实际测试前还必须执行
   `python3 scripts/verify_configs.py --check --require-checkpoints`。
2. 启动入口固定为仓根 `train.py` / `test.py`。相对数据、权重与输出路径均
   以本仓根目录解析；日志必须记录 `VERSION` 与当前 Git SHA（dirty 状态也记录）。
3. 对比实验按 optimizer iterations 对齐；随机种子、数据集、验证频率与各指标
   参数以对应 YAML 为准。1.0.0 配置不含 `gray_gt`，不得当作 1.1.0 结果。
4. 论文数字只能由 `python3 scripts/harvest_results.py` 从日志生成
   `Exp/metrics.yaml`；禁止手抄日志或改写既有 receipt。

迁移前已经存在、且没有生成器头的 `Exp/metrics.yaml` 仅保留为 legacy
evidence；只有能追溯到原始日志并由收割器重建的记录，才可升级为当前 receipt。

`--check` 不声明 checkpoint 已就绪；历史权重不属于 Git 恢复面。严格 checkpoint
门禁失败时，必须恢复配置指定的精确 artifact，不能用其他 variant 权重替代，也不能
声称测试闭环。数据集与 GPU 运行时可用性仍须在目标机器单独核验。
