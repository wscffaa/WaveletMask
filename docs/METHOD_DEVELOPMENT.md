# 方法开发落点（独立论文仓）

- 方法实现的 SSOT 是仓根 `<method_pkg>/`；网络、数据、模型与损失分别放在
  `archs/*_arch.py`、`data/*_dataset.py`、`models/*_model.py`、
  `losses/*_loss.py`，由注册器按后缀扫描。
- `basicofr/` 与 `basicsr/` 是固定的 1.0.0 框架副本，不放论文方法创新。
  框架升级必须通过母仓同步流程，并单独评估 1.1.0 评测口径变化。
- 新 variant 的最小闭环是：方法代码 → `Exp/options/` 配置 →
  `Exp/registry.yaml` 登记 → `scripts/verify_configs.py --check` →
  仓根入口运行 → `scripts/harvest_results.py` 生成 receipt。
- 公共契约默认保持向后兼容；确需改变数据、指标、checkpoint 或配置 schema 时，
  先在 `PAPER.yaml` 与实验记录中明确协议边界。
