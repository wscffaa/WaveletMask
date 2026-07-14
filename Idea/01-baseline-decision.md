# Baseline Decision: RTN (Swin Spatial)

## 选择结论

**Baseline**: RTN with Swin Transformer Spatial Module (Swin_Spatial_2)
**注册名**: `WaveletMaskBaseNet`（继承 `Video_Backbone`）

## 选择理由

1. **RTN 是 BasicOFR 框架的核心 baseline**：双向循环传播 + 光流对齐 + Gated Aggregation 的经典架构
2. **存在明确的改进点**：`Gated_Aggregation` 中的 `residual_indicator` 使用简单像素差分，是信息瓶颈
3. **架构简洁**：Swin RTN 的 Spatial 模块为 `Swin_Spatial_2`，不涉及 Mamba/SSM 等复杂组件，改动风险低
4. **成熟的训练配置**：已有完整的 REDS 合成退化训练 pipeline

## Baseline 架构

```
Input: (B, T, 3, H, W)
│
├── FlowEstimator (RAFT) → forward_flow, backward_flow
│
├── Backward Propagation (t-1 → 0):
│   ├── flow_warp(feat_prop, backward_flow)
│   ├── residual_indicator = |warped_pixel - curr_pixel|  ← 改进点
│   ├── Gated_Aggregation(feat_prop, curr_lr, indicator)
│   └── Swin_Spatial_2(aggregated) → cached_feats[i]
│
├── Forward Propagation (0 → t-1):
│   ├── flow_warp(feat_prop, forward_flow)
│   ├── residual_indicator = |warped_pixel - curr_pixel|  ← 改进点
│   ├── Gated_Aggregation(feat_prop, curr_lr, indicator)
│   ├── Swin_Spatial_2(aggregated)
│   └── concat(cached_feat, feat_prop) → conv → upsample → output
│
└── Output: tanh(sr_rlt + img_up(curr_lr))
```

## 关键超参

| 参数 | 值 | 说明 |
|------|----|------|
| num_feat | 16 | 特征通道数 |
| num_block | 6 | 残差块数 |
| flow_type | raft | 光流估计器 |
| Swin embed_dim | 64 (4*16) | Swin Transformer 嵌入维度 |
| Swin depths | [2,2,2] | 各阶段深度 |
| Swin num_heads | [4,4,4] | 注意力头数 |
| batch_size | 1 | 每 GPU 批大小 |
| total_iter | 200K | 总训练迭代数 |

## 未选择其他 Baseline 的原因

- **MambaOFR**: 有 SS2D + DCNv2 等复杂组件，改动 residual_indicator 需同时兼容 DCN 对齐，风险高
- **RRTN**: 递归结构太深，消融实验训练成本高
- **纯 CNN backbone**: 不具有代表性
