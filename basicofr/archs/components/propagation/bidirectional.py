"""双向 RNN 传播策略

RTN 和 MambaOFR 的共用传播模式：
  1. backward pass：从最后一帧向前传播，每帧做 align → aggregate → spatial
  2. forward pass：从第一帧向后传播，同上
  3. concat 双向特征后重建

这是一个工具类，用于新 idea 快速组合。现有 baseline 不需要改用此类。
"""

import torch
import torch.nn as nn

from ..arch_util import flow_warp


class BidirectionalRNNPropagation(nn.Module):
    """双向 RNN 传播

    将 alignment + aggregation + spatial 三个模块组合为完整的双向传播。

    Args:
        aggregation_fwd: 前向聚合模块
        aggregation_bwd: 后向聚合模块
        spatial_fwd: 前向空间修复模块
        spatial_bwd: 后向空间修复模块
        num_feat: 特征通道数
    """

    def __init__(self, aggregation_fwd, aggregation_bwd, spatial_fwd, spatial_bwd, num_feat):
        super().__init__()
        self.aggregation_fwd = aggregation_fwd
        self.aggregation_bwd = aggregation_bwd
        self.spatial_fwd = spatial_fwd
        self.spatial_bwd = spatial_bwd
        self.num_feat = num_feat

    def forward(self, lrs, forward_flow, backward_flow):
        """执行双向传播

        Args:
            lrs: (B, T, C, H, W) 输入序列
            forward_flow: (B, T-1, 2, H, W) 前向光流
            backward_flow: (B, T-1, 2, H, W) 后向光流

        Returns:
            backward_feats: list of (B, num_feat, H, W) 后向特征
            forward_feats: list of (B, num_feat, H, W) 前向特征
        """
        n, t, c, h, w = lrs.size()

        # ===== Backward pass =====
        backward_feats = []
        feat_prop = lrs.new_zeros(n, self.num_feat, h, w)
        residual_indicator = lrs.new_zeros(n, 1, h, w)

        for i in range(t - 1, -1, -1):
            curr_lr = lrs[:, i]
            if i < t - 1:
                flow = backward_flow[:, i]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
                pixel_prop = flow_warp(lrs[:, i + 1], flow.permute(0, 2, 3, 1))
                residual_indicator = torch.abs(pixel_prop[:, :1] - curr_lr[:, :1])
                aggregated = self.aggregation_bwd(feat_prop, curr_lr, residual_indicator, head=False)
            else:
                aggregated = self.aggregation_bwd(feat_prop, curr_lr, residual_indicator, head=True)

            feat_prop = self.spatial_bwd(aggregated)
            backward_feats.append(feat_prop)

        backward_feats = backward_feats[::-1]

        # ===== Forward pass =====
        forward_feats = []
        feat_prop = lrs.new_zeros(n, self.num_feat, h, w)
        residual_indicator = lrs.new_zeros(n, 1, h, w)

        for i in range(t):
            curr_lr = lrs[:, i]
            if i > 0:
                flow = forward_flow[:, i - 1]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
                pixel_prop = flow_warp(lrs[:, i - 1], flow.permute(0, 2, 3, 1))
                residual_indicator = torch.abs(pixel_prop[:, :1] - curr_lr[:, :1])
                aggregated = self.aggregation_fwd(feat_prop, curr_lr, residual_indicator, head=False)
            else:
                aggregated = self.aggregation_fwd(feat_prop, curr_lr, residual_indicator, head=True)

            feat_prop = self.spatial_fwd(aggregated)
            forward_feats.append(feat_prop)

        return backward_feats, forward_feats
