"""
WaveletMask Simple Architecture (Innovation A Simplified).

简化版：仅保留创新 A（高频掩码），使用固定权重融合替代 CNN 融合。
用于验证核心创新点的有效性。

Architecture: RTN + Simplified WaveletNoiseMask
Innovations: A (Local high-frequency mask) with fixed fusion

Integration Date: 2025-12-24
"""

import torch
import torch.nn.functional as F
from torch import nn

from basicsr.utils.registry import ARCH_REGISTRY

from basicofr.archs.flow import FlowEstimator
from basicofr.archs.components.arch_util import ResidualBlockNoBN, flow_warp, make_layer
from basicofr.archs.spatial.swin_spatial import Swin_Spatial_2
from .components.wavelet_mask import WaveletNoiseMask

from .waveletmask_base_arch import Video_Backbone, Gated_Aggregation, PSUpsample


@ARCH_REGISTRY.register()
class WaveletMaskSimpleNet(Video_Backbone):
    """WaveletMask Simple architecture - RTN with simplified wavelet mask (Innovation A only).

    简化版本:
    - 仅使用高频掩码（创新 A）
    - 移除闪烁检测（创新 B）
    - 使用固定权重融合替代 CNN 融合（简化创新 C）

    Args:
        num_feat (int): Number of feature channels. Default: 16.
        num_block (int): Number of residual blocks. Default: 6.
        flow_type (str): Type of flow estimator. Default: 'raft'.
        wavelet_weight (float): Wavelet mask weight in fusion. Default: 0.3.
    """

    def __init__(self,
                 num_feat=16,
                 num_block=6,
                 flow_type='raft',
                 wavelet_weight=0.3,
                 **kwargs):
        super().__init__(num_feat=num_feat, num_block=num_block, flow_type=flow_type, **kwargs)

        # 【创新 A 简化版】小波噪声掩码模块
        self.wavelet_mask = WaveletNoiseMask()

        # 融合权重（固定，不使用 CNN）
        self.wavelet_weight = wavelet_weight

    def _compute_fused_indicator(self, feat_prop, curr_lr, wavelet_mask):
        """计算融合后的掩码指示符

        简化融合策略:
        - spatial_indicator: 基于残差的传统指示符
        - wavelet_mask: 小波高频掩码
        - fused = (1 - w) * spatial + w * wavelet  (w=wavelet_weight)
        """
        # 计算传统空间域残差指示符
        warped_gray = feat_prop.mean(dim=1, keepdim=True)  # 简化的灰度
        curr_gray = curr_lr.mean(dim=1, keepdim=True)
        spatial_indicator = torch.abs(warped_gray - curr_gray)

        # 归一化到 [0, 1]
        spatial_indicator = spatial_indicator / (spatial_indicator.max() + 1e-8)
        wavelet_normalized = wavelet_mask / (wavelet_mask.max() + 1e-8)

        # 固定权重融合
        fused = (1 - self.wavelet_weight) * spatial_indicator + self.wavelet_weight * wavelet_normalized

        return fused

    def forward(self, lrs):
        """Bidirectional propagation with simplified wavelet mask (Innovation A).

        Args:
            lrs (Tensor): Input video (B, T, C, H, W)

        Returns:
            Tensor: Restored video (B, T, C, H, W)
        """
        n, t, c, h, w = lrs.size()

        assert h >= 64 and w >= 64, (
            'The height and width of input should be at least 64, '
            f'but got {h} and {w}.'
        )

        forward_flow, backward_flow = self.comp_flow(lrs)

        # 【创新 A】预计算小波掩码 (B, T, 1, H, W)
        wavelet_masks = self.wavelet_mask.compute_local_mask(lrs)

        # ===== Stage 1: backward propagation =====
        cached_feats = []
        feat_prop = lrs.new_zeros(n, self.num_feat, h, w)
        for i in range(t - 1, -1, -1):
            curr_lr = lrs[:, i, :, :, :]
            if i < t - 1:
                flow = backward_flow[:, i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))

                # 【简化融合】使用固定权重融合空间掩码和小波掩码
                wavelet_indicator = wavelet_masks[:, i, :, :, :]
                aggregated = self.Backward_Aggregation(feat_prop, curr_lr, wavelet_indicator, head=False)
            else:
                zero_indicator = lrs.new_zeros(n, 1, h, w)
                aggregated = self.Backward_Aggregation(feat_prop, curr_lr, zero_indicator, head=True)

            feat_prop = self.backward_resblocks(aggregated)
            cached_feats.append(feat_prop)

        cached_feats = cached_feats[::-1]

        # ===== Stage 2: forward propagation =====
        feat_prop = torch.zeros_like(feat_prop)
        outputs = []
        for i in range(t):
            curr_lr = lrs[:, i, :, :, :]
            if i > 0:
                flow = forward_flow[:, i - 1, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))

                wavelet_indicator = wavelet_masks[:, i, :, :, :]
                aggregated = self.Forward_Aggregation(feat_prop, curr_lr, wavelet_indicator, head=False)
            else:
                zero_indicator = lrs.new_zeros(n, 1, h, w)
                aggregated = self.Forward_Aggregation(feat_prop, curr_lr, zero_indicator, head=True)

            feat_prop = self.forward_resblocks(aggregated)

            # ===== Stage 3: feature fusion + output =====
            cat_feat = torch.cat([cached_feats[i], feat_prop], dim=1)
            sr_rlt = self.lrelu(self.concate(cat_feat))
            sr_rlt = self.lrelu(self.up1(sr_rlt))
            sr_rlt = self.lrelu(self.up2(sr_rlt))
            sr_rlt = self.lrelu(self.conv_hr(sr_rlt))
            sr_rlt = self.conv_last(sr_rlt)

            base = self.img_up(curr_lr)
            outputs.append(torch.tanh(sr_rlt + base))

        return torch.stack(outputs, dim=1)

    def visualiza_mask(self, lrs):
        """可视化论文所需的关键掩码。

        Returns:
            dict: {
                'spatial': 空间差异掩码 (B, T, 1, H, W) - baseline 使用的原始掩码
                'local': 局部高频掩码 (B, T, 1, H, W) - 创新A
            }
        """
        n, t, c, h, w = lrs.size()
        forward_flow, _ = self.comp_flow(lrs)

        # 计算 spatial 掩码（baseline 对比用）
        spatial_masks = []
        for i in range(t):
            if i > 0:
                flow = forward_flow[:, i - 1, :, :, :]
                pixel_prop = flow_warp(lrs[:, i - 1, :, :, :], flow.permute(0, 2, 3, 1))
                curr_lr = lrs[:, i, :, :, :]
                spatial_mask = torch.abs(pixel_prop[:, :1, :, :] - curr_lr[:, :1, :, :])
            else:
                spatial_mask = lrs.new_zeros(n, 1, h, w)
            spatial_masks.append(spatial_mask)

        local_masks = self.wavelet_mask.compute_local_mask(lrs)

        return {
            'spatial': torch.stack(spatial_masks, dim=1),
            'local': local_masks,
        }
