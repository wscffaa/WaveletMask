"""
WaveletMask Local Fusion Architecture (Innovation A+C).

集成创新 A: 局部高频掩码（HH/HL/LH 子带几何平均）
集成创新 C: 像素级 alpha 融合（空间掩码 + 小波掩码）

Architecture: RTN + WaveletNoiseMask + PixelWiseMaskFusion
Innovations: A (Local mask) + C (Pixel-wise fusion)

Integration Date: 2025-12-23
"""

import torch
import torch.nn.functional as F
from torch import nn

from basicsr.utils.registry import ARCH_REGISTRY

from basicofr.archs.flow import FlowEstimator
from basicofr.archs.components.arch_util import ResidualBlockNoBN, flow_warp, make_layer
from basicofr.archs.spatial.swin_spatial import Swin_Spatial_2
from .components.wavelet_mask import WaveletNoiseMask, PixelWiseMaskFusion

from .waveletmask_base_arch import Video_Backbone, Gated_Aggregation, PSUpsample


@ARCH_REGISTRY.register()
class WaveletMaskLocalFusionNet(Video_Backbone):
    """WaveletMask Local Fusion architecture - RTN with local mask + fusion (Innovation A+C).

    在 forward 循环中：
    1. 计算原始空间域 residual_indicator
    2. 计算小波域高频掩码
    3. 使用 PixelWiseMaskFusion 融合两种掩码
    4. 用融合掩码替换 residual_indicator

    Args:
        num_feat (int): Number of feature channels. Default: 16.
        num_block (int): Number of residual blocks. Default: 6.
        flow_type (str): Type of flow estimator. Default: 'raft'.
        fusion_mid_channels (int): Fusion network hidden channels. Default: 32.
    """

    def __init__(self,
                 num_feat=16,
                 num_block=6,
                 flow_type='raft',
                 fusion_mid_channels=32,
                 **kwargs):
        super().__init__(num_feat=num_feat, num_block=num_block, flow_type=flow_type, **kwargs)

        # 【创新 A】小波噪声掩码模块
        self.wavelet_mask = WaveletNoiseMask()

        # 【创新 C】像素级掩码融合模块
        self.mask_fusion = PixelWiseMaskFusion(mid_channels=fusion_mid_channels)

    def forward(self, lrs):
        """Bidirectional propagation with fused mask (Innovation A+C).

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

        # 【创新 A】预计算局部高频小波掩码 (B, T, 1, H, W)
        wavelet_masks = self.wavelet_mask.compute_local_mask(lrs)

        # ===== Stage 1: backward propagation =====
        cached_feats = []
        feat_prop = lrs.new_zeros(n, self.num_feat, h, w)
        for i in range(t - 1, -1, -1):
            curr_lr = lrs[:, i, :, :, :]
            if i < t - 1:
                flow = backward_flow[:, i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
                pixel_prop = flow_warp(lrs[:, i + 1, :, :, :], flow.permute(0, 2, 3, 1))

                # 原始空间域掩码
                spatial_mask = torch.abs(pixel_prop[:, :1, :, :] - curr_lr[:, :1, :, :])

                # 【创新 C 集成点】融合空间掩码与小波掩码
                wavelet_mask_i = wavelet_masks[:, i, :, :, :]
                fused_indicator = self.mask_fusion(spatial_mask, wavelet_mask_i)  # (B, 1, H, W)

                aggregated = self.Backward_Aggregation(feat_prop, curr_lr, fused_indicator, head=False)
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
                pixel_prop = flow_warp(lrs[:, i - 1, :, :, :], flow.permute(0, 2, 3, 1))

                # 原始空间域掩码
                spatial_mask = torch.abs(pixel_prop[:, :1, :, :] - curr_lr[:, :1, :, :])

                # 【创新 C 集成点】融合空间掩码与小波掩码
                wavelet_mask_i = wavelet_masks[:, i, :, :, :]
                fused_indicator = self.mask_fusion(spatial_mask, wavelet_mask_i)

                aggregated = self.Forward_Aggregation(feat_prop, curr_lr, fused_indicator, head=False)
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
