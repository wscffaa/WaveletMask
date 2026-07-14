"""
WaveletMask Gated Flicker Architecture - Fixed Local-Flicker Fusion.

修复 waveletmask_full 中的掩码融合问题：
- 原版: torch.max(local, flicker) - flicker 误检测导致特征坍缩
- gated_flicker: 加权融合 + 阈值门控 - 更稳定的掩码融合

问题根因：
1. flicker_mask 在场景变化时误检测为"闪烁"
2. max 操作放大误检测，导致 fused_mask 值过高
3. Gated_Aggregation 过度抑制，feat_prop 坍缩到接近零

修复方案：
1. 加权融合: 0.7 * local + 0.3 * flicker（降低 flicker 权重）
2. 阈值门控: flicker 仅在高于 0.5 时生效

Architecture: RTN + WaveletNoiseMask (gated flicker) + PixelWiseMaskFusion
Innovations: A (Local mask) + B (Flicker mask, gated) + C (Pixel-wise fusion)

Integration Date: 2025-12-28
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
class WaveletMaskGatedFlickerNet(Video_Backbone):
    """WaveletMask Gated Flicker - 门控闪烁掩码架构.

    相比 WaveletMaskFullNet 的改进：
    1. 使用加权融合替代 max: local_weight * local + flicker_weight * flicker_gated
    2. 添加 flicker 阈值门控: flicker * (flicker > threshold)

    这避免了 flicker_mask 误检测导致的特征坍缩问题。

    Args:
        num_feat (int): Number of feature channels. Default: 16.
        num_block (int): Number of residual blocks. Default: 6.
        flow_type (str): Type of flow estimator. Default: 'raft'.
        fusion_mid_channels (int): Fusion network hidden channels. Default: 32.
        local_weight (float): Weight for local mask. Default: 0.7.
        flicker_weight (float): Weight for flicker mask. Default: 0.3.
        flicker_threshold (float): Threshold for flicker gating. Default: 0.5.
    """

    def __init__(self,
                 num_feat=16,
                 num_block=6,
                 flow_type='raft',
                 fusion_mid_channels=32,
                 local_weight=0.7,
                 flicker_weight=0.3,
                 flicker_threshold=0.5,
                 **kwargs):
        super().__init__(num_feat=num_feat, num_block=num_block, flow_type=flow_type, **kwargs)

        # 【创新 A+B】完整小波噪声掩码模块
        self.wavelet_mask = WaveletNoiseMask()

        # 【创新 C】像素级掩码融合模块
        self.mask_fusion = PixelWiseMaskFusion(mid_channels=fusion_mid_channels)

        # 【修复】加权融合参数
        self.local_weight = local_weight
        self.flicker_weight = flicker_weight
        self.flicker_threshold = flicker_threshold

    def _fuse_wavelet_masks(self, local: torch.Tensor, flicker: torch.Tensor) -> torch.Tensor:
        """门控加权融合：降低 flicker 误检测影响.

        Args:
            local: 局部高频掩码 [B, T, 1, H, W]
            flicker: 闪烁掩码 [B, T, 1, H, W]

        Returns:
            fused: 融合后的掩码 [B, T, 1, H, W]
        """
        # 阈值门控: 仅当 flicker > threshold 时生效
        flicker_gated = flicker * (flicker > self.flicker_threshold).float()

        # 加权融合（替代 max 操作）
        fused = self.local_weight * local + self.flicker_weight * flicker_gated

        # 归一化到 [0, 1]
        return torch.clamp(fused, 0.0, 1.0)

    def forward(self, lrs):
        """Bidirectional propagation with gated flicker mask fusion.

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

        # 【修复】分别计算 local 和 flicker 掩码，然后门控加权融合
        local_masks = self.wavelet_mask.compute_local_mask(lrs)    # [B, T, 1, H, W]
        flicker_masks = self.wavelet_mask.compute_flicker_mask(lrs)  # [B, T, 1, H, W]
        wavelet_masks = self._fuse_wavelet_masks(local_masks, flicker_masks)

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

                # 【创新 C】融合空间掩码与门控小波掩码
                wavelet_mask_i = wavelet_masks[:, i, :, :, :]
                fused_indicator = self.mask_fusion(spatial_mask, wavelet_mask_i)

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

                # 【创新 C】融合空间掩码与门控小波掩码
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
                'flicker': 闪烁掩码 (B, T, 1, H, W) - 创新B
                'fused': 门控融合后的掩码 (B, T, 1, H, W)
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

        # 分别计算 local 和 flicker 掩码
        local_masks = self.wavelet_mask.compute_local_mask(lrs)
        flicker_masks = self.wavelet_mask.compute_flicker_mask(lrs)
        fused_masks = self._fuse_wavelet_masks(local_masks, flicker_masks)

        return {
            'spatial': torch.stack(spatial_masks, dim=1),
            'local': local_masks,
            'flicker': flicker_masks,
            'fused': fused_masks,
        }

    def visualiza_feature(self, lrs):
        """Visualize feature energy for gated flicker model diagnostics.

        Returns:
            backward_gate_energy: Backward gate feature energy (B, T, 1, H, W)
            backward_state_energy: Backward hidden state energy (B, T, 1, H, W)
            forward_gate_energy: Forward gate feature energy (B, T, 1, H, W)
            forward_state_energy: Forward hidden state energy (B, T, 1, H, W)
        """
        n, t, c, h, w = lrs.size()

        assert h >= 64 and w >= 64, (
            f'Input resolution must be at least 64x64, got {h}x{w}'
        )

        forward_flow, backward_flow = self.comp_flow(lrs)

        # 使用门控融合的掩码
        local_masks = self.wavelet_mask.compute_local_mask(lrs)
        flicker_masks = self.wavelet_mask.compute_flicker_mask(lrs)
        wavelet_masks = self._fuse_wavelet_masks(local_masks, flicker_masks)

        # Backward features
        backward_gate_energies = []
        backward_state_energies = []
        feat_prop = lrs.new_zeros(n, self.num_feat, h, w)

        for i in range(t - 1, -1, -1):
            curr_lr = lrs[:, i, :, :, :]
            if i < t - 1:
                flow = backward_flow[:, i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
                pixel_prop = flow_warp(lrs[:, i + 1, :, :, :], flow.permute(0, 2, 3, 1))

                spatial_mask = torch.abs(pixel_prop[:, :1, :, :] - curr_lr[:, :1, :, :])
                wavelet_mask_i = wavelet_masks[:, i, :, :, :]
                fused_indicator = self.mask_fusion(spatial_mask, wavelet_mask_i)

                temp, latent_feat = self.Backward_Aggregation(
                    feat_prop, curr_lr, fused_indicator,
                    head=False, return_mask=True, return_feat=True
                )
            else:
                zero_indicator = lrs.new_zeros(n, 1, h, w)
                temp, latent_feat = self.Backward_Aggregation(
                    feat_prop, curr_lr, zero_indicator,
                    head=True, return_mask=True, return_feat=True
                )

            feat_prop = self.backward_resblocks(temp)

            gate_energy = latent_feat.abs().mean(dim=1, keepdim=True)
            state_energy = feat_prop.abs().mean(dim=1, keepdim=True)
            backward_gate_energies.append(gate_energy)
            backward_state_energies.append(state_energy)

        backward_gate_energies = backward_gate_energies[::-1]
        backward_state_energies = backward_state_energies[::-1]

        # Forward features
        forward_gate_energies = []
        forward_state_energies = []
        feat_prop = lrs.new_zeros(n, self.num_feat, h, w)

        for i in range(t):
            curr_lr = lrs[:, i, :, :, :]
            if i > 0:
                flow = forward_flow[:, i - 1, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
                pixel_prop = flow_warp(lrs[:, i - 1, :, :, :], flow.permute(0, 2, 3, 1))

                spatial_mask = torch.abs(pixel_prop[:, :1, :, :] - curr_lr[:, :1, :, :])
                wavelet_mask_i = wavelet_masks[:, i, :, :, :]
                fused_indicator = self.mask_fusion(spatial_mask, wavelet_mask_i)

                temp, latent_feat = self.Forward_Aggregation(
                    feat_prop, curr_lr, fused_indicator,
                    head=False, return_mask=True, return_feat=True
                )
            else:
                zero_indicator = lrs.new_zeros(n, 1, h, w)
                temp, latent_feat = self.Forward_Aggregation(
                    feat_prop, curr_lr, zero_indicator,
                    head=True, return_mask=True, return_feat=True
                )

            feat_prop = self.forward_resblocks(temp)

            gate_energy = latent_feat.abs().mean(dim=1, keepdim=True)
            state_energy = feat_prop.abs().mean(dim=1, keepdim=True)
            forward_gate_energies.append(gate_energy)
            forward_state_energies.append(state_energy)

        return (
            torch.stack(backward_gate_energies, dim=1),
            torch.stack(backward_state_energies, dim=1),
            torch.stack(forward_gate_energies, dim=1),
            torch.stack(forward_state_energies, dim=1)
        )
