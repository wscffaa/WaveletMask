"""
WaveletMask Flicker Fusion Architecture (Innovation B+C) - Ablation Variant.

消融实验变体：仅使用闪烁掩码 + 融合，不使用局部高频掩码
用于验证创新 A（局部噪声掩码）的必要性

集成创新 B: 全局闪烁掩码（LL 子带差异）
集成创新 C: 像素级 alpha 融合（空间掩码 + 闪烁掩码）

Architecture: RTN + WaveletNoiseMask (flicker only) + PixelWiseMaskFusion
Innovations: B (Flicker mask) + C (Pixel-wise fusion)
Ablation: Validates necessity of Innovation A

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


class FlickerOnlyMask(nn.Module):
    """Flicker-only mask generator (Innovation B only).

    仅计算低频 LL 子带差异作为闪烁掩码，不使用高频掩码。
    用于消融实验，验证创新 A 的必要性。
    """

    def __init__(self, upsample_mode='bilinear'):
        super().__init__()
        self.upsample_mode = upsample_mode

        # Haar LL kernel
        kernel = torch.tensor([[1, 1], [1, 1]], dtype=torch.float32) / 2.0
        self.register_buffer("haar_ll", kernel.view(1, 1, 2, 2))

    def _haar_ll(self, x):
        """Extract LL subband only."""
        if x.dim() == 3:
            x = x.unsqueeze(1)
        device_type = "cuda" if x.is_cuda else "cpu"
        with torch.amp.autocast(device_type=device_type, enabled=False):
            x_float = x.float()
            ll = F.conv2d(x_float, self.haar_ll, stride=2)
        return ll

    def _upsample(self, x, size):
        if self.upsample_mode in {"bilinear", "bicubic", "trilinear"}:
            return F.interpolate(x, size=size, mode=self.upsample_mode, align_corners=False)
        return F.interpolate(x, size=size, mode=self.upsample_mode)

    def compute_flicker_mask(self, frame_t, frame_t1):
        """Compute flicker mask from LL subband (Innovation B)."""
        _, _, h, w = frame_t.shape
        ll_t = self._haar_ll(frame_t)
        ll_t1 = self._haar_ll(frame_t1)
        diff = torch.abs(ll_t - ll_t1)
        return self._upsample(diff, size=(h, w))

    def forward(self, lrs):
        """Generate flicker-only masks for video sequence.

        Args:
            lrs (Tensor): Input video (B, T, C, H, W)

        Returns:
            masks (Tensor): Flicker masks (B, T, 1, H, W)
        """
        if lrs.dim() != 5:
            raise ValueError(f"Expected 5D input (B,T,C,H,W), got {lrs.dim()}D.")
        b, t, c, h, w = lrs.shape
        device = lrs.device
        dtype = lrs.dtype

        device_type = "cuda" if lrs.is_cuda else "cpu"
        masks = []

        with torch.amp.autocast(device_type=device_type, enabled=False):
            lrs_float = lrs.float()
            if c == 3:
                lrs_gray = (
                    0.299 * lrs_float[:, :, 0:1]
                    + 0.587 * lrs_float[:, :, 1:2]
                    + 0.114 * lrs_float[:, :, 2:3]
                )
            elif c == 1:
                lrs_gray = lrs_float[:, :, 0:1]
            else:
                lrs_gray = lrs_float.mean(dim=2, keepdim=True)

            for idx in range(t):
                if idx == 0 or idx == t - 1:
                    masks.append(torch.zeros(b, 1, h, w, device=device, dtype=dtype))
                    continue

                frame_t = lrs_gray[:, idx]
                frame_prev = lrs_gray[:, idx - 1]
                frame_next = lrs_gray[:, idx + 1]

                # 【创新 B】仅计算闪烁掩码
                flicker_prev = self.compute_flicker_mask(frame_t, frame_prev)
                flicker_next = self.compute_flicker_mask(frame_t, frame_next)
                flicker_mask = 0.5 * (flicker_prev + flicker_next)

                mask = flicker_mask.clamp(0.0, 1.0)
                masks.append(mask.to(dtype))

        return torch.stack(masks, dim=1)


@ARCH_REGISTRY.register()
class WaveletMaskFlickerFusionNet(Video_Backbone):
    """WaveletMask Flicker Fusion architecture - Ablation variant (Innovation B+C only).

    消融变体：仅使用闪烁掩码（创新 B）+ 融合（创新 C），不使用局部高频掩码（创新 A）
    用于验证创新 A 对划痕/灰尘检测的必要性。

    如果 Full > FlickerFusion，则证明创新 A 有贡献。

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

        # 【创新 B】仅闪烁掩码模块（消融：不使用创新 A）
        self.flicker_mask = FlickerOnlyMask(upsample_mode='bilinear')

        # 【创新 C】像素级掩码融合模块
        self.mask_fusion = PixelWiseMaskFusion(mid_channels=fusion_mid_channels)

    def forward(self, lrs):
        """Bidirectional propagation with flicker mask only (Innovation B+C, ablation).

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

        # 【创新 B】仅计算闪烁掩码（消融：不使用创新 A 的局部高频掩码）
        flicker_masks = self.flicker_mask(lrs)

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

                # 【创新 C 集成点】融合空间掩码与闪烁掩码（仅 B，无 A）
                flicker_mask_i = flicker_masks[:, i, :, :, :]
                fused_indicator = self.mask_fusion(spatial_mask, flicker_mask_i)

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

                # 【创新 C 集成点】融合空间掩码与闪烁掩码（仅 B，无 A）
                flicker_mask_i = flicker_masks[:, i, :, :, :]
                fused_indicator = self.mask_fusion(spatial_mask, flicker_mask_i)

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
                'flicker': 闪烁掩码 (B, T, 1, H, W) - 创新B
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

        flicker_masks = self.flicker_mask(lrs)

        return {
            'spatial': torch.stack(spatial_masks, dim=1),
            'flicker': flicker_masks,
        }
