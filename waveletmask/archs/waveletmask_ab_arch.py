"""
WaveletMask A+B Architecture (Innovation A+B, no fusion).

集成创新 A: 局部高频掩码（HH/HL/LH 子带几何平均）
集成创新 B: 全局闪烁掩码（LL 子带差异）
移除创新 C: 删除像素级 alpha 融合（已证明有害 -1.77% PSNR）

使用 WaveletNoiseMask 同时计算 A+B，torch.max() 融合
这只是baseline融合方法，不算创新点

Architecture: RTN + WaveletNoiseMask (full A+B) + max fusion
Innovations: A (Local mask) + B (Flicker mask)
Removed: C (Alpha CNN fusion - harmful)

Integration Date: 2025-12-25
"""

import torch
import torch.nn.functional as F
from torch import nn

from basicsr.utils.registry import ARCH_REGISTRY

from basicofr.archs.flow import FlowEstimator
from basicofr.archs.components.arch_util import ResidualBlockNoBN, flow_warp, make_layer
from basicofr.archs.spatial.swin_spatial import Swin_Spatial_2
from .components.wavelet_mask import WaveletNoiseMask


class Gated_Aggregation(nn.Module):
    """Original RTN gated aggregation: blend hidden state and current frame by residual."""

    def __init__(self, hidden_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True,
                 activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(Gated_Aggregation, self).__init__()

        self.activation = activation
        self.conv2d_projection_head = torch.nn.Conv2d(
            in_channels=3, out_channels=hidden_channels,
            kernel_size=kernel_size, stride=stride, padding=padding,
            dilation=dilation, bias=bias
        )
        self.conv2d_1 = torch.nn.Conv2d(
            in_channels=hidden_channels + hidden_channels + 1,
            out_channels=int(hidden_channels / 2),
            kernel_size=kernel_size, stride=stride, padding=padding,
            dilation=dilation, bias=bias
        )
        self.conv2d_2 = torch.nn.Conv2d(
            in_channels=int(hidden_channels / 2), out_channels=1,
            kernel_size=kernel_size, stride=stride, padding=padding,
            dilation=dilation, bias=bias
        )
        self.sigmoid = torch.nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)

    def forward(self, hidden_state, curr_lr, residual_indicator, head=False, return_mask=False, return_feat=False):
        latent_feature = self.conv2d_projection_head(curr_lr)
        x = self.activation(self.conv2d_1(torch.cat([hidden_state, latent_feature, residual_indicator], dim=1)))
        x = self.sigmoid(self.conv2d_2(x))

        if return_mask:
            if head:
                if return_feat:
                    return latent_feature, latent_feature
                return latent_feature, x
            if return_feat:
                return x * hidden_state + (1 - x) * latent_feature, latent_feature
            return x * hidden_state + (1 - x) * latent_feature, x

        if head:
            return latent_feature
        return x * hidden_state + (1 - x) * latent_feature


class PSUpsample(nn.Module):
    """Pixel-shuffle upsample (scale=1 is a conv refresh)."""

    def __init__(self, in_feat, out_feat, scale_factor):
        super(PSUpsample, self).__init__()
        self.scale_factor = scale_factor
        self.up_conv = nn.Conv2d(in_feat, out_feat * scale_factor * scale_factor, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.up_conv(x)
        return F.pixel_shuffle(x, upscale_factor=self.scale_factor)


@ARCH_REGISTRY.register()
class WaveletMaskABNet(nn.Module):
    """WaveletMask A+B architecture - RTN with complete wavelet mask (A+B only, no fusion).

    完整模型实现（仅创新 A+B）：
    1. 创新 A: 局部高频掩码 - 检测划痕/灰尘等高频噪声
    2. 创新 B: 全局闪烁掩码 - 检测亮度渐变/闪烁

    融合方式: torch.max() 按元素取最大值
    - 这是baseline融合方法，不算创新点
    - 避免了Alpha CNN融合器的有害影响 (-1.77% PSNR)

    WaveletNoiseMask.forward() 同时计算 local + flicker 掩码。
    forward() 中使用 torch.max() 按元素取最大值融合两个掩码。

    Args:
        num_feat (int): Number of feature channels. Default: 16.
        num_block (int): Number of residual blocks. Default: 6.
        flow_type (str): Type of flow estimator. Default: 'raft'.
    """

    def __init__(self,
                 num_feat=16,
                 num_block=6,
                 flow_type='raft',
                 **kwargs):
        super().__init__()
        self.num_feat = num_feat
        self.num_block = num_block

        # Flow alignment (raft/spynet/memflow)
        self.flow_estimator = FlowEstimator(estimator_type=flow_type, normalization='tanh')

        # Bidirectional Swin spatial modules
        embed_dim = max(4, int(num_feat) * 4)
        self.forward_resblocks = Swin_Spatial_2(
            embed_dim=embed_dim,
            depths=[2, 2, 2],
            num_heads=[4, 4, 4],
            mlp_ratio=2,
            in_chans=num_feat,
        )
        self.backward_resblocks = Swin_Spatial_2(
            embed_dim=embed_dim,
            depths=[2, 2, 2],
            num_heads=[4, 4, 4],
            mlp_ratio=2,
            in_chans=num_feat,
        )

        # Gated aggregation (single-scale)
        self.Forward_Aggregation = Gated_Aggregation(hidden_channels=num_feat, kernel_size=3, padding=1)
        self.Backward_Aggregation = Gated_Aggregation(hidden_channels=num_feat, kernel_size=3, padding=1)

        # Feature fusion and tail
        self.concate = nn.Conv2d(num_feat * 2, num_feat, kernel_size=3, stride=1, padding=1, bias=True)
        self.up1 = PSUpsample(num_feat, num_feat, scale_factor=1)
        self.up2 = PSUpsample(num_feat, num_feat, scale_factor=1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, kernel_size=3, stride=1, padding=1)
        self.conv_last = nn.Conv2d(num_feat, 3, kernel_size=3, stride=1, padding=1)

        # Global residual and activation
        self.img_up = nn.Upsample(scale_factor=1, mode='bilinear', align_corners=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # 【创新 A+B】完整小波噪声掩码模块（包含 local + flicker）
        self.wavelet_mask = WaveletNoiseMask()

    def comp_flow(self, lrs):
        """Estimate bidirectional flow and return tensors."""
        return self.flow_estimator.compute_flow(lrs)

    def forward(self, lrs):
        """Bidirectional propagation with A+B wavelet mask (no fusion).

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

        # 【创新 A+B】预计算完整小波掩码 (B, T, 1, H, W)
        # WaveletNoiseMask.forward() 同时计算 local + flicker 并用 max() 融合
        wavelet_masks = self.wavelet_mask(lrs)

        # ===== Stage 1: backward propagation (no upsample) =====
        cached_feats = []
        feat_prop = lrs.new_zeros(n, self.num_feat, h, w)
        for i in range(t - 1, -1, -1):
            curr_lr = lrs[:, i, :, :, :]
            if i < t - 1:
                flow = backward_flow[:, i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
                # 【创新 A+B】直接使用小波掩码作为residual_indicator
                # 无额外融合器，避免Alpha CNN的有害影响
                aggregated = self.Backward_Aggregation(feat_prop, curr_lr, wavelet_masks[:, i], head=False)
            else:
                aggregated = self.Backward_Aggregation(feat_prop, curr_lr, wavelet_masks[:, i], head=True)

            feat_prop = self.backward_resblocks(aggregated)
            cached_feats.append(feat_prop)

        cached_feats = cached_feats[::-1]

        # ===== Stage 2: forward propagation (no upsample) =====
        feat_prop = torch.zeros_like(feat_prop)
        outputs = []
        for i in range(t):
            curr_lr = lrs[:, i, :, :, :]
            if i > 0:
                flow = forward_flow[:, i - 1, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
                # 【创新 A+B】直接使用小波掩码作为residual_indicator
                aggregated = self.Forward_Aggregation(feat_prop, curr_lr, wavelet_masks[:, i], head=False)
            else:
                aggregated = self.Forward_Aggregation(feat_prop, curr_lr, wavelet_masks[:, i], head=True)

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
