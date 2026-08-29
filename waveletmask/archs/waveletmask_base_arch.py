"""
WaveletMask Base Architecture (Baseline RTN).

This is the unmodified RTN architecture used as baseline for comparison.
No wavelet mask innovations are integrated in this variant.

Architecture: RTN + Swin Spatial
Innovations: None (baseline)

Integration Date: 2025-12-23
"""

import torch
import torch.nn.functional as F
from torch import nn

from basicsr.utils.registry import ARCH_REGISTRY

from basicofr.archs.flow import FlowEstimator
from basicofr.archs.components.arch_util import ResidualBlockNoBN, flow_warp, make_layer
from basicofr.archs.spatial.swin_spatial import Swin_Spatial_2


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


class Video_Backbone(nn.Module):
    """RTN-Swin backbone.

    Design notes:
    1. Bidirectional propagation after flow alignment; gated aggregation updates hidden state.
    2. Each direction uses Swin Transformer as spatial restoration module.
    3. Concatenate bidirectional features, then pixel-shuffle tail (scale=1) + global residual.
    """

    def __init__(self, num_feat=16, num_block=6, flow_type='raft', use_checkpoint=False, **kwargs):
        super(Video_Backbone, self).__init__()
        self.num_feat = num_feat
        self.num_block = num_block

        # Flow alignment (raft/spynet/memflow)
        self.flow_estimator = FlowEstimator(estimator_type=flow_type, normalization='tanh')

        # Bidirectional Swin spatial modules
        # Swin_Spatial_2(nearest+conv) 的输出通道为 embed_dim/4，为确保与 num_feat 对齐：
        #   embed_dim = 4 * num_feat  → out_channels = num_feat
        embed_dim = max(4, int(num_feat) * 4)
        self.forward_resblocks = Swin_Spatial_2(
            embed_dim=embed_dim,
            depths=[2, 2, 2],
            num_heads=[4, 4, 4],
            mlp_ratio=2,
            in_chans=num_feat,
            use_checkpoint=use_checkpoint,
        )
        self.backward_resblocks = Swin_Spatial_2(
            embed_dim=embed_dim,
            depths=[2, 2, 2],
            num_heads=[4, 4, 4],
            mlp_ratio=2,
            in_chans=num_feat,
            use_checkpoint=use_checkpoint,
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

    def comp_flow(self, lrs):
        """Estimate bidirectional flow and return tensors."""
        return self.flow_estimator.compute_flow(lrs)

    def forward(self, lrs):
        """Bidirectional propagation for Swin RTN."""
        n, t, c, h, w = lrs.size()

        assert h >= 64 and w >= 64, (
            'The height and width of input should be at least 64, '
            f'but got {h} and {w}.'
        )

        forward_flow, backward_flow = self.comp_flow(lrs)

        # ===== Stage 1: backward propagation (no upsample) =====
        cached_feats = []
        feat_prop = lrs.new_zeros(n, self.num_feat, h, w)
        residual_indicator = lrs.new_zeros(n, 1, h, w)
        for i in range(t - 1, -1, -1):
            curr_lr = lrs[:, i, :, :, :]
            if i < t - 1:
                flow = backward_flow[:, i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
                pixel_prop = flow_warp(lrs[:, i + 1, :, :, :], flow.permute(0, 2, 3, 1))
                residual_indicator = torch.abs(pixel_prop[:, :1, :, :] - curr_lr[:, :1, :, :])
                aggregated = self.Backward_Aggregation(feat_prop, curr_lr, residual_indicator, head=False)
            else:
                aggregated = self.Backward_Aggregation(feat_prop, curr_lr, residual_indicator, head=True)

            feat_prop = self.backward_resblocks(aggregated)
            cached_feats.append(feat_prop)

        cached_feats = cached_feats[::-1]

        # ===== Stage 2: forward propagation (no upsample) =====
        feat_prop = torch.zeros_like(feat_prop)
        residual_indicator = torch.zeros_like(residual_indicator)
        outputs = []
        for i in range(t):
            curr_lr = lrs[:, i, :, :, :]
            if i > 0:
                flow = forward_flow[:, i - 1, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
                pixel_prop = flow_warp(lrs[:, i - 1, :, :, :], flow.permute(0, 2, 3, 1))
                residual_indicator = torch.abs(pixel_prop[:, :1, :, :] - curr_lr[:, :1, :, :])
                aggregated = self.Forward_Aggregation(feat_prop, curr_lr, residual_indicator, head=False)
            else:
                aggregated = self.Forward_Aggregation(feat_prop, curr_lr, residual_indicator, head=True)

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
            }
        """
        n, t, c, h, w = lrs.size()
        forward_flow, backward_flow = self.comp_flow(lrs)

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

        return {'spatial': torch.stack(spatial_masks, dim=1)}

    def visualiza_feature(self, lrs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Visualize feature energy for RTN diagnostics.

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

        # ===== Backward features =====
        backward_gate_energies = []
        backward_state_energies = []
        feat_prop = lrs.new_zeros(n, self.num_feat, h, w)
        residual_indicator = lrs.new_zeros(n, 1, h, w)

        for i in range(t - 1, -1, -1):
            curr_lr = lrs[:, i, :, :, :]
            if i < t - 1:
                flow = backward_flow[:, i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
                pixel_prop = flow_warp(lrs[:, i + 1, :, :, :], flow.permute(0, 2, 3, 1))
                residual_indicator = torch.abs(pixel_prop[:, :1, :, :] - curr_lr[:, :1, :, :])
                temp, latent_feat = self.Backward_Aggregation(
                    feat_prop, curr_lr, residual_indicator,
                    head=False, return_mask=True, return_feat=True
                )
            else:
                temp, latent_feat = self.Backward_Aggregation(
                    feat_prop, curr_lr, residual_indicator,
                    head=True, return_mask=True, return_feat=True
                )

            feat_prop = self.backward_resblocks(temp)

            # Use abs() to avoid sign cancellation
            gate_energy = latent_feat.abs().mean(dim=1, keepdim=True)
            state_energy = feat_prop.abs().mean(dim=1, keepdim=True)
            backward_gate_energies.append(gate_energy)
            backward_state_energies.append(state_energy)

        backward_gate_energies = backward_gate_energies[::-1]
        backward_state_energies = backward_state_energies[::-1]

        # ===== Forward features =====
        forward_gate_energies = []
        forward_state_energies = []
        feat_prop = lrs.new_zeros(n, self.num_feat, h, w)
        residual_indicator = lrs.new_zeros(n, 1, h, w)

        for i in range(t):
            curr_lr = lrs[:, i, :, :, :]
            if i > 0:
                flow = forward_flow[:, i - 1, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
                pixel_prop = flow_warp(lrs[:, i - 1, :, :, :], flow.permute(0, 2, 3, 1))
                residual_indicator = torch.abs(pixel_prop[:, :1, :, :] - curr_lr[:, :1, :, :])
                temp, latent_feat = self.Forward_Aggregation(
                    feat_prop, curr_lr, residual_indicator,
                    head=False, return_mask=True, return_feat=True
                )
            else:
                temp, latent_feat = self.Forward_Aggregation(
                    feat_prop, curr_lr, residual_indicator,
                    head=True, return_mask=True, return_feat=True
                )

            feat_prop = self.forward_resblocks(temp)

            gate_energy = latent_feat.abs().mean(dim=1, keepdim=True)
            state_energy = feat_prop.abs().mean(dim=1, keepdim=True)
            forward_gate_energies.append(gate_energy)
            forward_state_energies.append(state_energy)

        backward_gate_energy = torch.stack(backward_gate_energies, dim=1)
        backward_state_energy = torch.stack(backward_state_energies, dim=1)
        forward_gate_energy = torch.stack(forward_gate_energies, dim=1)
        forward_state_energy = torch.stack(forward_state_energies, dim=1)

        return backward_gate_energy, backward_state_energy, forward_gate_energy, forward_state_energy


class ConvResBlock(nn.Module):
    """Conv + ResBlock combo (legacy compatibility)."""

    def __init__(self, in_feat, out_feat=64, num_block=30):
        super(ConvResBlock, self).__init__()
        conv_resblock = [
            nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            make_layer(ResidualBlockNoBN, num_block, num_feat=out_feat),
        ]
        self.conv_resblock = nn.Sequential(*conv_resblock)

    def forward(self, x):
        return self.conv_resblock(x)


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
class WaveletMaskBaseNet(Video_Backbone):
    """WaveletMask Base architecture - unmodified RTN for baseline comparison.

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
        super().__init__(num_feat=num_feat, num_block=num_block, flow_type=flow_type, **kwargs)
