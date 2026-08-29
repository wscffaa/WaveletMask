"""
WaveletMask Flicker Architecture (Innovation B only).

消融实验变体：仅使用闪烁掩码，不使用融合器
用于独立验证创新 B（全局闪烁掩码）的有效性

集成创新 B: 全局闪烁掩码（LL 子带差异）
不使用 C: 删除像素级 Alpha CNN 融合（已证明有害）

Architecture: RTN + Flicker-only Mask
Innovations: B (Flicker mask only)
Ablation: Validates Innovation B independently

Integration Date: 2025-12-25
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


class PSUpsample(nn.Module):
    """Pixel-shuffle upsample (scale=1 is a conv refresh)."""

    def __init__(self, in_feat, out_feat, scale_factor):
        super(PSUpsample, self).__init__()
        self.scale_factor = scale_factor
        self.up_conv = nn.Conv2d(in_feat, out_feat * scale_factor * scale_factor, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.up_conv(x)
        return F.pixel_shuffle(x, upscale_factor=self.scale_factor)


class FlickerOnlyMask(nn.Module):
    """Flicker-only mask generator (Innovation B only).

    仅计算低频 LL 子带差异作为闪烁掩码。
    用于消融实验，独立验证创新 B 的有效性。

    Args:
        upsample_mode: Upsampling mode. Default: 'bilinear'.
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
                    # 边界帧无法计算前后帧差异，使用零掩码
                    masks.append(torch.zeros(b, 1, h, w, device=device, dtype=dtype))
                    continue

                frame_t = lrs_gray[:, idx]
                frame_prev = lrs_gray[:, idx - 1]
                frame_next = lrs_gray[:, idx + 1]

                # 【创新 B】计算闪烁掩码（仅使用低频LL子带）
                # LL子带差异最小值：两个方向都有变化才认为是闪烁
                ll_t = self._haar_ll(frame_t)
                ll_prev = self._haar_ll(frame_prev)
                ll_next = self._haar_ll(frame_next)

                ll_diff_prev = torch.abs(ll_t - ll_prev)
                ll_diff_next = torch.abs(ll_t - ll_next)

                # 取最小值：两个方向都有差异才认为是闪烁
                flicker_mask = torch.min(ll_diff_prev, ll_diff_next)
                flicker_mask = self._upsample(flicker_mask, size=(h, w))

                mask = flicker_mask.clamp(0.0, 1.0)
                masks.append(mask.to(dtype))

        return torch.stack(masks, dim=1)


@ARCH_REGISTRY.register()
class WaveletMaskFlickerNet(nn.Module):
    """WaveletMask Flicker architecture - Innovation B only (no fusion).

    消融变体：仅使用闪烁掩码（创新 B），不使用融合器
    用于独立验证创新 B 的有效性，判断B是否值得保留

    Design notes:
    - 【创新 B】全局闪烁掩码：基于低频LL子带最小差分
    - 【移除 C】删除Alpha CNN融合器（已证明-1.77% PSNR有害）
    - 直接使用闪烁掩码作为 residual_indicator

    如果 FlickerNet 的性能不错（≥ +0.3dB），则保留创新 B
    如果 FlickerNet 的性能微弱（< +0.1dB），则删除创新 B，仅发表创新 A

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

        # 【创新 B】仅闪烁掩码模块（消融：不使用融合器）
        self.flicker_mask = FlickerOnlyMask(upsample_mode='bilinear')

    def comp_flow(self, lrs):
        """Estimate bidirectional flow and return tensors."""
        return self.flow_estimator.compute_flow(lrs)

    def forward(self, lrs):
        """Bidirectional propagation with flicker mask only (Innovation B).

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

        # 【创新 B】预计算闪烁掩码 (B, T, 1, H, W)
        flicker_masks = self.flicker_mask(lrs)

        # ===== Stage 1: backward propagation (no upsample) =====
        cached_feats = []
        feat_prop = lrs.new_zeros(n, self.num_feat, h, w)
        for i in range(t - 1, -1, -1):
            curr_lr = lrs[:, i, :, :, :]
            if i < t - 1:
                flow = backward_flow[:, i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
                # 【创新 B】直接使用闪烁掩码作为residual_indicator
                aggregated = self.Backward_Aggregation(feat_prop, curr_lr, flicker_masks[:, i], head=False)
            else:
                aggregated = self.Backward_Aggregation(feat_prop, curr_lr, flicker_masks[:, i], head=True)

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
                # 【创新 B】直接使用闪烁掩码作为residual_indicator
                aggregated = self.Forward_Aggregation(feat_prop, curr_lr, flicker_masks[:, i], head=False)
            else:
                aggregated = self.Forward_Aggregation(feat_prop, curr_lr, flicker_masks[:, i], head=True)

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
