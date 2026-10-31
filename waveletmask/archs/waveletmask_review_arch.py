"""WaveletMask Round-1 review ablation architecture."""

from __future__ import annotations

import torch

from basicsr.utils.registry import ARCH_REGISTRY

from basicofr.archs.components.arch_util import flow_warp

from .components.wavelet_mask.review_wavelet_mask import ConfigurableWaveletNoiseMask
from .waveletmask_base_arch import Video_Backbone


@ARCH_REGISTRY.register()
class WaveletMaskReviewNet(Video_Backbone):
    """RTN backbone with configurable WaveletMask review switches."""

    def __init__(
        self,
        num_feat=16,
        num_block=6,
        flow_type='raft',
        wavelet_backend='haar',
        wavelet_levels=1,
        fusion_type='max',
        fusion_mid_channels=16,
        flow_compensated_wavelet=False,
        **kwargs,
    ):
        super().__init__(num_feat=num_feat, num_block=num_block, flow_type=flow_type, **kwargs)
        self.flow_compensated_wavelet = bool(flow_compensated_wavelet)
        self.wavelet_mask = ConfigurableWaveletNoiseMask(
            wavelet=wavelet_backend,
            levels=wavelet_levels,
            fusion_type=fusion_type,
            fusion_mid_channels=fusion_mid_channels,
            flow_compensated=self.flow_compensated_wavelet,
        )

    def forward(self, lrs):
        n, t, _, h, w = lrs.size()
        assert h >= 64 and w >= 64, (
            'The height and width of input should be at least 64, '
            f'but got {h} and {w}.'
        )

        forward_flow, backward_flow = self.comp_flow(lrs)
        wavelet_masks = self.wavelet_mask(
            lrs,
            forward_flow=forward_flow,
            backward_flow=backward_flow,
        )

        cached_feats = []
        feat_prop = lrs.new_zeros(n, self.num_feat, h, w)
        for i in range(t - 1, -1, -1):
            curr_lr = lrs[:, i]
            if i < t - 1:
                flow = backward_flow[:, i]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
                aggregated = self.Backward_Aggregation(feat_prop, curr_lr, wavelet_masks[:, i], head=False)
            else:
                aggregated = self.Backward_Aggregation(feat_prop, curr_lr, wavelet_masks[:, i], head=True)
            feat_prop = self.backward_resblocks(aggregated)
            cached_feats.append(feat_prop)

        cached_feats = cached_feats[::-1]
        feat_prop = torch.zeros_like(feat_prop)
        outputs = []
        for i in range(t):
            curr_lr = lrs[:, i]
            if i > 0:
                flow = forward_flow[:, i - 1]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
                aggregated = self.Forward_Aggregation(feat_prop, curr_lr, wavelet_masks[:, i], head=False)
            else:
                aggregated = self.Forward_Aggregation(feat_prop, curr_lr, wavelet_masks[:, i], head=True)
            feat_prop = self.forward_resblocks(aggregated)

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
        n, t, _, h, w = lrs.size()
        forward_flow, backward_flow = self.comp_flow(lrs)

        spatial_masks = []
        for i in range(t):
            if i > 0:
                flow = forward_flow[:, i - 1]
                pixel_prop = flow_warp(lrs[:, i - 1], flow.permute(0, 2, 3, 1))
                spatial_mask = torch.abs(pixel_prop[:, :1] - lrs[:, i, :1])
            else:
                spatial_mask = lrs.new_zeros(n, 1, h, w)
            spatial_masks.append(spatial_mask)

        local, flicker, fused = self.wavelet_mask.compute_masks(
            lrs,
            forward_flow=forward_flow,
            backward_flow=backward_flow,
        )
        return {
            'spatial': torch.stack(spatial_masks, dim=1),
            'local': local,
            'flicker': flicker,
            'fused': fused,
        }
