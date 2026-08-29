"""Configurable WaveletMask branch for Round-1 review ablations."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from basicofr.archs.components.arch_util import flow_warp

from .generic_dwt import StaticDWT2d
from .review_fusion import build_review_fusion


class ConfigurableWaveletNoiseMask(nn.Module):
    """WaveletMask local/flicker branch with backend, fusion, and FC-DWT switches."""

    def __init__(
        self,
        high_freq_channels: int = 16,
        low_freq_channels: int = 16,
        wavelet: str = 'haar',
        levels: int = 1,
        fusion_type: str = 'max',
        fusion_mid_channels: int = 16,
        flow_compensated: bool = False,
    ):
        super().__init__()
        self.wavelet = wavelet
        self.levels = int(levels)
        self.fusion_type = fusion_type
        self.flow_compensated = bool(flow_compensated)
        self.dwt = StaticDWT2d(wavelet=wavelet, levels=levels)
        self.mask_fusion = build_review_fusion(fusion_type, mid_channels=fusion_mid_channels)

        self.high_freq_conv = nn.Sequential(
            nn.Conv2d(3, high_freq_channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(high_freq_channels, 1, 3, 1, 1),
            nn.Sigmoid(),
        )
        self.low_freq_conv = nn.Sequential(
            nn.Conv2d(1, low_freq_channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(low_freq_channels, 1, 3, 1, 1),
            nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        self.high_freq_conv[-2].bias.data.fill_(-5.0)
        self.low_freq_conv[-2].bias.data.fill_(-5.0)

    @staticmethod
    def _no_autocast(tensor: torch.Tensor):
        autocast = getattr(torch, 'autocast', None)
        if autocast is not None:
            device_type = 'cuda' if tensor.is_cuda else 'cpu'
            return autocast(device_type=device_type, enabled=False)
        if tensor.is_cuda:
            return torch.cuda.amp.autocast(enabled=False)
        return nullcontext()

    @staticmethod
    def _to_gray(frames: torch.Tensor) -> torch.Tensor:
        if frames.size(1) == 3:
            return 0.299 * frames[:, 0] + 0.587 * frames[:, 1] + 0.114 * frames[:, 2]
        if frames.size(1) == 1:
            return frames[:, 0]
        return frames.mean(dim=1)

    @staticmethod
    def _compute_high_freq_diff(
        curr: Dict[str, torch.Tensor],
        prev: Dict[str, torch.Tensor],
        next_: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        eps = 1e-8
        return torch.cat([
            torch.sqrt(torch.abs(curr['LH'] - prev['LH']) * torch.abs(curr['LH'] - next_['LH']) + eps),
            torch.sqrt(torch.abs(curr['HL'] - prev['HL']) * torch.abs(curr['HL'] - next_['HL']) + eps),
            torch.sqrt(torch.abs(curr['HH'] - prev['HH']) * torch.abs(curr['HH'] - next_['HH']) + eps),
        ], dim=1)

    @staticmethod
    def _compute_low_freq_diff(
        curr: Dict[str, torch.Tensor],
        prev: Dict[str, torch.Tensor],
        next_: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return torch.min(torch.abs(curr['LL'] - prev['LL']), torch.abs(curr['LL'] - next_['LL']))

    def _neighbor_frames(
        self,
        lqs: torch.Tensor,
        frame_idx: int,
        forward_flow: torch.Tensor | None,
        backward_flow: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.flow_compensated:
            return lqs[:, frame_idx - 1], lqs[:, frame_idx + 1]
        if forward_flow is None or backward_flow is None:
            raise ValueError('flow_compensated=True requires forward_flow and backward_flow.')
        prev_frame = flow_warp(lqs[:, frame_idx - 1], forward_flow[:, frame_idx - 1].permute(0, 2, 3, 1))
        next_frame = flow_warp(lqs[:, frame_idx + 1], backward_flow[:, frame_idx].permute(0, 2, 3, 1))
        return prev_frame, next_frame

    def compute_masks(
        self,
        lqs: torch.Tensor,
        forward_flow: torch.Tensor | None = None,
        backward_flow: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if lqs.dim() != 5:
            raise ValueError(f'Expected [B,T,C,H,W], got {tuple(lqs.shape)}.')

        b, t, _, h, w = lqs.shape
        local_masks = []
        flicker_masks = []

        with self._no_autocast(lqs):
            lqs_float = lqs.float()
            for i in range(t):
                if i == 0 or i == t - 1:
                    zero = torch.zeros(b, 1, h, w, device=lqs.device, dtype=torch.float32)
                    local_masks.append(zero)
                    flicker_masks.append(zero)
                    continue

                curr_gray = self._to_gray(lqs_float[:, i])
                prev_frame, next_frame = self._neighbor_frames(lqs_float, i, forward_flow, backward_flow)
                prev_gray = self._to_gray(prev_frame)
                next_gray = self._to_gray(next_frame)

                curr_coeff = self.dwt(curr_gray)
                prev_coeff = self.dwt(prev_gray)
                next_coeff = self.dwt(next_gray)

                high_diff = self._compute_high_freq_diff(curr_coeff, prev_coeff, next_coeff)
                high_mask = self.high_freq_conv(high_diff)
                high_mask = F.interpolate(high_mask, size=(h, w), mode='bilinear', align_corners=False)
                local_masks.append(high_mask)

                low_diff = self._compute_low_freq_diff(curr_coeff, prev_coeff, next_coeff)
                low_mask = self.low_freq_conv(low_diff)
                low_mask = F.interpolate(low_mask, size=(h, w), mode='bilinear', align_corners=False)
                flicker_masks.append(low_mask)

        local = torch.stack(local_masks, dim=1).to(dtype=lqs.dtype)
        flicker = torch.stack(flicker_masks, dim=1).to(dtype=lqs.dtype)
        if self.mask_fusion is None:
            fused = torch.max(local, flicker)
        else:
            fused = self.mask_fusion(local, flicker)
        return local, flicker, fused.to(dtype=lqs.dtype)

    def forward(
        self,
        lqs: torch.Tensor,
        forward_flow: torch.Tensor | None = None,
        backward_flow: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _, _, fused = self.compute_masks(lqs, forward_flow=forward_flow, backward_flow=backward_flow)
        return fused
