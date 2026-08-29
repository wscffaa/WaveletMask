"""Stronger local/flicker mask fusion alternatives for Round-1 review."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _flatten_time(mask_a: torch.Tensor, mask_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...] | None]:
    if mask_a.shape != mask_b.shape:
        raise ValueError(f'Mask shape mismatch: {tuple(mask_a.shape)} vs {tuple(mask_b.shape)}.')
    if mask_a.dim() == 4:
        return mask_a, mask_b, None
    if mask_a.dim() != 5:
        raise ValueError(f'Expected 4D/5D masks, got {mask_a.dim()}D.')
    b, t, c, h, w = mask_a.shape
    return mask_a.reshape(b * t, c, h, w), mask_b.reshape(b * t, c, h, w), (b, t, c, h, w)


def _restore_time(mask: torch.Tensor, shape: tuple[int, ...] | None) -> torch.Tensor:
    if shape is None:
        return mask
    return mask.reshape(shape)


def _blend(local: torch.Tensor, flicker: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    return torch.clamp(alpha * local + (1.0 - alpha) * flicker, 0.0, 1.0)


def _pair_stats(local: torch.Tensor, flicker: torch.Tensor) -> torch.Tensor:
    delta = local - flicker
    return torch.cat([
        F.adaptive_avg_pool2d(local, 1),
        F.adaptive_avg_pool2d(flicker, 1),
        F.adaptive_avg_pool2d(torch.abs(delta), 1),
        F.adaptive_max_pool2d(local, 1),
        F.adaptive_max_pool2d(flicker, 1),
    ], dim=1)


class FixedBiasFusion(nn.Module):
    """A constant local-vs-flicker gate used for bias-sweep review ablations."""

    def __init__(self, bias: float):
        super().__init__()
        self.register_buffer('logit_bias', torch.tensor(float(bias), dtype=torch.float32))

    def forward(self, local_mask: torch.Tensor, flicker_mask: torch.Tensor) -> torch.Tensor:
        local, flicker, original_shape = _flatten_time(local_mask, flicker_mask)
        alpha = torch.sigmoid(self.logit_bias).to(device=local.device, dtype=local.dtype).view(1, 1, 1, 1)
        fused = _blend(local, flicker, alpha)
        return _restore_time(fused, original_shape)


class ScalarGateFusion(nn.Module):
    """A tiny review-only gate that predicts one fusion weight per frame."""

    def __init__(self, mid_channels: int = 8, init_bias: float = 0.0):
        super().__init__()
        hidden = max(4, int(mid_channels))
        self.gate = nn.Sequential(
            nn.Conv2d(5, hidden, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden, 1, 1),
            nn.Sigmoid(),
        )
        self._init_weights(init_bias)

    def _init_weights(self, init_bias: float) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.zeros_(self.gate[-2].weight)
        self.gate[-2].bias.data.fill_(float(init_bias))

    def _gate_inputs(self, local: torch.Tensor, flicker: torch.Tensor) -> torch.Tensor:
        return _pair_stats(local, flicker)

    def forward(self, local_mask: torch.Tensor, flicker_mask: torch.Tensor) -> torch.Tensor:
        local, flicker, original_shape = _flatten_time(local_mask, flicker_mask)
        alpha = self.gate(self._gate_inputs(local, flicker))
        fused = _blend(local, flicker, alpha)
        return _restore_time(fused, original_shape)


class NormalizationAwareFusion(ScalarGateFusion):
    """Frame-wise normalized scalar gate for scale-sensitive reviewer checks."""

    @staticmethod
    def _normalize(mask: torch.Tensor) -> torch.Tensor:
        mean = mask.mean(dim=(-2, -1), keepdim=True)
        std = mask.std(dim=(-2, -1), keepdim=True, unbiased=False).clamp_min(1e-6)
        return (mask - mean) / std

    def _gate_inputs(self, local: torch.Tensor, flicker: torch.Tensor) -> torch.Tensor:
        return _pair_stats(self._normalize(local), self._normalize(flicker))


class ChannelAttentionFusion(nn.Module):
    """Fuse LHFM and GFM masks with per-frame channel attention."""

    def __init__(self, mid_channels: int = 8):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(4, mid_channels, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_channels, 2, 1),
        )

    def forward(self, local_mask: torch.Tensor, flicker_mask: torch.Tensor) -> torch.Tensor:
        local, flicker, original_shape = _flatten_time(local_mask, flicker_mask)
        pair = torch.cat([local, flicker], dim=1)
        stats = torch.cat([
            F.adaptive_avg_pool2d(pair, 1),
            F.adaptive_max_pool2d(pair, 1),
        ], dim=1)
        weights = torch.softmax(self.attention(stats), dim=1)
        fused = _blend(local, flicker, weights[:, 0:1])
        return _restore_time(fused, original_shape)


class ResidualFusion(nn.Module):
    """Learn a bounded residual around the max-fusion baseline."""

    def __init__(self, mid_channels: int = 16, residual_scale: float = 0.1):
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.residual = nn.Sequential(
            nn.Conv2d(4, mid_channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_channels, 1, 1),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, local_mask: torch.Tensor, flicker_mask: torch.Tensor) -> torch.Tensor:
        local, flicker, original_shape = _flatten_time(local_mask, flicker_mask)
        base = torch.max(local, flicker)
        features = torch.cat([local, flicker, base, torch.abs(local - flicker)], dim=1)
        fused = torch.clamp(base + self.residual_scale * torch.tanh(self.residual(features)), 0.0, 1.0)
        return _restore_time(fused, original_shape)


def build_review_fusion(fusion_type: str, mid_channels: int = 16) -> nn.Module | None:
    if fusion_type == 'max':
        return None
    if fusion_type == 'bias0':
        return FixedBiasFusion(0.0)
    if fusion_type == 'bias2':
        return FixedBiasFusion(2.0)
    if fusion_type == 'scalar_gate':
        return ScalarGateFusion(mid_channels=mid_channels, init_bias=0.0)
    if fusion_type == 'normalization_aware':
        return NormalizationAwareFusion(mid_channels=mid_channels, init_bias=0.0)
    if fusion_type == 'channel_attention':
        return ChannelAttentionFusion(mid_channels=max(4, mid_channels))
    if fusion_type == 'residual':
        return ResidualFusion(mid_channels=mid_channels)
    raise ValueError(f'Unsupported review fusion type: {fusion_type}.')
