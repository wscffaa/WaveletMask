import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm


@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize convolution and linear layers with Kaiming normal weights."""
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


def make_layer(basic_block, num_basic_block, **kwargs):
    """Stack identical blocks and return ``nn.Sequential``."""
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwargs))
    return nn.Sequential(*layers)


class ResidualBlockNoBN(nn.Module):
    """Residual block with two conv layers and ReLU, no batch norm."""

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


class Upsample(nn.Sequential):
    """Sub-pixel upsample module supporting scale 2^n or 3."""

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'Unsupported scale {scale}.')
        super().__init__(*m)


def flow_warp(x,
              flow,
              interp_mode='bilinear',
              padding_mode='zeros',
              align_corners=True):
    """Warp ``x`` (N,C,H,W) according to optical flow ``flow`` (N,H,W,2)."""
    assert x.size()[-2:] == flow.size()[1:3]
    _, _, h, w = x.size()
    grid_y, grid_x = torch.meshgrid(
        torch.arange(0, h, device=x.device, dtype=x.dtype),
        torch.arange(0, w, device=x.device, dtype=x.dtype),
        indexing='ij')
    grid = torch.stack((grid_x, grid_y), 2)
    vgrid = grid + flow
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    return F.grid_sample(
        x,
        vgrid_scaled,
        mode=interp_mode,
        padding_mode=padding_mode,
        align_corners=align_corners)


class ColorTemporalNormalizer(nn.Module):
    """平滑帧间亮度，并提供光度一致性评分帮助光流加权."""

    def __init__(self, kernel_size=5, decay=0.1, eps=1e-6):
        super().__init__()
        if kernel_size < 1:
            raise ValueError('kernel_size must be positive.')
        self.kernel_size = kernel_size
        self.decay = decay
        self.eps = eps

    def forward(self, frames: torch.Tensor):
        """返回校正后的序列与光度评分.

        Args:
            frames (Tensor): ``(B, T, C, H, W)`` 的视频片段.

        Returns:
            Tuple[Tensor, Tensor]: ``(corrected_frames, luminance_score)``，
            其中 ``luminance_score`` 为 ``(B, T, 1, 1, 1)``.
        """
        b, t, c, h, w = frames.shape
        frame_mean = frames.mean(dim=(3, 4), keepdim=True)
        smoothed_mean = self._temporal_smooth(frame_mean)
        offset = smoothed_mean - frame_mean
        corrected = frames + offset

        mean_offset = offset.abs().mean(dim=2, keepdim=True)
        score = torch.exp(-mean_offset / (self.decay + self.eps)).clamp_(0.0, 1.0)
        return corrected, score

    def _temporal_smooth(self, tensor: torch.Tensor):
        """对 ``(B, T, C, 1, 1)`` 的统计量做时间平滑."""
        b, t, c, _, _ = tensor.shape
        kernel = min(self.kernel_size, t)
        if kernel % 2 == 0:
            kernel = max(1, kernel - 1)
        if kernel == 1:
            return tensor

        pad = kernel // 2
        reshaped = tensor.permute(0, 2, 1, 3, 4).reshape(b * c, 1, t)
        padded = F.pad(reshaped, (pad, pad), mode='replicate')
        smoothed = F.avg_pool1d(padded, kernel_size=kernel, stride=1)
        smoothed = smoothed.view(b, c, t, 1, 1).permute(0, 2, 1, 3, 4)
        return smoothed


__all__ = [
    'default_init_weights',
    'make_layer',
    'ResidualBlockNoBN',
    'Upsample',
    'flow_warp',
    'ColorTemporalNormalizer',
]
