"""基于小波变换的噪声掩码检测模块

用于检测老电影中的局部噪声（划痕、灰尘）和全局噪声（闪烁、色偏）。
核心思想：利用前后帧的几何均值差异区分噪声和运动。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from contextlib import nullcontext

from .haar_dwt import HaarDWT2d


class WaveletNoiseMask(nn.Module):
    """基于小波变换的噪声掩码检测

    核心思路：
    - 高频子带（LH/HL/HH）：检测局部噪声（划痕、灰尘）
    - 低频子带（LL）：检测全局闪烁（亮度突变）

    使用前后帧的几何均值差异来区分噪声和运动：
    - 几何均值: sqrt(|curr - prev| * |curr - next|)
    - 当且仅当当前帧与前后帧都不同时才有高响应

    Args:
        high_freq_channels: 高频处理网络的中间通道数
        low_freq_channels: 低频处理网络的中间通道数
    """

    def __init__(self, high_freq_channels: int = 16, low_freq_channels: int = 16):
        super().__init__()
        self.dwt = HaarDWT2d()

        # ========== 高频子带处理（检测划痕等局部噪声） ==========
        # 输入：3 通道（LH/HL/HH 差异）
        self.high_freq_conv = nn.Sequential(
            nn.Conv2d(3, high_freq_channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(high_freq_channels, 1, 3, 1, 1),
            nn.Sigmoid()
        )

        # ========== 低频子带处理（检测全局闪烁） ==========
        # 输入：1 通道（LL 差异）
        self.low_freq_conv = nn.Sequential(
            nn.Conv2d(1, low_freq_channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(low_freq_channels, 1, 3, 1, 1),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        """初始化权重

        关键：最后一层 Sigmoid 前的 conv bias 初始化为负值，
        使未训练时输出接近 0，不干扰空间掩码。
        sigmoid(-5) ≈ 0.007
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # 最后一层 bias 初始化为 -5，使 sigmoid 输出接近 0
        # 这样未训练时小波掩码不会干扰空间掩码
        self.high_freq_conv[-2].bias.data.fill_(-5.0)
        self.low_freq_conv[-2].bias.data.fill_(-5.0)

    @staticmethod
    def _no_autocast(tensor: torch.Tensor):
        """在小波分支中局部禁用 AMP，避免 dtype 不一致导致的 conv2d 报错。"""
        autocast = getattr(torch, 'autocast', None)
        if autocast is not None:
            device_type = 'cuda' if tensor.is_cuda else 'cpu'
            return autocast(device_type=device_type, enabled=False)
        if tensor.is_cuda:
            return torch.cuda.amp.autocast(enabled=False)
        return nullcontext()

    def _compute_masks(
        self,
        lqs: torch.Tensor,
        *,
        compute_local: bool,
        compute_flicker: bool,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if lqs.dim() != 5:
            raise ValueError(f'Expected 5D input [B, T, C, H, W], got {lqs.dim()}D.')
        if not compute_local and not compute_flicker:
            raise ValueError('At least one of compute_local/compute_flicker must be True.')

        b, t, c, h, w = lqs.shape
        device = lqs.device
        out_dtype = lqs.dtype

        local_masks = [] if compute_local else None
        flicker_masks = [] if compute_flicker else None

        with self._no_autocast(lqs):
            lqs_float = lqs.float()

            # 转换为灰度图: [B, T, H, W]
            if c == 3:
                lqs_gray = (
                    0.299 * lqs_float[:, :, 0]
                    + 0.587 * lqs_float[:, :, 1]
                    + 0.114 * lqs_float[:, :, 2]
                )
            elif c == 1:
                lqs_gray = lqs_float[:, :, 0]
            else:
                lqs_gray = lqs_float.mean(dim=2)

            for i in range(t):
                if i == 0 or i == t - 1:
                    # 边界帧使用零掩码（无法计算前后帧差异）
                    zero = torch.zeros(b, 1, h, w, device=device, dtype=torch.float32)
                    if local_masks is not None:
                        local_masks.append(zero)
                    if flicker_masks is not None:
                        flicker_masks.append(zero)
                    continue

                curr = lqs_gray[:, i]  # [B, H, W]
                prev = lqs_gray[:, i - 1]
                next_ = lqs_gray[:, i + 1]

                # 小波分解（在 float32 下进行，避免 AMP dtype 问题）
                curr_c = self.dwt(curr)
                prev_c = self.dwt(prev)
                next_c = self.dwt(next_)

                if local_masks is not None:
                    high_freq_diff = self._compute_high_freq_diff(curr_c, prev_c, next_c)
                    high_mask = self.high_freq_conv(high_freq_diff)  # [B, 1, H/2, W/2]
                    high_mask = F.interpolate(high_mask, size=(h, w), mode='bilinear', align_corners=False)
                    local_masks.append(high_mask)

                if flicker_masks is not None:
                    low_freq_diff = self._compute_low_freq_diff(curr_c, prev_c, next_c)
                    low_mask = self.low_freq_conv(low_freq_diff)  # [B, 1, H/2, W/2]
                    low_mask = F.interpolate(low_mask, size=(h, w), mode='bilinear', align_corners=False)
                    flicker_masks.append(low_mask)

        local = torch.stack(local_masks, dim=1).to(dtype=out_dtype) if local_masks is not None else None
        flicker = torch.stack(flicker_masks, dim=1).to(dtype=out_dtype) if flicker_masks is not None else None
        return local, flicker

    def compute_local_mask(self, lqs: torch.Tensor) -> torch.Tensor:
        """仅计算局部高频掩码（创新 A）。

        Args:
            lqs: 输入视频序列 [B, T, C, H, W]，值域 [0, 1] 或 [-1, 1]

        Returns:
            local_mask: 局部高频掩码 [B, T, 1, H, W]
        """
        local, _ = self._compute_masks(lqs, compute_local=True, compute_flicker=False)
        assert local is not None
        return local

    def compute_flicker_mask(self, lqs: torch.Tensor) -> torch.Tensor:
        """仅计算全局闪烁掩码（创新 B）。

        Args:
            lqs: 输入视频序列 [B, T, C, H, W]

        Returns:
            flicker_mask: 闪烁掩码 [B, T, 1, H, W]
        """
        _, flicker = self._compute_masks(lqs, compute_local=False, compute_flicker=True)
        assert flicker is not None
        return flicker

    def forward(self, lqs: torch.Tensor) -> torch.Tensor:
        """计算完整小波域噪声掩码（创新 A+B）。

        Returns:
            wavelet_mask: 融合后的掩码 [B, T, 1, H, W]
        """
        local, flicker = self._compute_masks(lqs, compute_local=True, compute_flicker=True)
        assert local is not None and flicker is not None
        return torch.max(local, flicker)

    def _compute_high_freq_diff(
        self,
        curr: Dict[str, torch.Tensor],
        prev: Dict[str, torch.Tensor],
        next_: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """计算高频子带几何均值差异

        几何均值：sqrt(|curr - prev| * |curr - next|)
        当且仅当当前帧与前后帧都不同时才有高响应，可区分噪声和运动。

        Args:
            curr, prev, next_: 小波系数字典

        Returns:
            高频差异特征 [B, 3, H/2, W/2]
        """
        eps = 1e-8

        # LH 子带（水平高频）
        lh_diff = torch.sqrt(
            torch.abs(curr['LH'] - prev['LH']) *
            torch.abs(curr['LH'] - next_['LH']) + eps
        )

        # HL 子带（垂直高频）
        hl_diff = torch.sqrt(
            torch.abs(curr['HL'] - prev['HL']) *
            torch.abs(curr['HL'] - next_['HL']) + eps
        )

        # HH 子带（对角高频）
        hh_diff = torch.sqrt(
            torch.abs(curr['HH'] - prev['HH']) *
            torch.abs(curr['HH'] - next_['HH']) + eps
        )

        return torch.cat([lh_diff, hl_diff, hh_diff], dim=1)

    def _compute_low_freq_diff(
        self,
        curr: Dict[str, torch.Tensor],
        prev: Dict[str, torch.Tensor],
        next_: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """计算低频子带差异（检测全局闪烁）

        当前帧与前后帧的 LL 差异取最小值：
        - 如果两个方向都有差异 → 闪烁
        - 如果只有一个方向有差异 → 可能是渐变，不是闪烁

        Args:
            curr, prev, next_: 小波系数字典

        Returns:
            低频差异特征 [B, 1, H/2, W/2]
        """
        ll_diff_prev = torch.abs(curr['LL'] - prev['LL'])
        ll_diff_next = torch.abs(curr['LL'] - next_['LL'])

        # 取最小值：两个方向都有差异才认为是闪烁
        return torch.min(ll_diff_prev, ll_diff_next)
