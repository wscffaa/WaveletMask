"""Static 2D DWT backends for WaveletMask review ablations."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


_SQRT2_INV = 0.7071067811865476

_FILTERS: dict[str, tuple[list[float], list[float]]] = {
    'haar': (
        [_SQRT2_INV, _SQRT2_INV],
        [_SQRT2_INV, -_SQRT2_INV],
    ),
    'db2': (
        [-0.12940952255092145, 0.22414386804185735, 0.8365163037378079, 0.48296291314469025],
        [-0.48296291314469025, 0.8365163037378079, -0.22414386804185735, -0.12940952255092145],
    ),
    'sym2': (
        [-0.12940952255092145, 0.22414386804185735, 0.8365163037378079, 0.48296291314469025],
        [-0.48296291314469025, 0.8365163037378079, -0.22414386804185735, -0.12940952255092145],
    ),
    'coif1': (
        [-0.015655728135791993, -0.07273261951252645, 0.3848648468648578, 0.8525720202116004, 0.3378976624574818, -0.07273261951252645],
        [-0.015655728135791993, 0.07273261951252645, 0.3848648468648578, -0.8525720202116004, 0.3378976624574818, 0.07273261951252645],
    ),
}


class StaticDWT2d(nn.Module):
    """Single-channel 2D DWT with fixed orthogonal analysis filters.

    The review ablations only need forward subbands. For level-2 Haar, the second
    level is computed from LL and upsampled back to the first-level subband size
    so the existing mask heads stay unchanged.
    """

    def __init__(self, wavelet: str = 'haar', levels: int = 1):
        super().__init__()
        normalized_wavelet = wavelet.lower()
        if normalized_wavelet in {'dual_tree_complex', 'dual-tree-complex'}:
            normalized_wavelet = 'dtcwt'
        if normalized_wavelet == 'dtcwt':
            if levels != 1:
                raise ValueError('DTCWT review ablation currently supports only one level.')
            try:
                from pytorch_wavelets import DTCWTForward
            except Exception as exc:
                raise ValueError(
                    'DTCWT review backend requires pytorch_wavelets in the active environment.'
                ) from exc
            self.wavelet = normalized_wavelet
            self.levels = int(levels)
            self.dtcwt = DTCWTForward(
                J=1,
                biort='near_sym_a',
                qshift='qshift_a',
                skip_hps=False,
                include_scale=False,
                o_dim=2,
                ri_dim=-1,
                mode='symmetric',
            )
            return
        if normalized_wavelet not in _FILTERS:
            allowed = ', '.join(sorted(_FILTERS))
            raise ValueError(f'Unsupported wavelet backend: {wavelet}. Allowed: {allowed}, dtcwt.')
        if levels not in (1, 2):
            raise ValueError(f'Unsupported DWT levels: {levels}. Allowed: 1 or 2.')
        if levels == 2 and normalized_wavelet != 'haar':
            raise ValueError('Two-level review ablation is defined only for Haar.')

        self.wavelet = normalized_wavelet
        self.levels = int(levels)
        low, high = _FILTERS[normalized_wavelet]
        kernels = self._build_kernels(low, high)
        self.register_buffer('kernels', kernels)

    @staticmethod
    def _build_kernels(low: list[float], high: list[float]) -> torch.Tensor:
        lo = torch.tensor(low, dtype=torch.float32)
        hi = torch.tensor(high, dtype=torch.float32)
        kernels = torch.stack([
            torch.outer(lo, lo),
            torch.outer(lo, hi),
            torch.outer(hi, lo),
            torch.outer(hi, hi),
        ], dim=0)
        return kernels.view(4, 1, len(low), len(low))

    def _forward_one_level(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        kernel_size = self.kernels.shape[-1]
        if kernel_size > 2:
            pad = (kernel_size - 2) // 2
            x = F.pad(x, (pad, pad, pad, pad), mode='reflect')
        out = F.conv2d(x, self.kernels, stride=2)
        return {
            'LL': out[:, 0:1],
            'LH': out[:, 1:2],
            'HL': out[:, 2:3],
            'HH': out[:, 3:4],
        }

    @staticmethod
    def _upsample_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=ref.shape[-2:], mode='bilinear', align_corners=False)

    def _forward_dtcwt(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        yl, yh = self.dtcwt(x)
        band = yh[0]
        magnitude = torch.sqrt(torch.square(band[..., 0]) + torch.square(band[..., 1]) + 1e-8)
        # pytorch_wavelets orders orientations as 15, 45, 75, 105, 135, 165 degrees.
        # Group the six directional responses back into LH / HL / HH pairs so the
        # review mask heads keep the same three-band contract as the orthogonal DWT backends.
        return {
            'LL': F.avg_pool2d(yl, kernel_size=2, stride=2),
            'LH': magnitude[:, :, (0, 5)].mean(dim=2),
            'HL': magnitude[:, :, (2, 3)].mean(dim=2),
            'HH': magnitude[:, :, (1, 4)].mean(dim=2),
        }

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        if x.dim() != 4 or x.size(1) != 1:
            raise ValueError(f'Expected [B,1,H,W] or [B,H,W], got {tuple(x.shape)}.')
        if self.wavelet == 'dtcwt':
            return self._forward_dtcwt(x)

        level1 = self._forward_one_level(x)
        if self.levels == 1:
            return level1

        level2 = self._forward_one_level(level1['LL'])
        return {
            'LL': self._upsample_like(level2['LL'], level1['LL']),
            'LH': 0.5 * (level1['LH'] + self._upsample_like(level2['LH'], level1['LH'])),
            'HL': 0.5 * (level1['HL'] + self._upsample_like(level2['HL'], level1['HL'])),
            'HH': 0.5 * (level1['HH'] + self._upsample_like(level2['HH'], level1['HH'])),
        }
