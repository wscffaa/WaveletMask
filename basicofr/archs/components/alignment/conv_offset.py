"""
Channel-Attention Enhanced Offset Predictor for Deformable Convolution.

Source: DefMamba
Migration: From basicsr/archs/defmamba_arch.py:191-217

This module predicts spatial offsets with channel attention mechanism,
enabling content-aware deformable sampling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['ConvOffset']


class ConvOffset(nn.Module):
    """Offset predictor with channel attention.

    Predicts 3-channel output: [offset_y, offset_x, path_index]
    Used in deformable spatial sampling and path ordering.

    Args:
        embed_dim (int): Input/output channel dimension
        kernel_size (int): Depthwise convolution kernel size
        padding (int): Convolution padding size

    Source: basicsr/archs/defmamba_arch.py:191-217
    """

    def __init__(self, embed_dim: int, kernel_size: int, padding: int):
        super().__init__()
        self.conv1 = nn.Conv2d(embed_dim, embed_dim, kernel_size, 1, padding, groups=embed_dim)
        self.ca = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 16),
            nn.GELU(),
            nn.Linear(embed_dim // 16, embed_dim),
            nn.Sigmoid()
        )
        self.ln = nn.LayerNorm(embed_dim)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv2d(embed_dim, 3, 1, 1, 0, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        x_c = F.adaptive_avg_pool2d(x, (1, 1))
        x_c = self.ca(x_c.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = x1 * x_c.expand_as(x)
        x = self.gelu(self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
        x = self.conv2(x)
        return x
