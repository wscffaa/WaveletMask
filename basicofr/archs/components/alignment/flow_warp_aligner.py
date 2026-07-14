"""纯光流 Warp 对齐器

RTN 使用的最简对齐策略：仅通过光流进行特征 warp，无可变形卷积。
可与 SecondOrderDeformableAlignment 互换。

接口约定：
    forward(feat, flow) -> aligned_feat
"""

import torch
import torch.nn as nn

from ..arch_util import flow_warp


class FlowWarpAligner(nn.Module):
    """纯光流对齐器

    最简单的时序对齐方式，仅用 bilinear grid_sample 实现光流 warp。
    无可学习参数。
    """

    def __init__(self):
        super().__init__()

    def forward(self, feat: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """对齐特征

        Args:
            feat: 待对齐特征 (B, C, H, W)
            flow: 光流 (B, 2, H, W)

        Returns:
            对齐后的特征 (B, C, H, W)
        """
        return flow_warp(feat, flow.permute(0, 2, 3, 1))
