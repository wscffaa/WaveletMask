"""像素级掩码融合模块

将空间域掩码和小波域掩码进行像素级自适应融合。
"""
import torch
import torch.nn as nn


class PixelWiseMaskFusion(nn.Module):
    """像素级掩码融合网络

    使用 CNN 学习像素级融合权重 alpha，实现自适应融合：
    fused = alpha * spatial_mask + (1 - alpha) * wavelet_mask

    网络会根据局部特征自动决定更信任哪个分支的检测结果。

    Args:
        mid_channels: 中间层通道数
    """

    def __init__(self, mid_channels: int = 32):
        super().__init__()

        # 4 层 CNN，输入 2 通道（spatial + wavelet），输出 1 通道（alpha）
        self.fusion_net = nn.Sequential(
            nn.Conv2d(2, mid_channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_channels, mid_channels // 2, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_channels // 2, 1, 1),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        """初始化权重

        最后一层偏置初始化为正值，使初始 alpha ≈ 1（偏向空间掩码）。
        这样未训练时融合结果接近空间掩码，不受未训练小波掩码干扰。
        sigmoid(5) ≈ 0.993
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # 最后一层 bias 初始化为 +5，使 alpha ≈ 1（偏向空间掩码）
        self.fusion_net[-2].bias.data.fill_(5.0)

    def forward(
        self,
        spatial_mask: torch.Tensor,
        wavelet_mask: torch.Tensor
    ) -> torch.Tensor:
        """像素级融合两个掩码

        Args:
            spatial_mask: 空间域掩码 [B, 1, H, W] 或 [B, T, 1, H, W]
            wavelet_mask: 小波域掩码 [B, 1, H, W] 或 [B, T, 1, H, W]

        Returns:
            fused_mask: 融合后的掩码（与输入 shape 对齐）
        """
        if spatial_mask.dim() == 4:
            combined = torch.cat([spatial_mask, wavelet_mask], dim=1)  # [B, 2, H, W]
            alpha = self.fusion_net(combined)  # [B, 1, H, W]
            return alpha * spatial_mask + (1 - alpha) * wavelet_mask

        if spatial_mask.dim() != 5:
            raise ValueError(f'Expected 4D/5D mask tensors, got {spatial_mask.dim()}D.')

        b, t, _, h, w = spatial_mask.shape

        spatial = spatial_mask.view(b * t, 1, h, w)
        wavelet = wavelet_mask.view(b * t, 1, h, w)

        combined = torch.cat([spatial, wavelet], dim=1)  # [B*T, 2, H, W]
        alpha = self.fusion_net(combined)  # [B*T, 1, H, W]
        fused = alpha * spatial + (1 - alpha) * wavelet

        return fused.view(b, t, 1, h, w)

    def forward_with_alpha(
        self,
        spatial_mask: torch.Tensor,
        wavelet_mask: torch.Tensor
    ) -> tuple:
        """融合并返回 alpha 权重（用于可视化）

        Args:
            spatial_mask: 空间域掩码 [B, T, 1, H, W]
            wavelet_mask: 小波域掩码 [B, T, 1, H, W]

        Returns:
            fused_mask: 融合后的掩码 [B, T, 1, H, W]
            alpha: 融合权重 [B, T, 1, H, W]
        """
        if spatial_mask.dim() == 4:
            combined = torch.cat([spatial_mask, wavelet_mask], dim=1)
            alpha = self.fusion_net(combined)
            fused = alpha * spatial_mask + (1 - alpha) * wavelet_mask
            return fused, alpha

        if spatial_mask.dim() != 5:
            raise ValueError(f'Expected 4D/5D mask tensors, got {spatial_mask.dim()}D.')

        b, t, _, h, w = spatial_mask.shape

        spatial = spatial_mask.view(b * t, 1, h, w)
        wavelet = wavelet_mask.view(b * t, 1, h, w)

        combined = torch.cat([spatial, wavelet], dim=1)
        alpha = self.fusion_net(combined)

        fused = alpha * spatial + (1 - alpha) * wavelet

        return fused.view(b, t, 1, h, w), alpha.view(b, t, 1, h, w)
