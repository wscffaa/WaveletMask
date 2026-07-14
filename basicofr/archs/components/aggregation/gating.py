"""门控聚合模块

提供多种门控聚合实现，用于视频修复中的时序特征融合。
"""

import torch
import torch.nn.functional as F
from torch import nn

from ..alignment.dcn import DCNv2PackFlowGuided

__all__ = ['MultiScaleGatedAggregation', 'GatedAggregation', 'GatedAggregationDCN']


class GatedAggregation(nn.Module):
    """基础门控聚合模块

    根据残差指示调节隐藏状态与当前帧的融合比例。

    Args:
        hidden_channels: 隐藏通道数
        kernel_size: 卷积核大小
        stride: 步长
        padding: 填充
        dilation: 膨胀率
        bias: 是否使用偏置
        activation: 激活函数
    """

    def __init__(
        self,
        hidden_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = True,
        activation: nn.Module = nn.LeakyReLU(0.2, inplace=True),
    ):
        super().__init__()

        self.activation = activation
        self.proj = nn.Conv2d(3, hidden_channels, kernel_size, stride, padding, dilation, bias=bias)
        self.conv1 = nn.Conv2d(
            hidden_channels + hidden_channels + 1,
            hidden_channels // 2,
            kernel_size, stride, padding, dilation, bias=bias
        )
        self.conv2 = nn.Conv2d(hidden_channels // 2, 1, kernel_size, stride, padding, dilation, bias=bias)
        self.sigmoid = nn.Sigmoid()
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def forward(
        self,
        hidden_state: torch.Tensor,
        curr_lr: torch.Tensor,
        residual_indicator: torch.Tensor,
        head: bool = False,
        return_mask: bool = False,
        return_feat: bool = False,
    ):
        """门控聚合前向传播"""
        latent_feature = self.proj(curr_lr)
        x = self.activation(self.conv1(torch.cat([hidden_state, latent_feature, residual_indicator], dim=1)))
        gate = self.sigmoid(self.conv2(x))

        if return_mask:
            if head:
                if return_feat:
                    return latent_feature, latent_feature
                return latent_feature, gate
            blended = gate * hidden_state + (1 - gate) * latent_feature
            if return_feat:
                return blended, latent_feature
            return blended, gate

        if head:
            return latent_feature
        return gate * hidden_state + (1 - gate) * latent_feature


class GatedAggregationDCN(nn.Module):
    """带 DCNv2 对齐的门控聚合模块

    在基础门控聚合的基础上增加 DCNv2 流引导对齐。

    Args:
        hidden_channels: 隐藏通道数
        kernel_size: 卷积核大小
        stride: 步长
        padding: 填充
        dilation: 膨胀率
        bias: 是否使用偏置
        activation: 激活函数
    """

    def __init__(
        self,
        hidden_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = True,
        activation: nn.Module = nn.LeakyReLU(0.2, inplace=True),
    ):
        super().__init__()

        self.activation = activation

        # 当前帧投影
        self.proj = nn.Conv2d(3, hidden_channels, kernel_size, stride, padding, dilation, bias=bias)

        # 门控融合（含 pre_mask 通道）
        fusion_channels = hidden_channels * 2 + 2
        self.fusion = nn.Sequential(
            nn.Conv2d(fusion_channels, fusion_channels, kernel_size, stride, padding, dilation, bias=bias),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(fusion_channels, hidden_channels // 2, kernel_size, stride, padding, dilation, bias=bias),
        )

        # 门控输出
        self.gate = nn.Conv2d(hidden_channels // 2, 1, kernel_size, stride, padding, dilation, bias=bias)

        # DCNv2 流引导对齐
        self.align = DCNv2PackFlowGuided(
            hidden_channels, hidden_channels,
            kernel_size=3, stride=1, padding=1,
            deformable_groups=1, pa_frames=2
        )

        self.sigmoid = nn.Sigmoid()
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')

    def forward(
        self,
        hidden_state: torch.Tensor,
        curr_lr: torch.Tensor,
        residual_indicator: torch.Tensor,
        hidden_state_o: torch.Tensor = None,
        flow: torch.Tensor = None,
        pre_mask: torch.Tensor = None,
        head: bool = False,
        return_mask: bool = False,
        return_feat: bool = False,
    ):
        """门控聚合前向传播（含 DCN 对齐）"""
        if pre_mask is None:
            pre_mask = residual_indicator.new_ones(residual_indicator.shape)

        latent_feature = self.proj(curr_lr)

        # DCNv2 流引导对齐
        if hidden_state_o is not None and flow is not None:
            hidden_state = self.align(hidden_state_o, hidden_state, latent_feature, flow)

        # 门控融合
        gate_feat = self.activation(self.fusion(
            torch.cat([hidden_state, latent_feature, residual_indicator, pre_mask], dim=1)
        ))
        gate = self.sigmoid(self.gate(gate_feat))

        # 特征混合
        blended = gate * hidden_state + (1 - gate) * latent_feature
        blended = latent_feature if head else blended

        if return_mask:
            if return_feat:
                return blended, gate, latent_feature
            return blended, gate
        return blended, gate


class MultiScaleGatedAggregation(nn.Module):
    """Multi-scale gated fusion between recurrent hidden states and current LR frame.

    The module expands the original gating design with dilated convolution and
    pooled context so large artifacts receive stronger responses.
    """

    def __init__(
        self,
        hidden_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        activation: nn.Module = None,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.activation = activation if activation is not None else nn.LeakyReLU(0.2, inplace=True)

        # Project RGB frame into hidden feature space.
        self.proj = nn.Conv2d(3, hidden_channels, kernel_size=kernel_size, padding=padding)

        # Local pathway (3x3) keeps fine textures.
        in_channels = hidden_channels * 2 + 1
        self.local_conv = nn.Conv2d(in_channels, hidden_channels, kernel_size=kernel_size, padding=padding)

        # Dilated pathway (3x3 dilation=2) enlarges receptive field.
        self.dilated_conv = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=kernel_size, padding=2, dilation=2)

        # Low-frequency pooling pathway highlights broad corruptions.
        self.avg_pool = nn.AvgPool2d(5, stride=1, padding=2)
        self.pool_conv = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=kernel_size, padding=padding)

        # Fuse multi-scale signals and predict gate.
        self.fuse_conv = nn.Conv2d(hidden_channels * 3, hidden_channels // 2, kernel_size=1)
        self.gate_conv = nn.Conv2d(hidden_channels // 2, 1, kernel_size=kernel_size, padding=padding)
        self.sigmoid = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        hidden_state: torch.Tensor,
        curr_lr: torch.Tensor,
        residual_indicator: torch.Tensor,
        head: bool = False,
        return_mask: bool = False,
        return_feat: bool = False,
    ):
        latent_feature = self.proj(curr_lr)
        feat = torch.cat([hidden_state, latent_feature, residual_indicator], dim=1)
        local_feat = self.activation(self.local_conv(feat))
        dilated_feat = self.activation(self.dilated_conv(local_feat))

        pooled_feat = self.avg_pool(local_feat)
        pooled_feat = self.activation(self.pool_conv(pooled_feat))
        pooled_feat = F.interpolate(pooled_feat, size=local_feat.shape[-2:], mode='bilinear', align_corners=False)

        fused = torch.cat([local_feat, dilated_feat, pooled_feat], dim=1)
        fused = self.activation(self.fuse_conv(fused))
        gate = self.sigmoid(self.gate_conv(fused))

        if head:
            blended = latent_feature
        else:
            blended = gate * hidden_state + (1 - gate) * latent_feature

        if return_mask:
            if head:
                if return_feat:
                    return blended, blended
                return blended, gate
            if return_feat:
                return blended, latent_feature
            return blended, gate

        if head:
            return blended
        return blended
