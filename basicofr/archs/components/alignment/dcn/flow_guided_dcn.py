"""DCNv2 流引导可变形对齐模块

基于 BasicVSR++ 的 Flow-guided Deformable Alignment，
使用 torchvision.ops.deform_conv2d 实现可变形卷积。

Ref:
    BasicVSR++: Improving Video Super-Resolution with Enhanced Propagation and Alignment.
"""

import math

import torch
import torch.nn as nn
import torchvision
from torch.nn.modules.utils import _pair, _single

__all__ = ['DCNv2PackFlowGuided', 'ModulatedDeformConv', 'ModulatedDeformConvPack']


class ModulatedDeformConv(nn.Module):
    """可调制可变形卷积基类"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        deformable_groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.with_bias = bias
        self.transposed = False
        self.output_padding = _single(0)

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, *self.kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self._init_weights()

    def _init_weights(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1.0 / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()


class ModulatedDeformConvPack(ModulatedDeformConv):
    """封装的可调制可变形卷积，行为类似普通卷积层

    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        kernel_size: 卷积核大小
        stride: 步长
        padding: 填充
        dilation: 膨胀率
        groups: 分组数
        deformable_groups: 可变形分组数
        bias: 是否使用偏置
    """

    _version = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv_offset = nn.Conv2d(
            self.in_channels,
            self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=_pair(self.stride),
            padding=_pair(self.padding),
            dilation=_pair(self.dilation),
            bias=True,
        )
        self._init_weights()

    def _init_weights(self):
        super()._init_weights()
        if hasattr(self, 'conv_offset'):
            self.conv_offset.weight.data.zero_()
            self.conv_offset.bias.data.zero_()


class DCNv2PackFlowGuided(ModulatedDeformConvPack):
    """流引导可变形对齐模块

    基于光流信息引导的可变形卷积对齐，用于视频超分辨率和修复任务。

    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        kernel_size: 卷积核大小
        stride: 步长
        padding: 填充
        dilation: 膨胀率
        groups: 分组数
        deformable_groups: 可变形分组数
        bias: 是否使用偏置
        max_residue_magnitude: 偏移残差的最大幅度，默认 10
        pa_frames: 并行 warp 的帧数，默认 2

    Ref:
        BasicVSR++: Improving Video Super-Resolution with Enhanced Propagation and Alignment.
    """

    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)
        self.pa_frames = kwargs.pop('pa_frames', 2)

        super().__init__(*args, **kwargs)

        # 偏移预测网络
        self.conv_offset = nn.Sequential(
            nn.Conv2d(
                (1 + self.pa_frames // 2) * self.in_channels + self.pa_frames,
                self.out_channels, 3, 1, 1
            ),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 2 * 9 * self.deformable_groups, 3, 1, 1),
        )

        self._init_offset()

    def _init_offset(self):
        ModulatedDeformConv._init_weights(self)
        if hasattr(self, 'conv_offset'):
            self.conv_offset[-1].weight.data.zero_()
            self.conv_offset[-1].bias.data.zero_()

    def forward(self, x: torch.Tensor, x_flow_warpeds: torch.Tensor,
                x_current: torch.Tensor, flows: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x: 输入特征 (B, C, H, W)
            x_flow_warpeds: 光流 warp 后的特征
            x_current: 当前帧特征
            flows: 光流场

        Returns:
            对齐后的特征 (B, C, H, W)
        """
        out = self.conv_offset(torch.cat([x_flow_warpeds, x_current, flows], dim=1))
        o1, o2 = torch.chunk(out, 2, dim=1)

        # 计算偏移量
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))

        if self.pa_frames == 2:
            offset = offset + flows[0].flip(1).repeat(1, offset.size(1) // 2, 1, 1)
        elif self.pa_frames == 4:
            offset1, offset2 = torch.chunk(offset, 2, dim=1)
            offset1 = offset1 + flows[0].flip(1).repeat(1, offset1.size(1) // 2, 1, 1)
            offset2 = offset2 + flows[1].flip(1).repeat(1, offset2.size(1) // 2, 1, 1)
            offset = torch.cat([offset1, offset2], dim=1)
        elif self.pa_frames == 6:
            offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
            offset1, offset2, offset3 = torch.chunk(offset, 3, dim=1)
            offset1 = offset1 + flows[0].flip(1).repeat(1, offset1.size(1) // 2, 1, 1)
            offset2 = offset2 + flows[1].flip(1).repeat(1, offset2.size(1) // 2, 1, 1)
            offset3 = offset3 + flows[2].flip(1).repeat(1, offset3.size(1) // 2, 1, 1)
            offset = torch.cat([offset1, offset2, offset3], dim=1)

        # 可变形卷积（不使用 mask）
        return torchvision.ops.deform_conv2d(
            x,
            offset,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            mask=None,
        )
