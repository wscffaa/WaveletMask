"""可变形卷积组件

包含：
- ModulatedDeformConv2d: 纯 PyTorch 实现的调制可变形卷积
- SecondOrderDeformableAlignment: 二阶可变形对齐模块（RRTN 核心）
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init as nn_init


def _pair(value) -> Tuple[int, int]:
    """将单个值转换为二元组"""
    if isinstance(value, tuple):
        return value
    return value, value


def constant_init(module: nn.Module, val: float, bias: float = 0):
    """用常数初始化模块权重"""
    nn_init.constant_(module.weight, val)
    if module.bias is not None:
        nn_init.constant_(module.bias, bias)


def modulated_deform_conv2d(
    x: torch.Tensor,
    offset: torch.Tensor,
    mask: Optional[torch.Tensor],
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
    deform_groups: int = 1
) -> torch.Tensor:
    """纯 PyTorch 实现的调制可变形卷积

    Args:
        x: 输入特征 (N, C_in, H, W)
        offset: 偏移量 (N, 2*deform_groups*kH*kW, H_out, W_out)
        mask: 调制掩码 (N, deform_groups*kH*kW, H_out, W_out)
        weight: 卷积权重 (C_out, C_in/groups, kH, kW)
        bias: 偏置项 (C_out,)
        stride: 步长
        padding: 填充
        dilation: 膨胀率
        groups: 分组数
        deform_groups: 可变形分组数

    Returns:
        输出特征 (N, C_out, H_out, W_out)
    """
    stride_h, stride_w = _pair(stride)
    pad_h, pad_w = _pair(padding)
    dil_h, dil_w = _pair(dilation)

    n, c_in, h_in, w_in = x.shape
    out_channels, _, k_h, k_w = weight.shape
    if c_in % deform_groups != 0:
        raise ValueError('in_channels must be divisible by deform_groups.')
    if c_in % groups != 0:
        raise ValueError('in_channels must be divisible by groups.')

    n_off, off_channels, h_out, w_out = offset.shape
    if n_off != n:
        raise ValueError('Batch size of input and offset must match.')
    if off_channels != 2 * deform_groups * k_h * k_w:
        raise ValueError('Offset channels mismatch kernel size/deform_groups.')

    if mask is None:
        device = offset.device
        dtype = offset.dtype
        mask = torch.ones(n, deform_groups * k_h * k_w, h_out, w_out, device=device, dtype=dtype)
    else:
        if mask.shape[0] != n or mask.shape[1] != deform_groups * k_h * k_w:
            raise ValueError('Mask shape mismatch.')

    x_padded = F.pad(x, (pad_w, pad_w, pad_h, pad_h))
    h_pad, w_pad = x_padded.shape[2:]

    x_padded = x_padded.view(n, deform_groups, c_in // deform_groups, h_pad, w_pad)
    x_padded = x_padded.reshape(n * deform_groups, c_in // deform_groups, h_pad, w_pad)

    offset = offset.view(n, deform_groups, k_h, k_w, 2, h_out, w_out)
    mask = mask.view(n, deform_groups, k_h, k_w, h_out, w_out)

    dtype = offset.dtype
    device = offset.device
    base_y = torch.arange(h_out, device=device, dtype=dtype) * stride_h
    base_x = torch.arange(w_out, device=device, dtype=dtype) * stride_w
    base_y = base_y.view(1, 1, h_out, 1)
    base_x = base_x.view(1, 1, 1, w_out)

    cols = []
    height_normalizer = max(h_pad - 1, 1)
    width_normalizer = max(w_pad - 1, 1)

    for kh in range(k_h):
        for kw in range(k_w):
            offset_y = offset[:, :, kh, kw, 0, :, :]
            offset_x = offset[:, :, kh, kw, 1, :, :]
            y = base_y + kh * dil_h + offset_y
            x_coord = base_x + kw * dil_w + offset_x

            y = (y / height_normalizer) * 2 - 1
            x_coord = (x_coord / width_normalizer) * 2 - 1
            grid = torch.stack((x_coord, y), dim=-1)  # (n, deform_groups, h_out, w_out, 2)
            grid = grid.view(n * deform_groups, h_out, w_out, 2)

            sampled = F.grid_sample(
                x_padded,
                grid,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=True)
            sampled = sampled.view(n, deform_groups, c_in // deform_groups, h_out, w_out)

            sampled = sampled * mask[:, :, kh, kw].unsqueeze(2)
            cols.append(sampled)

    cols = torch.stack(cols, dim=3)  # (n, deform_groups, c_group, k_h*k_w, h_out, w_out)
    cols = cols.view(n, deform_groups, (c_in // deform_groups) * k_h * k_w, h_out, w_out)
    cols = cols.reshape(n, c_in * k_h * k_w, h_out, w_out)

    cols = cols.view(n, groups, (c_in // groups) * k_h * k_w, h_out, w_out)
    cols = cols.permute(0, 1, 3, 4, 2)  # (n, groups, h_out, w_out, cin_group * k*k)
    weight = weight.view(groups, out_channels // groups, (c_in // groups) * k_h * k_w)

    out = torch.einsum('nghwc,goc->nghwo', cols, weight)
    out = out.permute(0, 1, 4, 2, 3).contiguous()
    out = out.view(n, out_channels, h_out, w_out)

    if bias is not None:
        out = out + bias.view(1, -1, 1, 1)
    return out


class ModulatedDeformConv2d(nn.Module):
    """纯 PyTorch 实现的调制可变形卷积层

    相比 CUDA 实现更具可移植性，适用于不支持自定义 CUDA 算子的环境。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        deform_groups: int = 1
    ):
        super().__init__()
        k_h, k_w = _pair(kernel_size)
        stride_h, stride_w = _pair(stride)
        pad_h, pad_w = _pair(padding)
        dil_h, dil_w = _pair(dilation)
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups.')
        if in_channels % deform_groups != 0:
            raise ValueError('in_channels must be divisible by deform_groups.')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (k_h, k_w)
        self.stride = (stride_h, stride_w)
        self.padding = (pad_h, pad_w)
        self.dilation = (dil_h, dil_w)
        self.groups = groups
        self.deform_groups = deform_groups

        weight_shape = (out_channels, in_channels // groups, k_h, k_w)
        self.weight = nn.Parameter(torch.empty(weight_shape))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor, offset: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return modulated_deform_conv2d(
            x,
            offset,
            mask,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.deform_groups)


class SecondOrderDeformableAlignment(ModulatedDeformConv2d):
    """二阶可变形对齐模块（RRTN 核心组件）

    利用当前帧 t 与前两帧 t-1、t-2 的条件信息联合学习偏移量和调制掩码，
    实现更精确的时序对齐。相比一阶对齐，二阶对齐能够更好地处理复杂运动。

    Args:
        *args: ModulatedDeformConv2d 的位置参数
        max_residue_magnitude: 残差偏移的最大幅度（通过 tanh 约束）
        **kwargs: ModulatedDeformConv2d 的关键字参数
    """

    def __init__(self, *args, max_residue_magnitude: int = 10, **kwargs):
        self.max_residue_magnitude = max_residue_magnitude
        super().__init__(*args, **kwargs)

        # 偏移预测网络：输入为 3*out_channels + 4（光流通道）
        self.conv_offset = nn.Sequential(
            nn.Conv2d(3 * self.out_channels + 4, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 27 * self.deform_groups, 3, 1, 1),
        )
        self.init_offset()

    def init_offset(self):
        """初始化偏移预测网络的最后一层为零"""
        constant_init(self.conv_offset[-1], val=0, bias=0)

    def forward(
        self,
        x: torch.Tensor,
        extra_feat: torch.Tensor,
        flow_1: torch.Tensor,
        flow_2: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: 输入特征（包含 t-1 和 t-2 帧的传播特征）
            extra_feat: 条件特征（warped t-1, current, warped t-2 拼接）
            flow_1: t-1 到 t 的光流
            flow_2: t-2 到 t 的光流

        Returns:
            对齐后的特征
        """
        extra_feat = torch.cat([extra_feat, flow_1, flow_2], dim=1)
        out = self.conv_offset(extra_feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)

        # 使用 tanh 约束残差偏移的范围
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((o1, o2), dim=1))
        offset_1, offset_2 = torch.chunk(offset, 2, dim=1)

        # 将光流作为基础偏移，残差偏移作为修正
        offset_1 = offset_1 + flow_1.flip(1).repeat(1, offset_1.size(1) // 2, 1, 1)
        offset_2 = offset_2 + flow_2.flip(1).repeat(1, offset_2.size(1) // 2, 1, 1)
        offset = torch.cat([offset_1, offset_2], dim=1)

        mask = torch.sigmoid(mask)

        return modulated_deform_conv2d(
            x, offset, mask, self.weight, self.bias,
            self.stride, self.padding, self.dilation,
            self.groups, self.deform_groups)


__all__ = [
    'ModulatedDeformConv2d',
    'modulated_deform_conv2d',
    'SecondOrderDeformableAlignment',
    'constant_init',
]
