"""动态卷积模块

用于 MambaIR 中的通道注意力块 (CAB)。
"""

from collections.abc import Iterable
import itertools
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init
from torch.nn.modules.utils import _pair


class AttentionLayer(nn.Module):
    """注意力层：根据输入特征动态加权多个卷积核"""

    def __init__(self, c_dim: int, hidden_dim: int, nof_kernels: int):
        super().__init__()
        self.global_pooling = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.prompt_param = nn.Parameter(torch.rand(1, nof_kernels, hidden_dim))
        self.to_scores = nn.Sequential(
            nn.Linear(c_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        x = self.global_pooling(x)
        out = self.to_scores(x).unsqueeze(1)
        scores = torch.mul(out, self.prompt_param).sum(-1)
        return F.softmax(scores / temperature, dim=-1)


class DynamicConvolution(nn.Module):
    """动态卷积：使用注意力机制加权多个卷积核

    Args:
        nof_kernels: 卷积核数量
        extend: 隐藏层扩展因子
        in_channels: 输入通道数
        out_channels: 输出通道数
        kernel_size: 卷积核大小
        stride: 步长
        padding: 填充
        dilation: 膨胀率
        groups: 分组数
        temperature: softmax 温度
        bias: 是否使用偏置
    """

    def __init__(
        self,
        nof_kernels: int,
        extend: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        temperature: float = 1.0,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.temperature = temperature
        self.conv_args = {'stride': stride, 'padding': padding, 'dilation': dilation}
        self.nof_kernels = nof_kernels
        self.attention = AttentionLayer(in_channels, max(1, in_channels * extend), nof_kernels)
        self.kernel_size = _pair(kernel_size)
        self.kernels_weights = nn.Parameter(
            torch.Tensor(nof_kernels, out_channels, in_channels // self.groups, *self.kernel_size),
            requires_grad=True,
        )
        if bias:
            self.kernels_bias = nn.Parameter(torch.Tensor(nof_kernels, out_channels), requires_grad=True)
        else:
            self.register_parameter('kernels_bias', None)
        self._init_weights()

    def _init_weights(self):
        for i_kernel in range(self.nof_kernels):
            init.kaiming_uniform_(self.kernels_weights[i_kernel], a=math.sqrt(5))
        if self.kernels_bias is not None:
            bound = 1 / math.sqrt(self.kernels_weights[0, 0].numel())
            nn.init.uniform_(self.kernels_bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        alphas = self.attention(x, self.temperature)
        agg_weights = torch.sum(
            torch.mul(self.kernels_weights.unsqueeze(0), alphas.view(batch_size, -1, 1, 1, 1, 1)),
            dim=1,
        )
        agg_weights = agg_weights.view(-1, *agg_weights.shape[-3:])
        if self.kernels_bias is not None:
            agg_bias = torch.sum(torch.mul(self.kernels_bias.unsqueeze(0), alphas.view(batch_size, -1, 1)), dim=1)
            agg_bias = agg_bias.view(-1)
        else:
            agg_bias = None
        x_grouped = x.view(1, -1, *x.shape[-2:])
        out = F.conv2d(
            x_grouped,
            agg_weights,
            agg_bias,
            groups=self.groups * batch_size,
            **self.conv_args,
        )
        out = out.view(batch_size, -1, *out.shape[-2:])
        return out


def dynamic_convolution_generator(nof_kernels, extend):
    """创建动态卷积生成器"""
    class FlexibleKernelsDynamicConvolution:
        def __init__(self, Base, nof_kernels, extend):
            if isinstance(nof_kernels, Iterable):
                self.nof_kernels_it = iter(nof_kernels)
            else:
                self.nof_kernels_it = itertools.cycle([nof_kernels])
            self.Base = Base
            self.extend = extend

        def __call__(self, *args, **kwargs):
            return self.Base(next(self.nof_kernels_it), self.extend, *args, **kwargs)

    return FlexibleKernelsDynamicConvolution(DynamicConvolution, nof_kernels, extend)
