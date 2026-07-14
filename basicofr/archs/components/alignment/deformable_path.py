"""可变形路径选择模块 (Deformable Path Selection)

从 DefMamba 移植的可变形路径选择机制，用于自适应扫描顺序。

核心组件：
1. ConvOffset: 预测 2D 空间偏移量 + 1D 路径索引
2. DeformableLayer: 可变形空间采样 + 路径重排序
3. DeformableLayerReverse: 路径逆变换
4. DeformablePathTrans: 自定义 autograd.Function 实现可微路径重排

Ref:
    DefMamba: Deformable State Space Model
"""

import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_


class DeformablePathTrans(torch.autograd.Function):
    """可变形路径变换（可微分）

    基于预测的路径索引对特征序列进行重排序。
    使用 topk 排序实现可微分的重排操作。
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, de_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播

        Args:
            x: 输入特征 (B, C, N)
            de_index: 路径索引 (B, N)，值越小优先级越高

        Returns:
            x_out: 重排后的特征 (B, N, C)
            indices: 排序索引 (B, N)
        """
        B, C, N = x.shape
        # 按索引值排序，获取排列顺序
        _, indices = torch.topk(de_index, k=N, dim=-1, largest=False)
        # 根据索引重排特征
        x_gathered = torch.gather(x, 2, indices.unsqueeze(1).expand(-1, C, -1)).contiguous()
        x_out = x_gathered.permute(0, 2, 1).contiguous()
        ctx.save_for_backward(x, de_index, indices)
        return x_out, indices

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor, grad_indices: torch.Tensor):
        """反向传播

        使用 scatter_add_ 将梯度分散回原始位置。
        """
        x, de_index, indices = ctx.saved_tensors
        grad_x = torch.zeros_like(x)
        # 将梯度分散回原始位置
        grad_x.scatter_add_(
            2,
            indices.unsqueeze(1).expand(-1, x.shape[1], -1),
            grad_output.permute(0, 2, 1).contiguous()
        ).contiguous()
        # 路径索引的梯度
        grad_de_index = (grad_output.permute(0, 2, 1).contiguous() - grad_x).mean(dim=1)
        grad_de_index = grad_de_index.view_as(de_index)
        return grad_x, grad_de_index


class ConvOffset(nn.Module):
    """偏移量预测网络

    使用深度可分离卷积 + 通道注意力预测 2D 空间偏移量和 1D 路径索引。

    Args:
        embed_dim: 输入通道数
        kernel_size: 卷积核大小
        padding: 填充大小
    """

    def __init__(self, embed_dim: int, kernel_size: int, padding: int):
        super().__init__()
        # 深度可分离卷积
        self.conv1 = nn.Conv2d(
            embed_dim, embed_dim, kernel_size, 1, padding, groups=embed_dim
        )
        # 通道注意力
        squeeze_dim = max(embed_dim // 16, 4)  # 确保至少有 4 个通道
        self.ca = nn.Sequential(
            nn.Linear(embed_dim, squeeze_dim),
            nn.GELU(),
            nn.Linear(squeeze_dim, embed_dim),
            nn.Sigmoid()
        )
        self.ln = nn.LayerNorm(embed_dim)
        self.gelu = nn.GELU()
        # 输出 3 通道：2 用于空间偏移，1 用于路径索引
        self.conv2 = nn.Conv2d(embed_dim, 3, 1, 1, 0, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            x: 输入特征 (B, C, H, W)

        Returns:
            offset: 偏移量和路径索引 (B, 3, H, W)
        """
        x1 = self.conv1(x)
        # 全局平均池化 -> 通道注意力
        x_c = F.adaptive_avg_pool2d(x, (1, 1))
        x_c = self.ca(x_c.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = x1 * x_c.expand_as(x)
        # 归一化 + 激活
        x = self.gelu(self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
        x = self.conv2(x)
        return x


class DeformableLayer(nn.Module):
    """可变形空间采样层

    根据预测的偏移量进行空间采样，并根据路径索引进行序列重排。
    支持阶段自适应卷积核（不同 stage 使用不同大小的卷积核）。

    Args:
        stage: 当前阶段索引 (0-3)，用于选择卷积核大小
        embed_dim: 特征维度
        debug: 是否开启调试模式
    """

    # 阶段自适应卷积核大小：浅层大核（大感受野），深层小核（局部精细）
    STAGE_KERNELS = [9, 7, 5, 3]

    def __init__(self, stage: int = 0, embed_dim: int = 192, debug: bool = False):
        super().__init__()
        self.stage = stage
        self.embed_dim = embed_dim
        self.debug = debug

        # 根据 stage 选择卷积核大小
        kk = self.STAGE_KERNELS[min(stage, len(self.STAGE_KERNELS) - 1)]
        pad_size = kk // 2 if kk != 1 else 0

        # 偏移量预测网络
        self.conv_offset = ConvOffset(embed_dim, kk, pad_size)

        # 可学习的相对位置编码表
        self.rpe_table = nn.Parameter(torch.zeros(embed_dim, 7, 7))
        trunc_normal_(self.rpe_table, std=0.01)

    @torch.no_grad()
    def _get_ref_points(
        self, H: int, W: int, B: int,
        dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        """生成参考点网格（用于空间采样）

        Args:
            H, W: 空间尺寸
            B: batch size
            dtype: 数据类型
            device: 设备

        Returns:
            ref: 参考点坐标 (B, H, W, 2)，归一化到 [-1, 1]
        """
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device),
            torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        # 归一化到 [-1, 1]
        ref[..., 1].div_(W - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B, -1, -1, -1)
        return ref

    @torch.no_grad()
    def _get_key_ref_points(
        self, H: int, W: int, B: int,
        dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        """生成 key 参考点（用于位置编码）"""
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0, H, H, dtype=dtype, device=device),
            torch.linspace(0, W, W, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B, -1, -1, -1)
        return ref

    @torch.no_grad()
    def _get_path_ref_points(
        self, N: int, B: int,
        dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        """生成路径参考点（用于序列重排）"""
        ref_path = torch.linspace(0.5, N - 0.5, N, dtype=dtype, device=device)
        ref_path.div_(N - 1.0).mul_(2.0).sub_(1.0)
        ref = ref_path[None, ...].expand(B, -1)
        return ref

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table', 'rpe_table'}

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播

        Args:
            x: 输入特征 (B, C, H, W)

        Returns:
            x: 重排后的特征 (B, N, C)
            indices: 排序索引 (B, N)
        """
        dtype, device = x.dtype, x.device
        B, C, H, W = x.size()
        N = H * W

        # 预测偏移量和路径索引
        offset = self.conv_offset(x).contiguous()  # (B, 3, H, W)
        offset, de_index = torch.split(offset, [2, 1], dim=1)
        Hk, Wk = offset.size(2), offset.size(3)

        # 限制偏移量范围
        offset_range = torch.tensor(
            [1.0 / (Hk - 1.0), 1.0 / (Wk - 1.0)],
            device=device
        ).reshape(1, 2, 1, 1)
        offset = offset.tanh().mul(offset_range)

        # 转换为采样坐标
        offset = offset.permute(0, 2, 3, 1).contiguous()  # (B, H, W, 2)
        reference = self._get_ref_points(Hk, Wk, B, dtype, device)

        # 处理路径索引
        de_index = de_index.tanh().flatten(1)  # (B, N)
        path_reference = self._get_path_ref_points(N, B, dtype, device)

        # 计算采样位置
        pos = offset + reference
        path_pos = de_index + path_reference

        # 空间可变形采样
        x_sampled = F.grid_sample(
            input=x,
            grid=pos[..., (1, 0)],  # y, x -> x, y
            mode='bilinear',
            align_corners=True
        )  # (B, C, H, W)

        # 相对位置编码
        rpe_table = self.rpe_table
        rpe_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
        rpe_bias = F.interpolate(
            rpe_bias, size=(H, W), mode='bilinear', align_corners=False
        )
        key_grid = self._get_key_ref_points(H, W, B, dtype, device)
        displacement = (key_grid - pos) * 0.5
        pos_bias = F.grid_sample(
            input=rpe_bias,
            grid=displacement[..., (1, 0)],
            mode='bilinear',
            align_corners=True
        )

        # 添加位置编码
        x = x_sampled + pos_bias
        x = x.flatten(2)  # (B, C, N)

        # 路径重排
        x, indices = DeformablePathTrans.apply(x, path_pos)  # (B, N, C)

        return x, indices


class DeformableLayerReverse(nn.Module):
    """可变形路径逆变换

    根据 DeformableLayer 输出的索引，将重排后的序列恢复到原始顺序。
    """

    def __init__(self):
        super().__init__()

    def forward(
        self, x: torch.Tensor, indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """逆变换

        Args:
            x: 输入特征 (B, C, H, W) 或 (B, C, N)
            indices: 排序索引 (B, N)

        Returns:
            x: 恢复顺序后的特征 (B, C, N)
        """
        if indices is None:
            return x.flatten(2) if x.dim() == 4 else x

        x = x.flatten(2)
        B, C, N = x.size()

        # 计算逆索引
        index_re = torch.zeros_like(indices, device=x.device)
        index_re.scatter_add_(
            1,
            indices,
            torch.arange(indices.size(-1), device=x.device).unsqueeze(0).expand(indices.size(0), -1)
        )

        # 根据逆索引恢复顺序
        x = torch.gather(x, 2, index_re.unsqueeze(1).expand(-1, C, -1))
        return x


# 导出接口
__all__ = [
    'DeformablePathTrans',
    'ConvOffset',
    'DeformableLayer',
    'DeformableLayerReverse',
]
