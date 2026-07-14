"""RRTN (Recursive Recurrent Transformer Network) — WACV 2024

Adapted from "Restoring Degraded Old Films with Recursive Recurrent
Transformer Networks" by Shan Lin et al.

Architecture: BasicVSR++ propagation + Swin spatial backbone + noise mask prior.
1. 二阶可变形对齐：联合利用 t-1 和 t-2 帧信息学习更精准的几何变换
2. 四分支递归传播：backward_1 → forward_1 → backward_2 → forward_2
3. 噪声掩码驱动：使用光流残差的几何均值作为显式噪声先验
4. 通道自适应：每个分支的空间模块通道数随轮次增加
"""

import torch
import torch.nn.functional as F
from torch import nn

from basicsr.utils.registry import ARCH_REGISTRY

from .components.arch_util import flow_warp
from .components.alignment.deform import SecondOrderDeformableAlignment
from .components.reconstruction import PixelShufflePack, ConvResBlock
from .flow.flow_estimator import FlowEstimator
from .spatial.swin_spatial import Swin_Spatial_2

class RRTNBackbone(nn.Module):
    """RRTN 视频骨干网络

    核心流程：
    1. 利用光流估计器计算前后向光流，生成噪声掩码（残差指示器）
    2. 构建四条双向传播分支，使用二阶可变形对齐和 Swin 空间修复
    3. 将多分支特征串联后通过 PixelShuffle 重建，并叠加全局残差

    Args:
        num_feat: 特征通道数
        num_block: 残差块数量
        input_channel: 输入通道数（1=灰度，3=RGB）
        input_size: 输入图像尺寸（用于 Swin 位置编码）
        num_recursion: 递归轮数（默认 2，即四分支）
        max_residue_magnitude: 可变形偏移的最大残差幅度
        downscale_first: 是否先下采样再处理
        flow_type: 光流估计器类型 (raft/spynet/memflow)
        cpu_cache_length: 超过此长度启用 CPU 缓存
    """

    def __init__(
        self,
        num_feat: int = 16,
        num_block: int = 6,
        input_channel: int = 1,
        input_size: int = 128,
        num_recursion: int = 2,
        max_residue_magnitude: int = 10,
        downscale_first: bool = False,
        flow_type: str = 'raft',
        cpu_cache_length: int = 100,
        use_backbone_proj: bool = True,
    ):
        super().__init__()
        self.num_feat = num_feat
        self.num_block = num_block
        self.input_channel = input_channel
        self.num_recursion = num_recursion
        self.downscale_first = downscale_first
        self.cpu_cache_length = cpu_cache_length
        self.use_backbone_proj = use_backbone_proj

        # ========== 光流估计器 ==========
        self.flow_estimator = FlowEstimator(
            estimator_type=flow_type,
            normalization='sigmoid'
        )

        # ========== 特征提取（输入包含噪声掩码通道） ==========
        self.feat_extract = ConvResBlock(input_channel + 1, num_feat, 5)

        # ========== 四分支传播模块 ==========
        self.deform_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()
        if self.use_backbone_proj:
            self.backbone_proj = nn.ModuleDict()
        modules = ['backward_1', 'forward_1', 'backward_2', 'forward_2']

        for i, module in enumerate(modules):
            # 二阶可变形对齐
            self.deform_align[module] = SecondOrderDeformableAlignment(
                2 * num_feat,
                num_feat,
                3,
                padding=1,
                deform_groups=16,
                max_residue_magnitude=max_residue_magnitude
            )
            # Swin 空间修复（通道数随分支递增），追加投影层映射回 num_feat
            self.backbone[module] = Swin_Spatial_2(
                img_size=input_size,
                embed_dim=64,
                depths=[2, 2, 2],
                num_heads=[4, 4, 4],
                mlp_ratio=2,
                in_chans=(2 + i) * num_feat,
                out_chans=None if self.use_backbone_proj else num_feat
            )
            if self.use_backbone_proj:
                self.backbone_proj[module] = nn.Conv2d((2 + i) * num_feat, num_feat, 1, 1, 0)

        # ========== 重建模块 ==========
        self.reconstruction = ConvResBlock(5 * num_feat, num_feat, 5)
        scale_factor = 1
        self.upsample1 = PixelShufflePack(num_feat, num_feat, scale_factor)
        self.upsample2 = PixelShufflePack(num_feat, num_feat, scale_factor)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, input_channel, 3, 1, 1)
        self.img_upsample = nn.Identity()
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.cpu_cache = False

    def compute_flow(self, lqs: torch.Tensor):
        """估计双向光流

        Args:
            lqs: 输入序列 (B, T, C, H, W)

        Returns:
            flows_forward: 前向光流 (B, T-1, 2, H, W)
            flows_backward: 后向光流 (B, T-1, 2, H, W)
        """
        n, t, c, h, w = lqs.size()
        # 单通道扩展为三通道（光流估计器需要）
        if c == 1:
            lqs = lqs.repeat(1, 1, 3, 1, 1)
        return self.flow_estimator.compute_flow(lqs)

    def comp_flow(self, lqs: torch.Tensor):
        """compute_flow 的别名，兼容 debug 脚本"""
        return self.compute_flow(lqs)

    def compute_noise_mask(self, lqs: torch.Tensor, flows_forward: torch.Tensor, flows_backward: torch.Tensor):
        """计算噪声掩码（残差指示器）

        使用光流对齐后的残差几何均值作为噪声先验：
        noise_mask = sqrt(|warped_next - curr| * |warped_prev - curr|)

        Args:
            lqs: 输入序列 (B, T, C, H, W)
            flows_forward: 前向光流
            flows_backward: 后向光流

        Returns:
            noise_mask: 噪声掩码 (B, T, 1, H, W)
        """
        n, t, c, h, w = lqs.size()
        residual_indicators = lqs.new_zeros(n, t, 2, h, w)

        for i in range(t):
            lq_current = lqs[:, i, :, :, :]

            # backward 方向：对齐 frame[i+1] 到 frame[i]
            if i < t - 1:
                flow = flows_backward[:, i, :, :, :]
                warped_lq_next = flow_warp(lqs[:, i + 1, :, :, :], flow.permute(0, 2, 3, 1))
            else:
                # 边界帧处理
                flow = flows_backward[:, -1, :, :, :]
                warped_lq_next = flow_warp(lqs[:, -3, :, :, :], flow.permute(0, 2, 3, 1))
            residual_indicators[:, i, 0, :, :] = torch.abs(warped_lq_next[:, 0, :, :] - lq_current[:, 0, :, :])

            # forward 方向：对齐 frame[i-1] 到 frame[i]
            if i > 0:
                flow = flows_forward[:, i - 1, :, :, :]
                warped_lq_previous = flow_warp(lqs[:, i - 1, :, :, :], flow.permute(0, 2, 3, 1))
            else:
                # 边界帧处理
                flow = flows_forward[:, 0, :, :, :]
                warped_lq_previous = flow_warp(lqs[:, 2, :, :, :], flow.permute(0, 2, 3, 1))
            residual_indicators[:, i, 1, :, :] = torch.abs(warped_lq_previous[:, 0, :, :] - lq_current[:, 0, :, :])

        # 几何均值作为噪声掩码
        noise_mask = torch.sqrt(residual_indicators[:, :, :1, :, :] * residual_indicators[:, :, 1:, :, :])
        return noise_mask

    def propagate(self, feats: dict, flows: torch.Tensor, module_name: str):
        """双向时域传播与二阶对齐

        Args:
            feats: 特征字典，包含 'spatial' 和各分支特征
            flows: 光流 (B, T-1, 2, H, W)
            module_name: 分支名称 (backward_1/forward_1/backward_2/forward_2)

        Returns:
            更新后的 feats 字典
        """
        n, t, _, h, w = flows.size()

        frame_idx = list(range(0, t + 1))
        flow_idx = list(range(-1, t))
        mapping_idx = list(range(0, len(feats['spatial'])))
        mapping_idx += mapping_idx[::-1]

        if 'backward' in module_name:
            frame_idx = frame_idx[::-1]
            flow_idx = frame_idx

        feat_prop = flows.new_zeros(n, self.num_feat, h, w)

        for i, idx in enumerate(frame_idx):
            feat_current = feats['spatial'][mapping_idx[idx]]
            if self.cpu_cache:
                feat_current = feat_current.cuda()
                feat_prop = feat_prop.cuda()

            if i > 0:
                flow_n1 = flows[:, flow_idx[i], :, :, :]
                if self.cpu_cache:
                    flow_n1 = flow_n1.cuda()

                cond_n1 = flow_warp(feat_prop, flow_n1.permute(0, 2, 3, 1))

                feat_n2 = torch.zeros_like(feat_prop)
                flow_n2 = torch.zeros_like(flow_n1)
                cond_n2 = torch.zeros_like(cond_n1)

                # 二阶对齐：使用 t-2 帧信息
                if i > 1:
                    feat_n2 = feats[module_name][-2]
                    if self.cpu_cache:
                        feat_n2 = feat_n2.cuda()

                    flow_n2 = flows[:, flow_idx[i - 1], :, :, :]
                    if self.cpu_cache:
                        flow_n2 = flow_n2.cuda()

                    # 光流累积
                    flow_n2 = flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1))
                    cond_n2 = flow_warp(feat_n2, flow_n2.permute(0, 2, 3, 1))

                cond = torch.cat([cond_n1, feat_current, cond_n2], dim=1)
                feat_prop = torch.cat([feat_prop, feat_n2], dim=1)
                feat_prop = self.deform_align[module_name](feat_prop, cond, flow_n1, flow_n2)

            # 聚合所有分支特征
            feat = [feat_current] + [
                feats[k][idx]
                for k in feats if k not in ['spatial', module_name]
            ] + [feat_prop]
            if self.cpu_cache:
                feat = [f.cuda() for f in feat]

            feat = torch.cat(feat, dim=1)
            spatial_feat = self.backbone[module_name](feat)
            if self.use_backbone_proj:
                spatial_feat = self.backbone_proj[module_name](spatial_feat)
            feat_prop = feat_prop + spatial_feat
            feats[module_name].append(feat_prop)

            if self.cpu_cache:
                feats[module_name][-1] = feats[module_name][-1].cpu()
                torch.cuda.empty_cache()

        if 'backward' in module_name:
            feats[module_name] = feats[module_name][::-1]

        return feats

    def upsample(self, lqs: torch.Tensor, feats: dict):
        """PixelShuffle 重建高清序列

        Args:
            lqs: 输入序列 (B, T, C, H, W)
            feats: 特征字典

        Returns:
            outputs: 输出序列 (B, T, C, H, W)
        """
        outputs = []
        num_outputs = len(feats['spatial'])
        mapping_idx = list(range(0, num_outputs))
        mapping_idx += mapping_idx[::-1]

        for i in range(0, lqs.size(1)):
            hr = [feats[k].pop(0) for k in feats if k != 'spatial']
            hr.insert(0, feats['spatial'][mapping_idx[i]])
            hr = torch.cat(hr, dim=1)
            if self.cpu_cache:
                hr = hr.cuda()

            hr = self.reconstruction(hr)
            hr = self.lrelu(self.upsample1(hr))
            hr = self.lrelu(self.upsample2(hr))
            hr = self.lrelu(self.conv_hr(hr))
            hr = self.conv_last(hr)

            # 全局残差连接 + tanh（与 RTN/MambaOFR 保持一致）
            if not self.downscale_first:
                hr = torch.tanh(hr + self.img_upsample(lqs[:, i, :, :, :]))
            else:
                hr = torch.tanh(hr + lqs[:, i, :, :, :])

            if self.cpu_cache:
                hr = hr.cpu()
                torch.cuda.empty_cache()

            outputs.append(hr)

        return torch.stack(outputs, dim=1)

    def visualiza_mask(self, lqs: torch.Tensor):
        """可视化噪声掩码"""
        n, t, c, h, w = lqs.size()

        if not self.downscale_first:
            lqs_downsample = lqs.clone()
        else:
            lqs_downsample = F.interpolate(
                lqs.view(-1, c, h, w), scale_factor=0.25,
                mode='bicubic').view(n, t, c, h // 4, w // 4)

        flows_forward, flows_backward = self.compute_flow(lqs_downsample)
        noise_mask = self.compute_noise_mask(lqs, flows_forward, flows_backward)
        return noise_mask, noise_mask

    def visualiza_feature(self, lqs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """可视化特征能量（老电影修复诊断工具）

        计算双向传播中每帧的门控特征能量和隐藏状态能量，用于分析老电影修复效果。

        【老电影修复背景】
        老电影损伤（划痕、污点、闪烁）通常是时间稀疏的，不会连续多帧在同一位置都有损伤。
        双向传播从时间的"前后"两个方向寻找干净的参考信息来修复损伤帧。

        【能量计算原理】
        使用 L1 能量: abs().mean(dim=1)，避免正负抵消，反映真实激活强度。

        【Gate Energy 门控能量】（RRTN 使用空间特征代替）
        - 高能量区域 = 当前帧空间特征激活强 = 局部信息丰富
        - 低能量区域 = 空间特征激活弱 = 需要时域传播补充

        【State Energy 状态能量】
        - 高能量区域 = 累积特征强 = 信息成功传播到此
        - 低能量区域 = 信息被抑制 = 该方向可能有损伤阻断

        【诊断场景】
        - 单帧划痕: 前向高、后向高 → 两个方向都能提供修复信息
        - 连续损伤: 前向低、后向高 → 过去帧也有损伤，主要依赖后向
        - 场景切换: 前向低、后向高 → 前一场景内容不同，应信任后向
        - 静态区域: 前向低、后向低 → 无需复杂传播

        Note:
            RRTN 架构基于 BasicVSR++，没有显式的门控模块。
            这里用空间特征能量替代 gate_energy，传播特征能量作为 state_energy。

        Args:
            lqs: 输入序列 (B, T, C, H, W)

        Returns:
            backward_gate_energy: 后向空间特征能量 (B, T, 1, H, W)
            backward_state_energy: 后向传播特征能量 (B, T, 1, H, W)
            forward_gate_energy: 前向空间特征能量 (B, T, 1, H, W)
            forward_state_energy: 前向传播特征能量 (B, T, 1, H, W)
        """
        n, t, c, h, w = lqs.size()

        if not self.downscale_first:
            lqs_downsample = lqs.clone()
        else:
            lqs_downsample = F.interpolate(
                lqs.view(-1, c, h, w), scale_factor=0.25,
                mode='bicubic').view(n, t, c, h // 4, w // 4)

        flows_forward, flows_backward = self.compute_flow(lqs_downsample)
        noise_mask = self.compute_noise_mask(lqs, flows_forward, flows_backward)
        lqs_with_noise_mask = torch.cat([lqs, noise_mask], dim=2)

        feats = {}
        feats_ = self.feat_extract(lqs_with_noise_mask.view(-1, c + 1, h, w))
        h_, w_ = feats_.shape[2:]
        feats_ = feats_.view(n, t, -1, h_, w_)
        feats['spatial'] = [feats_[:, i, :, :, :] for i in range(0, t)]

        # 只运行第一轮双向传播
        for direction in ['backward', 'forward']:
            module = f'{direction}_1'
            feats[module] = []
            flows = flows_backward if direction == 'backward' else flows_forward
            feats = self.propagate(feats, flows, module)

        # 计算能量：使用 abs() 避免正负抵消
        # gate_energy: 使用空间特征（当前帧提取的特征）
        # state_energy: 使用传播后的特征
        backward_gate_energy = torch.stack(
            [f.abs().mean(dim=1, keepdim=True) for f in feats['spatial']], dim=1
        )
        backward_state_energy = torch.stack(
            [f.abs().mean(dim=1, keepdim=True) for f in feats['backward_1']], dim=1
        )
        forward_gate_energy = torch.stack(
            [f.abs().mean(dim=1, keepdim=True) for f in feats['spatial']], dim=1
        )
        forward_state_energy = torch.stack(
            [f.abs().mean(dim=1, keepdim=True) for f in feats['forward_1']], dim=1
        )

        return backward_gate_energy, backward_state_energy, forward_gate_energy, forward_state_energy

    def forward(self, lqs: torch.Tensor) -> torch.Tensor:
        """前向传播

        Args:
            lqs: 输入序列 (B, T, C, H, W)

        Returns:
            outputs: 输出序列 (B, T, C, H, W)
        """
        n, t, c, h, w = lqs.size()

        # 长序列启用 CPU 缓存
        if t > self.cpu_cache_length and lqs.is_cuda:
            self.cpu_cache = True
        else:
            self.cpu_cache = False

        if not self.downscale_first:
            lqs_downsample = lqs.clone()
        else:
            lqs_downsample = F.interpolate(
                lqs.view(-1, c, h, w), scale_factor=0.25,
                mode='bicubic').view(n, t, c, h // 4, w // 4)

        # ========== 阶段一：光流与噪声掩码 ==========
        assert lqs_downsample.size(3) >= 64 and lqs_downsample.size(4) >= 64, (
            f'Input size must be at least 64x64, but got {h}x{w}.')
        flows_forward, flows_backward = self.compute_flow(lqs_downsample)
        noise_mask = self.compute_noise_mask(lqs, flows_forward, flows_backward)
        lqs_with_noise_mask = torch.cat([lqs, noise_mask], dim=2)

        # ========== 阶段二：空间特征提取 ==========
        feats = {}
        if self.cpu_cache:
            feats['spatial'] = []
            for i in range(0, t):
                feat = self.feat_extract(lqs_with_noise_mask[:, i, :, :, :]).cpu()
                feats['spatial'].append(feat)
                torch.cuda.empty_cache()
        else:
            feats_ = self.feat_extract(lqs_with_noise_mask.view(-1, c + 1, h, w))
            h_, w_ = feats_.shape[2:]
            feats_ = feats_.view(n, t, -1, h_, w_)
            feats['spatial'] = [feats_[:, i, :, :, :] for i in range(0, t)]

        # ========== 阶段三：四分支递归传播 ==========
        for iter_ in [1, 2]:
            for direction in ['backward', 'forward']:
                module = f'{direction}_{iter_}'
                feats[module] = []

                if direction == 'backward':
                    flows = flows_backward
                elif flows_forward is not None:
                    flows = flows_forward
                else:
                    flows = flows_backward.flip(1)

                feats = self.propagate(feats, flows, module)
                if self.cpu_cache:
                    del flows
                    torch.cuda.empty_cache()

        # ========== 阶段四：重建输出 ==========
        return self.upsample(lqs, feats)


@ARCH_REGISTRY.register()
class RRTNRestorationNet(RRTNBackbone):
    """RRTN 修复网络（BasicSR 注册入口）

    保持与 RRTN 的参数接口兼容。
    """

    def __init__(self, flow_type: str = 'raft', **kwargs):
        super().__init__(flow_type=flow_type, **kwargs)


@ARCH_REGISTRY.register()
class RRTNRestorationNetLegacy(RRTNBackbone):
    """Checkpoint-compatible RRTN variant for converted baseline weights."""

    def __init__(self, flow_type: str = 'raft', **kwargs):
        super().__init__(flow_type=flow_type, use_backbone_proj=False, **kwargs)
