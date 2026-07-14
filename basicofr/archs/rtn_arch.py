import torch
from torch import nn

from basicsr.utils.registry import ARCH_REGISTRY

from .flow import FlowEstimator
from .components.arch_util import flow_warp
from .components.aggregation.gating import GatedAggregation
from .components.reconstruction import PSUpsample, ConvResBlock
from .spatial.swin_spatial import Swin_Spatial_2




class Video_Backbone(nn.Module):
    """RTN-Swin 骨干

    经典设计回顾：
    1. 光流对齐后进行双向传播，门控聚合控制隐藏状态更新。
    2. 每个方向均使用 Swin Transformer 作为空间修复模块。
    3. 拼接双向特征后通过恒等尺度的 Pixel-Shuffle 尾部生成输出，并添加全局残差。
    """

    def __init__(self, num_feat=16, num_block=6, flow_type='raft', **kwargs):
        super(Video_Backbone, self).__init__()
        self.num_feat = num_feat
        self.num_block = num_block

        # 光流对齐（支持 raft/spynet/memflow）
        self.flow_estimator = FlowEstimator(estimator_type=flow_type, normalization='tanh')

        # 双向 Swin 空间模块
        self.forward_resblocks = Swin_Spatial_2(embed_dim=64, depths=[2, 2, 2], num_heads=[4, 4, 4], mlp_ratio=2, in_chans=num_feat)
        self.backward_resblocks = Swin_Spatial_2(embed_dim=64, depths=[2, 2, 2], num_heads=[4, 4, 4], mlp_ratio=2, in_chans=num_feat)

        # 门控聚合（原始单尺度版本）
        self.Forward_Aggregation = GatedAggregation(hidden_channels=num_feat, kernel_size=3, padding=1)
        self.Backward_Aggregation = GatedAggregation(hidden_channels=num_feat, kernel_size=3, padding=1)

        # 特征融合与尾部
        self.concate = nn.Conv2d(num_feat * 2, num_feat, kernel_size=3, stride=1, padding=1, bias=True)
        self.up1 = PSUpsample(num_feat, num_feat, scale_factor=1)
        self.up2 = PSUpsample(num_feat, num_feat, scale_factor=1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, kernel_size=3, stride=1, padding=1)
        self.conv_last = nn.Conv2d(num_feat, 3, kernel_size=3, stride=1, padding=1)

        # 全局残差与激活
        self.img_up = nn.Upsample(scale_factor=1, mode='bilinear', align_corners=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def comp_flow(self, lrs):
        """估计双向光流并返回张量"""
        return self.flow_estimator.compute_flow(lrs)

    def forward(self, lrs):
        """Swin RTN 的双向传播主流程"""
        n, t, c, h, w = lrs.size()

        assert h >= 64 and w >= 64, (
            'The height and width of input should be at least 64, '
            f'but got {h} and {w}.'
        )

        forward_flow, backward_flow = self.comp_flow(lrs)

        # ========== 第一阶段：后向传播（无上采样）==========
        cached_feats = []
        feat_prop = lrs.new_zeros(n, self.num_feat, h, w)
        residual_indicator = lrs.new_zeros(n, 1, h, w)
        for i in range(t - 1, -1, -1):
            curr_lr = lrs[:, i, :, :, :]
            if i < t - 1:
                flow = backward_flow[:, i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
                pixel_prop = flow_warp(lrs[:, i + 1, :, :, :], flow.permute(0, 2, 3, 1))
                residual_indicator = torch.abs(pixel_prop[:, :1, :, :] - curr_lr[:, :1, :, :])
                aggregated = self.Backward_Aggregation(feat_prop, curr_lr, residual_indicator, head=False)
            else:
                aggregated = self.Backward_Aggregation(feat_prop, curr_lr, residual_indicator, head=True)

            feat_prop = self.backward_resblocks(aggregated)
            cached_feats.append(feat_prop)

        cached_feats = cached_feats[::-1]

        # ========== 第二阶段：前向传播（无上采样）==========
        feat_prop = torch.zeros_like(feat_prop)
        residual_indicator = torch.zeros_like(residual_indicator)
        outputs = []
        for i in range(t):
            curr_lr = lrs[:, i, :, :, :]
            if i > 0:
                flow = forward_flow[:, i - 1, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
                pixel_prop = flow_warp(lrs[:, i - 1, :, :, :], flow.permute(0, 2, 3, 1))
                residual_indicator = torch.abs(pixel_prop[:, :1, :, :] - curr_lr[:, :1, :, :])
                aggregated = self.Forward_Aggregation(feat_prop, curr_lr, residual_indicator, head=False)
            else:
                aggregated = self.Forward_Aggregation(feat_prop, curr_lr, residual_indicator, head=True)

            feat_prop = self.forward_resblocks(aggregated)

            # ========== 第三阶段：特征融合 + 输出 ==========
            cat_feat = torch.cat([cached_feats[i], feat_prop], dim=1)
            sr_rlt = self.lrelu(self.concate(cat_feat))
            sr_rlt = self.lrelu(self.up1(sr_rlt))
            sr_rlt = self.lrelu(self.up2(sr_rlt))
            sr_rlt = self.lrelu(self.conv_hr(sr_rlt))
            sr_rlt = self.conv_last(sr_rlt)

            base = self.img_up(curr_lr)
            outputs.append(torch.tanh(sr_rlt + base))

        return torch.stack(outputs, dim=1)

    def visualiza_mask(self, lrs):
        """可视化门控掩码（backward, forward）"""
        n, t, c, h, w = lrs.size()

        assert h >= 64 and w >= 64, (
            'The height and width of input should be at least 64, '
            f'but got {h} and {w}.'
        )

        forward_flow, backward_flow = self.comp_flow(lrs)

        # ========== 后向门控 ==========
        backward_mask = []
        feat_prop = lrs.new_zeros(n, self.num_feat, h, w)
        residual_indicator = lrs.new_zeros(n, 1, h, w)
        for i in range(t - 1, -1, -1):
            curr_lr = lrs[:, i, :, :, :]
            if i < t - 1:
                flow = backward_flow[:, i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
                pixel_prop = flow_warp(lrs[:, i + 1, :, :, :], flow.permute(0, 2, 3, 1))
                residual_indicator = torch.abs(pixel_prop[:, :1, :, :] - curr_lr[:, :1, :, :])
                temp, learned_mask = self.Backward_Aggregation(feat_prop, curr_lr, residual_indicator, head=False, return_mask=True)
                feat_prop = self.backward_resblocks(temp)
            else:
                temp, learned_mask = self.Backward_Aggregation(feat_prop, curr_lr, residual_indicator, head=True, return_mask=True)
                feat_prop = self.backward_resblocks(temp)
            backward_mask.append(learned_mask)
        backward_mask = backward_mask[::-1]

        # ========== 前向门控 ==========
        forward_mask = []
        feat_prop = torch.zeros_like(feat_prop)
        residual_indicator = torch.zeros_like(residual_indicator)
        for i in range(t):
            curr_lr = lrs[:, i, :, :, :]
            if i > 0:
                flow = forward_flow[:, i - 1, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
                pixel_prop = flow_warp(lrs[:, i - 1, :, :, :], flow.permute(0, 2, 3, 1))
                residual_indicator = torch.abs(pixel_prop[:, :1, :, :] - curr_lr[:, :1, :, :])
                temp, learned_mask = self.Forward_Aggregation(feat_prop, curr_lr, residual_indicator, head=False, return_mask=True)
                feat_prop = self.forward_resblocks(temp)
            else:
                temp, learned_mask = self.Forward_Aggregation(feat_prop, curr_lr, residual_indicator, head=True, return_mask=True)
                feat_prop = self.forward_resblocks(temp)
            forward_mask.append(learned_mask)

        return torch.stack(backward_mask, dim=1), torch.stack(forward_mask, dim=1)

    def visualiza_feature(self, lrs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """可视化特征能量（老电影修复诊断工具）

        计算双向传播中每帧的门控特征能量和隐藏状态能量，用于分析老电影修复效果。

        【老电影修复背景】
        老电影损伤（划痕、污点、闪烁）通常是时间稀疏的，不会连续多帧在同一位置都有损伤。
        双向传播从时间的"前后"两个方向寻找干净的参考信息来修复损伤帧。

        【能量计算原理】
        使用 L1 能量: abs().mean(dim=1)，避免正负抵消，反映真实激活强度。

        【Gate Energy 门控能量】
        - 高能量区域 = 门控正在积极决策 = 可能是损伤区域
        - 低能量区域 = 门控决策简单 = 干净区域

        【State Energy 状态能量】
        - 高能量区域 = 累积特征强 = 信息成功传播到此
        - 低能量区域 = 信息被抑制 = 该方向可能有损伤阻断

        【诊断场景】
        - 单帧划痕: 前向高、后向高 → 两个方向都能提供修复信息
        - 连续损伤: 前向低、后向高 → 过去帧也有损伤，主要依赖后向
        - 场景切换: 前向低、后向高 → 前一场景内容不同，应信任后向
        - 静态区域: 前向低、后向低 → 无需复杂传播

        Args:
            lrs: 输入序列 (B, T, C, H, W)

        Returns:
            backward_gate_energy: 后向门控特征能量 (B, T, 1, H, W)
            backward_state_energy: 后向隐藏状态能量 (B, T, 1, H, W)
            forward_gate_energy: 前向门控特征能量 (B, T, 1, H, W)
            forward_state_energy: 前向隐藏状态能量 (B, T, 1, H, W)
        """
        n, t, c, h, w = lrs.size()

        assert h >= 64 and w >= 64, (
            f'输入分辨率至少为 64x64，当前为 {h}x{w}'
        )

        forward_flow, backward_flow = self.comp_flow(lrs)

        # ========== 后向传播特征 ==========
        backward_gate_energies = []
        backward_state_energies = []
        feat_prop = lrs.new_zeros(n, self.num_feat, h, w)
        residual_indicator = lrs.new_zeros(n, 1, h, w)

        for i in range(t - 1, -1, -1):
            curr_lr = lrs[:, i, :, :, :]
            if i < t - 1:
                flow = backward_flow[:, i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
                pixel_prop = flow_warp(lrs[:, i + 1, :, :, :], flow.permute(0, 2, 3, 1))
                residual_indicator = torch.abs(pixel_prop[:, :1, :, :] - curr_lr[:, :1, :, :])
                temp, latent_feat = self.Backward_Aggregation(
                    feat_prop, curr_lr, residual_indicator,
                    head=False, return_mask=True, return_feat=True
                )
            else:
                temp, latent_feat = self.Backward_Aggregation(
                    feat_prop, curr_lr, residual_indicator,
                    head=True, return_mask=True, return_feat=True
                )

            feat_prop = self.backward_resblocks(temp)

            # 计算能量：使用 abs() 避免正负抵消
            gate_energy = latent_feat.abs().mean(dim=1, keepdim=True)  # (B, 1, H, W)
            state_energy = feat_prop.abs().mean(dim=1, keepdim=True)   # (B, 1, H, W)
            backward_gate_energies.append(gate_energy)
            backward_state_energies.append(state_energy)

        backward_gate_energies = backward_gate_energies[::-1]
        backward_state_energies = backward_state_energies[::-1]

        # ========== 前向传播特征 ==========
        forward_gate_energies = []
        forward_state_energies = []
        feat_prop = lrs.new_zeros(n, self.num_feat, h, w)
        residual_indicator = lrs.new_zeros(n, 1, h, w)

        for i in range(t):
            curr_lr = lrs[:, i, :, :, :]
            if i > 0:
                flow = forward_flow[:, i - 1, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
                pixel_prop = flow_warp(lrs[:, i - 1, :, :, :], flow.permute(0, 2, 3, 1))
                residual_indicator = torch.abs(pixel_prop[:, :1, :, :] - curr_lr[:, :1, :, :])
                temp, latent_feat = self.Forward_Aggregation(
                    feat_prop, curr_lr, residual_indicator,
                    head=False, return_mask=True, return_feat=True
                )
            else:
                temp, latent_feat = self.Forward_Aggregation(
                    feat_prop, curr_lr, residual_indicator,
                    head=True, return_mask=True, return_feat=True
                )

            feat_prop = self.forward_resblocks(temp)

            # 计算能量
            gate_energy = latent_feat.abs().mean(dim=1, keepdim=True)
            state_energy = feat_prop.abs().mean(dim=1, keepdim=True)
            forward_gate_energies.append(gate_energy)
            forward_state_energies.append(state_energy)

        backward_gate_energy = torch.stack(backward_gate_energies, dim=1)
        backward_state_energy = torch.stack(backward_state_energies, dim=1)
        forward_gate_energy = torch.stack(forward_gate_energies, dim=1)
        forward_state_energy = torch.stack(forward_state_energies, dim=1)

        return backward_gate_energy, backward_state_energy, forward_gate_energy, forward_state_energy
@ARCH_REGISTRY.register()
class RTNRestorationNet(Video_Backbone):
    """Swin 基础版 RTN，用作对比实验"""

    def __init__(self,
                 num_feat=16,
                 num_block=6,
                 flow_type='raft',
                 **kwargs):
        super().__init__(num_feat=num_feat, num_block=num_block, flow_type=flow_type, **kwargs)
