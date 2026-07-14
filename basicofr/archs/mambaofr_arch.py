"""MambaOFR: 基于 Mamba 状态空间模型的老电影修复网络

RTN 的 Mamba 变体，使用 MambaIR 替换 Swin 作为空间修复模块，
并增加 DCNv2 流引导对齐和 pre_mask 传播机制。

核心特点：
1. MambaIR 空间修复：利用 SSM 的线性复杂度处理长程依赖
2. DCNv2 流引导对齐：在光流 warp 后进行局部形变校正
3. pre_mask 传播：跨帧传播门控掩码以增强时序一致性
4. 双向 RNN 传播：backward → forward 两阶段特征传播

Ref:
    - MambaIR: A Simple Baseline for Image Restoration with State Space Model
    - BasicVSR++: Improving Video Super-Resolution with Enhanced Propagation and Alignment
"""

from functools import lru_cache
import warnings

import torch
import torch.nn as nn

from basicsr.utils.registry import ARCH_REGISTRY

from .flow import FlowEstimator
from .spatial.mamba import MambaIR
from .components import flow_warp, GatedAggregationDCN
from .components.reconstruction import PSUpsample


@lru_cache(maxsize=1)
def _load_optional_paf():
    try:
        from basicofr.archs.ideas._archive.scsegamba.components.paf import PAF
    except ImportError:
        warnings.warn(
            "`fusion_type='paf'` 不可用，回退到 baseline `concat` 融合。",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    return PAF

@ARCH_REGISTRY.register()
class MambaOFRNet(nn.Module):
    """MambaOFR 视频修复网络

    Args:
        num_feat: 特征通道数，默认 16
        num_block: 残差块数量（未使用，保持接口兼容）
        flow_type: 光流估计器类型，'raft'/'spynet'/'memflow'
        mamba_embed_dim: MambaIR 嵌入维度
        mamba_depths: MambaIR 各层深度
        mamba_d_state: MambaIR 状态维度
        mamba_mlp_ratio: MambaIR MLP 扩展比
        mamba_drop_path: MambaIR DropPath 率
    """

    def __init__(
        self,
        num_feat: int = 16,
        num_block: int = 6,
        flow_type: str = 'raft',
        mamba_embed_dim: int = 64,
        mamba_depths: list = None,
        mamba_d_state: int = 16,
        mamba_mlp_ratio: float = 1.2,
        mamba_drop_path: float = 0.1,  # 原始 MambaOFR/MambaIR 默认值
        fusion_type: str = "concat",
        scan_mode: str = "standard",
        ffn_type: str = "cab",
        use_checkpoint: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.num_feat = num_feat
        self.num_block = num_block
        self.fusion_type = fusion_type
        self.paf_module = None
        mamba_kwargs = {
            key: kwargs.pop(key)
            for key in ("scan_len", "shift_len", "use_shuffle_attn", "nss_use_shift")
            if key in kwargs
        }

        # ========== 光流估计器 ==========
        self.flow_estimator = FlowEstimator(estimator_type=flow_type, normalization='tanh')

        # ========== 双向 MambaIR 空间修复 ==========
        if mamba_depths is None:
            mamba_depths = [2, 2, 2]
        self.forward_resblocks = MambaIR(
            embed_dim=mamba_embed_dim,
            depths=mamba_depths,
            d_state=mamba_d_state,
            mlp_ratio=mamba_mlp_ratio,
            in_chans=num_feat,
            drop_rate=0.0, drop_path_rate=mamba_drop_path,
            scan_mode=scan_mode,
            ffn_type=ffn_type,
            use_checkpoint=use_checkpoint,
            **mamba_kwargs,
        )
        self.backward_resblocks = MambaIR(
            embed_dim=mamba_embed_dim,
            depths=mamba_depths,
            d_state=mamba_d_state,
            mlp_ratio=mamba_mlp_ratio,
            in_chans=num_feat,
            drop_rate=0.0, drop_path_rate=mamba_drop_path,
            scan_mode=scan_mode,
            ffn_type=ffn_type,
            use_checkpoint=use_checkpoint,
            **mamba_kwargs,
        )

        # ========== 门控聚合（带 DCN 对齐） ==========
        self.Forward_Aggregation = GatedAggregationDCN(hidden_channels=num_feat, kernel_size=3, padding=1)
        self.Backward_Aggregation = GatedAggregationDCN(hidden_channels=num_feat, kernel_size=3, padding=1)

        # ========== PAF 融合模块（可选） ==========
        if fusion_type == "paf":
            paf_cls = _load_optional_paf()
            if paf_cls is not None:
                self.paf_module = paf_cls(in_channels=num_feat, mid_channels=max(8, num_feat // 4))

        # ========== 特征融合与重建 ==========
        self.concate = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=True)
        self.up1 = PSUpsample(num_feat, num_feat, scale_factor=1)
        self.up2 = PSUpsample(num_feat, num_feat, scale_factor=1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, 3, 3, 1, 1)

        # ========== 全局残差 ==========
        self.img_up = nn.Upsample(scale_factor=1, mode='bilinear', align_corners=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def comp_flow(self, lrs: torch.Tensor):
        """计算双向光流"""
        return self.flow_estimator.compute_flow(lrs)

    def forward(self, lrs: torch.Tensor) -> torch.Tensor:
        """RNN 双向传播主流程

        Args:
            lrs: 输入低分辨率序列 (B, T, C, H, W)

        Returns:
            修复后的序列 (B, T, C, H, W)
        """
        n, t, c, h, w = lrs.size()

        assert h >= 64 and w >= 64, (
            f'输入分辨率至少为 64x64，当前为 {h}x{w}'
        )

        forward_flow, backward_flow = self.comp_flow(lrs)

        # ========== 第一阶段：后向传播 ==========
        cached_feats = []
        feat_prop = lrs.new_zeros(n, self.num_feat, h, w)
        residual_indicator = lrs.new_zeros(n, 1, h, w)
        pre_mask = residual_indicator.new_ones(residual_indicator.shape)

        for i in range(t - 1, -1, -1):
            curr_lr = lrs[:, i, :, :, :]
            if i < t - 1:
                flow = backward_flow[:, i, :, :, :]
                feat_prop_o = feat_prop
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))

                pixel_prop = flow_warp(lrs[:, i + 1, :, :, :], flow.permute(0, 2, 3, 1))
                residual_indicator = torch.abs(pixel_prop[:, :1, :, :] - curr_lr[:, :1, :, :])
                pre_mask = flow_warp(pre_mask, flow.permute(0, 2, 3, 1))

                feat_prop, pre_mask = self.Backward_Aggregation(
                    feat_prop, curr_lr, residual_indicator,
                    feat_prop_o, flow, pre_mask=pre_mask, head=False
                )
            else:
                feat_prop, pre_mask = self.Backward_Aggregation(
                    feat_prop, curr_lr, residual_indicator,
                    pre_mask=pre_mask, head=True
                )

            feat_prop = self.backward_resblocks(feat_prop)
            cached_feats.append(feat_prop)

        cached_feats = cached_feats[::-1]

        # ========== 第二阶段：前向传播 ==========
        outputs = []
        feat_prop = lrs.new_zeros(n, self.num_feat, h, w)
        residual_indicator = lrs.new_zeros(n, 1, h, w)
        pre_mask = residual_indicator.new_ones(residual_indicator.shape)

        for i in range(t):
            curr_lr = lrs[:, i, :, :, :]
            if i > 0:
                flow = forward_flow[:, i - 1, :, :, :]
                feat_prop_o = feat_prop
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))

                pixel_prop = flow_warp(lrs[:, i - 1, :, :, :], flow.permute(0, 2, 3, 1))
                residual_indicator = torch.abs(pixel_prop[:, :1, :, :] - curr_lr[:, :1, :, :])
                pre_mask = flow_warp(pre_mask, flow.permute(0, 2, 3, 1))

                feat_prop, pre_mask = self.Forward_Aggregation(
                    feat_prop, curr_lr, residual_indicator,
                    feat_prop_o, flow, pre_mask=pre_mask, head=False
                )
            else:
                feat_prop, pre_mask = self.Forward_Aggregation(
                    feat_prop, curr_lr, residual_indicator,
                    pre_mask=pre_mask, head=True
                )

            feat_prop = self.forward_resblocks(feat_prop)

            # ========== 第三阶段：特征融合 + 重建 ==========
            backward_feat = cached_feats[i]
            forward_feat = feat_prop

            # PAF 预融合（可选）
            if self.paf_module is not None:
                backward_feat = self.paf_module(backward_feat, forward_feat)
                forward_feat = self.paf_module(forward_feat, backward_feat)

            cat_feat = torch.cat([backward_feat, forward_feat], dim=1)
            sr_rlt = self.lrelu(self.concate(cat_feat))
            sr_rlt = self.lrelu(self.up1(sr_rlt))
            sr_rlt = self.lrelu(self.up2(sr_rlt))
            sr_rlt = self.lrelu(self.conv_hr(sr_rlt))
            sr_rlt = self.conv_last(sr_rlt)

            base = self.img_up(curr_lr)
            outputs.append(torch.tanh(sr_rlt + base))

        return torch.stack(outputs, dim=1)

    # ============================== 可视化接口 ==============================

    def visualiza_mask(self, lrs: torch.Tensor):
        """返回门控掩码用于可视化（命名保持与 RTN 一致供 Model 调用）

        Args:
            lrs: 输入序列 (B, T, C, H, W)

        Returns:
            (backward_mask, forward_mask): 各帧的门控掩码
        """
        n, t, c, h, w = lrs.size()
        forward_flow, backward_flow = self.comp_flow(lrs)

        # 后向掩码
        backward_mask = []
        feat_prop = lrs.new_zeros(n, self.num_feat, h, w)
        residual_indicator = lrs.new_zeros(n, 1, h, w)
        pre_mask = residual_indicator.new_ones(residual_indicator.shape)

        for i in range(t - 1, -1, -1):
            curr_lr = lrs[:, i, :, :, :]
            if i < t - 1:
                flow = backward_flow[:, i, :, :, :]
                feat_prop_o = feat_prop  # 保存 warp 前的特征用于 DCN 对齐
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
                pixel_prop = flow_warp(lrs[:, i + 1, :, :, :], flow.permute(0, 2, 3, 1))
                residual_indicator = torch.abs(pixel_prop[:, :1, :, :] - curr_lr[:, :1, :, :])
                pre_mask = flow_warp(pre_mask, flow.permute(0, 2, 3, 1))
                temp, learned_mask = self.Backward_Aggregation(
                    feat_prop, curr_lr, residual_indicator,
                    feat_prop_o, flow, pre_mask=pre_mask, head=False, return_mask=True
                )
                pre_mask = learned_mask
            else:
                temp, learned_mask = self.Backward_Aggregation(
                    feat_prop, curr_lr, residual_indicator,
                    pre_mask=pre_mask, head=True, return_mask=True
                )
                pre_mask = learned_mask
            feat_prop = self.backward_resblocks(temp)
            backward_mask.append(learned_mask)
        backward_mask = backward_mask[::-1]

        # 前向掩码
        forward_mask = []
        feat_prop = lrs.new_zeros(n, self.num_feat, h, w)
        residual_indicator = lrs.new_zeros(n, 1, h, w)
        pre_mask = residual_indicator.new_ones(residual_indicator.shape)

        for i in range(t):
            curr_lr = lrs[:, i, :, :, :]
            if i > 0:
                flow = forward_flow[:, i - 1, :, :, :]
                feat_prop_o = feat_prop  # 保存 warp 前的特征用于 DCN 对齐
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
                pixel_prop = flow_warp(lrs[:, i - 1, :, :, :], flow.permute(0, 2, 3, 1))
                residual_indicator = torch.abs(pixel_prop[:, :1, :, :] - curr_lr[:, :1, :, :])
                pre_mask = flow_warp(pre_mask, flow.permute(0, 2, 3, 1))
                temp, learned_mask = self.Forward_Aggregation(
                    feat_prop, curr_lr, residual_indicator,
                    feat_prop_o, flow, pre_mask=pre_mask, head=False, return_mask=True
                )
                pre_mask = learned_mask
            else:
                temp, learned_mask = self.Forward_Aggregation(
                    feat_prop, curr_lr, residual_indicator,
                    pre_mask=pre_mask, head=True, return_mask=True
                )
                pre_mask = learned_mask
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
        pre_mask = residual_indicator.new_ones(residual_indicator.shape)

        for i in range(t - 1, -1, -1):
            curr_lr = lrs[:, i, :, :, :]
            if i < t - 1:
                flow = backward_flow[:, i, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
                pixel_prop = flow_warp(lrs[:, i + 1, :, :, :], flow.permute(0, 2, 3, 1))
                residual_indicator = torch.abs(pixel_prop[:, :1, :, :] - curr_lr[:, :1, :, :])
                pre_mask = flow_warp(pre_mask, flow.permute(0, 2, 3, 1))
                temp, learned_mask, latent_feat = self.Backward_Aggregation(
                    feat_prop, curr_lr, residual_indicator,
                    pre_mask=pre_mask, head=False, return_mask=True, return_feat=True
                )
                pre_mask = learned_mask
            else:
                temp, learned_mask, latent_feat = self.Backward_Aggregation(
                    feat_prop, curr_lr, residual_indicator,
                    pre_mask=pre_mask, head=True, return_mask=True, return_feat=True
                )
                pre_mask = learned_mask

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
        pre_mask = residual_indicator.new_ones(residual_indicator.shape)

        for i in range(t):
            curr_lr = lrs[:, i, :, :, :]
            if i > 0:
                flow = forward_flow[:, i - 1, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
                pixel_prop = flow_warp(lrs[:, i - 1, :, :, :], flow.permute(0, 2, 3, 1))
                residual_indicator = torch.abs(pixel_prop[:, :1, :, :] - curr_lr[:, :1, :, :])
                pre_mask = flow_warp(pre_mask, flow.permute(0, 2, 3, 1))
                temp, learned_mask, latent_feat = self.Forward_Aggregation(
                    feat_prop, curr_lr, residual_indicator,
                    pre_mask=pre_mask, head=False, return_mask=True, return_feat=True
                )
                pre_mask = learned_mask
            else:
                temp, learned_mask, latent_feat = self.Forward_Aggregation(
                    feat_prop, curr_lr, residual_indicator,
                    pre_mask=pre_mask, head=True, return_mask=True, return_feat=True
                )
                pre_mask = learned_mask

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
