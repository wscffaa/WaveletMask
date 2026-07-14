"""递归多分支传播策略

RRTN 的传播模式：
  backward_1 → forward_1 → backward_2 → forward_2
  每个分支使用二阶可变形对齐，特征逐分支累积。

这是一个工具类描述，用于新 idea 参考。
"""

import torch
import torch.nn as nn

from ..arch_util import flow_warp


class RecursivePropagation(nn.Module):
    """递归多分支传播

    将多轮 backward/forward 传播组合为完整的递归传播策略。
    参考 RRTN/BasicVSR++ 的多分支设计。

    Args:
        num_feat: 特征通道数
        num_recursion: 递归轮数（默认 2 → 4 分支）
        deform_align: ModuleDict，各分支的对齐模块
        backbone: ModuleDict，各分支的空间修复模块
    """

    def __init__(self, num_feat, num_recursion=2, deform_align=None, backbone=None):
        super().__init__()
        self.num_feat = num_feat
        self.num_recursion = num_recursion
        self.deform_align = deform_align or nn.ModuleDict()
        self.backbone = backbone or nn.ModuleDict()

    def propagate(self, feats, flows, module_name):
        """单分支传播（与 RRTN.propagate 相同的接口）

        Args:
            feats: 特征字典
            flows: 光流 (B, T-1, 2, H, W)
            module_name: 分支名称

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

            if i > 0:
                flow_n1 = flows[:, flow_idx[i]]
                cond_n1 = flow_warp(feat_prop, flow_n1.permute(0, 2, 3, 1))

                feat_n2 = torch.zeros_like(feat_prop)
                flow_n2 = torch.zeros_like(flow_n1)
                cond_n2 = torch.zeros_like(cond_n1)

                if i > 1:
                    feat_n2 = feats[module_name][-2]
                    flow_n2 = flows[:, flow_idx[i - 1]]
                    flow_n2 = flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1))
                    cond_n2 = flow_warp(feat_n2, flow_n2.permute(0, 2, 3, 1))

                cond = torch.cat([cond_n1, feat_current, cond_n2], dim=1)
                feat_prop = torch.cat([feat_prop, feat_n2], dim=1)
                feat_prop = self.deform_align[module_name](feat_prop, cond, flow_n1, flow_n2)

            feat = [feat_current] + [
                feats[k][idx]
                for k in feats if k not in ['spatial', module_name]
            ] + [feat_prop]

            feat = torch.cat(feat, dim=1)
            feat_prop = feat_prop + self.backbone[module_name](feat)
            feats[module_name].append(feat_prop)

        if 'backward' in module_name:
            feats[module_name] = feats[module_name][::-1]

        return feats

    def forward(self, feats, flows_forward, flows_backward):
        """执行完整的递归多分支传播

        Args:
            feats: 特征字典，必须包含 'spatial' 键
            flows_forward: 前向光流
            flows_backward: 后向光流

        Returns:
            更新后的 feats 字典
        """
        for iter_ in range(1, self.num_recursion + 1):
            for direction in ['backward', 'forward']:
                module = f'{direction}_{iter_}'
                feats[module] = []
                flows = flows_backward if direction == 'backward' else flows_forward
                feats = self.propagate(feats, flows, module)
        return feats
