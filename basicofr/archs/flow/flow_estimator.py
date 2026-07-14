import argparse
import math
import os
import torch
import torch.nn.functional as F

# 项目根目录 (从 basicofr/archs/flow/ 向上 3 级)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

if __name__ == '__main__':
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
    from basicofr.archs.flow.raft import RAFT
    from basicofr.archs.flow.spynet import load_spynet
else:
    from .raft import RAFT
    from .spynet import load_spynet


def load_raft(pretrained_path=None, device=None):
    """Load RAFT optical flow network with provided checkpoint path."""
    opts = argparse.Namespace()
    if pretrained_path is None:
        pretrained_path = os.path.join(_PROJECT_ROOT, 'pretrained_model/flow/raft-sintel.pth')
    opts.model = pretrained_path
    opts.dataset = None
    opts.small = False
    opts.mixed_precision = False
    opts.alternate_corr = False

    state_dict = torch.load(opts.model, map_location='cpu', weights_only=True)
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    if isinstance(state_dict, dict):
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            if key.startswith('module.'):
                new_key = key[len('module.'):]
            cleaned_state_dict[new_key] = value
        state_dict = cleaned_state_dict
    model = RAFT(opts)
    model.load_state_dict(state_dict, strict=False)
    if device is not None:
        model = model.to(device)
    model.eval()
    return model


def _memory_readout(query_key, mem_key, mem_value, scale, train_avg_length):
    """MemFlow memory readout（参考 MemoryManager.match_memory）。

    只使用已存储的 memory（不传 current key/value），
    与 inference_core_predict.py 保持一致。

    Args:
        query_key: (B, C, L) 当前帧 query
        mem_key: (B, C, L_mem) 已存储 memory key，或 None
        mem_value: (B, C, L_mem) 已存储 memory value，或 None
        scale: attention scale factor
        train_avg_length: 训练时的平均序列长度

    Returns:
        readout: (B, C, L) 或 0（无 memory 时）
    """
    if mem_key is None or mem_value is None:
        return 0

    scaled = scale * math.log(max(mem_key.shape[-1], 2), max(train_avg_length, 2))

    try:
        from flash_attn import flash_attn_func
        if query_key.is_cuda:
            q = query_key.permute(0, 2, 1).unsqueeze(2).to(torch.bfloat16)
            k = mem_key.permute(0, 2, 1).unsqueeze(2).to(torch.bfloat16)
            v = mem_value.permute(0, 2, 1).unsqueeze(2).to(torch.bfloat16)
            readout = flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=scaled, causal=False)
            readout = readout.squeeze(2).permute(0, 2, 1).to(query_key.dtype)
            return readout
    except (ImportError, RuntimeError):
        pass

    # Standard attention fallback
    sim = torch.einsum('b c l, b c t -> b t l', query_key * scaled, mem_key)
    affinity = sim.softmax(dim=-1)
    readout = mem_value @ affinity
    return readout


class FlowEstimator:
    """统一光流估计器，支持 RAFT、SpyNet 和 MemFlow。

    MemFlow 使用 MemFlowNet_P 模型，参考原始 inference_core_predict.py 实现：
    - step(): encode_context → memory readout → motion_prompt + readout → update_block → flow
    - set_memory(): encode_features → CorrBlock(coords + estimated_flow) → memory update
    """

    DEFAULT_WEIGHTS = {
        'raft': 'pretrained_model/flow/raft-sintel.pth',
        'spynet': 'pretrained_model/flow/spynet_sintel_final-3d2a1287.pth',
        'memflow': 'pretrained_model/flow/MemFlowNet_P_sintel.pth',
    }

    def __init__(self, estimator_type='raft', device=None, normalization='tanh', pretrained_path=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.normalization = normalization
        self.estimator_type = estimator_type.lower()

        if pretrained_path is None:
            if self.estimator_type not in self.DEFAULT_WEIGHTS:
                raise ValueError(f"Unknown estimator_type: {estimator_type}.")
            pretrained_path = os.path.join(_PROJECT_ROOT, self.DEFAULT_WEIGHTS[self.estimator_type])

        if self.estimator_type == 'raft':
            self.model = load_raft(pretrained_path=pretrained_path, device=device)
        elif self.estimator_type == 'spynet':
            self.model = load_spynet(pretrained_path=pretrained_path, device=device)
        elif self.estimator_type == 'memflow':
            from .memflow import MemFlowNet_P
            self.model = MemFlowNet_P(pretrained_path=pretrained_path)
            if device is not None:
                self.model = self.model.to(device)
        else:
            raise ValueError(f"Unknown estimator_type: {estimator_type}.")
        self.model.eval()

    def compute_flow(self, lrs):
        """计算双向光流。

        Args:
            lrs (torch.Tensor): shape (n, t, c, h, w)
        Returns:
            tuple: (forward_flow, backward_flow)，均为 (n, t-1, 2, h, w)
        """
        if self.estimator_type == 'memflow':
            return self.compute_flow_sequential(lrs)

        self.model = self.model.to(lrs.device)
        n, t, c, h, w = lrs.size()

        if self.normalization == 'tanh':
            lrs = (lrs + 1.) / 2

        forward_lrs = lrs[:, 1:, :, :, :].reshape(-1, c, h, w)
        backward_lrs = lrs[:, :-1, :, :, :].reshape(-1, c, h, w)

        with torch.no_grad():
            if self.estimator_type == 'raft':
                _, forward_flow = self.model(forward_lrs * 255, backward_lrs * 255, iters=24, test_mode=True)
                forward_flow = forward_flow.view(n, t - 1, 2, h, w)
                _, backward_flow = self.model(backward_lrs * 255, forward_lrs * 255, iters=24, test_mode=True)
                backward_flow = backward_flow.view(n, t - 1, 2, h, w)
            else:  # spynet
                forward_flow = self.model(forward_lrs, backward_lrs).view(n, t - 1, 2, h, w)
                backward_flow = self.model(backward_lrs, forward_lrs).view(n, t - 1, 2, h, w)

        return forward_flow, backward_flow

    def _memflow_predict(self, image, mem_key, mem_value):
        """MemFlowNet_P 单帧光流预测（参考 inference_core_predict.step）。

        Args:
            image: (B, C, H, W) 单帧图像，已归一化到 [-1,1]
            mem_key: (B, C, L_mem) 已存储 memory key，或 None
            mem_value: (B, C, L_mem) 已存储 memory value，或 None

        Returns:
            delta_flow: (B, 2, Hf, Wf) 低分辨率 flow
            flow_up: (B, 2, H, W) 上采样 flow
        """
        B, _, H, W = image.shape

        # encode_context 接受单帧 (B, C, H, W)
        inp = self.model.cnet(image)
        inp = torch.relu(inp)
        query, key = self.model.att.to_qk(inp).chunk(2, dim=1)

        # Memory readout
        query_flat = query.flatten(start_dim=2)  # (B, C, L)
        memory_readout = _memory_readout(
            query_flat, mem_key, mem_value,
            scale=self.model.att.scale,
            train_avg_length=self.model.train_avg_length
        )

        Hf, Wf = H // 8, W // 8
        motion_features_global = self.model.motion_prompt.repeat(B, 1, Hf, Wf)

        if not isinstance(memory_readout, int):
            readout_spatial = memory_readout.view(B, -1, Hf, Wf)
            motion_features_global = motion_features_global + \
                self.model.update_block.aggregator.gamma * readout_spatial

        # concat_flow 处理
        if hasattr(self.model.cfg, 'concat_flow') and self.model.cfg.concat_flow:
            forward_warp_flow = torch.zeros(B, 2, Hf, Wf,
                                            device=image.device, dtype=image.dtype)
            motion_features_global = torch.cat([motion_features_global, forward_warp_flow], dim=1)

        _, up_mask, delta_flow = self.model.update_block(inp, motion_features_global)

        if hasattr(self.model.cfg, 'concat_flow') and self.model.cfg.concat_flow:
            pass  # forward_warp_flow is zero, no-op

        flow_up = self.model.upsample_flow(delta_flow, up_mask)

        return delta_flow, flow_up, key

    def _memflow_update_memory(self, images, flow, current_key, mem_key, mem_value, max_mem_frames):
        """MemFlowNet_P memory 更新（参考 inference_core_predict.set_memory）。

        用已估计的 flow 索引 CorrBlock 构建 memory value。

        Args:
            images: (B, 2, C, H, W) 双帧图像
            flow: (B, 2, Hf, Wf) 低分辨率 delta flow
            current_key: (B, C, Hf, Wf) 当前帧的 context key
            mem_key: (B, C, L_mem) 已存储 memory key，或 None
            mem_value: (B, C, L_mem) 已存储 memory value，或 None
            max_mem_frames: 最大 memory 帧数

        Returns:
            mem_key: (B, C, L_new) 更新后的 memory key
            mem_value: (B, C, L_new) 更新后的 memory value
        """
        from .memflow import CorrBlock

        coords0, coords1, fmaps = self.model.encode_features(images)

        # 用已估计的 flow 索引 CorrBlock
        corr_fn = CorrBlock(fmaps[:, 0], fmaps[:, 1],
                            num_levels=self.model.cfg.corr_levels,
                            radius=self.model.cfg.corr_radius)
        corr = corr_fn(coords1 + flow)
        corr = torch.nan_to_num(corr, nan=0.0)  # 防止低分辨率 fmap 的数值退化
        _, current_value = self.model.update_block.get_motion_and_value(flow, corr)
        current_value = torch.nan_to_num(current_value, nan=0.0)

        # 将 key 和 value flatten 后加入 memory
        key_flat = current_key.flatten(start_dim=2)      # (B, C, Hf*Wf)
        value_flat = current_value.flatten(start_dim=2)   # (B, C, Hf*Wf)

        if mem_key is None:
            mem_key = key_flat
            mem_value = value_flat
        else:
            mem_key = torch.cat([mem_key, key_flat], dim=-1)
            mem_value = torch.cat([mem_value, value_flat], dim=-1)
            # 限制 memory 大小
            hw = key_flat.shape[-1]
            max_len = max_mem_frames * hw
            if mem_key.shape[-1] > max_len:
                mem_key = mem_key[:, :, -max_len:]
                mem_value = mem_value[:, :, -max_len:]

        return mem_key, mem_value

    def compute_flow_sequential(self, lrs):
        """序列化光流计算（MemFlowNet_P with memory cross-attention）。

        参考原始 inference_core_predict.py 的两步流程：
        1. step(): 估计光流（不用 CorrBlock）
        2. set_memory(): 用已估计的 flow 更新 memory（用 CorrBlock）

        Args:
            lrs (torch.Tensor): shape (n, t, c, h, w)
        Returns:
            tuple: (forward_flow, backward_flow)，均为 (n, t-1, 2, h, w)
        """
        self.model = self.model.to(lrs.device)
        n, t, c, h, w = lrs.size()

        # 归一化到 [-1, 1]
        if self.normalization == 'tanh':
            lrs_n1 = lrs.clone()
        elif self.normalization == 'sigmoid':
            lrs_n1 = 2 * lrs - 1.0
        else:
            lrs_n1 = lrs.clone()

        forward_flows = []
        backward_flows = []
        max_mem = min(t, 20)

        with torch.no_grad():
            # === Forward: frame i+1 → frame i ===
            mem_key_fw = None
            mem_val_fw = None
            for i in range(t - 1):
                img1 = lrs_n1[:, i + 1, :, :, :].contiguous()  # 当前帧
                img2 = lrs_n1[:, i, :, :, :].contiguous()       # 参考帧

                # Step 1: 估计光流（无 CorrBlock）
                delta_flow, flow_up, current_key = self._memflow_predict(
                    img1, mem_key_fw, mem_val_fw)
                forward_flows.append(flow_up)

                # Step 2: 用已估计的 flow 更新 memory（用 CorrBlock + 已估计 flow）
                frames = torch.stack([img1, img2], dim=1)  # (B, 2, C, H, W)
                mem_key_fw, mem_val_fw = self._memflow_update_memory(
                    frames, delta_flow, current_key,
                    mem_key_fw, mem_val_fw, max_mem)

            # === Backward: frame i → frame i+1 ===
            mem_key_bw = None
            mem_val_bw = None
            for i in range(t - 2, -1, -1):
                img1 = lrs_n1[:, i, :, :, :].contiguous()
                img2 = lrs_n1[:, i + 1, :, :, :].contiguous()

                delta_flow, flow_up, current_key = self._memflow_predict(
                    img1, mem_key_bw, mem_val_bw)
                backward_flows.insert(0, flow_up)

                frames = torch.stack([img1, img2], dim=1)
                mem_key_bw, mem_val_bw = self._memflow_update_memory(
                    frames, delta_flow, current_key,
                    mem_key_bw, mem_val_bw, max_mem)

        forward_flow = torch.stack(forward_flows, dim=1)    # (n, t-1, 2, h, w)
        backward_flow = torch.stack(backward_flows, dim=1)  # (n, t-1, 2, h, w)
        return forward_flow, backward_flow

    def to(self, device):
        """移动模型到指定设备"""
        self.device = device
        self.model = self.model.to(device)
        return self
