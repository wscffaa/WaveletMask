"""WaveletMask 模型

继承 RTNGANModel，扩展 visualiza_mask 处理以支持 dict 返回格式：
- local: 局部高频掩码 (创新A)
- flicker: 闪烁掩码 (创新B)
- fused: 融合后的掩码
"""

import torch
from basicsr.utils.registry import MODEL_REGISTRY

from basicofr.models.rtn_model import RTNGANModel


@MODEL_REGISTRY.register()
class WaveletMaskModel(RTNGANModel):
    """WaveletMask 专用模型，支持论文所需的掩码可视化。

    visualiza_mask 返回 dict 格式：
    - base: {}
    - local/simple: {'local': tensor}
    - local_fusion: {'local': tensor}
    - flicker_fusion: {'flicker': tensor}
    - full: {'local': tensor, 'flicker': tensor, 'fused': tensor}
    - gated_flicker: {'local': tensor, 'flicker': tensor, 'fused': tensor}
    """

    def test(self):
        """重写 test 方法，支持 dict 格式掩码返回。"""
        use_ema = hasattr(self, 'ema_decay') and self.ema_decay > 0
        net = self.net_g_ema if use_ema else self.net_g

        prev_training_state = net.training
        net.eval()

        collect_debug = getattr(self, '_collect_visual_debug', False)
        self._last_visual_masks = None
        self._last_visual_features = None

        with torch.no_grad():
            self.output = net(self.lq)

            if collect_debug:
                masks = None
                features = None

                # 处理 visualiza_mask dict 返回
                if hasattr(net, 'visualiza_mask'):
                    try:
                        mask_outputs = net.visualiza_mask(self.lq)
                        masks = self._parse_mask_outputs(mask_outputs)
                    except Exception:
                        masks = None

                # 处理 visualiza_feature（与父类相同）
                if hasattr(net, 'visualiza_feature'):
                    try:
                        feature_outputs = net.visualiza_feature(self.lq)
                    except Exception:
                        feature_outputs = None
                    if feature_outputs is not None:
                        features = {}
                        if isinstance(feature_outputs, (list, tuple)):
                            if len(feature_outputs) == 4:
                                b_feat, b_state, f_feat, f_state = feature_outputs
                            elif len(feature_outputs) == 2:
                                b_feat = b_state = None
                                f_feat, f_state = feature_outputs
                            else:
                                b_feat = b_state = f_feat = f_state = None
                        else:
                            b_feat = b_state = None
                            f_feat = f_state = feature_outputs
                        if b_feat is not None:
                            features['backward_mean_mask'] = b_feat.detach().cpu()[0]
                        if b_state is not None:
                            features['backward_state'] = b_state.detach().cpu()[0]
                        if f_feat is not None:
                            features['forward_mean_mask'] = f_feat.detach().cpu()[0]
                        if f_state is not None:
                            features['forward_state'] = f_state.detach().cpu()[0]

                self._last_visual_masks = masks
                self._last_visual_features = features

        if not use_ema and prev_training_state:
            net.train()

    def _parse_mask_outputs(self, mask_outputs):
        """解析 visualiza_mask 返回值（dict 格式）。

        支持的 dict key：
        - 'local': 局部高频掩码 (创新A)
        - 'flicker': 闪烁掩码 (创新B)
        - 'fused': 融合后的掩码

        Returns:
            dict: 掩码字典，key 为掩码名称，value 为 tensor
        """
        if not isinstance(mask_outputs, dict):
            return None

        if not mask_outputs:
            return None

        masks = {}
        for name, tensor in mask_outputs.items():
            if tensor is not None:
                masks[name] = tensor.detach().cpu()[0]

        return masks if masks else None
