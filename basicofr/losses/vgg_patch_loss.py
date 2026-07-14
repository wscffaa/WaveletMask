"""Runtime patches for third-party losses to remove deprecated torchvision usage."""

from __future__ import annotations

import os
from collections import OrderedDict
from typing import Tuple

import torch
from torch import nn
from torchvision import models as vgg

from basicsr.archs import vgg_arch


def _resolve_default_weights(vgg_type: str) -> Tuple[object, bool]:
    """Return the default weights enum member for the requested VGG variant."""
    attr_name = f'{vgg_type.replace("-", "_").upper()}_WEIGHTS'
    weights_enum = getattr(vgg, attr_name, None)
    if weights_enum is None:
        return None, False
    default = getattr(weights_enum, 'DEFAULT', None)
    if default is not None:
        return default, True
    fallback = getattr(weights_enum, 'IMAGENET1K_V1', None)
    if fallback is not None:
        return fallback, True
    return None, True


def _instantiate_vgg(vgg_type: str) -> nn.Module:
    """Create a torchvision VGG model using the modern weights API when available."""
    weights, has_enum = _resolve_default_weights(vgg_type)
    try:
        if has_enum:
            return getattr(vgg, vgg_type)(weights=weights)
        return getattr(vgg, vgg_type)(weights=weights)
    except TypeError:
        # Older torchvision versions still expect the deprecated flag.
        return getattr(vgg, vgg_type)(pretrained=True)


def patch_vgg_feature_extractor() -> None:
    """Monkey-patch BasicSR's VGGFeatureExtractor to use the new torchvision weights API."""
    if getattr(vgg_arch, '_rtn_vgg_patched', False):
        return

    def _patched_init(self,
                      layer_name_list,
                      vgg_type: str = 'vgg19',
                      use_input_norm: bool = True,
                      range_norm: bool = False,
                      requires_grad: bool = False,
                      remove_pooling: bool = False,
                      pooling_stride: int = 2):
        nn.Module.__init__(self)

        self.layer_name_list = layer_name_list
        self.use_input_norm = use_input_norm
        self.range_norm = range_norm

        names = vgg_arch.NAMES[vgg_type.replace('_bn', '')]
        if 'bn' in vgg_type:
            names = vgg_arch.insert_bn(names)
        self.names = names

        max_idx = 0
        for name in layer_name_list:
            idx = self.names.index(name)
            if idx > max_idx:
                max_idx = idx

        if os.path.exists(vgg_arch.VGG_PRETRAIN_PATH):
            vgg_net = getattr(vgg, vgg_type)(weights=None)
            state_dict = torch.load(vgg_arch.VGG_PRETRAIN_PATH, map_location=lambda storage, loc: storage)
            vgg_net.load_state_dict(state_dict)
        else:
            vgg_net = _instantiate_vgg(vgg_type)

        features = vgg_net.features[:max_idx + 1]
        modified_net = OrderedDict()
        for key, layer in zip(self.names, features):
            if 'pool' in key:
                if remove_pooling:
                    continue
                modified_net[key] = nn.MaxPool2d(kernel_size=2, stride=pooling_stride)
            else:
                modified_net[key] = layer
        self.vgg_net = nn.Sequential(modified_net)

        if not requires_grad:
            self.vgg_net.eval()
            for param in self.parameters():
                param.requires_grad = False
        else:
            self.vgg_net.train()
            for param in self.parameters():
                param.requires_grad = True

        if self.use_input_norm:
            self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    vgg_arch.VGGFeatureExtractor.__init__ = _patched_init  # type: ignore
    vgg_arch._rtn_vgg_patched = True


__all__ = ['patch_vgg_feature_extractor']
