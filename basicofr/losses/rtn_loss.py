import os
import torch
import torch.nn as nn

try:  # torchvision >= 0.13
    from torchvision.models import vgg19, VGG19_Weights
except ImportError:  # fallback for older versions
    from torchvision.models import vgg19  # type: ignore
    VGG19_Weights = None  # type: ignore

from basicsr.archs.vgg_arch import VGG_PRETRAIN_PATH
from basicsr.utils.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register()
class AdversarialLoss(nn.Module):
    """Flexible adversarial loss supporting NS-GAN, LS-GAN, and hinge variants."""

    def __init__(self,
                 loss_type='nsgan',
                 target_real_label=1.0,
                 target_fake_label=0.0,
                 loss_weight=1.0):
        super().__init__()
        self.loss_type = loss_type.lower()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.loss_weight = loss_weight

        if self.loss_type == 'nsgan':
            self.criterion = nn.BCEWithLogitsLoss()
        elif self.loss_type == 'lsgan':
            self.criterion = nn.MSELoss()
        elif self.loss_type == 'hinge':
            self.criterion = nn.ReLU()
        else:
            raise ValueError(f'Unsupported GAN loss type: {loss_type}')

    def forward(self, outputs, is_real, is_disc=False):
        if self.loss_type == 'hinge':
            if is_disc:
                if is_real:
                    outputs = -outputs
                return self.loss_weight * self.criterion(1 + outputs).mean()
            return self.loss_weight * (-outputs).mean()

        labels = (self.real_label if is_real else self.fake_label).expand_as(outputs)
        return self.loss_weight * self.criterion(outputs, labels)


def _load_vgg19_backbone():
    """优先从本地权重加载 VGG19，否则回退到 torchvision 内置。"""
    if os.path.exists(VGG_PRETRAIN_PATH):
        model = vgg19(weights=None)
        state_dict = torch.load(VGG_PRETRAIN_PATH, map_location='cpu', weights_only=False)
        model.load_state_dict(state_dict)
        return model

    if VGG19_Weights is not None:
        weights = getattr(VGG19_Weights, 'DEFAULT', None) or VGG19_Weights.IMAGENET1K_V1
        return vgg19(weights=weights)

    return vgg19(pretrained=True)


class _VGG19Features(nn.Module):
    """VGG19 backbone sliced into blocks for perceptual supervision."""

    def __init__(self, requires_grad=False):
        super().__init__()
        vgg = _load_vgg19_backbone()
        features = vgg.features

        self.slice1 = nn.Sequential(*[features[x] for x in range(2)])
        self.slice2 = nn.Sequential(*[features[x] for x in range(2, 7)])
        self.slice3 = nn.Sequential(*[features[x] for x in range(7, 12)])
        self.slice4 = nn.Sequential(*[features[x] for x in range(12, 21)])
        self.slice5 = nn.Sequential(*[features[x] for x in range(21, 30)])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        h_relu1 = self.slice1(x)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]


@LOSS_REGISTRY.register()
class VGGLoss(nn.Module):
    """Multi-layer perceptual loss computed from frozen VGG19 features."""

    def __init__(self, weights=None, requires_grad=False, device=None, loss_weight=1.0):
        super().__init__()
        self.vgg = _VGG19Features(requires_grad=requires_grad)
        if device is not None:
            self.vgg = self.vgg.to(device)
        self.vgg.eval()
        self.criterion = nn.L1Loss()
        self.weights = weights or [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        self.loss_weight = loss_weight

    def forward(self, x, y):
        x_vgg = self.vgg(x)
        y_vgg = self.vgg(y)
        loss = 0.0
        for feat_x, feat_y, w in zip(x_vgg, y_vgg, self.weights):
            loss = loss + w * self.criterion(feat_x, feat_y.detach())
        return self.loss_weight * loss
