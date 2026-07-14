import torch
import torch.nn as nn

from basicsr.utils.registry import ARCH_REGISTRY


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)
    return module


class _BaseDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

    def init_weights(self, init_type='normal', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('InstanceNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight.data, 1.0)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':
                    if hasattr(m, 'reset_parameters'):
                        m.reset_parameters()
                else:
                    raise NotImplementedError(f'Initialization method {init_type} is not implemented.')
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        for module in self.children():
            if hasattr(module, 'init_weights'):
                module.init_weights(init_type, gain)


@ARCH_REGISTRY.register()
class RTNDiscriminator(_BaseDiscriminator):
    """3D convolutional discriminator tailored for RTN video restoration."""

    def __init__(self, in_channels=3, base_channels=64, use_sigmoid=False, use_spectral_norm=True, init_weights=True):
        super().__init__()
        self.use_sigmoid = use_sigmoid

        self.conv = nn.Sequential(
            spectral_norm(
                nn.Conv3d(in_channels, base_channels, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=1,
                          bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv3d(base_channels, base_channels * 2, kernel_size=(3, 5, 5), stride=(1, 2, 2),
                          padding=(1, 2, 2), bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv3d(base_channels * 2, base_channels * 4, kernel_size=(3, 5, 5), stride=(1, 2, 2),
                          padding=(1, 2, 2), bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv3d(base_channels * 4, base_channels * 4, kernel_size=(3, 5, 5), stride=(1, 2, 2),
                          padding=(1, 2, 2), bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv3d(base_channels * 4, base_channels * 4, kernel_size=(3, 5, 5), stride=(1, 2, 2),
                          padding=(1, 2, 2), bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(base_channels * 4, base_channels * 4, kernel_size=(3, 5, 5), stride=(1, 2, 2),
                      padding=(1, 2, 2))
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        x = x.transpose(1, 2)  # B, C, T, H, W
        feat = self.conv(x)
        if self.use_sigmoid:
            feat = torch.sigmoid(feat)
        return feat.transpose(1, 2)


@ARCH_REGISTRY.register()
class RTNDiscriminatorLite(_BaseDiscriminator):
    """Channel-reduced discriminator variant for RTN."""

    def __init__(self, in_channels=3, base_channels=64, use_sigmoid=False, use_spectral_norm=True, init_weights=True):
        super().__init__()
        self.use_sigmoid = use_sigmoid

        self.conv = nn.Sequential(
            spectral_norm(
                nn.Conv3d(in_channels, base_channels, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=1,
                          bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv3d(base_channels, base_channels * 2, kernel_size=(3, 5, 5), stride=(1, 2, 2),
                          padding=(1, 2, 2), bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv3d(base_channels * 2, base_channels * 2, kernel_size=(3, 5, 5), stride=(1, 2, 2),
                          padding=(1, 2, 2), bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv3d(base_channels * 2, base_channels * 2, kernel_size=(3, 5, 5), stride=(1, 2, 2),
                          padding=(1, 2, 2), bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(base_channels * 2, base_channels * 2, kernel_size=(3, 5, 5), stride=(1, 2, 2),
                      padding=(1, 2, 2))
        )

        if init_weights:
            self.init_weights()

    def forward(self, x):
        x = x.transpose(1, 2)
        feat = self.conv(x)
        if self.use_sigmoid:
            feat = torch.sigmoid(feat)
        return feat.transpose(1, 2)
