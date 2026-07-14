import torch.nn as nn
import torch.nn.functional as F


class PSUpsample(nn.Module):
    """Conv + PixelShuffle 重建头。"""

    def __init__(self, in_feat, out_feat, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor
        self.up_conv = nn.Conv2d(
            in_feat,
            out_feat * scale_factor * scale_factor,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        x = self.up_conv(x)
        return F.pixel_shuffle(x, upscale_factor=self.scale_factor)


PixelShufflePack = PSUpsample
