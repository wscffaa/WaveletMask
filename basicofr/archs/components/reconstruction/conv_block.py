import torch.nn as nn

from ..arch_util import ResidualBlockNoBN, make_layer


class ConvResBlock(nn.Module):
    """卷积 + 无 BN 残差块堆叠。"""

    def __init__(self, in_feat, out_feat=64, num_block=30):
        super().__init__()
        self.conv_resblock = nn.Sequential(
            nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            make_layer(ResidualBlockNoBN, num_block, num_feat=out_feat),
        )

    def forward(self, x):
        return self.conv_resblock(x)
