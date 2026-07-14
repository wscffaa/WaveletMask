"""MambaIR 空间模块

基于状态空间模型 (SSM) 的图像修复模块，源自 MambaIR 论文。

核心组件（保留在此目录）：
- MambaIR: 主图像修复网络
- SS2D: 2D 选择性状态空间模块
- VSSBlock: 视觉状态空间块
- CAB: 通道注意力块
- DynamicConvolution: 动态卷积

实验性架构请使用 basicofr.archs.ideas 下的对应模块：
- FreqMamba: basicofr.archs.ideas.freqmamba
- WaveMamba: basicofr.archs.ideas.wavemamba
- DefMamba: basicofr.archs.ideas.defmamba
- LidarMamba: basicofr.archs.ideas.lidarmamba
- LocalMamba: basicofr.archs.ideas.localmamba
- MaIR: basicofr.archs.ideas.mair
"""

from pkgutil import extend_path

# 支持从扩展目录注入 mamba 子模块（见 basicofr.__init__ 的说明）
__path__ = extend_path(__path__, __name__)

# 核心组件
from .mambair import (
    MambaIR,
    SS2D,
    VSSBlock,
    CAB,
    BasicLayer,
    ResidualGroup,
    PatchEmbed,
    PatchUnEmbed,
    Upsample,
    UpsampleOneStep,
)
from .dynamic_conv import DynamicConvolution

__all__ = [
    # MambaIR 核心
    'MambaIR',
    'SS2D',
    'VSSBlock',
    'CAB',
    'BasicLayer',
    'ResidualGroup',
    'PatchEmbed',
    'PatchUnEmbed',
    'Upsample',
    'UpsampleOneStep',
    'DynamicConvolution',
]
