"""时序对齐模块

- deform: 可变形卷积（ModulatedDeformConv2d, SecondOrderDeformableAlignment）
- deformable_path: 可变形路径
- conv_offset: 卷积偏移量
- dcn: DCN 实现（DCNv2PackFlowGuided）
"""

from .deform import (
    ModulatedDeformConv2d,
    modulated_deform_conv2d,
    SecondOrderDeformableAlignment,
    constant_init,
)
from .conv_offset import *
from . import dcn

__all__ = [
    'ModulatedDeformConv2d',
    'modulated_deform_conv2d',
    'SecondOrderDeformableAlignment',
    'constant_init',
    'dcn',
]

try:
    from .dcn import DCNv2PackFlowGuided
    __all__.append('DCNv2PackFlowGuided')
except (ImportError, AttributeError):
    pass
