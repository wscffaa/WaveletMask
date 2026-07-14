"""DCN 可变形卷积模块

提供流引导可变形对齐，用于视频修复和超分辨率任务中的特征对齐。

支持的模块：
- DCNv2PackFlowGuided: 基于 torchvision DCNv2 的流引导对齐
"""

from .flow_guided_dcn import DCNv2PackFlowGuided, ModulatedDeformConv, ModulatedDeformConvPack

__all__ = [
    'DCNv2PackFlowGuided',
    'ModulatedDeformConv',
    'ModulatedDeformConvPack',
]
