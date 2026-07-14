"""空间修复模块

- swin_spatial: Swin Transformer 空间修复
- swin_util: Swin 工具函数
- mamba: MambaIR 状态空间模型空间修复
"""

from __future__ import annotations

from pkgutil import extend_path

# 支持从扩展目录注入空间模块（见 basicofr.__init__ 的说明）
__path__ = extend_path(__path__, __name__)

from .swin_spatial import Swin_Spatial_2
from .swin_util import (
    BasicLayer,
    Mlp,
    PatchEmbed,
    PatchUnEmbed,
    RSTB,
    SwinTransformerBlock,
    WindowAttention,
    window_partition,
    window_reverse,
)

__all__ = [
    'Swin_Spatial_2',
    'BasicLayer',
    'Mlp',
    'PatchEmbed',
    'PatchUnEmbed',
    'RSTB',
    'SwinTransformerBlock',
    'WindowAttention',
    'window_partition',
    'window_reverse',
]

# 可选依赖：MambaIR 需要 mamba_ssm，且模块通常来自 `ofr_projects/**/basicofr/archs/spatial/mamba`。
try:
    from .mamba import MambaIR, DynamicConvolution  # noqa: F401
except ModuleNotFoundError as e:
    missing = e.name or ''
    missing_root = missing.split('.')[0]
    # 1) 未启用扩展（mamba 子包不存在） 2) mamba_ssm 缺失：两者都视为可选，静默跳过
    if missing == f'{__name__}.mamba' or missing_root == 'mamba_ssm':
        pass
    else:
        raise
else:
    __all__ += [
        'MambaIR',
        'DynamicConvolution',
    ]
