"""通用架构组件

- arch_util: 基础工具函数
- discriminator: 判别器
- timm_compat: timm 兼容层

说明：
- 为了支持将"实验代码/创新点"隔离到 `basicofr/archs/ideas/<project>/`，本模块会对部分组件做**可选导入**：
  - 未启用扩展时（仅 core）也能正常 import/训练 RTN 基线
  - 项目专用组件应放在对应的 ideas/{project}/ 目录下
"""

from __future__ import annotations

import warnings
from importlib import import_module
from pkgutil import extend_path

# 支持从扩展目录注入自定义组件模块（见 basicofr.__init__ 的说明）
__path__ = extend_path(__path__, __name__)

# -------------------- Core exports (always available) --------------------
from .arch_util import (
    default_init_weights,
    make_layer,
    ResidualBlockNoBN,
    Upsample,
    flow_warp,
    ColorTemporalNormalizer,
)
from .timm_compat import DropPath, to_2tuple, trunc_normal_
from . import discriminator  # noqa: F401

__all__ = [
    'default_init_weights',
    'make_layer',
    'ResidualBlockNoBN',
    'Upsample',
    'flow_warp',
    'ColorTemporalNormalizer',
    'DropPath',
    'to_2tuple',
    'trunc_normal_',
]


def _optional_import(rel_name: str):
    """Import optional module from this package.

    Optional components should not break lightweight import/test environments.
    """
    fq_name = f'{__name__}.{rel_name}'
    try:
        return import_module(fq_name)
    except ModuleNotFoundError as e:
        warnings.warn(f'跳过可选组件 {fq_name}: {e}')
        return None


# -------------------- Optional exports (provided by core extensions) --------------------
_gating = _optional_import('aggregation.gating')
if _gating is not None:
    MultiScaleGatedAggregation = _gating.MultiScaleGatedAggregation
    GatedAggregation = _gating.GatedAggregation
    GatedAggregationDCN = _gating.GatedAggregationDCN
    __all__ += [
        'MultiScaleGatedAggregation',
        'GatedAggregation',
        'GatedAggregationDCN',
    ]

_deform = _optional_import('alignment.deform')
if _deform is not None:
    ModulatedDeformConv2d = _deform.ModulatedDeformConv2d
    modulated_deform_conv2d = _deform.modulated_deform_conv2d
    SecondOrderDeformableAlignment = _deform.SecondOrderDeformableAlignment
    constant_init = _deform.constant_init
    __all__ += [
        'ModulatedDeformConv2d',
        'modulated_deform_conv2d',
        'SecondOrderDeformableAlignment',
        'constant_init',
    ]

_dcn = _optional_import('alignment.dcn')
if _dcn is not None:
    DCNv2PackFlowGuided = getattr(_dcn, 'DCNv2PackFlowGuided', None)
    if DCNv2PackFlowGuided is not None:
        __all__ += ['DCNv2PackFlowGuided']
