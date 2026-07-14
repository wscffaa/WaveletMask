"""BasicOFR - 老电影修复底层框架

基于 BasicSR 构建的视频修复核心框架，提供：
- RTN 双向传播架构
- 光流估计（RAFT/SpyNet）
- 老电影退化模拟
- GAN 训练流程
"""

from __future__ import annotations

import importlib
import warnings
from pkgutil import extend_path
from types import ModuleType

# ===================== 抑制第三方库的兼容性警告 =====================
# timm.models.layers 已废弃，应使用 timm.layers（timm 内部兼容层触发）
warnings.filterwarnings(
    'ignore',
    message='Importing from timm.models.layers is deprecated',
    category=FutureWarning,
    module='timm.models.layers'
)
# ================================================================

# ===================== NumPy 2.0 兼容性补丁 =====================
# 在任何其他导入之前应用，以修复依赖库（mamba_ssm, imgaug 等）的兼容性问题
import numpy as np

if not hasattr(np, 'float_'):
    np.float_ = np.float64

if not hasattr(np, 'sctypes'):
    np.sctypes = {
        'int': [np.int8, np.int16, np.int32, np.int64],
        'uint': [np.uint8, np.uint16, np.uint32, np.uint64],
        'float': [np.float16, np.float32, np.float64],
        'complex': [np.complex64, np.complex128],
        'others': [bool, object, bytes, str, np.void],
    }
# ================================================================

__version__ = '0.1.0'
__author__ = 'BasicOFR Team'

__all__ = ['archs', 'models', 'losses', 'data', 'metrics']

# 允许通过在 sys.path 中追加“扩展根目录”来为 BasicOFR 注入实验代码：
#   OFR_EXT_PATH=ofr_projects/<exp> python train.py -opt ...
# 扩展目录结构示例：
#   ofr_projects/<exp>/basicofr/archs/*_arch.py
__path__ = extend_path(__path__, __name__)


def __getattr__(name: str) -> ModuleType:
    """按需加载子模块，避免导入时触发可选依赖。"""
    if name in __all__:
        return importlib.import_module(f'{__name__}.{name}')
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
