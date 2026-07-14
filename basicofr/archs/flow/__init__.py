"""光流估计模块

- flow_estimator: 统一光流估计器
- raft: RAFT 光流网络
- spynet: SpyNet 光流网络
- memflow: MemFlow 光流网络
"""

from pkgutil import extend_path

# 支持从扩展目录注入 flow 模块（见 basicofr.__init__ 的说明）
__path__ = extend_path(__path__, __name__)

from .flow_estimator import FlowEstimator, load_raft
from .spynet import SpyNet, load_spynet
from .memflow import MemFlowNet

# 向后兼容
RAFTFlowEstimator = FlowEstimator

__all__ = [
    'FlowEstimator',
    'RAFTFlowEstimator',
    'load_raft',
    'SpyNet',
    'load_spynet',
    'MemFlowNet',
]
