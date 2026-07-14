"""BasicSR metrics - 兼容层。

所有评价指标统一由 basicofr.metrics (pyiqa) 实现。
此模块仅提供 calculate_metric 函数和必要的工具函数。
"""
from copy import deepcopy

from basicsr.utils.registry import METRIC_REGISTRY

__all__ = ['calculate_metric']


def calculate_metric(data, opt):
    """Calculate metric from data and options.

    Args:
        opt (dict): Configuration. It must contain:
            type (str): Model type.
    """
    opt = deepcopy(opt)
    metric_type = opt.pop('type')
    metric = METRIC_REGISTRY.get(metric_type)(**data, **opt)
    return metric
