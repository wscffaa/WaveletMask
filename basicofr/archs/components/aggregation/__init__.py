"""特征聚合模块

- gating: 多尺度门控聚合（MultiScaleGatedAggregation, GatedAggregation, GatedAggregationDCN）
"""

from .gating import (
    MultiScaleGatedAggregation,
    GatedAggregation,
    GatedAggregationDCN,
)

__all__ = [
    'MultiScaleGatedAggregation',
    'GatedAggregation',
    'GatedAggregationDCN',
]
