"""数据加载模块

- train_dataset/test_dataset: 活跃 OFR 数据集
- rtn_dataset: 历史兼容入口
- degradations/: 退化函数
- utils/: 数据工具
"""

from __future__ import annotations

import importlib
from os import path as osp
from pkgutil import extend_path

from basicsr.utils import scandir

__path__ = extend_path(__path__, __name__)


def _collect_dataset_filenames() -> list[str]:
    filenames: list[str] = []
    for data_folder in list(__path__):
        if not osp.isdir(data_folder):
            continue
        for v in scandir(data_folder):
            if v.endswith('_dataset.py'):
                filenames.append(osp.splitext(osp.basename(v))[0])
    seen: set[str] = set()
    uniq: list[str] = []
    for name in filenames:
        if name in seen:
            continue
        seen.add(name)
        uniq.append(name)
    return uniq


_data_modules = [importlib.import_module(f'basicofr.data.{file_name}') for file_name in _collect_dataset_filenames()]

from .base_dataset import OFRBaseDataset
from .test_dataset import OFRTestDataset, OFRTestDatasetGray
from .train_dataset import OFRTrainDataset

# 向后兼容别名
Film_dataset_1 = OFRTrainDataset
Film_SRWOV_dataset = OFRTestDataset
RRTN_Film_SRWOV_dataset = OFRTestDatasetGray
Film_dataset_damage = OFRTrainDataset
Film_dataset_damage_color = OFRTrainDataset
FilmDatasetMambaOFR = OFRTrainDataset

__all__ = [
    'OFRBaseDataset',
    'OFRTrainDataset',
    'OFRTestDataset',
    'OFRTestDatasetGray',
    'Film_dataset_1',
    'Film_SRWOV_dataset',
    'RRTN_Film_SRWOV_dataset',
    'Film_dataset_damage',
    'Film_dataset_damage_color',
    'FilmDatasetMambaOFR',
]
