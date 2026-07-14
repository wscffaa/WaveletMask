from __future__ import annotations

import operator
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
import torchvision.transforms as transforms
from torch.utils import data as data

from .utils import augment, paired_random_crop


def getfilelist_with_length(file_path: str) -> List[tuple]:
    all_file = []
    for dir_path, _, files in os.walk(file_path):
        for file_name in files:
            path = os.path.join(dir_path, file_name)
            all_file.append((path, len(os.listdir(dir_path))))
    all_file.sort(key=operator.itemgetter(0))
    return all_file


def getfolderlist(file_path: str) -> List[tuple]:
    all_folder = []
    for dir_path, _, files in os.walk(file_path):
        if not files:
            continue
        rerank = sorted(files)
        if rerank[0].endswith('.avi'):
            continue
        all_folder.append((os.path.join(dir_path, rerank[0]), len(files)))
    all_folder.sort(key=operator.itemgetter(0))
    return all_folder


def resize_368_short_side(img: np.ndarray, short_side: int = 368, align: int = 16) -> np.ndarray:
    h, w = img.shape[:2]
    if w < h:
        new_w = short_side
        new_h = int(short_side * h / w)
    else:
        new_h = short_side
        new_w = int(short_side * w / h)

    new_h = new_h // align * align
    new_w = new_w // align * align
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


class OFRBaseDataset(data.Dataset, ABC):

    def __init__(self, data_config):
        super().__init__()
        self.data_config = data_config
        self.opt = data_config
        self.scale = data_config.get('scale', 1)
        self.gt_root = data_config.get('dataroot_gt')
        self.lq_root = data_config.get('dataroot_lq', self.gt_root)
        self.is_train = data_config.get('is_train', False)
        self.io_backend_opt = data_config.get('io_backend')

        raw_num_frame = data_config.get('num_frame', None if self.is_train else -1)
        if raw_num_frame is None or raw_num_frame <= 0:
            self.num_frame = None
            self.num_half_frames = None
        else:
            self.num_frame = int(raw_num_frame)
            self.num_half_frames = self.num_frame // 2

        self.interval_list = data_config.get('interval_list', [1])
        self.random_reverse = data_config.get('random_reverse', False)

        self.normalizing = data_config.get('normalizing', False)
        channel_count = int(data_config.get('num_channels', 3))
        self.normalize_transform = (
            transforms.Normalize(tuple([0.5] * channel_count), tuple([0.5] * channel_count))
            if self.normalizing else None
        )

        self.cache_data = bool(data_config.get('cache_data', False)) and not self.is_train
        self.gt_cache: Dict[str, np.ndarray] = {}
        self.lq_cache: Dict[str, np.ndarray] = {}
        resize_short_side = data_config.get('resize_short_side', 368)
        self.resize_short_side = (
            int(resize_short_side)
            if resize_short_side is not None and int(resize_short_side) > 0 else None
        )
        self._clip_key_cache: Dict[tuple, list] = {}

    @staticmethod
    def _filter_folder_entries(entries):
        valid_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}
        filtered_entries = []
        for path, length in entries:
            file_name = os.path.basename(path)
            if file_name.startswith('.'):
                continue
            if os.path.splitext(file_name)[1].lower() not in valid_exts:
                continue
            filtered_entries.append((path, length))
        return filtered_entries

    def _read_image_cached(self, img_path: str, cache_store: Dict[str, np.ndarray]) -> np.ndarray:
        if self.cache_data and img_path in cache_store:
            return cache_store[img_path]

        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f'无法读取图像: {img_path}')
        img = img.astype(np.float32) / 255.

        if self.cache_data:
            cache_store[img_path] = img
        return img

    def _load_frames(
        self,
        frame_paths: Sequence[str],
        cache_store: Optional[Dict[str, np.ndarray]] = None,
    ) -> List[np.ndarray]:
        if cache_store is None:
            cache_store = self.gt_cache
        return [self._read_image_cached(frame_path, cache_store) for frame_path in frame_paths]

    def _random_crop(self, img_gts, img_lqs, clip_name: str):
        gt_size = self.data_config.get('gt_size')
        if gt_size is None:
            raise ValueError('Training dataset requires gt_size to be specified.')
        gt_size_w, gt_size_h = gt_size
        return paired_random_crop(img_gts, img_lqs, gt_size_w, gt_size_h, self.scale, clip_name)

    def _augment(self, img_lqs, img_gts):
        frames = list(img_lqs) + list(img_gts)
        frames = augment(frames, self.data_config['use_flip'], self.data_config['use_rot'])
        clip_len = len(img_lqs)
        return frames[:clip_len], frames[clip_len:]

    def _get_clip_keys(self, root_path: Optional[str], train: Optional[bool] = None):
        if root_path is None:
            return []

        is_train = self.is_train if train is None else train
        cache_key = (root_path, is_train)
        if cache_key in self._clip_key_cache:
            return self._clip_key_cache[cache_key]

        if is_train:
            entries = getfilelist_with_length(root_path)
        else:
            entries = self._filter_folder_entries(getfolderlist(root_path))

        self._clip_key_cache[cache_key] = entries
        return entries

    def __len__(self):
        return len(getattr(self, 'lq_frames', []))

    @abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError
