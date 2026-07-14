from __future__ import annotations

import os
from typing import Dict, Optional

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from basicsr.utils import get_root_logger
from basicsr.utils.registry import DATASET_REGISTRY

from .base_dataset import OFRBaseDataset, resize_368_short_side
from .utils import img2tensor


@DATASET_REGISTRY.register()
class OFRTestDataset(OFRBaseDataset):
    """Film_SRWOV_dataset 的整理版实现。"""

    def __init__(self, data_config):
        super().__init__(data_config)
        if 'resize_short_side' not in data_config:
            self.resize_short_side = None

        self.scale = data_config.get('scale', 1)
        self.num_frame = None if data_config.get('num_frame', -1) <= 0 else int(data_config['num_frame'])
        self.grayscale = data_config.get('grayscale', False)
        self.align = int(data_config.get('align', 8))
        self.lq_frames = self._get_clip_keys(self.lq_root)

        logger = get_root_logger()
        logger.info(f'[OFRTestDataset] Loaded {len(self.lq_frames)} clips, align={self.align}')

    def _get_clip_keys(self, root_path: Optional[str], train: Optional[bool] = None):
        if root_path is None:
            return []
        cache_key = ('srwov', root_path)
        if cache_key in self._clip_key_cache:
            return self._clip_key_cache[cache_key]

        all_clips = []
        exts = ('.jpg', '.png', '.jpeg')

        def _list_frames(dir_path: str):
            return sorted([f for f in os.listdir(dir_path) if f.lower().endswith(exts)])

        def _safe_key_component(name: str) -> str:
            return name if not name.isdigit() else f'clip_{name}'

        for first in sorted(os.listdir(root_path)):
            first_path = os.path.join(root_path, first)
            if not os.path.isdir(first_path):
                continue

            flat_frames = _list_frames(first_path)
            if flat_frames:
                all_clips.append({
                    'key': first,
                    'video_name': first,
                    'frames_dir': first_path,
                    'frames': flat_frames,
                })
                continue

            for second in sorted(os.listdir(first_path)):
                second_path = os.path.join(first_path, second)
                if not os.path.isdir(second_path):
                    continue

                frames_path = os.path.join(second_path, 'frames')
                if os.path.isdir(frames_path):
                    frames = _list_frames(frames_path)
                    if frames:
                        all_clips.append({
                            'key': f'{first}/{_safe_key_component(second)}',
                            'video_name': f'{first}_{second}',
                            'frames_dir': frames_path,
                            'frames': frames,
                            'video_id': first,
                            'clip_id': second,
                        })
                        continue

                frames = _list_frames(second_path)
                if frames:
                    all_clips.append({
                        'key': f'{first}/{_safe_key_component(second)}',
                        'video_name': f'{first}_{second}',
                        'frames_dir': second_path,
                        'frames': frames,
                        'video_id': first,
                        'clip_id': second,
                    })

        self._clip_key_cache[cache_key] = all_clips
        return all_clips

    def _read_image_cached(self, img_path: str, cache_store: Dict[str, np.ndarray]) -> np.ndarray:
        if self.cache_data and img_path in cache_store:
            return cache_store[img_path]

        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f'无法读取图像: {img_path}')

        if self.grayscale:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        img = img.astype(np.float32) / 255.
        if self.cache_data:
            cache_store[img_path] = img
        return img

    def __getitem__(self, index):
        clip_info = self.lq_frames[index]
        frames_dir = clip_info['frames_dir']
        frames = clip_info['frames']

        if self.num_frame and len(frames) > self.num_frame:
            frames = frames[:self.num_frame]

        key = clip_info.get('key', 'sample')
        frame_list = list(range(len(frames)))

        img_lqs = []
        orig_h = None
        orig_w = None
        pad_h = 0
        pad_w = 0
        for frame_name in frames:
            img_path = os.path.join(frames_dir, frame_name)
            img = self._read_image_cached(img_path, self.lq_cache)
            if self.resize_short_side is not None:
                img = resize_368_short_side(img, short_side=self.resize_short_side, align=self.align)
            else:
                h, w = img.shape[:2]
                if orig_h is None:
                    orig_h, orig_w = h, w
                    pad_h = (self.align - (orig_h % self.align)) % self.align
                    pad_w = (self.align - (orig_w % self.align)) % self.align
                else:
                    if h != orig_h or w != orig_w:
                        raise ValueError(
                            f'Film_SRWOV_dataset expects consistent frame size within a clip, '
                            f'but got {h}x{w} vs {orig_h}x{orig_w} for {key}.'
                        )

                if pad_h or pad_w:
                    img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
            img_lqs.append(img)

        img_results = img2tensor(img_lqs)
        if self.normalize_transform is not None:
            img_results = [self.normalize_transform(t) for t in img_results]

        img_lqs = torch.stack(img_results, dim=0)
        return {
            'lq': img_lqs,
            'gt': img_lqs,
            'key': key,
            'frame_list': frame_list,
            'video_name': clip_info.get('video_name', str(key).replace('/', '_')),
            'name_list': frames,
            'orig_h': orig_h if orig_h is not None else img_lqs.size(2),
            'orig_w': orig_w if orig_w is not None else img_lqs.size(3),
            'pad_h': pad_h,
            'pad_w': pad_w,
        }


@DATASET_REGISTRY.register()
class OFRTestDatasetGray(OFRTestDataset):
    """RRTN_Film_SRWOV_dataset 的整理版实现。"""

    def __init__(self, data_config):
        super().__init__(data_config)
        self.gt_root = data_config.get('dataroot_gt')
        self.gt_cache: Dict[str, np.ndarray] = {}

        ch = 1 if self.grayscale else 3
        self.normalize_transform = (
            transforms.Normalize(tuple([0.5] * ch), tuple([0.5] * ch))
            if self.normalizing else None
        )

        logger = get_root_logger()
        logger.info(
            f'[OFRTestDatasetGray] Loaded {len(self.lq_frames)} clips, '
            f'grayscale={self.grayscale}, channels={ch}, align={self.align}'
        )

    @staticmethod
    def _ensure_channel_dim(img: np.ndarray) -> np.ndarray:
        if img.ndim == 2:
            return img[:, :, np.newaxis]
        return img

    def _read_image_cached(
        self,
        img_path: str,
        cache_store: Optional[Dict[str, np.ndarray]] = None,
    ) -> np.ndarray:
        if cache_store is None:
            cache_store = self.lq_cache

        if self.cache_data and img_path in cache_store:
            return cache_store[img_path]

        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f'无法读取图像: {img_path}')

        if self.grayscale:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = gray[:, :, np.newaxis]

        img = self._ensure_channel_dim(img.astype(np.float32) / 255.)
        if self.cache_data:
            cache_store[img_path] = img
        return img

    def _resolve_gt_path(self, clip_info: dict, frame_name: str) -> Optional[str]:
        if not self.gt_root:
            return None

        candidates = []
        if 'video_id' in clip_info and 'clip_id' in clip_info:
            video_id = clip_info['video_id']
            clip_id = clip_info['clip_id']
            candidates.extend([
                os.path.join(self.gt_root, video_id, clip_id, 'frames', frame_name),
                os.path.join(self.gt_root, video_id, clip_id, frame_name),
            ])
        else:
            candidates.append(os.path.join(self.gt_root, clip_info['key'], frame_name))

        for path in candidates:
            if os.path.isfile(path):
                return path

        raise FileNotFoundError(
            f'无法在 GT 根目录中找到匹配帧: key={clip_info.get("key")}, frame={frame_name}'
        )

    def __getitem__(self, index):
        clip_info = self.lq_frames[index]
        frames_dir = clip_info['frames_dir']
        frames = clip_info['frames']

        if self.num_frame and len(frames) > self.num_frame:
            frames = frames[:self.num_frame]

        key = clip_info.get('key', 'sample')
        frame_list = list(range(len(frames)))

        img_lqs = []
        img_gts = [] if self.gt_root else None
        orig_h = None
        orig_w = None
        pad_h = 0
        pad_w = 0

        for frame_name in frames:
            img_path = os.path.join(frames_dir, frame_name)
            img = self._read_image_cached(img_path, self.lq_cache)

            if self.resize_short_side is not None:
                img = self._ensure_channel_dim(
                    resize_368_short_side(img, short_side=self.resize_short_side, align=self.align)
                )
            else:
                h, w = img.shape[:2]
                if orig_h is None:
                    orig_h, orig_w = h, w
                    pad_h = (self.align - (orig_h % self.align)) % self.align
                    pad_w = (self.align - (orig_w % self.align)) % self.align
                elif h != orig_h or w != orig_w:
                    raise ValueError(
                        f'RRTN_Film_SRWOV_dataset expects consistent frame size within a clip, '
                        f'but got {h}x{w} vs {orig_h}x{orig_w} for {key}.'
                    )

                if pad_h or pad_w:
                    img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')

            img_lqs.append(img)

            if img_gts is None:
                continue

            gt_path = self._resolve_gt_path(clip_info, frame_name)
            gt = self._read_image_cached(gt_path, self.gt_cache)
            if self.resize_short_side is not None:
                gt = self._ensure_channel_dim(
                    resize_368_short_side(gt, short_side=self.resize_short_side, align=self.align)
                )
            elif pad_h or pad_w:
                gt = np.pad(gt, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')

            if gt.shape[:2] != img.shape[:2]:
                raise ValueError(
                    f'GT/LQ size mismatch for {key}/{frame_name}: '
                    f'LQ={img.shape[:2]}, GT={gt.shape[:2]}'
                )
            img_gts.append(gt)

        lq_results = img2tensor(img_lqs)
        gt_results = img2tensor(img_gts) if img_gts is not None else None

        if self.normalize_transform is not None:
            lq_results = [self.normalize_transform(t) for t in lq_results]
            if gt_results is not None:
                gt_results = [self.normalize_transform(t) for t in gt_results]

        img_lqs = torch.stack(lq_results, dim=0)
        img_gts = torch.stack(gt_results, dim=0) if gt_results is not None else img_lqs

        return {
            'lq': img_lqs,
            'gt': img_gts,
            'key': key,
            'frame_list': frame_list,
            'video_name': clip_info.get('video_name', str(key).replace('/', '_')),
            'name_list': frames,
            'orig_h': orig_h if orig_h is not None else img_lqs.size(2),
            'orig_w': orig_w if orig_w is not None else img_lqs.size(3),
            'pad_h': pad_h,
            'pad_w': pad_w,
        }
