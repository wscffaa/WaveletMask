"""MambaOFR 数据集

独立于 RTN 数据管线的视频数据集实现，特点：
1. 清晰的目录结构扫描
2. 验证阶段也支持归一化
3. 更好的缓存机制
4. 现代化代码风格
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from basicsr.utils import get_root_logger
from basicsr.utils.registry import DATASET_REGISTRY
from torch.utils import data as data

from .degradations.degradation import degradation_video_list_4
from .utils import augment, img2tensor, paired_random_crop


def _scan_clip_frames(root: str) -> Dict[str, List[str]]:
    """遍历根目录下的所有序列，返回 {clip_name: [frame_name, ...]} 映射

    支持两种目录结构：
    1. root/clip_name/frame.png
    2. root/frame.png（直接存放帧）
    """
    clip_map: Dict[str, List[str]] = {}
    root_path = Path(root)

    # 检查根目录下是否直接存放帧文件
    root_files = sorted([p.name for p in root_path.iterdir() if p.is_file()])
    if root_files:
        clip_map[root_path.name] = root_files

    # 扫描子目录
    for clip_dir in sorted([p for p in root_path.iterdir() if p.is_dir()]):
        frame_names = sorted([p.name for p in clip_dir.iterdir() if p.is_file()])
        if frame_names:
            clip_map[clip_dir.name] = frame_names

    return clip_map


def _resize_short_side(img: np.ndarray, short_size: int = 368, align: int = 16) -> np.ndarray:
    """保持纵横比缩放，短边对齐到 short_size，并向下取整到 align 的倍数"""
    h, w = img.shape[:2]
    if w < h:
        new_w = short_size
        new_h = int(short_size * h / w)
    else:
        new_h = short_size
        new_w = int(short_size * w / h)
    if align > 0:
        new_h = new_h // align * align
        new_w = new_w // align * align
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def _frame_number(name: str) -> int:
    """将帧文件名转换为整数编号，失败时返回 0"""
    stem = Path(name).stem
    try:
        return int(stem)
    except ValueError:
        return 0


@DATASET_REGISTRY.register()
class FilmDatasetMambaOFR(data.Dataset):
    """MambaOFR 风格的视频数据集

    使用在线退化，支持训练和验证阶段的归一化。

    Args:
        data_config: 数据集配置字典，包含：
            - dataroot_gt: GT 数据路径
            - dataroot_lq: LQ 数据路径（可选，默认与 GT 相同）
            - is_train: 是否为训练模式
            - num_frame: 采样帧数（<=0 表示使用全序列）
            - gt_size: 裁剪尺寸 [w, h]
            - scale: 上采样倍数
            - interval_list: 时间间隔列表
            - random_reverse: 是否随机反转
            - use_flip: 是否使用翻转增强
            - use_rot: 是否使用旋转增强
            - normalizing: 是否归一化到 [-1, 1]
            - texture_template: 退化纹理模板路径
            - cache_data: 是否缓存数据（仅验证阶段）
    """

    def __init__(self, data_config: dict):
        super().__init__()

        self.data_config = data_config
        self.opt = data_config

        self.scale = data_config.get('scale', 1)
        self.gt_root = data_config['dataroot_gt']
        self.lq_root = data_config.get('dataroot_lq', self.gt_root)
        self.is_train = data_config.get('is_train', False)
        self.texture_template = data_config.get('texture_template', 'datasets/VSR/noise_data')
        self.use_flip = data_config.get('use_flip', False)
        self.use_rot = data_config.get('use_rot', False)
        self.normalizing = data_config.get('normalizing', False)

        # 帧数配置
        raw_num_frame = data_config.get('num_frame', None)
        if raw_num_frame is None or raw_num_frame <= 0:
            self.num_frame = None
            self.num_half_frames = None
        else:
            self.num_frame = int(raw_num_frame)
            self.num_half_frames = self.num_frame // 2

        # 裁剪尺寸
        self.gt_size = data_config.get('gt_size')
        if self.gt_size is not None:
            self.gt_size_w, self.gt_size_h = self.gt_size
        else:
            self.gt_size_w = self.gt_size_h = None

        # 时序增强
        self.interval_list = data_config.get('interval_list', [1])
        self.random_reverse = data_config.get('random_reverse', False)

        # 扫描序列结构
        self.clip_frames = _scan_clip_frames(self.gt_root)
        self.clip_order = sorted(self.clip_frames.keys())
        if not self.clip_frames:
            raise ValueError(f'未在 {self.gt_root} 下找到有效视频序列。')

        # 构建索引
        if self.is_train:
            self.samples = self._build_train_indices()
        else:
            self.samples = self._build_eval_indices()

        # 日志
        logger = get_root_logger()
        if self.is_train:
            logger.info('[MambaOFR Dataset] 使用在线退化模式')
            if self.num_frame:
                interval_str = ','.join(str(x) for x in self.interval_list)
                logger.info(
                    f'[MambaOFR Dataset] interval_list: [{interval_str}]; random_reverse={self.random_reverse}.'
                )
            else:
                logger.info('[MambaOFR Dataset] 使用全序列训练，禁用时间采样。')

        # 缓存（仅验证阶段）
        self.cache_data = bool(data_config.get('cache_data', False)) and not self.is_train
        self.gt_cache: Dict[str, Dict[str, np.ndarray]] = {}
        self.lq_cache: Dict[str, Dict[str, np.ndarray]] = {}
        if self.cache_data:
            self._preload_cache()

        # 归一化变换
        self.normalize_transform = transforms.Normalize(
            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        ) if self.normalizing else None

    def _build_train_indices(self) -> List[Dict[str, str]]:
        """训练集：按帧展开，每个样本对应一个中心帧"""
        samples: List[Dict[str, str]] = []
        for clip, frame_names in self.clip_frames.items():
            for frame_name in frame_names:
                samples.append({
                    'clip': clip,
                    'frame_name': frame_name,
                    'length': len(frame_names)
                })
        return samples

    def _build_eval_indices(self) -> List[Dict[str, str]]:
        """验证集：每个序列返回一次，以首帧作为 key 标识"""
        samples: List[Dict[str, str]] = []
        for clip, frame_names in self.clip_frames.items():
            samples.append({
                'clip': clip,
                'frame_name': frame_names[0],
                'length': len(frame_names)
            })
        return samples

    def _read_image(
        self,
        root_dir: str,
        clip: str,
        frame_name: str,
        cache_store: Dict[str, Dict[str, np.ndarray]]
    ) -> np.ndarray:
        """读取单帧，并可选缓存到内存"""
        if self.cache_data and clip in cache_store and frame_name in cache_store[clip]:
            return cache_store[clip][frame_name]

        img_path = os.path.join(root_dir, clip, frame_name)
        if not os.path.exists(img_path):
            # 根目录直接存帧时 clip == 根目录名，需要降一级
            alt_path = os.path.join(root_dir, frame_name)
            if os.path.exists(alt_path):
                img_path = alt_path

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f'无法读取图像: {img_path}')
        img = img.astype(np.float32) / 255.

        if self.cache_data:
            cache_store.setdefault(clip, {})[frame_name] = img
        return img

    def _preload_cache(self):
        """预加载验证集帧，减少磁盘 IO"""
        for clip, frame_names in self.clip_frames.items():
            for frame_name in frame_names:
                self._read_image(self.gt_root, clip, frame_name, self.gt_cache)
                self._read_image(self.lq_root, clip, frame_name, self.lq_cache)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[index]
        clip_name = sample['clip']
        center_frame_name = sample['frame_name']
        all_frames = self.clip_frames[clip_name]
        current_len = len(all_frames)

        try:
            center_pos = all_frames.index(center_frame_name)
        except ValueError:
            center_pos = 0

        # ========== 帧采样 ==========
        if self.is_train and self.num_frame:
            interval = random.choice(self.interval_list)
            start_frame_idx = center_pos - self.num_half_frames * interval
            end_frame_idx = start_frame_idx + (self.num_frame - 1) * interval

            # 确保不越界
            while start_frame_idx < 0 or end_frame_idx > current_len - 1:
                center_pos = random.randint(
                    self.num_half_frames * interval,
                    current_len - self.num_half_frames * interval
                )
                start_frame_idx = center_pos - self.num_half_frames * interval
                end_frame_idx = start_frame_idx + (self.num_frame - 1) * interval

            frame_indices = list(range(start_frame_idx, end_frame_idx + 1, interval))
        elif self.is_train:
            # 全序列训练
            frame_indices = list(range(current_len))
        else:
            # 验证模式
            if self.num_frame and current_len > self.num_frame:
                frame_indices = list(range(self.num_frame))
                current_len = len(frame_indices)
            else:
                frame_indices = list(range(current_len))

        frame_names = [all_frames[i] for i in frame_indices]
        frame_list = [_frame_number(name) for name in frame_names]

        # 随机反转
        if self.random_reverse and self.is_train and random.random() < 0.5:
            frame_indices.reverse()
            frame_names.reverse()
            frame_list.reverse()

        # ========== 读取帧数据 ==========
        img_gts: List[np.ndarray] = []
        img_lqs: List[np.ndarray] = []

        for frame_name in frame_names:
            img_gt = self._read_image(self.gt_root, clip_name, frame_name, self.gt_cache)
            img_gts.append(img_gt)
            if not self.is_train:
                img_lq = self._read_image(self.lq_root, clip_name, frame_name, self.lq_cache)
                img_lqs.append(img_lq)

        # ========== 训练/验证处理 ==========
        if self.is_train:
            if self.gt_size_w is None or self.gt_size_h is None:
                raise ValueError('训练模式必须提供 gt_size。')

            # 在线退化：先裁剪，再退化
            img_lqs = img_gts
            img_gts, img_lqs = paired_random_crop(
                img_gts, img_lqs, self.gt_size_w, self.gt_size_h, self.scale, clip_name
            )

            # degradation_video_list_4：使用 add_sharpening 对 GT 进行锐化
            img_lqs, img_gts = degradation_video_list_4(
                img_gts,
                texture_url=self.texture_template
            )
        else:
            # 验证集：resize_short_side
            img_gts = [_resize_short_side(img) for img in img_gts]
            img_lqs = [_resize_short_side(img) for img in img_lqs]

        # ========== 数据增强与转换 ==========
        img_lqs.extend(img_gts)
        if self.is_train:
            img_lqs = augment(img_lqs, self.use_flip, self.use_rot)

        img_results = img2tensor(img_lqs)

        # 归一化（训练和验证均可）
        if self.normalize_transform is not None:
            img_results = [self.normalize_transform(t) for t in img_results]

        clip_len = len(frame_names)
        img_lqs = torch.stack(img_results[:clip_len], dim=0)
        img_gts = torch.stack(img_results[clip_len:], dim=0)

        key = f'{clip_name}/{center_frame_name}'
        return {
            'lq': img_lqs,
            'gt': img_gts,
            'key': key,
            'frame_list': frame_list,
            'video_name': clip_name,
            'name_list': frame_names
        }

    def __len__(self) -> int:
        return len(self.samples)
