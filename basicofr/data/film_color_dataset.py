"""
老电影修复 Dataset - 彩色版本

独立于 rtn_dataset.py，支持新的老电影损伤退化。
基于 Film_dataset_1 的结构，集成 FilmDamageGenerator。

特点：GT 保留彩色，LQ 为灰度，用于训练彩色化+修复联合任务。

Author: BasicOFR Team
Date: 2025-12-24
"""

import operator
import os
import random
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils import data

from basicsr.utils import get_root_logger
from basicsr.utils.registry import DATASET_REGISTRY

from .degradations.film_degradation import film_degradation_video_list
from .utils import augment, img2tensor, paired_random_crop


def getfilelist_with_length(file_path: str) -> List[tuple]:
    """获取文件列表及其所在目录的文件数量"""
    all_file = []
    for dir_path, _, files in os.walk(file_path):
        for f in files:
            t = os.path.join(dir_path, f)
            all_file.append((t, len(os.listdir(dir_path))))
    all_file.sort(key=operator.itemgetter(0))
    return all_file


def getfolderlist(file_path: str) -> List[tuple]:
    """获取文件夹列表（首文件路径 + 文件数量）"""
    all_folder = []
    for dir_path, _, files in os.walk(file_path):
        if not files:
            continue
        rerank = sorted(files)
        if rerank[0].endswith('.avi'):
            continue
        t = os.path.join(dir_path, rerank[0])
        all_folder.append((t, len(files)))
    all_folder.sort(key=operator.itemgetter(0))
    return all_folder


def resize_and_align(img: np.ndarray, short_side: int = 368, align: int = 16) -> np.ndarray:
    """保持纵横比缩放，短边对齐到指定值，并对齐到 align 的倍数

    Args:
        img: numpy 数组 (H, W, C)，float32 格式 [0, 1]
        short_side: 短边目标长度
        align: 对齐倍数

    Returns:
        缩放后的 numpy 数组
    """
    h, w = img.shape[:2]

    if w < h:
        new_w = short_side
        new_h = int(short_side * h / w)
    else:
        new_h = short_side
        new_w = int(short_side * w / h)

    # 对齐
    new_h = new_h // align * align
    new_w = new_w // align * align

    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


@DATASET_REGISTRY.register()
class Film_dataset_damage_color(data.Dataset):
    """支持老电影损伤退化的 Dataset - 彩色版本

    特点：GT 保留彩色，LQ 为灰度，用于训练彩色化+修复联合任务。

    相比 Film_dataset_1，主要改进：
    1. 使用 film_degradation_video_list 替代 degradation_video_list_4
    2. 支持划痕/污渍/灰尘/毛发/闪烁等老电影特有退化
    3. 支持帧间时序一致性（划痕延续）
    4. GT 保留原始彩色

    配置示例:
        type: Film_dataset_damage_color
        dataroot_gt: datasets/REDS/train_sharp
        dataroot_lq: datasets/REDS/train_sharp  # 在线生成 LQ
        num_frame: 5
        gt_size: [256, 256]
        texture_template: datasets/noise_data

        film_damage:
          enabled: true
          assets_root: datasets/film_damage_assets
          scratch:
            enabled: true
            prob: 0.7
            num_range: [1, 5]
          dirt:
            enabled: true
            prob: 0.5
          dust:
            enabled: true
            prob: 0.6
          hair:
            enabled: true
            prob: 0.3
          flicker:
            enabled: true
            prob: 0.4
            intensity_range: [-0.15, 0.15]
    """

    def __init__(self, data_config: dict):
        super().__init__()

        self.data_config = data_config
        self.opt = data_config

        self.scale = data_config.get('scale', 1)
        self.gt_root = data_config['dataroot_gt']
        self.lq_root = data_config.get('dataroot_lq', self.gt_root)
        self.is_train = data_config.get('is_train', False)

        # 帧数配置
        raw_num_frame = data_config.get('num_frame', None)
        if raw_num_frame is None or raw_num_frame <= 0:
            self.num_frame = None
            self.num_half_frames = None
        else:
            self.num_frame = int(raw_num_frame)
            self.num_half_frames = self.num_frame // 2

        # 文件列表
        if self.is_train:
            self.lq_frames = getfilelist_with_length(self.lq_root)
            self.gt_frames = getfilelist_with_length(self.gt_root)
        else:
            self.lq_frames = getfolderlist(self.lq_root)
            self.gt_frames = getfolderlist(self.gt_root)

        # 时间增强配置
        self.interval_list = data_config.get('interval_list', [1])
        self.random_reverse = data_config.get('random_reverse', False)

        # 归一化配置
        self.normalizing = data_config.get('normalizing', False)
        self.normalize_transform = transforms.Normalize(
            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        ) if self.normalizing else None

        # 纹理配置
        self.texture_url = data_config.get('texture_template', './noise_data')

        # 老电影损伤配置
        self.film_damage_config = data_config.get('film_damage', {})
        self.degradation_strength = float(self.film_damage_config.get('degradation_strength', 1.0))

        # 日志
        logger = get_root_logger()
        if self.is_train:
            if self.num_frame:
                interval_str = ','.join(str(x) for x in self.interval_list)
                logger.info(
                    f'[Film_dataset_damage_color] Temporal augmentation interval: [{interval_str}]; '
                    f'Random reverse: {self.random_reverse}'
                )
            else:
                logger.info('[Film_dataset_damage_color] Using full clips (num_frame <= 0)')

            if self.film_damage_config.get('enabled', False):
                logger.info(
                    f'[Film_dataset_damage_color] Film damage enabled: '
                    f'scratch={self.film_damage_config.get("scratch", {}).get("enabled", True)}, '
                    f'dirt={self.film_damage_config.get("dirt", {}).get("enabled", True)}, '
                    f'dust={self.film_damage_config.get("dust", {}).get("enabled", True)}, '
                    f'hair={self.film_damage_config.get("hair", {}).get("enabled", True)}, '
                    f'flicker={self.film_damage_config.get("flicker", {}).get("enabled", True)}'
                )
            else:
                logger.info('[Film_dataset_damage_color] Film damage disabled, using standard degradation')

        if self.normalizing:
            logger.info(f'[Film_dataset_damage_color] Normalization enabled')

    def __len__(self):
        return len(self.gt_frames)

    def __getitem__(self, index: int) -> dict:
        gt_size = self.data_config.get('gt_size')
        if gt_size is not None:
            gt_size_w, gt_size_h = gt_size
        else:
            gt_size_w = gt_size_h = None

        key = self.gt_frames[index][0]
        current_len = self.gt_frames[index][1]

        # 获取目录路径
        current_gt_root = os.path.dirname(os.path.dirname(self.gt_frames[index][0]))
        current_lq_root = os.path.dirname(os.path.dirname(self.lq_frames[index][0]))

        clip_name, frame_name = key.split('/')[-2:]
        key = f"{clip_name}/{frame_name}"
        center_frame_idx = int(frame_name[:-4])

        new_clip_sequence = sorted(os.listdir(os.path.join(current_gt_root, clip_name)))

        # 验证集截断
        if not self.is_train and self.num_frame and len(new_clip_sequence) > self.num_frame:
            new_clip_sequence = new_clip_sequence[:self.num_frame]
            current_len = len(new_clip_sequence)

        # 确定帧列表
        if self.is_train and self.num_frame:
            interval = random.choice(self.interval_list)

            start_frame_idx = center_frame_idx - self.num_half_frames * interval
            end_frame_idx = start_frame_idx + (self.num_frame - 1) * interval

            while (start_frame_idx < 0) or (end_frame_idx > current_len - 1):
                center_frame_idx = random.randint(
                    self.num_half_frames * interval,
                    current_len - self.num_half_frames * interval
                )
                start_frame_idx = center_frame_idx - self.num_half_frames * interval
                end_frame_idx = start_frame_idx + (self.num_frame - 1) * interval

            frame_list = list(range(start_frame_idx, end_frame_idx + 1, interval))
            assert len(frame_list) == self.num_frame
        elif self.is_train:
            frame_list = list(range(current_len))
        else:
            frame_list = list(range(center_frame_idx, center_frame_idx + current_len))
            assert len(frame_list) == current_len

        # 随机翻转
        if self.random_reverse and random.random() < 0.5:
            frame_list.reverse()

        # 加载 GT 帧
        img_gts = []
        img_lqs = []

        for tmp_id, frame in enumerate(frame_list):
            img_gt_path = os.path.join(current_gt_root, clip_name, new_clip_sequence[tmp_id])
            img_gt = cv2.imread(img_gt_path)
            if img_gt is None:
                raise FileNotFoundError(f"无法读取图像: {img_gt_path}")
            img_gt = img_gt.astype(np.float32) / 255.0
            img_gts.append(img_gt)

            if not self.is_train:
                img_lq_path = os.path.join(current_lq_root, clip_name, new_clip_sequence[tmp_id])
                img_lq = cv2.imread(img_lq_path)
                if img_lq is None:
                    raise FileNotFoundError(f"无法读取图像: {img_lq_path}")
                img_lq = img_lq.astype(np.float32) / 255.0
                img_lqs.append(img_lq)

        # 训练阶段：在线生成退化
        if self.is_train:
            if gt_size is None:
                raise ValueError('Training dataset requires gt_size to be specified.')

            img_lqs = img_gts
            img_gts, img_lqs = paired_random_crop(
                img_gts, img_lqs, gt_size_w, gt_size_h, self.scale, clip_name
            )

            # 使用新的老电影退化流水线
            img_lqs, img_gts = film_degradation_video_list(
                img_gts,
                texture_url=self.texture_url,
                film_damage_config=self.film_damage_config,
                degradation_strength=self.degradation_strength,
            )
        else:
            # 验证阶段：resize
            for i in range(len(img_gts)):
                img_gts[i] = resize_and_align(img_gts[i])
                img_lqs[i] = resize_and_align(img_lqs[i])

        # 数据增强
        img_lqs.extend(img_gts)
        if self.is_train:
            img_lqs = augment(
                img_lqs,
                self.data_config.get('use_flip', True),
                self.data_config.get('use_rot', True),
            )

        # 转为 tensor
        img_results = img2tensor(img_lqs)

        # 归一化
        if self.normalize_transform is not None:
            img_results = [self.normalize_transform(t) for t in img_results]

        # 分离 LQ 和 GT
        clip_len = len(img_gts)
        if self.is_train:
            img_lqs = torch.stack(img_results[:clip_len], dim=0)
            img_gts = torch.stack(img_results[clip_len:], dim=0)
        else:
            img_lqs = torch.stack(img_results[:current_len], dim=0)
            img_gts = torch.stack(img_results[current_len:], dim=0)

        return {
            'lq': img_lqs,
            'gt': img_gts,
            'key': key,
            'frame_list': frame_list,
            'video_name': os.path.basename(current_lq_root),
            'name_list': new_clip_sequence,
        }
