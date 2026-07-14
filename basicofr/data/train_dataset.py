from __future__ import annotations

import os
import random

import torch
from basicsr.utils import get_root_logger
from basicsr.utils.registry import DATASET_REGISTRY

from .base_dataset import OFRBaseDataset, resize_368_short_side
from .degradations.core import (
    degradation_video_list_4_one_channel,
    degradation_video_list_5,
    standard_degradation_pipeline,
    transfer_1,
    transfer_2,
)
from .utils import img2tensor


def convert_to_L(img):
    frame_pil = transfer_1(img)
    frame_cv2 = transfer_2(frame_pil.convert('RGB'))
    return frame_cv2


@DATASET_REGISTRY.register()
class OFRTrainDataset(OFRBaseDataset):

    def __init__(self, data_config):
        super().__init__(data_config)
        self.gt_frames = self._get_clip_keys(self.gt_root, train=self.is_train)
        self.lq_frames = self._get_clip_keys(self.lq_root, train=self.is_train)

        if not self.is_train:
            self.gt_frames = self._get_clip_keys(self.gt_root, train=False)
            self.lq_frames = self._get_clip_keys(self.lq_root, train=False)
            if len(self.lq_frames) != len(self.gt_frames):
                raise ValueError(
                    f'OFRTrainDataset clip count mismatch: gt={len(self.gt_frames)}, '
                    f'lq={len(self.lq_frames)}; gt_root={self.gt_root}, lq_root={self.lq_root}'
                )

        if self.is_train and self.lq_root != self.gt_root:
            sentinel = os.path.join(self.lq_root, '.TRAIN_LQ_IS_PLACEHOLDER')
            if not os.path.exists(sentinel):
                raise RuntimeError(
                    f'OFRTrainDataset safety guard: missing {sentinel}. '
                    f'When lq_root differs from gt_root in train mode, the LQ directory '
                    f'must be a GT-mirror placeholder (training uses online degradation). '
                    f'If this LQ directory contains real pre-generated LQ, remove it and '
                    f'create the sentinel file to acknowledge the placeholder. '
                    f'lq_root={self.lq_root}'
                )

        logger = get_root_logger()
        if self.is_train and self.num_frame:
            interval_str = ','.join(str(x) for x in self.interval_list)
            logger.info(
                f'Temporal augmentation interval list: [{interval_str}]; '
                f'Random reverse is {self.random_reverse}.'
            )
        elif self.is_train:
            logger.info('Training dataset using full clips (num_frame <= 0); temporal augmentation is disabled.')
        elif 'interval_list' not in data_config:
            logger.info('Validation dataset missing interval_list; defaulting to [1].')

        if self.normalizing:
            logger.info(f'[OFRTrainDataset] Normalization enabled (train={self.is_train}).')
        if self.cache_data:
            logger.info('[OFRTrainDataset] Cache enabled for validation data.')

    def _build_frame_list(self, center_frame_idx: int, current_len: int):
        if self.is_train and self.num_frame:
            interval = random.choice(self.interval_list)
            start_frame_idx = center_frame_idx - self.num_half_frames * interval
            end_frame_idx = start_frame_idx + (self.num_frame - 1) * interval

            while (start_frame_idx < 0) or (end_frame_idx > current_len - 1):
                center_frame_idx = random.randint(
                    self.num_half_frames * interval,
                    current_len - self.num_half_frames * interval,
                )
                start_frame_idx = center_frame_idx - self.num_half_frames * interval
                end_frame_idx = start_frame_idx + (self.num_frame - 1) * interval

            frame_list = list(range(start_frame_idx, end_frame_idx + 1, interval))
            assert len(frame_list) == self.num_frame, f'Wrong length of frame list: {len(frame_list)}'
            return frame_list

        if self.is_train:
            return list(range(current_len))

        frame_list = list(range(center_frame_idx, center_frame_idx + current_len))
        assert len(frame_list) == current_len, f'Wrong length of frame list: {len(frame_list)}'
        return frame_list

    def _select_degradation(self, img_gts):
        version = self.opt.get('degradation_version', self.opt.get('degradation_pipeline', 'v4'))
        texture_url = self.opt['texture_template']

        if version in {'v4', 'standard', 'standard_degradation_pipeline', 'degradation_video_list_4'}:
            return standard_degradation_pipeline(img_gts, texture_url=texture_url)
        if version in {'v4_one_channel', 'one_channel', 'gray'}:
            return degradation_video_list_4_one_channel(img_gts, texture_url=texture_url)
        if version in {'v5', 'degradation_video_list_5'}:
            degree = int(self.opt.get('degradation_degree', self.opt.get('degree', 1)))
            return degradation_video_list_5(img_gts, degree=degree, texture_url=texture_url)

        raise ValueError(f'Unsupported degradation version: {version}')

    def __getitem__(self, index):
        key = self.gt_frames[index][0]
        current_len = self.gt_frames[index][1]

        current_gt_root = os.path.dirname(os.path.dirname(self.gt_frames[index][0]))
        current_lq_root = os.path.dirname(os.path.dirname(self.lq_frames[index][0]))

        clip_name, frame_name = key.split('/')[-2:]
        key = clip_name + '/' + frame_name
        center_frame_idx = int(frame_name[:-4])

        new_clip_sequence = sorted(os.listdir(os.path.join(current_gt_root, clip_name)))
        if not self.is_train and self.num_frame and len(new_clip_sequence) > self.num_frame:
            new_clip_sequence = new_clip_sequence[:self.num_frame]
            current_len = len(new_clip_sequence)

        frame_list = self._build_frame_list(center_frame_idx, current_len)
        if self.random_reverse and random.random() < 0.5:
            frame_list.reverse()

        gt_paths = [
            os.path.join(current_gt_root, clip_name, new_clip_sequence[tmp_id])
            for tmp_id, _ in enumerate(frame_list)
        ]
        img_gts = self._load_frames(gt_paths, self.gt_cache)
        img_lqs = []

        if not self.is_train:
            lq_paths = [
                os.path.join(current_lq_root, clip_name, new_clip_sequence[tmp_id])
                for tmp_id, _ in enumerate(frame_list)
            ]
            img_lqs = self._load_frames(lq_paths, self.lq_cache)

        if self.is_train:
            img_lqs = img_gts
            img_gts, img_lqs = self._random_crop(img_gts, img_lqs, clip_name)
            img_lqs, img_gts = self._select_degradation(img_gts)
        else:
            for i in range(len(img_gts)):
                if self.resize_short_side is not None:
                    img_gts[i] = resize_368_short_side(img_gts[i], short_side=self.resize_short_side)
                    img_lqs[i] = resize_368_short_side(img_lqs[i], short_side=self.resize_short_side)
                if self.data_config['name'] == 'colorization':
                    img_gts[i] = convert_to_L(img_gts[i])
                    img_lqs[i] = convert_to_L(img_lqs[i])

        if self.is_train:
            img_lqs, img_gts = self._augment(img_lqs, img_gts)

        img_results = img2tensor(list(img_lqs) + list(img_gts))
        if self.normalize_transform is not None:
            img_results = [self.normalize_transform(t) for t in img_results]

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
