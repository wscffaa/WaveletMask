import operator
import os
import random
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from skimage.color import rgb2lab
from torch.utils import data as data

from basicsr.utils import get_root_logger
from basicsr.utils.registry import DATASET_REGISTRY

from .degradations.degradation import (
    degradation_video_colorization,
    degradation_video_colorization_v2,
    degradation_video_colorization_v3,
    degradation_video_colorization_v4,
    degradation_video_list,
    degradation_video_list_2,
    degradation_video_list_3,
    degradation_video_list_4,
    degradation_video_list_simple_debug,
    transfer_1,
    transfer_2,
)
from .utils import Normalize_LAB, augment, img2tensor, paired_random_crop, to_mytensor

def getfilelist(file_path):
    all_file = []
    for dir,folder,file in os.walk(file_path):
        for i in file:
            t = "%s/%s"%(dir,i)
            all_file.append(t)
    all_file = sorted(all_file)
    return all_file

def getfilelist_with_length(file_path):
    all_file = []
    for dir,folder,file in os.walk(file_path):
        for i in file:
            t = "%s/%s"%(dir,i)
            all_file.append((t,len(os.listdir(dir))))

    all_file.sort(key = operator.itemgetter(0))
    return all_file

def getfolderlist(file_path):

    all_folder = []
    for dir,folder,file in os.walk(file_path):
        if len(file)==0:
            continue
        rerank = sorted(file)
        t = "%s/%s"%(dir,rerank[0])
        if t.endswith('.avi'):
            continue
        all_folder.append((t,len(file)))
    
    all_folder.sort(key = operator.itemgetter(0))
    # all_folder = sorted(all_folder)
    return all_folder


def convert_to_L(img):

    frame_pil = transfer_1(img)
    frame_cv2 = transfer_2(frame_pil.convert("RGB"))

    return frame_cv2


def resize_256_short_side(img):
    width, height = img.size

    if width<height:
        new_height =  int (256 * height / width)
        new_width = 256
    else:
        new_width =  int (256 * width / height)
        new_height = 256
    
    return img.resize((new_width,new_height),resample=Image.BILINEAR)


def resize_368_short_side(img, short_side: int = 368, align: int = 16):
    """保持纵横比缩放，短边对齐到指定尺寸，并向下取整到 align 的倍数

    改进版本：直接使用 cv2 处理 numpy 数组，避免 PIL 转换开销

    Args:
        img: numpy 数组 (H, W, C)，float32 格式 [0, 1]
        short_side: 短边目标尺寸
        align: 输出尺寸向下取整到该倍数

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

    # 对齐到指定倍数（避免下采样/patch 切分带来的尺寸不整除）
    new_h = new_h // align * align
    new_w = new_w // align * align

    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

# def getfolderlist_with_length(file_path):
#     all_folder = []
#     for dir,folder,file in os.walk(file_path):
#         for i in folder:
#             t = "%s/%s"%(dir,i)
#             all_folder.append((t,len(os.listdir(t))))
#     all_folder.sort(key = operator.itemgetter(0))
#     return all_folder


@DATASET_REGISTRY.register()
class Film_dataset_1(data.Dataset):  # 1 for REDS dataset

    def __init__(self, data_config):
        super(Film_dataset_1, self).__init__()

        self.data_config = data_config
        self.opt = data_config

        self.scale = data_config['scale']
        self.gt_root, self.lq_root = data_config['dataroot_gt'], data_config['dataroot_lq']
        self.is_train = data_config.get('is_train', False)

        raw_num_frame = data_config.get('num_frame', None)
        if raw_num_frame is None or raw_num_frame <= 0:
            self.num_frame = None
            self.num_half_frames = None
        else:
            self.num_frame = int(raw_num_frame)
            self.num_half_frames = self.num_frame // 2

        if self.is_train:
            self.lq_frames = getfilelist_with_length(self.lq_root)
            self.gt_frames = getfilelist_with_length(self.gt_root)
        else:
            ## Now: Append the first frame name, then load all frames based on the clip length
            self.lq_frames = self._filter_folder_entries(getfolderlist(self.lq_root))
            self.gt_frames = self._filter_folder_entries(getfolderlist(self.gt_root))
            if len(self.lq_frames) != len(self.gt_frames):
                raise ValueError(
                    f'Film_dataset_1 clip count mismatch: gt={len(self.gt_frames)}, lq={len(self.lq_frames)}; '
                    f'gt_root={self.gt_root}, lq_root={self.lq_root}'
                )
            # self.lq_frames = []
            # self.gt_frames = []
            # for i in range(len(self.lq_folders))
            #     val_frame_list_this = sorted(os.listdir(self.lq_folders[i]))
            #     first_frame_name = val_frame_list_this[0]
            #     clip_length = len(val_frame_list_this)
            #     self.lq_frames.append((os.path.join(self.lq_folders[i],f'{first_frame_name:08d}.png'),clip_length))
            #     self.gt_frames.append((os.path.join(self.gt_folders[i],f'{first_frame_name:08d}.png'),clip_length))

        # temporal augmentation configs
        self.interval_list = data_config.get('interval_list', [1])
        self.random_reverse = data_config.get('random_reverse', False)

        # ========== 归一化配置（训练和验证均可使用） ==========
        self.normalizing = data_config.get('normalizing', False)
        self.normalize_transform = transforms.Normalize(
            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        ) if self.normalizing else None

        # ========== 缓存机制（仅验证阶段，减少磁盘 IO） ==========
        self.cache_data = bool(data_config.get('cache_data', False)) and not self.is_train
        self.gt_cache: Dict[str, np.ndarray] = {}
        self.lq_cache: Dict[str, np.ndarray] = {}

        resize_short_side = data_config.get('resize_short_side', 368)
        self.resize_short_side = int(resize_short_side) if resize_short_side is not None and int(resize_short_side) > 0 else None

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
            logger.info(f'[Film_dataset_1] Normalization enabled (train={self.is_train}).')
        if self.cache_data:
            logger.info(f'[Film_dataset_1] Cache enabled for validation data.')

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
        """读取图像，支持缓存

        Args:
            img_path: 图像路径
            cache_store: 缓存字典

        Returns:
            归一化后的图像 numpy 数组 (H, W, C)，float32 [0, 1]
        """
        if self.cache_data and img_path in cache_store:
            return cache_store[img_path]

        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f'无法读取图像: {img_path}')
        img = img.astype(np.float32) / 255.

        if self.cache_data:
            cache_store[img_path] = img
        return img

    def __getitem__(self, index):

        gt_size = self.data_config.get('gt_size')
        if gt_size is not None:
            gt_size_w, gt_size_h = gt_size
        else:
            gt_size_w = gt_size_h = None

        key = self.gt_frames[index][0]
        current_len = self.gt_frames[index][1]

        ## Fetch the parent directory of clip name
        current_gt_root = os.path.dirname(os.path.dirname(self.gt_frames[index][0]))
        current_lq_root = os.path.dirname(os.path.dirname(self.lq_frames[index][0]))

        clip_name, frame_name = key.split('/')[-2:]  # key example: 000/00000000
        key = clip_name + "/" + frame_name
        center_frame_idx = int(frame_name[:-4])

        new_clip_sequence = sorted(os.listdir(os.path.join(current_gt_root, clip_name)))
        if not self.is_train and self.num_frame and len(new_clip_sequence) > self.num_frame:
            new_clip_sequence = new_clip_sequence[:self.num_frame]
            current_len = len(new_clip_sequence)
        if self.is_train and self.num_frame:
            # determine the frameing frames
            interval = random.choice(self.interval_list)

            # ensure not exceeding the borders
            start_frame_idx = center_frame_idx - self.num_half_frames * interval
            end_frame_idx = start_frame_idx + (self.num_frame - 1) * interval

            # each clip has 100 frames starting from 0 to 99. TODO: if the training clip is not 100 frames [√]
            # Training start frames should be 0
            while (start_frame_idx < 0) or (end_frame_idx > current_len-1):
                center_frame_idx = random.randint(self.num_half_frames * interval, current_len - self.num_half_frames *interval)
                start_frame_idx = (center_frame_idx - self.num_half_frames * interval)
                end_frame_idx = start_frame_idx + (self.num_frame - 1) * interval
            
            # frame_name = f'{center_frame_idx:08d}'
            frame_list = list(range(start_frame_idx, end_frame_idx + 1, interval))
            # Sample number should equal to the numer we set
            assert len(frame_list) == self.num_frame, (f'Wrong length of frame list: {len(frame_list)}')
        elif self.is_train:
            frame_list = list(range(current_len))
        else:
            frame_list = list(range(center_frame_idx, center_frame_idx + current_len))
            # Sample number should equal to the all frames number in on folder
            assert len(frame_list) == current_len, (f'Wrong length of frame list: {len(frame_list)}')

        # random reverse
        if self.random_reverse and random.random() < 0.5:
            frame_list.reverse()


        # get the GT frame (as the center frame)
        img_gts = []
        img_lqs = []
        for tmp_id, frame in enumerate(frame_list):

            # img_gt_path = os.path.join(current_gt_root, clip_name, f'{frame:05d}.png')
            img_gt_path = os.path.join(current_gt_root, clip_name, new_clip_sequence[tmp_id])
            img_gt = cv2.imread(img_gt_path)
            img_gt = img_gt.astype(np.float32) / 255.
            img_gts.append(img_gt)

            if not self.is_train:
                img_lq_path = os.path.join(current_lq_root, clip_name, new_clip_sequence[tmp_id])
                img_lq = cv2.imread(img_lq_path)
                img_lq = img_lq.astype(np.float32) / 255.
                img_lqs.append(img_lq)

        if self.is_train:
            if gt_size is None:
                raise ValueError('Training dataset requires gt_size to be specified.')

            img_lqs = img_gts
            img_gts, img_lqs = paired_random_crop(img_gts, img_lqs, gt_size_w, gt_size_h, self.scale, clip_name)
            img_lqs, img_gts = degradation_video_list_4(
                img_gts,
                texture_url=self.data_config['texture_template']
            )
        else:
            for i in range(len(img_gts)):
                # img_gts[i] = cv2.resize(img_gts[i], (gt_size_w, gt_size_h), interpolation = cv2.INTER_AREA)
                # img_lqs[i] = cv2.resize(img_lqs[i], (gt_size_w, gt_size_h), interpolation = cv2.INTER_AREA)
                if self.resize_short_side is not None:
                    img_gts[i] = resize_368_short_side(img_gts[i], short_side=self.resize_short_side)
                    img_lqs[i] = resize_368_short_side(img_lqs[i], short_side=self.resize_short_side)
                if self.data_config['name']=='colorization':
                    img_gts[i] = convert_to_L(img_gts[i])
                    img_lqs[i] = convert_to_L(img_lqs[i])

        # augmentation - flip, rotate
        img_lqs.extend(img_gts)
        if self.is_train:
            img_lqs = augment(img_lqs, self.data_config['use_flip'], self.data_config['use_rot'])

        img_results = img2tensor(img_lqs) ## List of tensor

        # ========== 归一化（训练和验证均可使用） ==========
        if self.normalize_transform is not None:
            img_results = [self.normalize_transform(t) for t in img_results]

        clip_len = len(img_gts)
        if self.is_train:
            img_lqs = torch.stack(img_results[:clip_len], dim=0)
            img_gts = torch.stack(img_results[clip_len:], dim=0)
        else:
            img_lqs = torch.stack(img_results[:current_len], dim=0)
            img_gts = torch.stack(img_results[current_len:], dim=0)           

        # img_lqs: (t, c, h, w)
        # img_gt: (t, c, h, w)
        # key: str
        return {
            'lq': img_lqs,
            'gt': img_gts,
            'key': key,
            'frame_list': frame_list,
            'video_name': os.path.basename(current_lq_root),
            'name_list': new_clip_sequence
        }

    def __len__(self):
        return len(self.lq_frames)


@DATASET_REGISTRY.register()
class Film_dataset_2(data.Dataset):  # 2 for Vimeo dataset

    def __init__(self, data_config):
        super(Film_dataset_2, self).__init__()
        self.data_config = data_config
        self.opt = data_config

        self.scale = data_config['scale']
        self.gt_root, self.lq_root = data_config['dataroot_gt'], data_config['dataroot_lq']
        self.is_train = data_config.get('is_train', False)
        
        ## TODO: dynamic frame num for different video clips
        self.num_frame = data_config['num_frame']
        self.num_half_frames = data_config['num_frame'] // 2


        self.lq_frames = getfolderlist(self.lq_root)
        self.gt_frames = getfolderlist(self.gt_root)

        # temporal augmentation configs
        self.interval_list = data_config['interval_list']
        self.random_reverse = data_config['random_reverse']
        interval_str = ','.join(str(x) for x in data_config['interval_list'])
        logger = get_root_logger()
        logger.info(f'Temporal augmentation interval list: [{interval_str}]; '
                    f'Random reverse is {self.random_reverse}.')

    def __getitem__(self, index):

        gt_size = self.data_config.get('gt_size', None)
        gt_size_w = gt_size[0]
        gt_size_h = gt_size[1]

        key = self.gt_frames[index][0]
        current_len = self.gt_frames[index][1]

        ## Fetch the parent directory of clip name
        current_gt_root = os.path.dirname(os.path.dirname(self.gt_frames[index][0]))
        current_lq_root = os.path.dirname(os.path.dirname(self.lq_frames[index][0]))

        clip_name, frame_name = key.split('/')[-2:]  # key example: 000/00000000
        key = clip_name + "/" + frame_name
        center_frame_idx = 1


        frame_list = list(range(center_frame_idx, center_frame_idx + current_len))
        # Sample number should equal to the all frames number in on folder
        assert len(frame_list) == current_len, (f'Wrong length of frame list: {len(frame_list)}')

        # random reverse
        if self.random_reverse and random.random() < 0.5:
            frame_list.reverse()


        # get the GT frame (as the center frame)
        img_gts = []
        img_lqs = []
        for frame in frame_list:

            img_gt_path = os.path.join(current_gt_root, clip_name, 'im%d.png'%(frame))
            img_gt = cv2.imread(img_gt_path)
            img_gt = img_gt.astype(np.float32) / 255.
            img_gts.append(img_gt)

            if not self.is_train:
                img_lq_path = os.path.join(current_lq_root, clip_name, f'{frame:08d}.png')
                img_lq = cv2.imread(img_lq_path)
                img_lq = img_lq.astype(np.float32) / 255.
                img_lqs.append(img_lq)

        if self.is_train:
            img_lqs = img_gts
            img_gts, img_lqs = paired_random_crop(img_gts, img_lqs, gt_size_w, gt_size_h, self.scale, clip_name)
            img_lqs, img_gts = degradation_video_list_3(img_gts, texture_url=self.data_config['texture_template'])
        else:
            for i in range(len(img_gts)):
                img_gts[i] = cv2.resize(img_gts[i], (gt_size_w, gt_size_h), interpolation = cv2.INTER_AREA)
                img_lqs[i] = cv2.resize(img_lqs[i], (gt_size_w, gt_size_h), interpolation = cv2.INTER_AREA)

        # augmentation - flip, rotate
        img_lqs.extend(img_gts)
        if self.is_train:
            img_lqs = augment(img_lqs, self.data_config['use_flip'], self.data_config['use_rot'])

        img_results = img2tensor(img_lqs) ## List of tensor

        # 仅训练阶段执行归一化；验证阶段保持 [0,1]（符合数据标准 4.2）
        if self.is_train and self.data_config['normalizing']:
            transform_normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            for i in range(len(img_results)):
                img_results[i] = transform_normalize(img_results[i])

        if self.is_train:
            img_lqs = torch.stack(img_results[:self.num_frame], dim=0)
            img_gts = torch.stack(img_results[self.num_frame:], dim=0)
        else:
            img_lqs = torch.stack(img_results[:current_len], dim=0)
            img_gts = torch.stack(img_results[current_len:], dim=0)           

        # img_lqs: (t, c, h, w)
        # img_gt: (t, c, h, w)
        # key: str
        return {'lq': img_lqs, 'gt': img_gts, 'key': key, 'frame_list': frame_list, 'video_name': os.path.basename(current_lq_root)}

    def __len__(self):
        return len(self.lq_frames)

@DATASET_REGISTRY.register()
class Film_dataset_3(data.Dataset):  # 3 for Vimeo dataset + Colorization

    def __init__(self, data_config):
        super(Film_dataset_3, self).__init__()
        self.data_config = data_config
        self.opt = data_config

        self.scale = data_config['scale']
        self.gt_root, self.lq_root = data_config['dataroot_gt'], data_config['dataroot_lq']
        self.is_train = data_config.get('is_train', False)
        
        ## TODO: dynamic frame num for different video clips
        self.num_frame = data_config['num_frame']
        self.num_half_frames = data_config['num_frame'] // 2


        self.lq_frames = getfolderlist(self.lq_root)
        self.gt_frames = getfolderlist(self.gt_root)

        # temporal augmentation configs
        self.interval_list = data_config['interval_list']
        self.random_reverse = data_config['random_reverse']
        interval_str = ','.join(str(x) for x in data_config['interval_list'])
        logger = get_root_logger()
        logger.info(f'Temporal augmentation interval list: [{interval_str}]; '
                    f'Random reverse is {self.random_reverse}.')

    def __getitem__(self, index):

        gt_size = self.data_config.get('gt_size', None)
        gt_size_w = gt_size[0]
        gt_size_h = gt_size[1]

        key = self.gt_frames[index][0]
        current_len = self.gt_frames[index][1]

        ## Fetch the parent directory of clip name
        current_gt_root = os.path.dirname(os.path.dirname(self.gt_frames[index][0]))
        current_lq_root = os.path.dirname(os.path.dirname(self.lq_frames[index][0]))

        clip_name, frame_name = key.split('/')[-2:]  # key example: 000/00000000
        key = clip_name + "/" + frame_name
        center_frame_idx = 1


        frame_list = list(range(center_frame_idx, center_frame_idx + current_len))
        # Sample number should equal to the all frames number in on folder
        assert len(frame_list) == current_len, (f'Wrong length of frame list: {len(frame_list)}')

        # random reverse
        if self.random_reverse and random.random() < 0.5:
            frame_list.reverse()


        # get the GT frame (as the center frame)
        img_gts = []
        img_lqs = []
        for frame in frame_list:

            img_gt_path = os.path.join(current_gt_root, clip_name, 'im%d.png'%(frame))
            img_gt = cv2.imread(img_gt_path)
            img_gt = img_gt.astype(np.float32) / 255.
            img_gts.append(img_gt)

            if not self.is_train:
                img_lq_path = os.path.join(current_lq_root, clip_name, f'{frame:08d}.png')
                img_lq = cv2.imread(img_lq_path)
                img_lq = img_lq.astype(np.float32) / 255.
                img_lqs.append(img_lq)

        if self.is_train:
            img_lqs = img_gts
            img_gts, img_lqs = paired_random_crop(img_gts, img_lqs, gt_size_w, gt_size_h, self.scale, clip_name)
            img_lqs, img_gts = degradation_video_colorization(img_gts)
        else:
            for i in range(len(img_gts)):
                img_gts[i] = cv2.resize(img_gts[i], (gt_size_w, gt_size_h), interpolation = cv2.INTER_AREA)
                img_lqs[i] = cv2.resize(img_lqs[i], (gt_size_w, gt_size_h), interpolation = cv2.INTER_AREA)

        # augmentation - flip, rotate
        img_lqs.extend(img_gts)
        if self.is_train:
            img_lqs = augment(img_lqs, self.data_config['use_flip'], self.data_config['use_rot'])

        img_results = img2tensor(img_lqs) ## List of tensor

        # 仅训练阶段执行归一化；验证阶段保持 [0,1]（符合数据标准 4.2）
        if self.is_train and self.data_config['normalizing']:
            transform_normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            for i in range(len(img_results)):
                img_results[i] = transform_normalize(img_results[i])

        if self.is_train:
            img_lqs = torch.stack(img_results[:self.num_frame], dim=0)
            img_gts = torch.stack(img_results[self.num_frame:], dim=0)
        else:
            img_lqs = torch.stack(img_results[:current_len], dim=0)
            img_gts = torch.stack(img_results[current_len:], dim=0)           

        # img_lqs: (t, c, h, w)
        # img_gt: (t, c, h, w)
        # key: str
        return {'lq': img_lqs, 'gt': img_gts, 'key': key, 'frame_list': frame_list, 'video_name': os.path.basename(current_lq_root)}

    def __len__(self):
        return len(self.lq_frames)


@DATASET_REGISTRY.register()
class Film_dataset_4(data.Dataset):  # 4 for REDS dataset + resize by 2

    def __init__(self, data_config):
        super(Film_dataset_4, self).__init__()
        self.data_config = data_config
        self.opt = data_config

        self.scale = data_config['scale']
        self.gt_root, self.lq_root = data_config['dataroot_gt'], data_config['dataroot_lq']
        self.is_train = data_config.get('is_train', False)
        
        ## TODO: dynamic frame num for different video clips
        self.num_frame = data_config['num_frame']
        self.num_half_frames = data_config['num_frame'] // 2

        if self.is_train:

            self.lq_frames = getfilelist_with_length(self.lq_root)
            self.gt_frames = getfilelist_with_length(self.gt_root)
        
        else:
            ## Now: Append the first frame name, then load all frames based on the clip length
            self.lq_frames = getfolderlist(self.lq_root)
            self.gt_frames = getfolderlist(self.gt_root)
            # self.lq_frames = []
            # self.gt_frames = []
            # for i in range(len(self.lq_folders))
            #     val_frame_list_this = sorted(os.listdir(self.lq_folders[i]))
            #     first_frame_name = val_frame_list_this[0]
            #     clip_length = len(val_frame_list_this)
            #     self.lq_frames.append((os.path.join(self.lq_folders[i],f'{first_frame_name:08d}.png'),clip_length))
            #     self.gt_frames.append((os.path.join(self.gt_folders[i],f'{first_frame_name:08d}.png'),clip_length))

        # temporal augmentation configs
        self.interval_list = data_config['interval_list']
        self.random_reverse = data_config['random_reverse']
        interval_str = ','.join(str(x) for x in data_config['interval_list'])
        logger = get_root_logger()
        logger.info(f'Temporal augmentation interval list: [{interval_str}]; '
                    f'Random reverse is {self.random_reverse}.')

    def __getitem__(self, index):

        gt_size = self.data_config.get('gt_size', None)
        gt_size_w = gt_size[0]
        gt_size_h = gt_size[1]

        key = self.gt_frames[index][0]
        current_len = self.gt_frames[index][1]

        ## Fetch the parent directory of clip name
        current_gt_root = os.path.dirname(os.path.dirname(self.gt_frames[index][0]))
        current_lq_root = os.path.dirname(os.path.dirname(self.lq_frames[index][0]))

        clip_name, frame_name = key.split('/')[-2:]  # key example: 000/00000000
        key = clip_name + "/" + frame_name
        center_frame_idx = int(frame_name[:-4])

        if self.is_train:
            # determine the frameing frames
            interval = random.choice(self.interval_list)

            # ensure not exceeding the borders
            start_frame_idx = center_frame_idx - self.num_half_frames * interval
            end_frame_idx = start_frame_idx + (self.num_frame - 1) * interval

            # each clip has 100 frames starting from 0 to 99. TODO: if the training clip is not 100 frames [√]
            # Training start frames should be 0
            while (start_frame_idx < 0) or (end_frame_idx > current_len-1):
                center_frame_idx = random.randint(self.num_half_frames * interval, current_len - self.num_half_frames *interval)
                start_frame_idx = (center_frame_idx - self.num_half_frames * interval)
                end_frame_idx = start_frame_idx + (self.num_frame - 1) * interval
            
            # frame_name = f'{center_frame_idx:08d}'
            frame_list = list(range(start_frame_idx, end_frame_idx + 1, interval))
            # Sample number should equal to the numer we set
            assert len(frame_list) == self.num_frame, (f'Wrong length of frame list: {len(frame_list)}')
        else:

            frame_list = list(range(center_frame_idx, center_frame_idx + current_len))
            # Sample number should equal to the all frames number in on folder
            assert len(frame_list) == current_len, (f'Wrong length of frame list: {len(frame_list)}')

        # random reverse
        if self.random_reverse and random.random() < 0.5:
            frame_list.reverse()


        # get the GT frame (as the center frame)
        img_gts = []
        img_lqs = []
        for frame in frame_list:

            img_gt_path = os.path.join(current_gt_root, clip_name, f'{frame:08d}.png')
            img_gt = cv2.imread(img_gt_path)
            # 先归一化再缩放，避免在 uint8 空间插值导致精度损失
            img_gt = img_gt.astype(np.float32) / 255.
            img_gt = cv2.resize(img_gt, (640, 360), interpolation=cv2.INTER_LANCZOS4)
            img_gts.append(img_gt)

            if not self.is_train:
                img_lq_path = os.path.join(current_lq_root, clip_name, f'{frame:08d}.png')
                img_lq = cv2.imread(img_lq_path)
                img_lq = img_lq.astype(np.float32) / 255.
                img_lqs.append(img_lq)

        if self.is_train:
            img_lqs = img_gts
            img_gts, img_lqs = paired_random_crop(img_gts, img_lqs, gt_size_w, gt_size_h, self.scale, clip_name)
            img_lqs, img_gts = degradation_video_list_4(img_gts, texture_url=self.data_config['texture_template'])
        else:
            for i in range(len(img_gts)):
                img_gts[i] = cv2.resize(img_gts[i], (gt_size_w, gt_size_h), interpolation = cv2.INTER_AREA)
                img_lqs[i] = cv2.resize(img_lqs[i], (gt_size_w, gt_size_h), interpolation = cv2.INTER_AREA)

        # augmentation - flip, rotate
        img_lqs.extend(img_gts)
        if self.is_train:
            img_lqs = augment(img_lqs, self.data_config['use_flip'], self.data_config['use_rot'])

        img_results = img2tensor(img_lqs) ## List of tensor

        # 仅训练阶段执行归一化；验证阶段保持 [0,1]（符合数据标准 4.2）
        if self.is_train and self.data_config['normalizing']:
            transform_normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            for i in range(len(img_results)):
                img_results[i] = transform_normalize(img_results[i])

        if self.is_train:
            img_lqs = torch.stack(img_results[:self.num_frame], dim=0)
            img_gts = torch.stack(img_results[self.num_frame:], dim=0)
        else:
            img_lqs = torch.stack(img_results[:current_len], dim=0)
            img_gts = torch.stack(img_results[current_len:], dim=0)           

        # img_lqs: (t, c, h, w)
        # img_gt: (t, c, h, w)
        # key: str
        return {'lq': img_lqs, 'gt': img_gts, 'key': key, 'frame_list': frame_list, 'video_name': os.path.basename(current_lq_root)}

    def __len__(self):
        return len(self.lq_frames)


@DATASET_REGISTRY.register()
class Film_dataset_5(data.Dataset):  # 5 for Vimeo dataset + Colorization + Convert_to_LAB

    def __init__(self, data_config):
        super(Film_dataset_5, self).__init__()
        self.data_config = data_config
        self.opt = data_config

        self.scale = data_config['scale']
        self.gt_root, self.lq_root = data_config['dataroot_gt'], data_config['dataroot_lq']
        self.is_train = data_config.get('is_train', False)
        
        ## TODO: dynamic frame num for different video clips
        self.num_frame = data_config['num_frame']
        self.num_half_frames = data_config['num_frame'] // 2


        self.lq_frames = getfolderlist(self.lq_root)
        self.gt_frames = getfolderlist(self.gt_root)

        # temporal augmentation configs
        self.interval_list = data_config['interval_list']
        self.random_reverse = data_config['random_reverse']
        interval_str = ','.join(str(x) for x in data_config['interval_list'])
        logger = get_root_logger()
        logger.info(f'Temporal augmentation interval list: [{interval_str}]; '
                    f'Random reverse is {self.random_reverse}.')

    def __getitem__(self, index):

        gt_size = self.data_config.get('gt_size', None)
        gt_size_w = gt_size[0]
        gt_size_h = gt_size[1]

        key = self.gt_frames[index][0]
        current_len = self.gt_frames[index][1]

        ## Fetch the parent directory of clip name
        current_gt_root = os.path.dirname(os.path.dirname(self.gt_frames[index][0]))
        current_lq_root = os.path.dirname(os.path.dirname(self.lq_frames[index][0]))

        clip_name, frame_name = key.split('/')[-2:]  # key example: 000/00000000
        key = clip_name + "/" + frame_name
        
        new_clip_sequence = sorted(os.listdir(os.path.join(current_gt_root, clip_name)))
        
        if self.is_train:
            center_frame_idx = 1
        else:
            center_frame_idx = int(frame_name[:-4])

        frame_list = list(range(center_frame_idx, center_frame_idx + current_len))
        # Sample number should equal to the all frames number in on folder
        assert len(frame_list) == current_len, (f'Wrong length of frame list: {len(frame_list)}')

        # random reverse
        if self.random_reverse and random.random() < 0.5:
            frame_list.reverse()


        # get the GT frame (as the center frame)
        img_gts = []
        img_lqs = []
        for tmp_id,frame in enumerate(frame_list):

            if self.is_train:
                img_gt_path = os.path.join(current_gt_root, clip_name, 'im%d.png'%(frame))
            else:
                img_gt_path = os.path.join(current_gt_root, clip_name, new_clip_sequence[tmp_id])

            img_gt=rgb2lab(Image.open(img_gt_path).convert("RGB"))
            img_gts.append(np.array(img_gt))
            # img_gt = cv2.imread(img_gt_path)
            # img_gt = img_gt.astype(np.float32) / 255.
            # img_gts.append(img_gt)

            if not self.is_train:
                img_lq_path = os.path.join(current_lq_root, clip_name, new_clip_sequence[tmp_id])

                img_lq=rgb2lab(Image.open(img_lq_path).convert("RGB"))
                img_lqs.append(np.array(img_lq))
                # img_lq = cv2.imread(img_lq_path)
                # img_lq = img_lq.astype(np.float32) / 255.
                # img_lqs.append(img_lq)

        if self.is_train:
            img_lqs = img_gts
            img_gts, img_lqs = paired_random_crop(img_gts, img_lqs, gt_size_w, gt_size_h, self.scale, clip_name)
            img_lqs, img_gts = degradation_video_colorization_v2(img_gts) ## LAB now
        else:
            # for i in range(len(img_gts)): ##TODO: inference stage: set the non-reference AB channel to 0 [√]
            #     img_gts[i] = cv2.resize(img_gts[i], (gt_size_w, gt_size_h), interpolation = cv2.INTER_AREA)
            #     img_lqs[i] = cv2.resize(img_lqs[i], (gt_size_w, gt_size_h), interpolation = cv2.INTER_AREA)
            img_gts = img_gts
            img_lqs = img_lqs 

        # augmentation - flip, rotate
        img_lqs.extend(img_gts)
        if self.is_train:
            img_lqs = augment(img_lqs, self.data_config['use_flip'], self.data_config['use_rot'])

        img_results = []
        for x in img_lqs:
            img_results.append(to_mytensor(x))

        # LAB 模式同样仅训练阶段归一化；验证阶段保持 [0,1]
        if self.is_train and self.data_config['normalizing']:
            # transform_normalize=transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
            for i in range(len(img_results)):
                img_results[i] = Normalize_LAB(img_results[i])

        if self.is_train:
            img_lqs = torch.stack(img_results[:self.num_frame], dim=0)
            img_gts = torch.stack(img_results[self.num_frame:], dim=0)
        else:
            img_lqs = torch.stack(img_results[:current_len], dim=0)
            img_gts = torch.stack(img_results[current_len:], dim=0)           

        # img_lqs: (t, c, h, w)
        # img_gt: (t, c, h, w)
        # key: str
        return {'lq': img_lqs, 'gt': img_gts, 'key': key, 'frame_list': frame_list, 'video_name': os.path.basename(current_lq_root)}

    def __len__(self):
        return len(self.lq_frames)



@DATASET_REGISTRY.register()
class Film_dataset_6(data.Dataset):  # 6 for DAVIS&YoutubeVOS dataset + Colorization + Convert_to_LAB

    def __init__(self, data_config):
        super(Film_dataset_6, self).__init__()
        self.data_config = data_config
        self.opt = data_config

        self.scale = data_config['scale']
        self.gt_root, self.lq_root = data_config['dataroot_gt'], data_config['dataroot_lq']
        self.is_train = data_config.get('is_train', False)
        

        self.num_frame = data_config['num_frame']
        self.num_half_frames = data_config['num_frame'] // 2


        self.lq_frames = getfolderlist(self.lq_root)
        self.gt_frames = getfolderlist(self.gt_root)

        # temporal augmentation configs
        self.interval_list = data_config['interval_list']
        self.random_reverse = data_config['random_reverse']
        interval_str = ','.join(str(x) for x in data_config['interval_list'])
        logger = get_root_logger()
        logger.info(f'Temporal augmentation interval list: [{interval_str}]; '
                    f'Random reverse is {self.random_reverse}.')

    def __getitem__(self, index):

        gt_size = self.data_config.get('gt_size', None)
        gt_size_w = gt_size[0]
        gt_size_h = gt_size[1]

        key = self.gt_frames[index][0]
        current_len = self.gt_frames[index][1]


        ## Fetch the parent directory of clip name
        current_gt_root = os.path.dirname(os.path.dirname(self.gt_frames[index][0]))
        current_lq_root = os.path.dirname(os.path.dirname(self.lq_frames[index][0]))

        clip_name, frame_name = key.split('/')[-2:]  # key example: 000/00000000
        key = clip_name + "/" + frame_name
        start_id = int(frame_name[:-4])

        if self.is_train:
            # determine the frameing frames
            interval = random.choice(self.interval_list)

            center_frame_idx = random.randint(self.num_half_frames * interval, current_len - self.num_half_frames *interval - 1)
            start_frame_idx = (center_frame_idx - self.num_half_frames * interval)
            end_frame_idx = start_frame_idx + (self.num_frame - 1) * interval
            # frame_name = f'{center_frame_idx:08d}'
            frame_list = list(range(start_frame_idx, end_frame_idx + 1, interval))
            # print(frame_list)
            # Sample number should equal to the numer we set
            new_clip_sequence = sorted(os.listdir(os.path.join(current_gt_root, clip_name)))
            assert len(frame_list) == self.num_frame, (f'Wrong length of frame list: {len(frame_list)}')

        else:
            frame_list = list(range(start_id, start_id + current_len))
            # Sample number should equal to the all frames number in on folder
            assert len(frame_list) == current_len, (f'Wrong length of frame list: {len(frame_list)}')

        # random reverse
        if self.random_reverse and random.random() < 0.5:
            frame_list.reverse()


        # get the GT frame (as the center frame)
        img_gts = []
        img_lqs = []
        for frame in frame_list:

            if self.is_train:
                # img_gt_path = os.path.join(current_gt_root, clip_name, f'{frame:05d}.jpg') ## Adaptive for DAVIS and YoutubeVOS
                img_gt_path = os.path.join(current_gt_root, clip_name, new_clip_sequence[frame])
                current_frame = Image.open(img_gt_path).convert("RGB")
                current_frame = resize_256_short_side(current_frame)
                img_gt=rgb2lab(current_frame)
                img_gts.append(np.array(img_gt))
            else:
                img_gt_path = os.path.join(current_gt_root, clip_name, f'{frame:08d}.png')
                img_gt=rgb2lab(Image.open(img_gt_path).convert("RGB"))
                img_gts.append(np.array(img_gt))
            # img_gt = cv2.imread(img_gt_path)
            # img_gt = img_gt.astype(np.float32) / 255.
            # img_gts.append(img_gt)

            if not self.is_train:
                img_lq_path = os.path.join(current_lq_root, clip_name, f'{frame:08d}.png')

                img_lq=rgb2lab(Image.open(img_lq_path).convert("RGB"))
                img_lqs.append(np.array(img_lq))
                # img_lq = cv2.imread(img_lq_path)
                # img_lq = img_lq.astype(np.float32) / 255.
                # img_lqs.append(img_lq)

        if self.is_train:
            img_lqs = img_gts
            img_gts, img_lqs = paired_random_crop(img_gts, img_lqs, gt_size_w, gt_size_h, self.scale, clip_name)
            img_lqs, img_gts = degradation_video_colorization_v2(img_gts) ## LAB now
        else:
            for i in range(len(img_gts)): ##TODO: inference stage: set the non-reference AB channel to 0 [√]
                img_gts[i] = cv2.resize(img_gts[i], (gt_size_w, gt_size_h), interpolation = cv2.INTER_AREA)
                img_lqs[i] = cv2.resize(img_lqs[i], (gt_size_w, gt_size_h), interpolation = cv2.INTER_AREA)

        # augmentation - flip, rotate
        img_lqs.extend(img_gts)
        if self.is_train:
            img_lqs = augment(img_lqs, self.data_config['use_flip'], self.data_config['use_rot'])

        img_results = []
        for x in img_lqs:
            img_results.append(to_mytensor(x))

        # LAB 模式同样仅训练阶段归一化；验证阶段保持 [0,1]
        if self.is_train and self.data_config['normalizing']:
            # transform_normalize=transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
            for i in range(len(img_results)):
                img_results[i] = Normalize_LAB(img_results[i])

        if self.is_train:
            img_lqs = torch.stack(img_results[:self.num_frame], dim=0)
            img_gts = torch.stack(img_results[self.num_frame:], dim=0)
        else:
            img_lqs = torch.stack(img_results[:current_len], dim=0)
            img_gts = torch.stack(img_results[current_len:], dim=0)           

        # img_lqs: (t, c, h, w)
        # img_gt: (t, c, h, w)
        # key: str
        return {'lq': img_lqs, 'gt': img_gts, 'key': key, 'frame_list': frame_list, 'video_name': os.path.basename(current_lq_root), 'clip_name': clip_name}

    def __len__(self):
        return len(self.lq_frames)

@DATASET_REGISTRY.register()
class Film_dataset_7(data.Dataset):  # 7 for DAVIS&YoutubeVOS dataset + Colorization + Convert_to_LAB, add extra long-term references (the final element)

    def __init__(self, data_config):
        super(Film_dataset_7, self).__init__()
        self.data_config = data_config
        self.opt = data_config

        self.scale = data_config['scale']
        self.gt_root, self.lq_root = data_config['dataroot_gt'], data_config['dataroot_lq']
        self.is_train = data_config.get('is_train', False)
        

        self.num_frame = data_config['num_frame']
        self.num_half_frames = data_config['num_frame'] // 2


        self.lq_frames = getfolderlist(self.lq_root)
        self.gt_frames = getfolderlist(self.gt_root)

        # temporal augmentation configs
        self.interval_list = data_config['interval_list']
        self.random_reverse = data_config['random_reverse']
        interval_str = ','.join(str(x) for x in data_config['interval_list'])
        logger = get_root_logger()
        logger.info(f'Temporal augmentation interval list: [{interval_str}]; '
                    f'Random reverse is {self.random_reverse}.')

    def __getitem__(self, index):

        gt_size = self.data_config.get('gt_size', None)
        gt_size_w = gt_size[0]
        gt_size_h = gt_size[1]

        key = self.gt_frames[index][0]
        current_len = self.gt_frames[index][1]


        ## Fetch the parent directory of clip name
        current_gt_root = os.path.dirname(os.path.dirname(self.gt_frames[index][0]))
        current_lq_root = os.path.dirname(os.path.dirname(self.lq_frames[index][0]))

        clip_name, frame_name = key.split('/')[-2:]  # key example: 000/00000000
        key = clip_name + "/" + frame_name
        start_id = int(frame_name[:-4])

        if self.is_train:
            # determine the frameing frames
            interval = random.choice(self.interval_list)

            center_frame_idx = random.randint(self.num_half_frames * interval, current_len - self.num_half_frames *interval - 1)
            start_frame_idx = (center_frame_idx - self.num_half_frames * interval)
            end_frame_idx = start_frame_idx + (self.num_frame - 1) * interval
            # frame_name = f'{center_frame_idx:08d}'
            frame_list = list(range(start_frame_idx, end_frame_idx + 1, interval))
            # print(frame_list)
            # Sample number should equal to the numer we set
            new_clip_sequence = sorted(os.listdir(os.path.join(current_gt_root, clip_name)))
            assert len(frame_list) == self.num_frame, (f'Wrong length of frame list: {len(frame_list)}')


        else:
            frame_list = list(range(start_id, start_id + current_len))
            # Sample number should equal to the all frames number in on folder
            assert len(frame_list) == current_len, (f'Wrong length of frame list: {len(frame_list)}')

        # random reverse
        if self.random_reverse and random.random() < 0.5:
            frame_list.reverse()

        if self.is_train:
            # Random for the reference frame
            frame_list.append(random.randint(0,current_len-1))            


        # get the GT frame (as the center frame)
        img_gts = []
        img_lqs = []
        for frame in frame_list:

            if self.is_train:
                # img_gt_path = os.path.join(current_gt_root, clip_name, f'{frame:05d}.jpg') ## Adaptive for DAVIS and YoutubeVOS
                img_gt_path = os.path.join(current_gt_root, clip_name, new_clip_sequence[frame])
                current_frame = Image.open(img_gt_path).convert("RGB")
                current_frame = resize_256_short_side(current_frame)
                img_gt=rgb2lab(current_frame)
                img_gts.append(np.array(img_gt))
            else:
                img_gt_path = os.path.join(current_gt_root, clip_name, f'{frame:08d}.png')
                img_gt=rgb2lab(Image.open(img_gt_path).convert("RGB"))
                img_gts.append(np.array(img_gt))
            # img_gt = cv2.imread(img_gt_path)
            # img_gt = img_gt.astype(np.float32) / 255.
            # img_gts.append(img_gt)

            if not self.is_train:
                img_lq_path = os.path.join(current_lq_root, clip_name, f'{frame:08d}.png')

                img_lq=rgb2lab(Image.open(img_lq_path).convert("RGB"))
                img_lqs.append(np.array(img_lq))
                # img_lq = cv2.imread(img_lq_path)
                # img_lq = img_lq.astype(np.float32) / 255.
                # img_lqs.append(img_lq)

        if self.is_train:
            img_lqs = img_gts
            img_gts, img_lqs = paired_random_crop(img_gts, img_lqs, gt_size_w, gt_size_h, self.scale, clip_name)
            img_lqs, img_gts = degradation_video_colorization_v3(img_gts) ## LAB now
        else:
            for i in range(len(img_gts)): ##TODO: inference stage: set the non-reference AB channel to 0 [√]
                img_gts[i] = cv2.resize(img_gts[i], (gt_size_w, gt_size_h), interpolation = cv2.INTER_AREA)
                img_lqs[i] = cv2.resize(img_lqs[i], (gt_size_w, gt_size_h), interpolation = cv2.INTER_AREA)

        # augmentation - flip, rotate
        img_lqs.extend(img_gts)
        if self.is_train:
            img_lqs = augment(img_lqs, self.data_config['use_flip'], self.data_config['use_rot'])

        img_results = []
        for x in img_lqs:
            img_results.append(to_mytensor(x))

        # LAB 模式同样仅训练阶段归一化；验证阶段保持 [0,1]
        if self.is_train and self.data_config['normalizing']:
            # transform_normalize=transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
            for i in range(len(img_results)):
                img_results[i] = Normalize_LAB(img_results[i])

        if self.is_train:
            img_lqs = torch.stack(img_results[:self.num_frame+1], dim=0)
            img_gts = torch.stack(img_results[self.num_frame+1:], dim=0)
        else:
            img_lqs = torch.stack(img_results[:current_len], dim=0)
            img_gts = torch.stack(img_results[current_len:], dim=0)           

        # img_lqs: (t, c, h, w)
        # img_gt: (t, c, h, w)
        # key: str
        return {'lq': img_lqs, 'gt': img_gts, 'key': key, 'frame_list': frame_list, 'video_name': os.path.basename(current_lq_root), 'clip_name': clip_name}

    def __len__(self):
        return len(self.lq_frames)



@DATASET_REGISTRY.register()
class Film_dataset_8(data.Dataset):  # 8 for DAVIS&YoutubeVOS dataset + Colorization + Convert_to_LAB, add extra long-term references (the final element), degradation_video_colorization_v4

    def __init__(self, data_config):
        super(Film_dataset_8, self).__init__()
        self.data_config = data_config
        self.opt = data_config

        self.scale = data_config['scale']
        self.gt_root, self.lq_root = data_config['dataroot_gt'], data_config['dataroot_lq']
        self.is_train = data_config.get('is_train', False)
        

        self.num_frame = data_config['num_frame']
        self.num_half_frames = data_config['num_frame'] // 2


        self.lq_frames = getfolderlist(self.lq_root)
        self.gt_frames = getfolderlist(self.gt_root)

        # temporal augmentation configs
        self.interval_list = data_config['interval_list']
        self.random_reverse = data_config['random_reverse']
        interval_str = ','.join(str(x) for x in data_config['interval_list'])
        logger = get_root_logger()
        logger.info(f'Temporal augmentation interval list: [{interval_str}]; '
                    f'Random reverse is {self.random_reverse}.')

    def __getitem__(self, index):

        gt_size = self.data_config.get('gt_size', None)
        gt_size_w = gt_size[0]
        gt_size_h = gt_size[1]

        key = self.gt_frames[index][0]
        current_len = self.gt_frames[index][1]


        ## Fetch the parent directory of clip name
        current_gt_root = os.path.dirname(os.path.dirname(self.gt_frames[index][0]))
        current_lq_root = os.path.dirname(os.path.dirname(self.lq_frames[index][0]))

        clip_name, frame_name = key.split('/')[-2:]  # key example: 000/00000000
        key = clip_name + "/" + frame_name
        start_id = int(frame_name[:-4])

        if self.is_train:
            # determine the frameing frames
            interval = random.choice(self.interval_list)

            center_frame_idx = random.randint(self.num_half_frames * interval, current_len - self.num_half_frames *interval - 1)
            start_frame_idx = (center_frame_idx - self.num_half_frames * interval)
            end_frame_idx = start_frame_idx + (self.num_frame - 1) * interval
            # frame_name = f'{center_frame_idx:08d}'
            frame_list = list(range(start_frame_idx, end_frame_idx + 1, interval))
            # print(frame_list)
            # Sample number should equal to the numer we set
            new_clip_sequence = sorted(os.listdir(os.path.join(current_gt_root, clip_name)))
            assert len(frame_list) == self.num_frame, (f'Wrong length of frame list: {len(frame_list)}')


        else:
            frame_list = list(range(start_id, start_id + current_len))
            # Sample number should equal to the all frames number in on folder
            assert len(frame_list) == current_len, (f'Wrong length of frame list: {len(frame_list)}')

        # random reverse
        if self.random_reverse and random.random() < 0.5:
            frame_list.reverse()

        if self.is_train:
            # Random for the reference frame
            frame_list.append(random.randint(0,current_len-1))            


        # get the GT frame (as the center frame)
        img_gts = []
        img_lqs = []
        for frame in frame_list:

            if self.is_train:
                # img_gt_path = os.path.join(current_gt_root, clip_name, f'{frame:05d}.jpg') ## Adaptive for DAVIS and YoutubeVOS
                img_gt_path = os.path.join(current_gt_root, clip_name, new_clip_sequence[frame])
                current_frame = Image.open(img_gt_path).convert("RGB")
                current_frame = resize_256_short_side(current_frame)
                img_gt=rgb2lab(current_frame)
                img_gts.append(np.array(img_gt))
            else:
                img_gt_path = os.path.join(current_gt_root, clip_name, f'{frame:08d}.png')
                img_gt=rgb2lab(Image.open(img_gt_path).convert("RGB"))
                img_gts.append(np.array(img_gt))
            # img_gt = cv2.imread(img_gt_path)
            # img_gt = img_gt.astype(np.float32) / 255.
            # img_gts.append(img_gt)

            if not self.is_train:
                img_lq_path = os.path.join(current_lq_root, clip_name, f'{frame:08d}.png')

                img_lq=rgb2lab(Image.open(img_lq_path).convert("RGB"))
                img_lqs.append(np.array(img_lq))
                # img_lq = cv2.imread(img_lq_path)
                # img_lq = img_lq.astype(np.float32) / 255.
                # img_lqs.append(img_lq)

        if self.is_train:
            img_lqs = img_gts
            img_gts, img_lqs = paired_random_crop(img_gts, img_lqs, gt_size_w, gt_size_h, self.scale, clip_name)
            img_lqs, img_gts = degradation_video_colorization_v4(img_gts) ## LAB now
        else:
            for i in range(len(img_gts)): ##TODO: inference stage: set the non-reference AB channel to 0 [√]
                img_gts[i] = cv2.resize(img_gts[i], (gt_size_w, gt_size_h), interpolation = cv2.INTER_AREA)
                img_lqs[i] = cv2.resize(img_lqs[i], (gt_size_w, gt_size_h), interpolation = cv2.INTER_AREA)

        # augmentation - flip, rotate
        img_lqs.extend(img_gts)
        if self.is_train:
            img_lqs = augment(img_lqs, self.data_config['use_flip'], self.data_config['use_rot'])

        img_results = []
        for x in img_lqs:
            img_results.append(to_mytensor(x))

        # LAB 模式同样仅训练阶段归一化；验证阶段保持 [0,1]
        if self.is_train and self.data_config['normalizing']:
            # transform_normalize=transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
            for i in range(len(img_results)):
                img_results[i] = Normalize_LAB(img_results[i])

        if self.is_train:
            img_lqs = torch.stack(img_results[:self.num_frame+1], dim=0)
            img_gts = torch.stack(img_results[self.num_frame+1:], dim=0)
        else:
            img_lqs = torch.stack(img_results[:current_len], dim=0)
            img_gts = torch.stack(img_results[current_len:], dim=0)           

        # img_lqs: (t, c, h, w)
        # img_gt: (t, c, h, w)
        # key: str
        return {'lq': img_lqs, 'gt': img_gts, 'key': key, 'frame_list': frame_list, 'video_name': os.path.basename(current_lq_root), 'clip_name': clip_name}

    def __len__(self):
        return len(self.lq_frames)


@DATASET_REGISTRY.register()
class Film_SRWOV_dataset(data.Dataset):
    """MambaOFR_SRWOV 真实世界视频数据集。

    支持两种目录结构（只统计/加载图像文件）：

    1) 原始结构（嵌套）：
        dataroot_lq/
        ├── video_id_1/
        │   └── clip_001/
        │       ├── video/          # 忽略
        │       └── frames/         # 只加载这个
        │           ├── 00000.jpg
        │           └── ...
        └── video_id_2/
            └── ...

    2) 扁平结构（例如 SRWOV_120）：
        dataroot_lq/
        ├── 000/
        │   ├── 00000.jpg
        │   └── ...
        ├── 001/
        └── ...

    配置示例:
        type: Film_SRWOV_dataset
        dataroot_lq: datasets/OldFilmRestoration/val/RealFilm/MambaOFR_SRWOV
        num_frame: -1
        normalizing: true
        grayscale: true

    """

    def __init__(self, data_config):
        super(Film_SRWOV_dataset, self).__init__()

        self.data_config = data_config
        self.opt = data_config

        self.lq_root = data_config['dataroot_lq']
        self.scale = data_config.get('scale', 1)

        raw_num_frame = data_config.get('num_frame', -1)
        self.num_frame = None if raw_num_frame <= 0 else int(raw_num_frame)

        # 扫描 frames/ 子目录
        self.lq_frames = self._scan_frames_dirs(self.lq_root)

        self.grayscale = data_config.get('grayscale', False)
        self.normalizing = data_config.get('normalizing', False)
        self.normalize_transform = transforms.Normalize(
            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        ) if self.normalizing else None

        self.cache_data = bool(data_config.get('cache_data', False))
        self.lq_cache: Dict[str, np.ndarray] = {}

        resize_short_side = data_config.get('resize_short_side')
        self.resize_short_side = int(resize_short_side) if resize_short_side is not None and int(resize_short_side) > 0 else None

        # 尺寸对齐倍数，默认 8（避免网络下采样时尺寸不整除报错）
        self.align = int(data_config.get('align', 8))

        logger = get_root_logger()
        logger.info(f'[Film_SRWOV_dataset] Loaded {len(self.lq_frames)} clips, align={self.align}')

    def _scan_frames_dirs(self, root_path):
        """扫描 clips 并返回帧目录与帧文件列表。"""
        all_clips = []
        exts = ('.jpg', '.png', '.jpeg')

        def _list_frames(dir_path: str):
            return sorted([
                f for f in os.listdir(dir_path)
                if f.lower().endswith(exts)
            ])

        def _safe_key_component(name: str) -> str:
            # RTNModel 的 folder 解析会把 "a/000" 识别为 "a"，导致不同 clip 覆盖；
            # 这里保证 key 的最后一段不为纯数字。
            return name if not name.isdigit() else f'clip_{name}'

        for first in sorted(os.listdir(root_path)):
            first_path = os.path.join(root_path, first)
            if not os.path.isdir(first_path):
                continue

            # 结构 2：root/<clip_id>/*.jpg
            flat_frames = _list_frames(first_path)
            if flat_frames:
                all_clips.append({
                    'key': first,  # 保持与目录名一致（例如 000）
                    'video_name': first,
                    'frames_dir': first_path,
                    'frames': flat_frames,
                })
                continue

            # 结构 1：root/<video_id>/<clip_id>/frames/*.jpg
            for second in sorted(os.listdir(first_path)):
                second_path = os.path.join(first_path, second)
                if not os.path.isdir(second_path):
                    continue

                # 优先使用 frames/ 子目录
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

                # 兼容：root/<video_id>/<clip_id>/*.jpg
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

        return all_clips

    def _read_image_cached(self, img_path: str) -> np.ndarray:
        if self.cache_data and img_path in self.lq_cache:
            return self.lq_cache[img_path]

        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f'无法读取图像: {img_path}')

        if self.grayscale:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        img = img.astype(np.float32) / 255.

        if self.cache_data:
            self.lq_cache[img_path] = img
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
            img = self._read_image_cached(img_path)
            if self.resize_short_side is not None:
                img = resize_368_short_side(img, short_side=self.resize_short_side, align=self.align)
            else:
                # 不缩放时按 align 向上 padding（reflect），避免裁切丢失内容
                h, w = img.shape[:2]
                if orig_h is None:
                    orig_h, orig_w = h, w
                    pad_h = (self.align - (orig_h % self.align)) % self.align
                    pad_w = (self.align - (orig_w % self.align)) % self.align
                else:
                    if h != orig_h or w != orig_w:
                        raise ValueError(
                            f'Film_SRWOV_dataset expects consistent frame size within a clip, '
                            f'but got {h}x{w} vs {orig_h}x{orig_w} for {video_id}/{clip_id}.'
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
            # 保存/评测时裁回原始尺寸（未 resize 情况下）
            'orig_h': orig_h if orig_h is not None else img_lqs.size(2),
            'orig_w': orig_w if orig_w is not None else img_lqs.size(3),
            'pad_h': pad_h,
            'pad_w': pad_w
        }

    def __len__(self):
        return len(self.lq_frames)
