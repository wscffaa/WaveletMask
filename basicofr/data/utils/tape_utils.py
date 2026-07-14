"""Utility helpers for TAPE dataset integration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch


def imfrombytes(content: bytes, flag: str = 'color', float32: bool = False) -> np.ndarray:
    """Decode images from raw bytes using OpenCV with RGB ordering."""
    img_np = np.frombuffer(content, np.uint8)
    imread_flags = {
        'color': cv2.IMREAD_COLOR,
        'grayscale': cv2.IMREAD_GRAYSCALE,
        'unchanged': cv2.IMREAD_UNCHANGED,
    }
    img = cv2.imdecode(img_np, imread_flags[flag])
    if flag == 'color':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if float32:
        img = img.astype(np.float32) / 255.0
    return img


def _center_crop(img: torch.Tensor, patch_size: int) -> torch.Tensor:
    _, _, h, w = img.shape
    h_start = max((h - patch_size) // 2, 0)
    w_start = max((w - patch_size) // 2, 0)
    return img[:, :, h_start:h_start + patch_size, w_start:w_start + patch_size]


def _random_crop(img: torch.Tensor, patch_size: int) -> torch.Tensor:
    _, _, h, w = img.shape
    if h == patch_size and w == patch_size:
        return img
    h_start = torch.randint(0, h - patch_size + 1, (1,)).item()
    w_start = torch.randint(0, w - patch_size + 1, (1,)).item()
    return img[:, :, h_start:h_start + patch_size, w_start:w_start + patch_size]


def _resolve_patch_size(patch_size: Union[int, Tuple[int, int]], h: int, w: int) -> Tuple[int, int]:
    if isinstance(patch_size, (tuple, list)):
        patch_h, patch_w = int(patch_size[0]), int(patch_size[1])
    else:
        patch_h = patch_w = int(patch_size)
    target_h = min(patch_h, h)
    target_w = min(patch_w, w)
    return target_h, target_w


def _sample_crop_params(
    img: torch.Tensor,
    patch_size: Union[int, Tuple[int, int]],
    crop_mode: str,
) -> Tuple[int, int, int, int]:
    _, _, h, w = img.shape
    target_h, target_w = _resolve_patch_size(patch_size, h, w)
    if crop_mode == 'center':
        h_start = max((h - target_h) // 2, 0)
        w_start = max((w - target_w) // 2, 0)
    elif crop_mode == 'random':
        if h == target_h:
            h_start = 0
        else:
            h_start = torch.randint(0, h - target_h + 1, (1,)).item()
        if w == target_w:
            w_start = 0
        else:
            w_start = torch.randint(0, w - target_w + 1, (1,)).item()
    else:
        raise ValueError(f'Unsupported crop_mode: {crop_mode}')
    return h_start, w_start, target_h, target_w


def crop(
    img: torch.Tensor,
    patch_size: Union[int, Tuple[int, int]] = 768,
    crop_mode: str = 'center',
    crop_params: Optional[Tuple[int, int, int, int]] = None,
) -> torch.Tensor:
    """Crop a stack of frames to `patch_size` using the requested mode."""
    _, _, h, w = img.shape
    if crop_params is None:
        h_start, w_start, target_h, target_w = _sample_crop_params(img, patch_size, crop_mode)
    else:
        h_start, w_start, target_h, target_w = crop_params
        target_h = min(target_h, h)
        target_w = min(target_w, w)
        h_start = max(min(h_start, h - target_h), 0)
        w_start = max(min(w_start, w - target_w), 0)
    if h == target_h and w == target_w:
        return img
    return img[:, :, h_start:h_start + target_h, w_start:w_start + target_w]


def resize(img: torch.Tensor, patch_size: int = 768) -> torch.Tensor:
    """Resize frames so the longer side matches `patch_size`, preserving aspect ratio."""
    _, _, h, w = img.shape
    if h <= patch_size and w <= patch_size:
        return img
    if h > w:
        new_h = patch_size
        new_w = int(w * patch_size / h)
    else:
        new_w = patch_size
        new_h = int(h * patch_size / w)
    return torch.nn.functional.interpolate(img, size=(new_h, new_w), mode='bilinear', align_corners=False)


def preprocess(
    imgs: Union[List[torch.Tensor], torch.Tensor],
    mode: str = 'crop',
    patch_size: Union[int, Tuple[int, int]] = 768,
    crop_mode: str = 'center',
    crop_params: Optional[Tuple[int, int, int, int]] = None,
) -> Union[List[torch.Tensor], torch.Tensor]:
    """Apply spatial preprocessing to a tensor or list of tensors."""
    if isinstance(imgs, list):
        if not imgs:
            return []
        shared_params = crop_params
        if mode == 'crop' and shared_params is None:
            shared_params = _sample_crop_params(imgs[0], patch_size, crop_mode)
        return [
            preprocess(img, mode=mode, patch_size=patch_size, crop_mode=crop_mode, crop_params=shared_params)
            for img in imgs
        ]
    if isinstance(imgs, torch.Tensor):
        if mode == 'crop':
            return crop(imgs, patch_size=patch_size, crop_mode=crop_mode, crop_params=crop_params)
        if mode == 'resize':
            return resize(imgs, patch_size=patch_size)
        raise ValueError(f'Unknown preprocess mode: {mode}')
    raise TypeError(f'Unsupported imgs type: {type(imgs)}')


def ensure_exists(path: Union[str, Path]) -> Path:
    """Validate that a path exists and return it as Path."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Path does not exist: {path}')
    return path
