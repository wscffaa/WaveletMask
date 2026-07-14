"""
老电影退化流水线

独立于 degradation.py，提供完整的老电影退化函数。
结合 FilmDamageGenerator 和原有退化方法。

Author: BasicOFR Team
Date: 2025-12-24
"""

import os
import random
from io import BytesIO
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageEnhance
from scipy import ndimage

from .film_damage import FilmDamageGenerator, FlickerGenerator
from .blend_modes import addition, subtract, multiply


# ============== 工具函数 ==============

def pil_to_np(img_pil: Image.Image) -> np.ndarray:
    """PIL -> numpy [C, H, W] float32 [0, 1]"""
    ar = np.array(img_pil)
    if len(ar.shape) == 3:
        ar = ar.transpose(2, 0, 1)
    else:
        ar = ar[None, ...]
    return ar.astype(np.float32) / 255.0


def np_to_pil(img_np: np.ndarray) -> Image.Image:
    """numpy [C, H, W] [0, 1] -> PIL"""
    ar = np.clip(img_np * 255, 0, 255).astype(np.uint8)
    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        ar = ar.transpose(1, 2, 0)
    return Image.fromarray(ar)


def transfer_cv2_to_pil(img_cv2: np.ndarray) -> Image.Image:
    """BGR float32 [0, 1] -> PIL 灰度"""
    ar = np.clip(img_cv2 * 255.0, 0, 255).astype(np.uint8)
    if ar.ndim == 2:
        return Image.fromarray(ar, mode='L')
    rgb = ar[..., ::-1]
    return Image.fromarray(rgb, mode='RGB').convert("L")


def transfer_pil_to_cv2(img_pil: Image.Image) -> np.ndarray:
    """PIL -> BGR float32 [0, 1]"""
    arr = np.asarray(img_pil, dtype=np.uint8)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.shape[-1] == 4:
        arr = arr[:, :, :3]
    return arr[..., ::-1].astype(np.float32) / 255.0


# ============== 纹理相关 ==============

_TEXTURE_CACHE = {}


def _clamp_strength(value: float, min_value: float = 0.5, max_value: float = 1.0) -> float:
    """约束退化强度范围"""
    return float(np.clip(value, min_value, max_value))


def _get_texture_files(path: str, only_001: bool = False) -> List[str]:
    """获取纹理文件列表（带缓存）"""
    key = (os.path.abspath(path), only_001)
    if key in _TEXTURE_CACHE:
        return _TEXTURE_CACHE[key]

    files = []
    for dirpath, _, filenames in os.walk(path):
        if only_001 and not dirpath.endswith("001"):
            continue
        for name in filenames:
            if name.lower().endswith(('.png', '.jpg', '.jpeg')):
                files.append(os.path.join(dirpath, name))
    files.sort()
    _TEXTURE_CACHE[key] = files
    return files


def texture_blending(
    image: Image.Image,
    texture: Image.Image,
    h: int,
    w: int,
) -> Image.Image:
    """纹理混合

    Args:
        image: 输入图像 (PIL)
        texture: 纹理图像 (PIL)
        h, w: 目标尺寸

    Returns:
        混合后的图像 (PIL 灰度)
    """
    # 纹理增强
    texture = texture.resize(
        (random.randint(w, w * 2), random.randint(h, h * 2)),
        resample=Image.BILINEAR
    )
    tw, th = texture.size
    top = random.randint(0, th - h)
    left = random.randint(0, tw - w)
    texture = texture.crop((left, top, left + w, top + h))

    # 转为 float 数组
    image_arr = np.array(image.convert('RGBA')).astype(float)
    texture_arr = np.array(texture.convert('RGBA')).astype(float)

    # 随机混合模式，使用正确的 blend_modes 函数
    mode = random.randint(0, 2)
    opacity = random.uniform(0.6, 1.0)

    if mode == 0:  # addition
        blended = addition(image_arr, texture_arr, opacity)
    elif mode == 1:  # subtract
        blended = subtract(image_arr, texture_arr, opacity)
    else:  # multiply
        blended = multiply(image_arr, texture_arr, opacity)

    blended = np.clip(blended, 0, 255).astype(np.uint8)[:, :, :3]
    return Image.fromarray(blended).convert("L")


# ============== 标准退化函数 ==============

def gaussian_blur(img: np.ndarray, kernel_size: int = 5, sigma: float = 2.0) -> np.ndarray:
    """高斯模糊"""
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)


def anisotropic_gaussian_kernel(ksize: int = 15, theta: float = 0, l1: float = 6, l2: float = 6) -> np.ndarray:
    """各向异性高斯核"""
    v = np.array([np.cos(theta), np.sin(theta)])
    V = np.array([[v[0], v[1]], [v[1], -v[0]]])
    D = np.array([[l1, 0], [0, l2]])
    Sigma = V @ D @ np.linalg.inv(V)

    center = ksize / 2.0 + 0.5
    k = np.zeros([ksize, ksize])
    for y in range(ksize):
        for x in range(ksize):
            cy, cx = y - center + 1, x - center + 1
            k[y, x] = np.exp(-0.5 * np.array([cx, cy]) @ np.linalg.inv(Sigma) @ np.array([cx, cy]))

    return k / k.sum()


def add_blur(img: np.ndarray, params: dict) -> np.ndarray:
    """添加模糊"""
    wd2 = 8.0
    wd = 2.8

    if params.get('type_value', 0.5) < 0.5:
        l1 = wd2 * params.get('l1_value', 0.5)
        l2 = wd2 * params.get('l2_value', 0.5)
        theta = params.get('angle_value', 0.5) * np.pi
        ksize = 2 * params.get('shape_value', 5) + 3
        k = anisotropic_gaussian_kernel(ksize, theta, max(0.1, l1), max(0.1, l2))
    else:
        ksize = 2 * params.get('shape_value', 5) + 3
        sigma = wd * params.get('l1_value', 0.5)
        k = cv2.getGaussianKernel(ksize, sigma)
        k = k @ k.T

    return ndimage.convolve(img, np.expand_dims(k, axis=2), mode='mirror')


def add_noise(img: np.ndarray, noise_std: float, noise_type: str = 'gaussian') -> np.ndarray:
    """添加噪声"""
    noise = np.random.normal(0, noise_std, img.shape).astype(np.float32)

    if noise_type == 'speckle':
        noisy = img + noise * img
    else:
        noisy = img + noise

    return np.clip(noisy, 0, 1).astype(np.float32)


def add_jpeg_artifact(img: Image.Image, quality: int) -> Image.Image:
    """添加 JPEG 压缩伪影"""
    img_arr = np.array(img)
    img_arr = np.clip(img_arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_arr)

    with BytesIO() as f:
        img.save(f, format='JPEG', quality=quality)
        f.seek(0)
        return Image.open(f).convert('L')


def random_scaling(img: Image.Image, x: int, y: int) -> Image.Image:
    """随机缩放"""
    methods = [Image.BICUBIC, Image.BILINEAR, Image.LANCZOS]
    return img.resize((x, y), random.choice(methods))


def add_downsampling(img: Image.Image, params: dict) -> Image.Image:
    """添加下采样退化"""
    w, h = img.size

    rnum = params.get('rnum', 0.5)
    if rnum > 0.8:
        sf = params.get('up_scale', 1.5)
    elif rnum < 0.7:
        sf = params.get('down_scale', 0.5)
    else:
        sf = 1.0

    new_w, new_h = int(sf * w), int(sf * h)
    return random_scaling(img, new_w, new_h)


def color_jitter(img: Image.Image) -> Image.Image:
    """颜色抖动"""
    img = img.convert("RGB")

    if random.random() < 0.5:
        factor = random.uniform(0.8, 1.2)
        img = ImageEnhance.Brightness(img).enhance(factor)

    if random.random() < 0.5:
        factor = random.uniform(0.9, 1.0)
        img = ImageEnhance.Contrast(img).enhance(factor)

    return img.convert('L')


def add_sharpening(img: np.ndarray, weight: float = 0.5, radius: int = 50, threshold: int = 10) -> np.ndarray:
    """USM 锐化"""
    if radius % 2 == 0:
        radius += 1
    blur = cv2.GaussianBlur(img, (radius, radius), 0)
    residual = img - blur
    mask = (np.abs(residual) > threshold / 255.0).astype('float32')
    soft_mask = cv2.GaussianBlur(mask, (radius, radius), 0)

    K = np.clip(img + weight * residual, 0, 1)
    return soft_mask * K + (1 - soft_mask) * img


# ============== 标准退化流水线 ==============

def standard_degradation(
    img: Image.Image,
    params: dict,
    distortion_prob: List[float],
) -> Image.Image:
    """标准退化流水线

    Args:
        img: 输入图像 (PIL)
        params: 退化参数
        distortion_prob: 各退化类型的概率

    Returns:
        退化后的图像 (PIL)
    """
    x, y = img.size

    # blur
    if distortion_prob[0] < 0.8:
        img_cv2 = transfer_pil_to_cv2(img.convert("RGB"))
        img_cv2 = add_blur(img_cv2, params)
        img = transfer_cv2_to_pil(img_cv2)

    # downsample
    if distortion_prob[1] < 0.8:
        img = add_downsampling(img, params)

    # noise
    if distortion_prob[2] < 0.8:
        img_np = pil_to_np(img)
        noise_type = 'speckle' if random.random() < 0.5 else 'gaussian'
        img_np = add_noise(img_np, params.get('noise_std', 0.03), noise_type)
        img = np_to_pil(img_np)

    # jpeg
    if distortion_prob[3] < 0.8:
        quality = params.get('jpeg_quality', 70)
        quality_var = random.randint(-15, 15)
        quality = int(np.clip(quality + quality_var, 40, 100))
        img = add_jpeg_artifact(img, quality)

    # 恢复原始尺寸
    img = random_scaling(img, x, y)

    return img


# ============== 主退化函数 ==============

def film_degradation_video_list(
    video_list: List[np.ndarray],
    texture_url: str = './noise_data',
    film_damage_config: Optional[dict] = None,
    degradation_strength: float = 1.0,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """老电影退化流水线

    完整流程：
    1. GT 帧序列
    2. 纹理叠加 + 标准退化（blur/noise/jpeg/downsample）+ 颜色抖动
    3. 老电影损伤（划痕/污渍/灰尘/毛发）
    4. 序列级闪烁
    5. LQ 帧序列

    Args:
        video_list: GT 帧列表 [H, W, C] float32 [0, 1]
        texture_url: 纹理目录
        film_damage_config: 老电影损伤配置
        degradation_strength: 标准退化强度（0.5~1.0，越小越轻）

    Returns:
        (degraded_list, gt_list)
    """
    config = film_damage_config or {}
    film_damage_enabled = config.get('enabled', False)

    # 初始化生成器
    if film_damage_enabled:
        assets_root = config.get('assets_root', 'datasets/film_damage_assets')
        film_damage_gen = FilmDamageGenerator(assets_root, config)
        flicker_gen = FlickerGenerator(config.get('flicker', {}))
    else:
        film_damage_gen = None
        flicker_gen = None

    # 纹理文件
    texture_files = _get_texture_files(texture_url)

    strength = _clamp_strength(degradation_strength)

    # 标准退化参数（每个视频固定）
    distortion_params = {
        'type_value': random.random(),
        'l1_value': random.random() * strength,
        'l2_value': random.random() * strength,
        'angle_value': random.random(),
        'shape_value': random.randint(2, max(2, int(2 + (11 - 2) * strength))),
        'noise_std': random.uniform(5.0 / 255., 10.0 / 255.) * strength,
        'jpeg_quality': random.randint(int(40 + (1 - strength) * 60), 100),
        'rnum': random.random(),
        'up_scale': random.uniform(1, 1 + strength),
        'down_scale': random.uniform(0.125 + (1 - strength) * 0.375, 1),
    }
    distortion_prob = [min(1.0, random.random() / strength) for _ in range(4)]

    gt_list = []
    base_degraded = []

    # 第一阶段：纹理叠加 + 标准退化
    for frame in video_list:
        h, w = frame.shape[:2]

        # GT 锐化
        gt_frame = add_sharpening(frame)
        gt_list.append(gt_frame)

        # 转为 PIL
        degraded_pil = transfer_cv2_to_pil(frame)

        # 纹理叠加
        if texture_files:
            texture_path = random.choice(texture_files)
            texture_pil = Image.open(texture_path).convert("L")
            degraded_pil = texture_blending(degraded_pil, texture_pil, h, w)

        # 标准退化
        degraded_pil = standard_degradation(degraded_pil, distortion_params, distortion_prob)

        # 颜色抖动
        degraded_pil = color_jitter(degraded_pil)

        # 转回 numpy
        degraded_cv2 = transfer_pil_to_cv2(degraded_pil.convert("RGB"))
        base_degraded.append(degraded_cv2)

    # 第二阶段：老电影损伤（帧间一致）
    if film_damage_gen:
        degraded_list = []
        persistent_scratch = None
        for degraded in base_degraded:
            degraded, persistent_scratch = film_damage_gen.apply(degraded, persistent_scratch)
            degraded_list.append(degraded)
    else:
        degraded_list = base_degraded

    # 第三阶段：序列级闪烁
    if flicker_gen:
        degraded_list = flicker_gen.apply(degraded_list)

    return degraded_list, gt_list
