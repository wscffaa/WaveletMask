"""基于 pyiqa 的统一图像质量评估指标。

统一使用 pyiqa 库实现所有老电影修复评价指标，替代之前分散的多库实现。
接口与 BasicSR METRIC_REGISTRY 兼容，可直接在 options/test 配置文件中使用。

## 支持的指标

全参考指标 (FR, 需要 GT):
- calculate_psnr: 峰值信噪比（越高越好）
- calculate_ssim: 结构相似性（越高越好）
- calculate_lpips: 感知相似度（越低越好）
- calculate_dists: 结构纹理相似度（越低越好）

无参考指标 (NR, 不需要 GT):
- calculate_niqe: 自然图像质量（越低越好，范围约 2-8）
- calculate_brisque: 盲图像质量（越低越好）
- calculate_clipiqa: CLIP 图像质量（越高越好）

数据集级指标:
- calculate_fid: Fréchet Inception Distance（越低越好）

## 依赖
- pyiqa: pip install pyiqa (>= 0.1.10)

## 变更记录
- 2024-12-29: 统一迁移至 pyiqa 实现，替代 lpips/piq/skvideo/BasicSR 多库方案
"""
import logging
import math
import numpy as np
import torch
from typing import Optional, Dict, Any

logger = logging.getLogger('basicsr')

from basicsr.metrics.metric_util import reorder_image, to_y_channel
from basicsr.utils.registry import METRIC_REGISTRY

# ===================== 模型缓存 =====================
_MODEL_CACHE: Dict[str, Any] = {}


def _get_device():
    """获取可用设备。"""
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def _get_model(metric_name: str, device: Optional[str] = None, **kwargs):
    """获取缓存的 pyiqa 模型。

    Args:
        metric_name: pyiqa 指标名称
        device: 计算设备
        **kwargs: 传递给 pyiqa.create_metric 的额外参数

    Returns:
        pyiqa 模型实例
    """
    import pyiqa

    if device is None:
        device = _get_device()

    cache_key = f"{metric_name}_{device}"
    if cache_key not in _MODEL_CACHE:
        _MODEL_CACHE[cache_key] = pyiqa.create_metric(metric_name, device=device, **kwargs)

    return _MODEL_CACHE[cache_key]


def _ensure_3ch(img: np.ndarray) -> np.ndarray:
    """确保图像为 3 通道 (H,W,3)。

    处理单通道灰度图 (H,W,1) 或 (H,W) → 复制为 3 通道灰度 RGB。
    3 通道图原样返回。
    """
    if img.ndim == 2:
        return np.stack([img] * 3, axis=-1)
    if img.ndim == 3 and img.shape[2] == 1:
        return np.concatenate([img] * 3, axis=-1)
    return img


def _img_to_tensor(img: np.ndarray, input_order: str = 'HWC',
                   crop_border: int = 0, normalize: bool = True) -> torch.Tensor:
    """将 numpy 图像转换为 tensor。

    Args:
        img: numpy array, [0, 255], BGR 或灰度格式
        input_order: 'HWC' 或 'CHW'
        crop_border: 裁剪边缘像素
        normalize: 是否归一化到 [0, 1]

    Returns:
        tensor: [1, 3, H, W], 范围 [0, 1]
    """
    img = reorder_image(img, input_order=input_order)
    img = _ensure_3ch(img)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]

    # BGR -> RGB
    if img.ndim == 3 and img.shape[2] == 3:
        img = img[..., ::-1].copy()

    # [0, 255] -> [0, 1]
    img = img.astype(np.float32)
    if normalize:
        img = img / 255.0

    # HWC -> CHW -> NCHW
    if img.ndim == 3:
        img = np.transpose(img, (2, 0, 1))
    else:
        img = img[np.newaxis, ...]

    return torch.from_numpy(img).unsqueeze(0)


def _is_grayscale(img: np.ndarray, tolerance: float = 3.0) -> bool:
    """检测图像是否为灰度图 (R≈G≈B)。

    使用平均差异检测，tolerance 默认为 3.0 以适应老电影修复场景。
    老电影修复模型输出可能是"近灰度"（R-G 差异约 1-2），需要较宽松的阈值。

    Args:
        img: numpy array, [0, 255], HWC 格式
        tolerance: 通道间平均差异阈值，默认 3.0

    Returns:
        bool: 是否为灰度图
    """
    if len(img.shape) == 2:
        return True
    if img.shape[2] == 1:
        return True
    # 使用平均差异而非最大差异，更稳定
    rg_diff = np.abs(img[..., 0].astype(float) - img[..., 1].astype(float)).mean()
    gb_diff = np.abs(img[..., 1].astype(float) - img[..., 2].astype(float)).mean()
    return rg_diff < tolerance and gb_diff < tolerance


def _to_grayscale_rgb(img: np.ndarray) -> np.ndarray:
    """将图像转换为灰度 RGB (3通道相同值)。

    支持 (H,W), (H,W,1), (H,W,3) 各种输入格式。
    """
    from PIL import Image
    # 处理单通道 (H,W,1) → (H,W)
    if img.ndim == 3 and img.shape[2] == 1:
        img = img.squeeze(axis=2)
    pil_img = Image.fromarray(img.astype(np.uint8))
    gray_rgb = pil_img.convert('L').convert('RGB')
    return np.array(gray_rgb)


def _adaptive_preprocess(img: np.ndarray, img2: np.ndarray) -> tuple:
    """感知指标的自适应预处理。

    当模型输出为灰度时，GT 也转为灰度 RGB，确保公平比较。
    """
    if _is_grayscale(img):
        if not _is_grayscale(img2):
            img2 = _to_grayscale_rgb(img2)
    return img, img2


# ===================== PSNR =====================
def calculate_psnr(img: np.ndarray, img2: np.ndarray, crop_border: int = 0,
                   input_order: str = 'HWC', test_y_channel: bool = False,
                   device: Optional[str] = None, **kwargs) -> float:
    """计算 PSNR（基于 pyiqa，默认 RGB 通道）。

    Args:
        img: 预测图像, numpy array, [0, 255], BGR
        img2: GT 图像, numpy array, [0, 255], BGR
        crop_border: 裁剪边缘像素
        input_order: 'HWC' 或 'CHW'
        test_y_channel: 是否在 Y 通道计算（默认 False，RGB 全通道）
        device: 计算设备

    Returns:
        float: PSNR 值（越高越好）
    """
    if device is None:
        device = _get_device()

    model = _get_model('psnr', device=device)

    # 统一为 HWC 格式，确保 3 通道
    img = _ensure_3ch(reorder_image(img, input_order=input_order))
    img2 = _ensure_3ch(reorder_image(img2, input_order=input_order))

    # 裁剪边缘
    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    # Y 通道计算
    if test_y_channel:
        img = to_y_channel(img)
        img2 = to_y_channel(img2)
        # Y 通道是单通道，需要扩展为 3 通道
        img = np.stack([img.squeeze()] * 3, axis=-1)
        img2 = np.stack([img2.squeeze()] * 3, axis=-1)

    # BGR -> RGB
    img_rgb = img[..., ::-1].copy() if not test_y_channel else img
    img2_rgb = img2[..., ::-1].copy() if not test_y_channel else img2

    # 转 tensor
    img_t = torch.from_numpy(img_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    img2_t = torch.from_numpy(img2_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        score = model(img_t, img2_t)

    return score.item()


# ===================== SSIM =====================
def calculate_ssim(img: np.ndarray, img2: np.ndarray, crop_border: int = 0,
                   input_order: str = 'HWC', test_y_channel: bool = False,
                   device: Optional[str] = None, **kwargs) -> float:
    """计算 SSIM（基于 pyiqa，默认 RGB 通道）。

    Args:
        img: 预测图像, numpy array, [0, 255], BGR
        img2: GT 图像, numpy array, [0, 255], BGR
        crop_border: 裁剪边缘像素
        input_order: 'HWC' 或 'CHW'
        test_y_channel: 是否在 Y 通道计算（默认 False，RGB 全通道）
        device: 计算设备

    Returns:
        float: SSIM 值（越高越好）
    """
    if device is None:
        device = _get_device()

    model = _get_model('ssim', device=device)

    # 统一为 HWC 格式，确保 3 通道
    img = _ensure_3ch(reorder_image(img, input_order=input_order))
    img2 = _ensure_3ch(reorder_image(img2, input_order=input_order))

    # 裁剪边缘
    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    # Y 通道计算
    if test_y_channel:
        img = to_y_channel(img)
        img2 = to_y_channel(img2)
        img = np.stack([img.squeeze()] * 3, axis=-1)
        img2 = np.stack([img2.squeeze()] * 3, axis=-1)

    # BGR -> RGB
    img_rgb = img[..., ::-1].copy() if not test_y_channel else img
    img2_rgb = img2[..., ::-1].copy() if not test_y_channel else img2

    # 转 tensor
    img_t = torch.from_numpy(img_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    img2_t = torch.from_numpy(img2_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        score = model(img_t, img2_t)

    return score.item()


# ===================== LPIPS =====================
def calculate_lpips(img: np.ndarray, img2: np.ndarray, crop_border: int = 0,
                    input_order: str = 'HWC', device: Optional[str] = None,
                    convert_to_grayscale: Optional[bool] = None, **kwargs) -> float:
    """计算 LPIPS（基于 pyiqa）。

    Args:
        img: 预测图像, numpy array, [0, 255], BGR
        img2: GT 图像, numpy array, [0, 255], BGR
        crop_border: 裁剪边缘像素
        input_order: 'HWC' 或 'CHW'
        device: 计算设备
        convert_to_grayscale: 是否转换为灰度后计算
            - True: 强制将两张图都转为灰度
            - False: 保持原始颜色
            - None: 自动检测（向后兼容）

    Returns:
        float: LPIPS 分数（越低越好）
    """
    if device is None:
        device = _get_device()

    model = _get_model('lpips', device=device)

    # 统一为 HWC 格式，确保 3 通道
    img = _ensure_3ch(reorder_image(img, input_order=input_order))
    img2 = _ensure_3ch(reorder_image(img2, input_order=input_order))

    # 裁剪边缘
    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    # BGR -> RGB
    img_rgb = img[..., ::-1].copy()
    img2_rgb = img2[..., ::-1].copy()

    # 灰度转换
    if convert_to_grayscale is True:
        # 强制灰度
        img_rgb = _to_grayscale_rgb(img_rgb)
        img2_rgb = _to_grayscale_rgb(img2_rgb)
    elif convert_to_grayscale is None:
        # 自动检测（向后兼容）
        img_rgb, img2_rgb = _adaptive_preprocess(img_rgb, img2_rgb)
    # convert_to_grayscale=False: 保持原样

    # 转 tensor，范围 [0, 1]
    img_t = torch.from_numpy(img_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    img2_t = torch.from_numpy(img2_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        score = model(img_t, img2_t)

    return score.item()


# ===================== DISTS =====================
def calculate_dists(img: np.ndarray, img2: np.ndarray, crop_border: int = 0,
                    input_order: str = 'HWC', device: Optional[str] = None,
                    convert_to_grayscale: Optional[bool] = None, **kwargs) -> float:
    """计算 DISTS（基于 pyiqa）。

    Args:
        img: 预测图像, numpy array, [0, 255], BGR
        img2: GT 图像, numpy array, [0, 255], BGR
        crop_border: 裁剪边缘像素
        input_order: 'HWC' 或 'CHW'
        device: 计算设备
        convert_to_grayscale: 是否转换为灰度后计算
            - True: 强制将两张图都转为灰度
            - False: 保持原始颜色
            - None: 自动检测（向后兼容）

    Returns:
        float: DISTS 分数（越低越好）
    """
    if device is None:
        device = _get_device()

    model = _get_model('dists', device=device)

    # 统一为 HWC 格式，确保 3 通道
    img = _ensure_3ch(reorder_image(img, input_order=input_order))
    img2 = _ensure_3ch(reorder_image(img2, input_order=input_order))

    # 裁剪边缘
    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    # BGR -> RGB
    img_rgb = img[..., ::-1].copy()
    img2_rgb = img2[..., ::-1].copy()

    # 灰度转换
    if convert_to_grayscale is True:
        # 强制灰度
        img_rgb = _to_grayscale_rgb(img_rgb)
        img2_rgb = _to_grayscale_rgb(img2_rgb)
    elif convert_to_grayscale is None:
        # 自动检测（向后兼容）
        img_rgb, img2_rgb = _adaptive_preprocess(img_rgb, img2_rgb)
    # convert_to_grayscale=False: 保持原样

    # 转 tensor，范围 [0, 1]
    img_t = torch.from_numpy(img_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    img2_t = torch.from_numpy(img2_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        score = model(img_t, img2_t)

    return score.item()


# ===================== NIQE =====================
def calculate_niqe(img: np.ndarray, crop_border: int = 0, input_order: str = 'HWC',
                   device: Optional[str] = None, **kwargs) -> float:
    """计算 NIQE（基于 pyiqa，MATLAB 兼容）。

    使用 pyiqa 的 niqe 实现，输出范围约 2-8，与 MATLAB 官方实现一致。

    Args:
        img: 输入图像, numpy array, [0, 255], BGR
        crop_border: 裁剪边缘像素
        input_order: 'HWC' 或 'CHW'
        device: 计算设备

    Returns:
        float: NIQE 分数（越低越好，范围约 2-8）
    """
    if device is None:
        device = _get_device()

    model = _get_model('niqe', device=device)

    img_t = _img_to_tensor(img, input_order, crop_border).to(device)

    try:
        with torch.no_grad():
            score = model(img_t)
        return score.item()
    except (AssertionError, RuntimeError, torch._C._LinAlgError) as e:
        logger.warning(f'NIQE calculation failed (degenerate input): {e}')
        return float('nan')


# ===================== BRISQUE =====================
def calculate_brisque(img: np.ndarray, crop_border: int = 0, input_order: str = 'HWC',
                      device: Optional[str] = None, **kwargs) -> float:
    """计算 BRISQUE（基于 pyiqa）。

    Args:
        img: 输入图像, numpy array, [0, 255], BGR
        crop_border: 裁剪边缘像素
        input_order: 'HWC' 或 'CHW'
        device: 计算设备

    Returns:
        float: BRISQUE 分数（越低越好）
    """
    if device is None:
        device = _get_device()

    model = _get_model('brisque', device=device)

    img_t = _img_to_tensor(img, input_order, crop_border).to(device)

    try:
        with torch.no_grad():
            score = model(img_t)
        return score.item()
    except (AssertionError, RuntimeError, torch._C._LinAlgError) as e:
        logger.warning(f'BRISQUE calculation failed (degenerate input): {e}')
        return float('nan')


# ===================== CLIP-IQA+ =====================
def calculate_clipiqa(img: np.ndarray, crop_border: int = 0, input_order: str = 'HWC',
                      device: Optional[str] = None, **kwargs) -> float:
    """计算 CLIP-IQA+（基于 pyiqa）。

    Args:
        img: 输入图像, numpy array, [0, 255], BGR
        crop_border: 裁剪边缘像素
        input_order: 'HWC' 或 'CHW'
        device: 计算设备

    Returns:
        float: CLIP-IQA+ 分数（越高越好）
    """
    if device is None:
        device = _get_device()

    model = _get_model('clipiqa+', device=device)

    img_t = _img_to_tensor(img, input_order, crop_border).to(device)

    with torch.no_grad():
        score = model(img_t)

    return score.item()


# ===================== FID =====================
def calculate_fid(real_path: str, fake_path: str, device: Optional[str] = None,
                  batch_size: int = 64, num_workers: int = 4, **kwargs) -> float:
    """计算 FID（基于 pyiqa）。

    这是数据集级别的指标，需要两个图像目录。

    Args:
        real_path: 真实图像目录路径
        fake_path: 生成图像目录路径
        device: 计算设备
        batch_size: 批次大小
        num_workers: 数据加载线程数

    Returns:
        float: FID 分数（越低越好）
    """
    import pyiqa

    if device is None:
        device = _get_device()

    fid_metric = pyiqa.create_metric('fid', device=device)

    score = fid_metric(real_path, fake_path, batch_size=batch_size, num_workers=num_workers)

    return score.item() if hasattr(score, 'item') else float(score)


# ===================== 注册到 BasicSR METRIC_REGISTRY =====================
# 强制覆盖 BasicSR 的注册，确保使用 pyiqa 统一实现
METRIC_REGISTRY._obj_map['calculate_psnr'] = calculate_psnr
METRIC_REGISTRY._obj_map['calculate_ssim'] = calculate_ssim
METRIC_REGISTRY._obj_map['calculate_lpips'] = calculate_lpips
METRIC_REGISTRY._obj_map['calculate_dists'] = calculate_dists
METRIC_REGISTRY._obj_map['calculate_niqe'] = calculate_niqe
METRIC_REGISTRY._obj_map['calculate_brisque'] = calculate_brisque
METRIC_REGISTRY._obj_map['calculate_clipiqa'] = calculate_clipiqa
# FID 不注册到 METRIC_REGISTRY，因为它是数据集级别指标，接口不同
