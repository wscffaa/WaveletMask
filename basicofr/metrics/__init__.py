"""BasicOFR metrics module - 图像质量评估指标。

基于 pyiqa 统一实现所有老电影修复评价指标。
与 BasicSR METRIC_REGISTRY 兼容，可直接在 options/test/*.yml 配置文件中使用。

## PSNR/SSIM 默认 RGB 通道

PSNR 和 SSIM 默认在 RGB 通道计算，与项目统一标准一致。
- test_y_channel 默认为 False
- 可通过配置 test_y_channel: true 切换到 Y 通道计算

## 支持的指标

全参考指标（需要 GT，自适应灰度）:
- calculate_psnr: 峰值信噪比（越高越好）
- calculate_ssim: 结构相似性（越高越好）
- calculate_lpips: 感知相似度（越低越好）
- calculate_dists: 结构纹理相似度（越低越好）

无参考指标:
- calculate_niqe: 自然图像质量（越低越好，范围约 2-8）
- calculate_brisque: 盲图像质量（越低越好）
- calculate_clipiqa: CLIP 图像质量（越高越好）

数据集级指标:
- calculate_fid: Fréchet Inception Distance（越低越好）

## 配置文件用法

```yaml
val:
  metrics:
    psnr:
      type: calculate_psnr
      crop_border: 0
    ssim:
      type: calculate_ssim
      crop_border: 0
    lpips:
      type: calculate_lpips
      crop_border: 0
    dists:
      type: calculate_dists
      crop_border: 0
    niqe:
      type: calculate_niqe
      crop_border: 0
    brisque:
      type: calculate_brisque
      crop_border: 0
    clipiqa:
      type: calculate_clipiqa
      crop_border: 0
```

## 依赖

仅需安装 pyiqa:
```bash
pip install pyiqa>=0.1.10
```

## 变更记录

- 2024-12-29: 统一迁移至 pyiqa 实现
  - 替代之前的 lpips/piq/skvideo/BasicSR 多库方案
  - 所有指标现在由 pyiqa 统一提供
  - NIQE 输出范围约 2-8，与 MATLAB 官方一致
"""

# 导入基于 pyiqa 的统一实现
from .pyiqa_metrics import (
    # 全参考指标
    calculate_psnr,
    calculate_ssim,
    calculate_lpips,
    calculate_dists,
    # 无参考指标
    calculate_niqe,
    calculate_brisque,
    calculate_clipiqa,
    # 数据集级指标
    calculate_fid,
)

# 导入工具函数（保持向后兼容）
from .pyiqa_metrics import (
    _is_grayscale as is_grayscale,
    _to_grayscale_rgb as to_grayscale_rgb,
    _adaptive_preprocess as adaptive_preprocess,
)

__all__ = [
    # 全参考指标
    'calculate_psnr',
    'calculate_ssim',
    'calculate_lpips',
    'calculate_dists',
    # 无参考指标
    'calculate_niqe',
    'calculate_brisque',
    'calculate_clipiqa',
    # 数据集级指标
    'calculate_fid',
    # 工具函数
    'is_grayscale',
    'to_grayscale_rgb',
    'adaptive_preprocess',
]
