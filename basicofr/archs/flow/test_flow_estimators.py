#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
光流估计器测试脚本
测试 RAFT、SpyNet 和 MemFlow 三个光流网络
"""
import sys
from pathlib import Path

# 添加项目根目录到 sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import glob
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm

from basicofr.archs.flow.flow_estimator import FlowEstimator


def flow_to_color(flow, max_flow=None):
    """将光流转换为彩色可视化图像

    Args:
        flow: 光流数组，shape (H, W, 2)
        max_flow: 最大光流幅值，用于归一化

    Returns:
        BGR 格式的可视化图像
    """
    u, v = flow[:, :, 0], flow[:, :, 1]

    if max_flow is None:
        rad = np.sqrt(u**2 + v**2)
        max_flow = np.max(rad)

    u = u / (max_flow + 1e-5)
    v = v / (max_flow + 1e-5)

    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[:, :, 0] = (np.arctan2(v, u) + np.pi) / (2 * np.pi) * 180
    hsv[:, :, 1] = 255
    hsv[:, :, 2] = np.clip(np.sqrt(u**2 + v**2) * 255, 0, 255)

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


def validate_flow(flow, name):
    """验证光流输出是否有效

    Args:
        flow: 光流张量
        name: 光流名称（用于打印）

    Returns:
        bool: 是否有效
    """
    if torch.isnan(flow).any():
        print(f"  ⚠ {name} 包含 NaN", flush=True)
        return False
    flow_mag = torch.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
    print(f"  ✓ {name} 幅值范围: [{flow_mag.min():.2f}, {flow_mag.max():.2f}]", flush=True)
    return True


def save_flow_visualization(flow_tensor, output_dir, prefix):
    """保存光流可视化结果

    Args:
        flow_tensor: 光流张量，shape (1, T-1, 2, H, W)
        output_dir: 输出目录
        prefix: 文件名前缀
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for i in tqdm(range(flow_tensor.shape[1]), desc=f"  保存 {prefix}"):
        flow_vis = flow_to_color(flow_tensor[0, i].permute(1, 2, 0).cpu().numpy())
        cv2.imwrite(str(output_dir / f'{prefix}_{i:03d}.png'), flow_vis)


def prepare_test_data(input_dir=None):
    """准备测试数据

    Args:
        input_dir: 输入图像目录（可选）

    Returns:
        torch.Tensor: 预处理后的图像张量，shape (1, T, 3, 256, 256)，范围 [-1, 1]
    """
    if input_dir is None:
        input_dir = Path(__file__).parent / 'test_images' / 'inputs'

    image_files = sorted(glob.glob(str(input_dir / '*.png')))

    if not image_files:
        print(f"❌ 未找到输入图像，生成合成测试数据...", flush=True)
        t, h, w = 5, 128, 128
        frames = []
        for i in range(t):
            x = np.linspace(0, 1, w) + i * 0.1
            y = np.linspace(0, 1, h).reshape(-1, 1)
            pattern = np.sin(x * 2 * np.pi) * np.cos(y * 2 * np.pi)
            pattern = ((pattern + 1) * 127.5).astype(np.uint8)
            frame = np.stack([pattern] * 3, axis=-1)
            frames.append(frame)
        print(f"✓ 生成 {t} 帧合成图像 ({h}x{w})", flush=True)
    else:
        print(f"✓ 找到 {len(image_files)} 帧图像", flush=True)
        frames = [cv2.imread(img_path) for img_path in image_files]

    # 转换为张量并归一化到 [-1, 1]
    imgs = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).unsqueeze(0).float()
    imgs = (imgs / 127.5) - 1.0

    # 下采样到 256x256
    b, t, c, h, w = imgs.shape
    imgs = imgs.view(b * t, c, h, w)
    imgs = F.interpolate(imgs, size=(256, 256), mode='bilinear', align_corners=False)
    imgs = imgs.view(b, t, c, 256, 256)

    return imgs


def test_single_estimator(estimator_type, imgs, output_base_dir):
    """测试单个光流估计器

    Args:
        estimator_type: 估计器类型 ('raft', 'spynet', 'memflow')
        imgs: 输入图像张量
        output_base_dir: 输出基础目录

    Returns:
        bool: 测试是否成功
    """
    output_dir = output_base_dir / estimator_type

    try:
        print(f"  - 加载 {estimator_type.upper()} 模型...", flush=True)
        estimator = FlowEstimator(estimator_type=estimator_type, normalization='tanh')

        print(f"  - 计算光流...", flush=True)
        forward_flow, backward_flow = estimator.compute_flow(imgs)

        print(f"  ✓ {estimator_type.upper()} 光流形状: forward={forward_flow.shape}, backward={backward_flow.shape}", flush=True)

        fwd_valid = validate_flow(forward_flow, f"{estimator_type.upper()} forward")
        bwd_valid = validate_flow(backward_flow, f"{estimator_type.upper()} backward")

        # 保存可视化结果
        save_flow_visualization(forward_flow, output_dir, 'forward')
        save_flow_visualization(backward_flow, output_dir, 'backward')
        print(f"  ✓ 已保存到: {output_dir}", flush=True)

        return fwd_valid and bwd_valid

    except Exception as e:
        print(f"  ⚠ {estimator_type.upper()} 测试失败: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return False


def test_flow_estimators(flow_modules=None):
    """测试光流估计器

    Args:
        flow_modules: 要测试的光流模块列表，如 ['raft', 'spynet', 'memflow']
                      默认测试全部三个模块
    """
    # 默认测试全部模块
    if flow_modules is None:
        flow_modules = ['raft', 'spynet', 'memflow']

    # 验证模块名称
    valid_modules = {'raft', 'spynet', 'memflow'}
    for module in flow_modules:
        if module.lower() not in valid_modules:
            raise ValueError(f"未知的光流模块: {module}，可选: {valid_modules}")

    flow_modules = [m.lower() for m in flow_modules]

    print("=" * 50)
    print("       光流估计器测试")
    print(f"       测试: {' / '.join([m.upper() for m in flow_modules])}")
    print("=" * 50 + "\n", flush=True)

    # 准备测试数据
    imgs = prepare_test_data()
    print(f"✓ 测试数据形状: {imgs.shape}", flush=True)

    # 输出目录
    output_base_dir = Path(__file__).parent / 'test_images' / 'flows'

    results = {}

    for idx, estimator_type in enumerate(flow_modules, 1):
        print(f"\n[{idx}/{len(flow_modules)}] 测试 {estimator_type.upper()} 估计器...", flush=True)
        results[estimator_type] = test_single_estimator(estimator_type, imgs, output_base_dir)

    # 打印测试结果汇总
    print("\n" + "=" * 50)
    print("       测试结果汇总")
    print("=" * 50)

    all_passed = True
    for estimator_type, passed in results.items():
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {estimator_type.upper():10s}: {status}")
        if not passed:
            all_passed = False

    print("=" * 50)
    if all_passed:
        print("✅ 所有测试通过！")
    else:
        print("⚠ 部分测试失败，请检查上方日志")

    return all_passed


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='光流估计器测试脚本')
    parser.add_argument(
        '--flow_modules',
        nargs='+',
        default=['raft', 'spynet', 'memflow'],
        choices=['raft', 'spynet', 'memflow'],
        help='要测试的光流模块列表，如: --flow_modules raft spynet'
    )

    args = parser.parse_args()
    test_flow_estimators(flow_modules=args.flow_modules)
