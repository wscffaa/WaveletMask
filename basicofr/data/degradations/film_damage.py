"""
老电影损伤生成器

基于 FilmDamageSimulator 实现，提供真实老电影退化效果：
- 划痕 (scratches): 垂直/斜线划痕，支持帧间延续
- 污渍 (dirt): 不规则形状污点
- 灰尘 (dust): 细小斑点
- 毛发 (hair): 长/短毛发
- 闪烁 (flicker): 全局/局部亮度波动

核心改进：
1. 使用真实扫描提取的素材库
2. Perlin 噪声控制空间分布
3. Gamma 分布采样数量/大小
4. 帧间时序一致性

Author: BasicOFR Team
Date: 2025-12-24
"""

import os
import random
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d


class PerlinNoise2D:
    """Perlin 噪声生成器

    用于控制 artifacts 的空间分布，使其更自然聚集。
    基于 FilmDamageSimulator 的实现。
    """

    @staticmethod
    def generate(shape: Tuple[int, int], res: Tuple[int, int]) -> np.ndarray:
        """生成 2D Perlin 噪声

        Args:
            shape: 输出形状 (H, W)
            res: 噪声分辨率 (res_y, res_x)

        Returns:
            归一化到 [0, 1] 的噪声数组
        """
        def f(t):
            return 6 * t**5 - 15 * t**4 + 10 * t**3

        delta = (res[0] / shape[0], res[1] / shape[1])
        d = (shape[0] // res[0], shape[1] // res[1])

        grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1

        # Gradients
        angles = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)
        gradients = np.dstack((np.cos(angles), np.sin(angles)))

        g00 = gradients[0:-1, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
        g10 = gradients[1:, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
        g01 = gradients[0:-1, 1:].repeat(d[0], 0).repeat(d[1], 1)
        g11 = gradients[1:, 1:].repeat(d[0], 0).repeat(d[1], 1)

        # Ramps
        n00 = np.sum(grid * g00, 2)
        n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, 2)
        n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, 2)
        n11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, 2)

        # Interpolation
        t = f(grid)
        n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
        n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11

        noise = np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)

        # Normalize to [0, 1]
        noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)

        return noise

    @staticmethod
    def sample_positions(noise: np.ndarray, num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """基于噪声分布采样位置

        Args:
            noise: Perlin 噪声数组
            num_samples: 采样数量

        Returns:
            (y_coords, x_coords) 采样坐标
        """
        # 将噪声作为概率分布
        prob = noise.ravel() / noise.sum()
        linear_idx = np.random.choice(noise.size, p=prob, size=num_samples)
        y, x = np.unravel_index(linear_idx, noise.shape)
        return y, x


class SyntheticAssetBank:
    """合成素材库管理器

    管理从 FilmDamageSimulator 提取的各类老电影损伤素材。
    支持懒加载和缓存机制。
    """

    # 素材类型映射
    ASSET_TYPES = {
        'scratches': ['scratches'],
        'dirt': ['dirt', 'stain', 'smut', 'spots'],
        'dust': ['dots', 'sprinkles'],
        'hair': ['hair'],
        'hair_short': ['hair-short', 'lint'],
    }

    def __init__(self, assets_root: str, cache_size: int = 100):
        """
        Args:
            assets_root: 素材根目录
            cache_size: 每类素材的缓存数量
        """
        self.assets_root = assets_root
        self.cache_size = cache_size

        # 文件列表缓存
        self._file_lists: Dict[str, List[str]] = {}
        # 素材缓存
        self._cache: Dict[str, List[np.ndarray]] = {}

        self._scan_assets()

    def _scan_assets(self):
        """扫描素材目录"""
        for asset_type, folders in self.ASSET_TYPES.items():
            files = []
            for folder in folders:
                folder_path = os.path.join(self.assets_root, folder)
                if os.path.exists(folder_path):
                    for fname in os.listdir(folder_path):
                        if fname.endswith('.png'):
                            files.append(os.path.join(folder_path, fname))
            self._file_lists[asset_type] = sorted(files)

    def _load_asset(self, path: str) -> np.ndarray:
        """加载单个素材（带 alpha 通道）"""
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"无法加载素材: {path}")

        # 提取 alpha 通道作为掩码
        if img.ndim == 3 and img.shape[2] == 4:
            alpha = img[:, :, 3].astype(np.float32) / 255.0
        elif img.ndim == 2:
            alpha = img.astype(np.float32) / 255.0
        else:
            alpha = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

        return alpha

    def get_assets(self, asset_type: str, n: int) -> List[np.ndarray]:
        """获取指定类型的素材

        Args:
            asset_type: 素材类型 ('scratches', 'dirt', 'dust', 'hair', 'hair_short')
            n: 需要的数量

        Returns:
            素材列表，每个元素是 [H, W] float32 掩码
        """
        if asset_type not in self._file_lists:
            raise ValueError(f"未知素材类型: {asset_type}")

        files = self._file_lists[asset_type]
        if not files:
            return []

        # 随机采样
        selected = random.choices(files, k=n)

        assets = []
        for path in selected:
            try:
                asset = self._load_asset(path)
                assets.append(asset)
            except Exception:
                continue

        return assets

    def get_scratches(self, n: int) -> List[np.ndarray]:
        return self.get_assets('scratches', n)

    def get_dirt(self, n: int) -> List[np.ndarray]:
        return self.get_assets('dirt', n)

    def get_dust(self, n: int) -> List[np.ndarray]:
        return self.get_assets('dust', n)

    def get_hair(self, n: int, hair_type: str = 'long') -> List[np.ndarray]:
        if hair_type == 'short':
            return self.get_assets('hair_short', n)
        return self.get_assets('hair', n)

    @property
    def available_types(self) -> List[str]:
        return [k for k, v in self._file_lists.items() if v]


class FilmDamageGenerator:
    """老电影损伤生成器

    生成划痕、污渍、灰尘、毛发等老电影特有损伤。
    支持帧间时序一致性（划痕延续）。
    """

    def __init__(
        self,
        assets_root: str,
        config: Optional[dict] = None,
    ):
        """
        Args:
            assets_root: 素材库根目录
            config: 配置字典，包含各类损伤的参数
        """
        self.config = config or {}
        self.asset_bank = SyntheticAssetBank(assets_root)
        self.perlin = PerlinNoise2D()

        # 默认配置
        self.scratch_config = self.config.get('scratch', {
            'enabled': True,
            'prob': 0.7,
            'num_range': [1, 5],
            'persistence': 0.7,
            'intensity_range': [0.6, 0.95],
        })

        self.dirt_config = self.config.get('dirt', {
            'enabled': True,
            'prob': 0.5,
            'num_range': [3, 15],
            'scale_range': [0.3, 1.5],
        })

        self.dust_config = self.config.get('dust', {
            'enabled': True,
            'prob': 0.6,
            'num_range': [10, 30],
            'scale_range': [0.2, 0.8],
        })

        self.hair_config = self.config.get('hair', {
            'enabled': True,
            'prob': 0.3,
            'num_range': [1, 3],
            'types': ['long', 'short'],
        })

    def _sample_num(self, num_range: List[int]) -> int:
        """使用 Gamma 分布采样数量"""
        mean = (num_range[0] + num_range[1]) / 2
        shape = 2.0
        scale = mean / shape
        num = int(np.random.gamma(shape, scale))
        return max(num_range[0], min(num_range[1], num))

    def _augment_asset(
        self,
        asset: np.ndarray,
        target_h: int,
        target_w: int,
        scale_range: List[float] = [0.5, 1.5],
        allow_rotation: bool = True,
    ) -> np.ndarray:
        """增强素材：缩放、旋转

        Args:
            asset: 输入素材
            target_h, target_w: 目标尺寸（用于计算相对缩放）
            scale_range: 缩放范围
            allow_rotation: 是否允许旋转

        Returns:
            增强后的素材
        """
        h, w = asset.shape[:2]

        # 随机缩放
        scale = random.uniform(*scale_range)
        # 相对于目标尺寸的缩放因子
        relative_scale = min(target_h, target_w) / max(h, w) * scale
        new_h = max(1, int(h * relative_scale))
        new_w = max(1, int(w * relative_scale))

        asset = cv2.resize(asset, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 随机旋转
        if allow_rotation:
            angle = random.uniform(0, 360)
            center = (new_w // 2, new_h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)

            # 计算旋转后的边界
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])
            rot_w = int(new_h * sin + new_w * cos)
            rot_h = int(new_h * cos + new_w * sin)

            M[0, 2] += (rot_w - new_w) / 2
            M[1, 2] += (rot_h - new_h) / 2

            asset = cv2.warpAffine(asset, M, (rot_w, rot_h))

        return asset

    def _blend_asset(
        self,
        frame: np.ndarray,
        asset: np.ndarray,
        y: int,
        x: int,
        intensity: float = 0.8,
        color: float = 1.0,
    ) -> np.ndarray:
        """将素材混合到帧上

        Args:
            frame: 目标帧 [H, W, C] float32 [0, 1]
            asset: 素材掩码 [h, w] float32 [0, 1]
            y, x: 中心位置
            intensity: 混合强度
            color: 损伤颜色 (0=黑, 1=白)

        Returns:
            混合后的帧
        """
        h, w = frame.shape[:2]
        ah, aw = asset.shape[:2]

        # 计算有效区域
        y1 = y - ah // 2
        y2 = y1 + ah
        x1 = x - aw // 2
        x2 = x1 + aw

        # 裁剪到帧边界
        asset_y1 = max(0, -y1)
        asset_y2 = ah - max(0, y2 - h)
        asset_x1 = max(0, -x1)
        asset_x2 = aw - max(0, x2 - w)

        y1 = max(0, y1)
        y2 = min(h, y2)
        x1 = max(0, x1)
        x2 = min(w, x2)

        if y1 >= y2 or x1 >= x2:
            return frame

        # 提取有效区域
        asset_crop = asset[asset_y1:asset_y2, asset_x1:asset_x2]

        # 混合
        alpha = asset_crop * intensity
        if frame.ndim == 3:
            alpha = alpha[:, :, np.newaxis]

        frame[y1:y2, x1:x2] = frame[y1:y2, x1:x2] * (1 - alpha) + color * alpha

        return frame

    def _apply_full_mask(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        intensity: float = 0.8,
        color: float = 1.0,
    ) -> np.ndarray:
        """将全尺寸掩码直接混合到帧上"""
        if mask is None:
            return frame
        alpha = np.clip(mask * intensity, 0, 1)
        if frame.ndim == 3:
            alpha = alpha[:, :, np.newaxis]
        return frame * (1 - alpha) + color * alpha

    def _generate_synthetic_scratch(self, h: int, w: int) -> np.ndarray:
        """合成细长划痕掩码（用于素材缺失时的回退）"""
        mask = np.zeros((h, w), dtype=np.float32)
        x1 = random.randint(0, max(0, w - 1))
        x2 = int(np.clip(x1 + random.randint(-w // 20, w // 20), 0, w - 1))
        thickness = random.randint(1, 2)
        cv2.line(mask, (x1, 0), (x2, h - 1), 1.0, thickness=thickness)

        # 生成纵向断裂
        if random.random() < 0.7:
            num_gaps = random.randint(1, 3)
            keep = np.ones(h, dtype=np.float32)
            for _ in range(num_gaps):
                gap_len = random.randint(max(1, h // 40), max(2, h // 15))
                gap_y = random.randint(0, max(0, h - gap_len))
                keep[gap_y:gap_y + gap_len] = 0.0
            mask = mask * keep[:, None]

        # 边缘轻微柔化
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        return np.clip(mask, 0, 1)

    def add_scratches(
        self,
        frame: np.ndarray,
        persistent_scratch: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """添加划痕

        Args:
            frame: 输入帧 [H, W, C] float32 [0, 1]
            persistent_scratch: 上一帧的划痕掩码（用于时序一致性）

        Returns:
            (退化后的帧, 当前划痕掩码)
        """
        if not self.scratch_config.get('enabled', True):
            return frame, None

        if random.random() > self.scratch_config.get('prob', 0.7):
            return frame, persistent_scratch

        h, w = frame.shape[:2]
        degraded = frame.copy()

        # 时序一致性：继承上一帧的划痕
        persistence = self.scratch_config.get('persistence', 0.7)
        if persistent_scratch is not None and random.random() < persistence:
            # 轻微偏移模拟抖动
            shift_x = random.randint(-2, 2)
            M = np.float32([[1, 0, shift_x], [0, 1, 0]])
            current_scratch = cv2.warpAffine(persistent_scratch, M, (w, h))
        else:
            current_scratch = np.zeros((h, w), dtype=np.float32)

        intensity_range = self.scratch_config.get('intensity_range', [0.6, 0.95])
        persistence_intensity = self.scratch_config.get('persistence_intensity', 0.7)

        # 应用持久划痕到当前帧
        if np.any(current_scratch):
            intensity = random.uniform(*intensity_range) * persistence_intensity
            color = 1.0 if random.random() > 0.3 else 0.0
            degraded = self._apply_full_mask(degraded, current_scratch, intensity, color)

        # 采样新划痕数量
        num_range = self.scratch_config.get('num_range', [1, 5])
        num_new = self._sample_num(num_range) if random.random() > persistence else max(0, self._sample_num([0, 2]))

        # 获取划痕素材
        scratches = self.asset_bank.get_scratches(num_new)

        if len(scratches) < num_new:
            missing = num_new - len(scratches)
            scratches.extend(self._generate_synthetic_scratch(h, w) for _ in range(missing))

        if scratches:
            # Perlin 噪声控制位置
            noise = self.perlin.generate((h, w), (2, 2))
            ys, xs = self.perlin.sample_positions(noise, len(scratches))

            for scratch, y, x in zip(scratches, ys, xs):
                # 划痕增强：主要是垂直方向，轻微旋转
                scratch = self._augment_asset(
                    scratch, h, w,
                    scale_range=[0.5, 1.2],
                    allow_rotation=False,  # 划痕保持垂直
                )

                # 轻微角度偏移
                angle = random.uniform(-5, 5)
                if abs(angle) > 0.5:
                    center = (scratch.shape[1] // 2, scratch.shape[0] // 2)
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    scratch = cv2.warpAffine(scratch, M, (scratch.shape[1], scratch.shape[0]))

                intensity = random.uniform(*intensity_range)
                color = 1.0 if random.random() > 0.3 else 0.0

                # 更新掩码
                self._blend_asset(
                    current_scratch[:, :, np.newaxis] if current_scratch.ndim == 2 else current_scratch,
                    scratch, int(y), int(x), intensity=1.0, color=1.0
                )

                # 应用到帧
                degraded = self._blend_asset(degraded, scratch, int(y), int(x), intensity, color)

        return np.clip(degraded, 0, 1), current_scratch

    def add_dirt(self, frame: np.ndarray) -> np.ndarray:
        """添加污渍"""
        if not self.dirt_config.get('enabled', True):
            return frame

        if random.random() > self.dirt_config.get('prob', 0.5):
            return frame

        h, w = frame.shape[:2]
        degraded = frame.copy()

        num_range = self.dirt_config.get('num_range', [3, 15])
        num = self._sample_num(num_range)

        dirts = self.asset_bank.get_dirt(num)

        if dirts:
            noise = self.perlin.generate((h, w), (4, 4))
            ys, xs = self.perlin.sample_positions(noise, len(dirts))

            scale_range = self.dirt_config.get('scale_range', [0.3, 1.5])

            for dirt, y, x in zip(dirts, ys, xs):
                dirt = self._augment_asset(dirt, h, w, scale_range)
                intensity = random.uniform(0.5, 0.9)
                color = random.choice([0.0, 0.2, 0.8, 1.0])
                degraded = self._blend_asset(degraded, dirt, int(y), int(x), intensity, color)

        return np.clip(degraded, 0, 1)

    def add_dust(self, frame: np.ndarray) -> np.ndarray:
        """添加灰尘"""
        if not self.dust_config.get('enabled', True):
            return frame

        if random.random() > self.dust_config.get('prob', 0.6):
            return frame

        h, w = frame.shape[:2]
        degraded = frame.copy()

        num_range = self.dust_config.get('num_range', [10, 30])
        num = self._sample_num(num_range)

        dusts = self.asset_bank.get_dust(num)

        if dusts:
            noise = self.perlin.generate((h, w), (8, 8))
            ys, xs = self.perlin.sample_positions(noise, len(dusts))

            scale_range = self.dust_config.get('scale_range', [0.2, 0.8])

            for dust, y, x in zip(dusts, ys, xs):
                dust = self._augment_asset(dust, h, w, scale_range)
                intensity = random.uniform(0.4, 0.8)
                color = 1.0 if random.random() > 0.5 else 0.0
                degraded = self._blend_asset(degraded, dust, int(y), int(x), intensity, color)

        return np.clip(degraded, 0, 1)

    def add_hair(self, frame: np.ndarray) -> np.ndarray:
        """添加毛发"""
        if not self.hair_config.get('enabled', True):
            return frame

        if random.random() > self.hair_config.get('prob', 0.3):
            return frame

        h, w = frame.shape[:2]
        degraded = frame.copy()

        num_range = self.hair_config.get('num_range', [1, 3])
        num = self._sample_num(num_range)

        hair_types = self.hair_config.get('types', ['long', 'short'])
        hair_type = random.choice(hair_types)

        hairs = self.asset_bank.get_hair(num, hair_type)

        if hairs:
            # 毛发分布更随机
            ys = np.random.randint(0, h, size=len(hairs))
            xs = np.random.randint(0, w, size=len(hairs))

            for hair, y, x in zip(hairs, ys, xs):
                hair = self._augment_asset(hair, h, w, scale_range=[0.3, 1.0])
                intensity = random.uniform(0.6, 0.9)
                degraded = self._blend_asset(degraded, hair, int(y), int(x), intensity, color=0.0)

        return np.clip(degraded, 0, 1)

    def apply(
        self,
        frame: np.ndarray,
        persistent_scratch: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """应用所有损伤

        Args:
            frame: 输入帧 [H, W, C] float32 [0, 1]
            persistent_scratch: 上一帧的划痕掩码

        Returns:
            (退化后的帧, 当前划痕掩码)
        """
        degraded = frame.copy()

        # 按顺序应用各种损伤
        degraded, current_scratch = self.add_scratches(degraded, persistent_scratch)
        degraded = self.add_dirt(degraded)
        degraded = self.add_dust(degraded)
        degraded = self.add_hair(degraded)

        return degraded, current_scratch


class FlickerGenerator:
    """闪烁生成器

    模拟老电影的亮度闪烁，支持：
    - global: 全局亮度波动
    - local: 局部区域闪烁
    - vignette: 渐晕效果
    """

    @staticmethod
    def _sanitize_range(value: Optional[list], fallback: List[float]) -> List[float]:
        if not value or len(value) != 2:
            return fallback
        low, high = float(value[0]), float(value[1])
        if low > high:
            low, high = high, low
        return [low, high]

    @staticmethod
    def _curve_to_range(curve: np.ndarray, target_range: List[float], reference: float) -> np.ndarray:
        if reference <= 1e-6:
            return np.full_like(curve, 1.0, dtype=np.float32)
        ratio = np.clip(curve / reference, -1.0, 1.0)
        low, high = target_range
        return (low + (ratio + 1.0) * 0.5 * (high - low)).astype(np.float32)

    def __init__(self, config: Optional[dict] = None):
        """
        Args:
            config: 闪烁配置
        """
        self.config = config or {}
        self.enabled = self.config.get('enabled', True)
        self.prob = self.config.get('prob', 0.4)
        self.intensity_range = self._sanitize_range(
            self.config.get('intensity_range'), [-0.15, 0.15]
        )
        self.modes = self.config.get('modes', ['global', 'local', 'vignette'])
        self.mode_weights = self.config.get('mode_weights')
        self.noise_ratio = float(self.config.get('noise_ratio', 0.3))
        self.jump_prob = float(self.config.get('jump_prob', 0.0))
        self.jump_range = self._sanitize_range(
            self.config.get('jump_range'), self.intensity_range
        )
        self.apply_gain_gamma = self.config.get('use_gain_gamma', True)

        base = max(abs(self.intensity_range[0]), abs(self.intensity_range[1]))
        self.gain_range = self.config.get('gain_range')
        if self.gain_range is None:
            self.gain_range = [1.0 - base * 0.5, 1.0 + base * 0.5]
        else:
            self.gain_range = self._sanitize_range(self.gain_range, [1.0, 1.0])

        self.gamma_range = self.config.get('gamma_range')
        if self.gamma_range is None:
            gamma_delta = min(0.15, base * 0.4)
            self.gamma_range = [max(0.1, 1.0 - gamma_delta), 1.0 + gamma_delta]
        else:
            self.gamma_range = self._sanitize_range(self.gamma_range, [1.0, 1.0])

        if self.mode_weights is None and self.modes:
            weights = []
            for mode in self.modes:
                if mode == 'global':
                    weights.append(0.7)
                elif mode == 'local':
                    weights.append(0.2)
                elif mode == 'vignette':
                    weights.append(0.1)
                else:
                    weights.append(1.0)
            self.mode_weights = weights

    def _generate_flicker_curve(self, T: int) -> np.ndarray:
        """生成平滑的闪烁曲线

        Args:
            T: 帧数

        Returns:
            长度为 T 的亮度偏移数组
        """
        # 低频正弦波 + 随机噪声
        t = np.linspace(0, 2 * np.pi * random.uniform(0.5, 2), T)
        intensity = (self.intensity_range[1] - self.intensity_range[0]) / 2
        if intensity <= 0:
            return np.zeros(T, dtype=np.float32)
        base = np.sin(t) * intensity
        noise = np.random.randn(T) * intensity * self.noise_ratio

        # 平滑滤波
        curve = gaussian_filter1d(base + noise, sigma=max(1, T // 10))

        # 叠加曝光跳变
        if self.jump_prob > 0:
            jump_offset = 0.0
            for i in range(T):
                if random.random() < self.jump_prob:
                    jump_offset += random.uniform(*self.jump_range)
                curve[i] += jump_offset

        curve = np.clip(curve, self.intensity_range[0], self.intensity_range[1])
        return curve.astype(np.float32)

    def _apply_global_flicker(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """全局亮度闪烁"""
        T = len(frames)
        curve = self._generate_flicker_curve(T)
        reference = max(abs(self.intensity_range[0]), abs(self.intensity_range[1]), 1e-6)
        if self.apply_gain_gamma:
            gain_curve = self._curve_to_range(curve, self.gain_range, reference)
            gamma_curve = self._curve_to_range(curve, self.gamma_range, reference)
        else:
            gain_curve = np.ones_like(curve, dtype=np.float32)
            gamma_curve = np.ones_like(curve, dtype=np.float32)

        result = []
        for frame, shift, gain, gamma in zip(frames, curve, gain_curve, gamma_curve):
            degraded = np.clip(frame * gain + shift, 0, 1)
            if abs(float(gamma) - 1.0) > 1e-3:
                degraded = np.power(np.clip(degraded, 1e-6, 1.0), gamma)
            result.append(degraded.astype(np.float32))

        return result

    def _apply_local_flicker(self, frame: np.ndarray) -> np.ndarray:
        """局部区域闪烁"""
        h, w = frame.shape[:2]
        degraded = frame.copy()

        num_regions = random.randint(1, 3)

        for _ in range(num_regions):
            # 随机矩形区域
            x1 = random.randint(0, w // 2)
            y1 = random.randint(0, h // 2)
            x2 = random.randint(x1 + w // 4, w)
            y2 = random.randint(y1 + h // 4, h)

            shift = random.uniform(*self.intensity_range)
            degraded[y1:y2, x1:x2] = np.clip(degraded[y1:y2, x1:x2] + shift, 0, 1)

        return degraded.astype(np.float32)

    def _apply_vignette_flicker(self, frame: np.ndarray) -> np.ndarray:
        """渐晕效果闪烁"""
        h, w = frame.shape[:2]

        y_coords, x_coords = np.ogrid[:h, :w]
        center_y, center_x = h / 2, w / 2

        dist_from_center = np.sqrt(
            (x_coords - center_x) ** 2 + (y_coords - center_y) ** 2
        )
        max_dist = np.sqrt(center_x ** 2 + center_y ** 2)

        # 渐晕掩码
        vignette = 1 - (dist_from_center / max_dist) ** 2

        shift = random.uniform(*self.intensity_range)

        if frame.ndim == 3:
            vignette = vignette[:, :, np.newaxis]

        degraded = frame + shift * (1 - vignette)

        return np.clip(degraded, 0, 1).astype(np.float32)

    def apply(
        self,
        frames: List[np.ndarray],
        mode: Optional[str] = None,
    ) -> List[np.ndarray]:
        """应用闪烁效果

        Args:
            frames: 帧列表
            mode: 闪烁模式 ('global', 'local', 'vignette', 'random', None)

        Returns:
            处理后的帧列表
        """
        if not self.enabled:
            return frames

        if random.random() > self.prob:
            return frames

        if mode is None or mode == 'random':
            if self.mode_weights and len(self.mode_weights) == len(self.modes):
                mode = random.choices(self.modes, weights=self.mode_weights, k=1)[0]
            else:
                mode = random.choice(self.modes)

        if mode == 'global':
            return self._apply_global_flicker(frames)

        elif mode == 'local':
            return [
                self._apply_local_flicker(f) if random.random() < 0.5 else f
                for f in frames
            ]

        elif mode == 'vignette':
            return [self._apply_vignette_flicker(f) for f in frames]

        return frames
