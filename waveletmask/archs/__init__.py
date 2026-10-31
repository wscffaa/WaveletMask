"""WaveletMask Architecture Package for BasicOFR.

9 个架构变体用于小波域掩码引导的老电影修复消融实验：

- WaveletMaskBaseNet: Baseline RTN (无创新)
- WaveletMaskSimpleNet: Innovation A (固定权重融合)
- WaveletMaskLocalNet: Innovation A (直接替换 indicator)
- WaveletMaskFlickerNet: Innovation B only (消融)
- WaveletMaskABNet: Innovation A+B (max 融合，无 CNN)
- WaveletMaskLocalFusionNet: Innovation A+C (local + CNN fusion)
- WaveletMaskFlickerFusionNet: Innovation B+C (flicker + CNN fusion)
- WaveletMaskFullNet: Innovation A+B+C (max + CNN fusion) ← BEST (PSNR 26.60)
- WaveletMaskGatedFlickerNet: Innovation A+B+C (门控加权 + CNN fusion)

Best variant: WaveletMaskFullNet (A+B+C), PSNR=26.60 vs base=24.98 (+1.62 dB)
组件位于 components/ 子目录，按需从 .components 导入。
"""

from .waveletmask_base_arch import WaveletMaskBaseNet
from .waveletmask_local_arch import WaveletMaskLocalNet
from .waveletmask_flicker_arch import WaveletMaskFlickerNet
from .waveletmask_ab_arch import WaveletMaskABNet
from .waveletmask_local_fusion_arch import WaveletMaskLocalFusionNet
from .waveletmask_full_arch import WaveletMaskFullNet
from .waveletmask_flicker_fusion_arch import WaveletMaskFlickerFusionNet
from .waveletmask_simple_arch import WaveletMaskSimpleNet
from .waveletmask_gated_flicker_arch import WaveletMaskGatedFlickerNet

__all__ = [
    'WaveletMaskBaseNet',
    'WaveletMaskLocalNet',
    'WaveletMaskFlickerNet',
    'WaveletMaskABNet',
    'WaveletMaskLocalFusionNet',
    'WaveletMaskFullNet',
    'WaveletMaskFlickerFusionNet',
    'WaveletMaskSimpleNet',
    'WaveletMaskGatedFlickerNet',
]
