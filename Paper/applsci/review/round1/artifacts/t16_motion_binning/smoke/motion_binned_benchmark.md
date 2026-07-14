# T6 Motion/Occlusion-Binned Benchmark

Motion descriptors are RAFT statistics on clean GT frames (camera proxy = median
flow vector; object proxy = residual after removing the median; occlusion =
forward-backward inconsistency). Restoration metrics join the per-clip values
already logged by the same-protocol seed-2021 RTN baseline and WaveletMask tests.
Clean-FP is the fraction of each detector's top-5% activation that falls outside
the LQ-GT residual proxy (lower is better).

- Clips: `6`
- Frames per clip: `7`
- Spatial size: `128`
- Flow type: `raft`
- Checkpoint: `projects/007-waveletmask/Exp/checkpoints/waveletmask_ab/net_g_best.pth`

## Binned Restoration Improvement (WaveletMask - RTN)

| Group | Bin | N | Motion | Occl. | dPSNR | dSSIM | dLPIPS | dDISTS |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| motion | low | 2 | 1.0833 | 0.0127 | -0.1671 | -0.0034 | 0.0099 | 0.0084 |
| motion | medium | 2 | 2.1679 | 0.0185 | 0.7176 | 0.0034 | 0.0108 | 0.0096 |
| motion | high | 2 | 3.5218 | 0.0671 | 0.5611 | 0.0020 | 0.0095 | 0.0063 |
| occlusion | low | 2 | 1.3918 | 0.0117 | 0.3624 | 0.0022 | 0.0146 | 0.0112 |
| occlusion | medium | 2 | 1.8594 | 0.0195 | 0.1882 | -0.0021 | 0.0062 | 0.0067 |
| occlusion | high | 2 | 3.5218 | 0.0671 | 0.5611 | 0.0020 | 0.0095 | 0.0063 |
| dominance | camera | 5 | 1.9672 | 0.0186 | 0.2950 | 0.0006 | 0.0113 | 0.0091 |
| dominance | object | 1 | 3.7099 | 0.1036 | 0.7481 | 0.0011 | 0.0042 | 0.0031 |

## Binned Detector Clean-FP (top-5% outside residual proxy)

| Group | Bin | N | Spatial | LHFM | GFM | Wavelet(max) |
|---|---|---:|---:|---:|---:|---:|
| motion | low | 2 | 0.3112 | 0.4493 | 0.2673 | 0.2673 |
| motion | medium | 2 | 0.7137 | 0.8073 | 0.7273 | 0.7273 |
| motion | high | 2 | 0.6429 | 0.7488 | 0.6756 | 0.6756 |
| occlusion | low | 2 | 0.7010 | 0.8161 | 0.6576 | 0.6576 |
| occlusion | medium | 2 | 0.3239 | 0.4405 | 0.3371 | 0.3371 |
| occlusion | high | 2 | 0.6429 | 0.7488 | 0.6756 | 0.6756 |
| dominance | camera | 5 | 0.5065 | 0.6228 | 0.5208 | 0.5208 |
| dominance | object | 1 | 0.8029 | 0.8966 | 0.7366 | 0.7366 |

## Clip-Level Spearman Correlations

| Pair | Spearman |
|---|---:|
| motion_mag vs delta_psnr | 0.7143 |
| occlusion_ratio vs delta_psnr | 0.2000 |
| motion_mag vs lhfm_top5_clean | 0.3143 |
| motion_mag vs gfm_top5_clean | 0.6000 |
| motion_mag vs spatial_top5_clean | 0.4286 |
| occlusion_ratio vs lhfm_top5_clean | -0.2571 |
| occlusion_ratio vs gfm_top5_clean | 0.0857 |

## Per-Clip Rows

| Clip | Motion | Camera | Object | Occl. | Bin | Dom. | dPSNR | LHFM FP | GFM FP |
|---|---:|---:|---:|---:|---|---|---:|---:|---:|
| 004 | 1.0023 | 0.8453 | 0.4397 | 0.0200 | low | camera | -0.4425 | 0.1863 | 0.0137 |
| 001 | 1.1643 | 1.0007 | 0.5083 | 0.0055 | low | camera | 0.1083 | 0.7122 | 0.5210 |
| 005 | 1.6193 | 1.5184 | 0.5910 | 0.0180 | medium | camera | 0.6165 | 0.9200 | 0.7941 |
| 002 | 2.7164 | 2.5436 | 0.8932 | 0.0190 | medium | camera | 0.8188 | 0.6946 | 0.6605 |
| 000 | 3.3337 | 3.1341 | 0.7412 | 0.0305 | high | camera | 0.3740 | 0.6010 | 0.6146 |
| 003 | 3.7099 | 1.4887 | 3.0517 | 0.1036 | high | object | 0.7481 | 0.8966 | 0.7366 |
