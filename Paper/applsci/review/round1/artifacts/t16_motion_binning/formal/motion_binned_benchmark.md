# T6 Motion/Occlusion-Binned Benchmark

Motion descriptors are RAFT statistics on clean GT frames (camera proxy = median
flow vector; object proxy = residual after removing the median; occlusion =
forward-backward inconsistency). Restoration metrics join the per-clip values
already logged by the same-protocol seed-2021 RTN baseline and WaveletMask tests.
Clean-FP is the fraction of each detector's top-5% activation that falls outside
the LQ-GT residual proxy (lower is better).

- Clips: `30`
- Frames per clip: `7`
- Spatial size: `128`
- Flow type: `raft`
- Checkpoint: `projects/007-waveletmask/Exp/checkpoints/waveletmask_ab/net_g_best.pth`

## Binned Restoration Improvement (WaveletMask - RTN)

| Group | Bin | N | Motion | Occl. | dPSNR | dSSIM | dLPIPS | dDISTS |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| motion | low | 10 | 0.7665 | 0.0085 | 0.0854 | 0.0009 | 0.0115 | 0.0127 |
| motion | medium | 10 | 1.8715 | 0.0269 | 0.4769 | 0.0036 | 0.0172 | 0.0135 |
| motion | high | 10 | 3.9024 | 0.0528 | 0.7687 | 0.0056 | 0.0143 | 0.0103 |
| occlusion | low | 10 | 0.8522 | 0.0052 | 0.1863 | 0.0018 | 0.0119 | 0.0133 |
| occlusion | medium | 10 | 2.0196 | 0.0215 | 0.4979 | 0.0018 | 0.0139 | 0.0113 |
| occlusion | high | 10 | 3.6686 | 0.0615 | 0.6469 | 0.0065 | 0.0172 | 0.0119 |
| dominance | camera | 29 | 2.1274 | 0.0269 | 0.4332 | 0.0034 | 0.0147 | 0.0125 |
| dominance | object | 1 | 3.7099 | 0.1036 | 0.7481 | 0.0011 | 0.0042 | 0.0031 |

## Binned Detector Clean-FP (top-5% outside residual proxy)

| Group | Bin | N | Spatial | LHFM | GFM | Wavelet(max) |
|---|---|---:|---:|---:|---:|---:|
| motion | low | 10 | 0.4845 | 0.6462 | 0.4398 | 0.4398 |
| motion | medium | 10 | 0.6372 | 0.7819 | 0.6265 | 0.6265 |
| motion | high | 10 | 0.5690 | 0.6808 | 0.5839 | 0.5839 |
| occlusion | low | 10 | 0.5427 | 0.7161 | 0.5152 | 0.5152 |
| occlusion | medium | 10 | 0.5729 | 0.7032 | 0.5414 | 0.5414 |
| occlusion | high | 10 | 0.5750 | 0.6896 | 0.5937 | 0.5937 |
| dominance | camera | 29 | 0.5553 | 0.6963 | 0.5437 | 0.5437 |
| dominance | object | 1 | 0.8029 | 0.8966 | 0.7366 | 0.7366 |

## Clip-Level Spearman Correlations

| Pair | Spearman |
|---|---:|
| motion_mag vs delta_psnr | 0.5849 |
| occlusion_ratio vs delta_psnr | 0.4919 |
| motion_mag vs lhfm_top5_clean | -0.0389 |
| motion_mag vs gfm_top5_clean | 0.2966 |
| motion_mag vs spatial_top5_clean | 0.1097 |
| occlusion_ratio vs lhfm_top5_clean | 0.0866 |
| occlusion_ratio vs gfm_top5_clean | 0.3431 |

## Per-Clip Rows

| Clip | Motion | Camera | Object | Occl. | Bin | Dom. | dPSNR | LHFM FP | GFM FP |
|---|---:|---:|---:|---:|---|---|---:|---:|---:|
| 013 | 0.2367 | 0.1889 | 0.1034 | 0.0090 | low | camera | 0.2001 | 0.8849 | 0.6400 |
| 016 | 0.2627 | 0.2382 | 0.1279 | 0.0000 | low | camera | 0.1037 | 0.8137 | 0.5785 |
| 009 | 0.3258 | 0.3158 | 0.0970 | 0.0000 | low | camera | 0.3772 | 0.4810 | 0.3415 |
| 019 | 0.3944 | 0.3875 | 0.0623 | 0.0013 | low | camera | 0.4173 | 0.5054 | 0.1151 |
| 021 | 0.9142 | 0.8775 | 0.1892 | 0.0051 | low | camera | -0.3326 | 0.6985 | 0.4107 |
| 004 | 1.0023 | 0.8453 | 0.4397 | 0.0200 | low | camera | -0.4425 | 0.1863 | 0.0137 |
| 017 | 1.0073 | 0.8493 | 0.3616 | 0.0098 | low | camera | 0.1366 | 0.7727 | 0.4654 |
| 018 | 1.1446 | 0.9598 | 0.6819 | 0.0311 | low | camera | 0.5264 | 0.7932 | 0.7522 |
| 001 | 1.1643 | 1.0007 | 0.5083 | 0.0055 | low | camera | 0.1083 | 0.7122 | 0.5210 |
| 014 | 1.2128 | 1.0030 | 0.5468 | 0.0036 | low | camera | -0.2407 | 0.6146 | 0.5600 |
| 015 | 1.2256 | 1.2622 | 0.3343 | 0.0147 | medium | camera | 0.3514 | 0.4868 | 0.4459 |
| 006 | 1.3272 | 1.2974 | 0.2501 | 0.0100 | medium | camera | 0.7310 | 0.9776 | 0.8663 |
| 020 | 1.5642 | 1.5230 | 0.4855 | 0.0148 | medium | camera | 0.8134 | 0.8029 | 0.5502 |
| 005 | 1.6193 | 1.5184 | 0.5910 | 0.0180 | medium | camera | 0.6165 | 0.9200 | 0.7941 |
| 024 | 1.6770 | 1.5686 | 0.7099 | 0.0076 | medium | camera | 0.3617 | 0.7005 | 0.6537 |
| 027 | 1.9990 | 1.7654 | 1.0558 | 0.0301 | medium | camera | -0.2448 | 0.9307 | 0.7434 |
| 029 | 2.1140 | 2.0345 | 0.7363 | 0.0209 | medium | camera | 0.8018 | 0.9063 | 0.5454 |
| 010 | 2.3101 | 2.2631 | 0.8861 | 0.0497 | medium | camera | 0.3327 | 0.8273 | 0.4976 |
| 023 | 2.3964 | 1.8125 | 1.0338 | 0.0804 | medium | camera | 0.6728 | 0.6732 | 0.6693 |
| 007 | 2.4824 | 2.2043 | 0.9844 | 0.0230 | medium | camera | 0.3328 | 0.5932 | 0.4995 |
| 025 | 2.5491 | 2.0044 | 1.2455 | 0.0278 | high | camera | 1.2926 | 0.8000 | 0.7532 |
| 002 | 2.7164 | 2.5436 | 0.8932 | 0.0190 | high | camera | 0.8188 | 0.6946 | 0.6605 |
| 028 | 2.9238 | 2.8480 | 0.4222 | 0.0270 | high | camera | 0.6390 | 0.7112 | 0.4078 |
| 000 | 3.3337 | 3.1341 | 0.7412 | 0.0305 | high | camera | 0.3740 | 0.6010 | 0.6146 |
| 012 | 3.4035 | 3.2032 | 1.1812 | 0.0934 | high | camera | 1.3498 | 0.7463 | 0.6498 |
| 022 | 3.6103 | 3.4634 | 0.6093 | 0.0316 | high | camera | 0.8384 | 0.3200 | 0.0654 |
| 003 | 3.7099 | 1.4887 | 3.0517 | 0.1036 | high | object | 0.7481 | 0.8966 | 0.7366 |
| 008 | 3.8227 | 3.7381 | 0.7761 | 0.0410 | high | camera | 0.4169 | 0.6498 | 0.6361 |
| 011 | 5.0298 | 4.9950 | 0.8929 | 0.0570 | high | camera | 0.3647 | 0.6859 | 0.6878 |
| 026 | 7.9248 | 7.8945 | 1.0737 | 0.0972 | high | camera | 0.8449 | 0.7024 | 0.6273 |
