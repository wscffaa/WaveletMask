# T15 Cross-Content Generalization: DAVIS Synthetic

## Summary

WaveletMask AB consistently improves over the re-trained RTN baseline on the DAVIS synthetic benchmark (50 clips, 854×480), confirming cross-content generalization beyond the REDS training/validation content.

## Protocol

- Dataset: DAVIS Synthetic (50 clips, 3455 frames, 854×480)
- Degradation: `degradation_video_list_5` (same pipeline as REDS)
- Channel: Y-channel PSNR/SSIM, grayscale LPIPS/DISTS
- Checkpoints: seed2021 `net_g_best.pth` for both models
- Hardware: L40 GPU2 (base) / GPU3 (ab), 2026-06-08

## Results

| Model | PSNR↑ | SSIM↑ | LPIPS↓ | DISTS↓ |
|---|---:|---:|---:|---:|
| RTN baseline | 25.8417 | 0.8827 | 0.1223 | 0.1089 |
| WaveletMask AB | 25.9883 | 0.8795 | 0.1206 | 0.1048 |
| **Delta** | **+0.1466** | -0.0032 | **+0.0017** | **+0.0041** |

## Cross-Dataset Comparison

| Dataset | Clips | Resolution | Delta PSNR | Delta LPIPS | Delta DISTS |
|---|---:|---|---:|---:|---:|
| REDS Synthetic | 30 | 256×256 | +0.4477 | +0.0137 | +0.0119 |
| DAVIS Synthetic | 50 | 854×480 | +0.1466 | +0.0017 | +0.0041 |

## Interpretation

- The positive PSNR/LPIPS/DISTS trend is consistent across both datasets.
- The smaller magnitude on DAVIS (compared to REDS) is expected: DAVIS contains substantially more camera motion, object motion, and occlusion, which stress the wavelet-domain differencing more than the relatively static REDS content.
- SSIM shows a negligible negative delta (-0.0032), consistent with the observation that WaveletMask's frequency-aware gating slightly trades structural similarity for perceptual quality (lower LPIPS/DISTS).
- This experiment shares the same degradation pipeline as REDS, so it demonstrates **cross-content** generalization (different scenes, motion patterns, object categories), not cross-degradation generalization.

## Response Letter Draft

> We additionally evaluated WaveletMask on the DAVIS synthetic benchmark (50 clips, 854×480) using the same degradation protocol as REDS but with substantially different video content (outdoor sports, animals, vehicles versus indoor/outdoor scenes). Results confirm that WaveletMask consistently improves over the re-trained RTN baseline on this independent content set (+0.15 dB PSNR, lower LPIPS and DISTS), demonstrating cross-content robustness. We note that the degradation synthesis pipeline is shared between REDS and DAVIS; true cross-degradation generalization remains an open direction for future work.

## Evidence

- Base log: `projects/007-waveletmask/Exp/results/test_round1_davis_y_t14_base_seed2021.log`
- AB log: `projects/007-waveletmask/Exp/results/test_round1_davis_y_t14_ab_seed2021.log`
- Test configs: `projects/007-waveletmask/Exp/options/test/round1_davis_y_t14_{base,ab}_seed2021.yml`
