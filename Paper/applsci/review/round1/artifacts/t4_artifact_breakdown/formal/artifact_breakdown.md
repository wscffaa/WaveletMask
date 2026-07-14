# T4 Artifact-Dominant Breakdown

Clip strata are derived from paired `LQ-GT` residual descriptors on the fixed SRWOV/REDS synthetic split.
They are not pixel-level generator labels.

- Frames sampled per clip: `20`
- Residual threshold: `0.1`
- Feature max side: `360`
- Base log: `projects/007-waveletmask/Exp/results/waveletmask_base_reds_synthetic_best/test_waveletmask_base_reds_synthetic_best_20260415_174118.log`
- WaveletMask log: `projects/007-waveletmask/Exp/results/waveletmask_ab_reds_synthetic_best/test_waveletmask_ab_reds_synthetic_best_20260415_191427.log`

| Stratum | Clips | RTN PSNR | WaveletMask PSNR | Delta PSNR | Delta SSIM | Delta LPIPS | Delta DISTS |
|---|---:|---:|---:|---:|---:|---:|---:|
| scratch | 5 | 27.47 | 28.11 | 0.64 | 0.0020 | 0.0110 | 0.0116 |
| dust | 8 | 26.10 | 26.55 | 0.45 | 0.0049 | 0.0196 | 0.0120 |
| flicker | 6 | 23.90 | 24.37 | 0.47 | 0.0055 | 0.0176 | 0.0153 |
| mixed | 11 | 26.83 | 27.17 | 0.34 | 0.0017 | 0.0102 | 0.0109 |

Positive deltas mean WaveletMask improves over the RTN baseline; for LPIPS/DISTS this is reported as the reduction in error.
