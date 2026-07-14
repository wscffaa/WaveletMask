# Paired Clip Statistics (Seed 2021)

Primary same-protocol paired comparison between the re-trained RTN-style baseline and WaveletMask.

- Baseline log: `projects/007-waveletmask/Exp/evidence/reds_round1_y_logs/L40/test_round1_reds_y_t1_base_seed2021_20260607_161215.log`
- WaveletMask log: `projects/007-waveletmask/Exp/results/round1_reds_y_t1_ab_seed2021/test_round1_reds_y_t1_ab_seed2021_20260607_155022.log`
- Number of paired clips: `30`
- Bootstrap draws: `20000`
- Permutation draws: `20000`

| Metric | Baseline mean | WaveletMask mean | Mean paired improvement | 95% bootstrap CI | Two-sided permutation p |
|---|---:|---:|---:|---:|---:|
| PSNR | 26.1648 | 26.6125 | 0.4477 | [0.2955, 0.5981] | 5e-05 |
| SSIM | 0.8923 | 0.8954 | 0.0031 | [0.0014, 0.0050] | 0.0015 |
| LPIPS | 0.1023 | 0.0886 | 0.0137 | [0.0108, 0.0166] | 5e-05 |
| DISTS | 0.0946 | 0.0827 | 0.0119 | [0.0103, 0.0136] | 5e-05 |

Improvement is defined so that positive values always favor WaveletMask:

- `PSNR`, `SSIM`: `WaveletMask - RTN`
- `LPIPS`, `DISTS`: `RTN - WaveletMask`

The permutation test uses random sign flips of the paired clip differences.
