# T20 Cross-Domain Frequency Decomposition Ablation

Residual proxy labels are derived from paired `LQ-GT` frames; they are diagnostic labels, not generator-side artifact annotations.

- GT root: `/mnt/data/Antigravity_OFR/datasets/OldFilmRestoration/val/REDS/GT/val_sharp`
- LQ root: `/mnt/data/Antigravity_OFR/datasets/OldFilmRestoration/val/REDS/LQ/val_sharp_degraded`
- Clips: `2`
- Frames per clip: `4`
- Analysis size: `128`
- Metric size: `64`
- Residual threshold: `0.1`
- Top-k fraction: `0.05`

| Method | Family | Status | Fold | Proxy area | AUPRC | Best F1 | Top-5 hit | Top-5 FPR | Lift | Inside/outside | Note |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| haar_low | haar_dwt | done |  | 0.3155 | 0.3512 | 0.4799 | 0.3834 | 0.6166 | 1.2151 | 1.1595 | deterministic frequency operator |
| haar_high | haar_dwt | done |  | 0.3155 | 0.3438 | 0.4804 | 0.3822 | 0.6178 | 1.2112 | 1.1399 | deterministic frequency operator |
| haar_fused | haar_dwt | done |  | 0.3155 | 0.3524 | 0.4831 | 0.3785 | 0.6215 | 1.1996 | 1.1558 | deterministic frequency operator |
| sobel | gradient_edge | done |  | 0.3155 | 0.3448 | 0.4815 | 0.3730 | 0.6270 | 1.1822 | 1.1412 | deterministic frequency operator |
| laplacian | gradient_edge | done |  | 0.3155 | 0.3417 | 0.4816 | 0.3852 | 0.6148 | 1.2209 | 1.1374 | deterministic frequency operator |
| gaussian_low | gaussian_pyramid | done |  | 0.3155 | 0.3759 | 0.4820 | 0.4060 | 0.5940 | 1.2867 | 1.2631 | deterministic frequency operator |
| gaussian_high | gaussian_pyramid | done |  | 0.3155 | 0.3494 | 0.4797 | 0.3840 | 0.6160 | 1.2170 | 1.1509 | deterministic frequency operator |
| gaussian_fused | gaussian_pyramid | done |  | 0.3155 | 0.3774 | 0.4872 | 0.4035 | 0.5965 | 1.2790 | 1.2261 | deterministic frequency operator |
| fourier_temporal_low | fourier_temporal | done |  | 0.3155 | 0.3742 | 0.5409 | 0.4072 | 0.5928 | 1.2906 | 1.2294 | deterministic frequency operator |
| fourier_temporal_high | fourier_temporal | done |  | 0.3155 | 0.3609 | 0.4842 | 0.3834 | 0.6166 | 1.2151 | 1.1831 | deterministic frequency operator |
| fourier_temporal_fused | fourier_temporal | done |  | 0.3155 | 0.3766 | 0.5377 | 0.4237 | 0.5763 | 1.3428 | 1.2147 | deterministic frequency operator |

## Qualitative Example Asset Directories

- `/mnt/data/Antigravity_OFR/projects/007-waveletmask/Paper/applsci/review/round1/artifacts/t20_cross_domain_frequency/smoke/examples/example_01_001_00000001.png`
- `/mnt/data/Antigravity_OFR/projects/007-waveletmask/Paper/applsci/review/round1/artifacts/t20_cross_domain_frequency/smoke/examples/example_02_000_00000065.png`
- `/mnt/data/Antigravity_OFR/projects/007-waveletmask/Paper/applsci/review/round1/artifacts/t20_cross_domain_frequency/smoke/examples/example_03_001_00000033.png`
- `/mnt/data/Antigravity_OFR/projects/007-waveletmask/Paper/applsci/review/round1/artifacts/t20_cross_domain_frequency/smoke/examples/example_04_000_00000033.png`
- `/mnt/data/Antigravity_OFR/projects/007-waveletmask/Paper/applsci/review/round1/artifacts/t20_cross_domain_frequency/smoke/examples/example_05_000_00000098.png`
- `/mnt/data/Antigravity_OFR/projects/007-waveletmask/Paper/applsci/review/round1/artifacts/t20_cross_domain_frequency/smoke/examples/example_06_001_00000065.png`

Interpretation rule: deterministic Haar support should be based on the fused row's balanced AUPRC, Top-5 hit, and Top-5 false-positive trade-off against Sobel/Laplacian, Gaussian pyramid, Fourier temporal, and learned-shallow rows.
