# T20 Cross-Domain Frequency Decomposition Ablation

Residual proxy labels are derived from paired `LQ-GT` frames; they are diagnostic labels, not generator-side artifact annotations.

- GT root: `/mnt/data/Antigravity_OFR/datasets/OldFilmRestoration/val/REDS/GT/val_sharp`
- LQ root: `/mnt/data/Antigravity_OFR/datasets/OldFilmRestoration/val/REDS/LQ/val_sharp_degraded`
- Clips: `30`
- Frames per clip: `20`
- Analysis size: `256`
- Metric size: `128`
- Residual threshold: `0.1`
- Top-k fraction: `0.05`

| Method | Family | Status | Fold | Proxy area | AUPRC | Best F1 | Top-5 hit | Top-5 FPR | Lift | Inside/outside | Note |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| gaussian_low | gaussian_pyramid | done |  | 0.2990 | 0.3244 | 0.4604 | 0.3696 | 0.6304 | 1.2361 | 1.1058 | deterministic frequency operator |
| gaussian_high | gaussian_pyramid | done |  | 0.2990 | 0.3163 | 0.4604 | 0.3695 | 0.6305 | 1.2358 | 1.0672 | deterministic frequency operator |
| gaussian_fused | gaussian_pyramid | done |  | 0.2990 | 0.3270 | 0.4604 | 0.3815 | 0.6185 | 1.2758 | 1.0853 | deterministic frequency operator |

## Qualitative Example Asset Directories

- `/mnt/data/Antigravity_OFR/projects/007-waveletmask/Paper/applsci/review/round1/artifacts/t20_cross_domain_frequency/formal/shards/shard_02/examples/example_01_026_00000026.png`
- `/mnt/data/Antigravity_OFR/projects/007-waveletmask/Paper/applsci/review/round1/artifacts/t20_cross_domain_frequency/formal/shards/shard_02/examples/example_02_022_00000006.png`
- `/mnt/data/Antigravity_OFR/projects/007-waveletmask/Paper/applsci/review/round1/artifacts/t20_cross_domain_frequency/formal/shards/shard_02/examples/example_03_026_00000098.png`
- `/mnt/data/Antigravity_OFR/projects/007-waveletmask/Paper/applsci/review/round1/artifacts/t20_cross_domain_frequency/formal/shards/shard_02/examples/example_04_005_00000092.png`
- `/mnt/data/Antigravity_OFR/projects/007-waveletmask/Paper/applsci/review/round1/artifacts/t20_cross_domain_frequency/formal/shards/shard_02/examples/example_05_022_00000011.png`
- `/mnt/data/Antigravity_OFR/projects/007-waveletmask/Paper/applsci/review/round1/artifacts/t20_cross_domain_frequency/formal/shards/shard_02/examples/example_06_011_00000041.png`

Interpretation rule: deterministic Haar support should be based on the fused row's balanced AUPRC, Top-5 hit, and Top-5 false-positive trade-off against Sobel/Laplacian, Gaussian pyramid, Fourier temporal, and learned-shallow rows.
