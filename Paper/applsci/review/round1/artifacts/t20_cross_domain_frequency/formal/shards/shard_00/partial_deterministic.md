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
| haar_low | haar_dwt | done |  | 0.2990 | 0.3194 | 0.4604 | 0.3755 | 0.6245 | 1.2557 | 1.0797 | deterministic frequency operator |
| haar_high | haar_dwt | done |  | 0.2990 | 0.3039 | 0.4606 | 0.3391 | 0.6609 | 1.1339 | 1.0191 | deterministic frequency operator |
| haar_fused | haar_dwt | done |  | 0.2990 | 0.3166 | 0.4604 | 0.3685 | 0.6315 | 1.2324 | 1.0517 | deterministic frequency operator |

## Qualitative Example Asset Directories

- `/mnt/data/Antigravity_OFR/projects/007-waveletmask/Paper/applsci/review/round1/artifacts/t20_cross_domain_frequency/formal/shards/shard_00/examples/example_01_026_00000026.png`
- `/mnt/data/Antigravity_OFR/projects/007-waveletmask/Paper/applsci/review/round1/artifacts/t20_cross_domain_frequency/formal/shards/shard_00/examples/example_02_022_00000006.png`
- `/mnt/data/Antigravity_OFR/projects/007-waveletmask/Paper/applsci/review/round1/artifacts/t20_cross_domain_frequency/formal/shards/shard_00/examples/example_03_026_00000098.png`
- `/mnt/data/Antigravity_OFR/projects/007-waveletmask/Paper/applsci/review/round1/artifacts/t20_cross_domain_frequency/formal/shards/shard_00/examples/example_04_005_00000092.png`
- `/mnt/data/Antigravity_OFR/projects/007-waveletmask/Paper/applsci/review/round1/artifacts/t20_cross_domain_frequency/formal/shards/shard_00/examples/example_05_022_00000011.png`
- `/mnt/data/Antigravity_OFR/projects/007-waveletmask/Paper/applsci/review/round1/artifacts/t20_cross_domain_frequency/formal/shards/shard_00/examples/example_06_011_00000041.png`

Interpretation rule: deterministic Haar support should be based on the fused row's balanced AUPRC, Top-5 hit, and Top-5 false-positive trade-off against Sobel/Laplacian, Gaussian pyramid, Fourier temporal, and learned-shallow rows.
