# T20 Cross-Domain Frequency Decomposition Ablation

Residual proxy labels are derived from paired `LQ-GT` frames; they are diagnostic labels, not generator-side artifact annotations.

- GT root: `/Users/caifeifan/VibeScholar/Antigravity_OFR/projects/007-waveletmask/datasets/OldFilmRestoration/val/REDS/GT/val_sharp`
- LQ root: `/Users/caifeifan/VibeScholar/Antigravity_OFR/projects/007-waveletmask/datasets/OldFilmRestoration/val/REDS/LQ/val_sharp_degraded`
- Clips: `30`
- Frames per clip: `20`
- Analysis size: `256`
- Metric size: `128`
- Residual threshold: `0.1`
- Top-k fraction: `0.05`

| Method | Family | Status | Fold | Proxy area | AUPRC | Top-5 hit | Top-5 FPR | Lift | Inside/outside | Note |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| haar_low | haar_dwt | done |  | 0.2990 | 0.3194 | 0.3755 | 0.6245 | 1.2557 | 1.0797 | deterministic frequency operator |
| haar_high | haar_dwt | done |  | 0.2990 | 0.3039 | 0.3391 | 0.6609 | 1.1339 | 1.0191 | deterministic frequency operator |
| haar_fused | haar_dwt | done |  | 0.2990 | 0.3166 | 0.3685 | 0.6315 | 1.2324 | 1.0517 | deterministic frequency operator |
| sobel | gradient_edge | done |  | 0.2990 | 0.3062 | 0.3470 | 0.6530 | 1.1603 | 1.0373 | deterministic frequency operator |
| laplacian | gradient_edge | done |  | 0.2990 | 0.2993 | 0.3352 | 0.6648 | 1.1211 | 1.0030 | deterministic frequency operator |
| gaussian_low | gaussian_pyramid | done |  | 0.2990 | 0.3244 | 0.3696 | 0.6304 | 1.2361 | 1.1058 | deterministic frequency operator |
| gaussian_high | gaussian_pyramid | done |  | 0.2990 | 0.3163 | 0.3695 | 0.6305 | 1.2358 | 1.0672 | deterministic frequency operator |
| gaussian_fused | gaussian_pyramid | done |  | 0.2990 | 0.3270 | 0.3815 | 0.6185 | 1.2758 | 1.0853 | deterministic frequency operator |
| fourier_temporal_low | fourier_temporal | done |  | 0.2990 | 0.4019 | 0.4541 | 0.5459 | 1.5187 | 1.3399 | deterministic frequency operator |
| fourier_temporal_high | fourier_temporal | done |  | 0.2990 | 0.3091 | 0.3402 | 0.6598 | 1.1376 | 1.0350 | deterministic frequency operator |
| fourier_temporal_fused | fourier_temporal | done |  | 0.2990 | 0.3885 | 0.4384 | 0.5616 | 1.4660 | 1.2649 | deterministic frequency operator |
| learned_shallow_frequency | learned_shallow | done | 0 | 0.2915 | 0.4401 | 0.5210 | 0.4790 | 1.7873 | 1.0323 | shallow conv frequency probe; val_clips=6; final_loss=0.989051 |
| learned_shallow_frequency | learned_shallow | done | 1 | 0.2678 | 0.4221 | 0.4915 | 0.5085 | 1.8353 | 1.0349 | shallow conv frequency probe; val_clips=6; final_loss=0.982011 |
| learned_shallow_frequency | learned_shallow | done | 2 | 0.3487 | 0.3952 | 0.5262 | 0.4738 | 1.5090 | 1.0077 | shallow conv frequency probe; val_clips=6; final_loss=1.062051 |
| learned_shallow_frequency | learned_shallow | done | 3 | 0.2423 | 0.4090 | 0.5539 | 0.4461 | 2.2856 | 1.0337 | shallow conv frequency probe; val_clips=6; final_loss=0.942357 |
| learned_shallow_frequency | learned_shallow | done | 4 | 0.3448 | 0.4660 | 0.5466 | 0.4534 | 1.5855 | 1.0331 | shallow conv frequency probe; val_clips=6; final_loss=0.985112 |

## Qualitative Example Asset Directories

- `/mnt/data/Antigravity_OFR/projects/007-waveletmask/Paper/applsci/review/round1/artifacts/t20_cross_domain_frequency/formal/shards/shard_00/examples/example_01_026_00000026.png`
- `/mnt/data/Antigravity_OFR/projects/007-waveletmask/Paper/applsci/review/round1/artifacts/t20_cross_domain_frequency/formal/shards/shard_00/examples/example_02_022_00000006.png`
- `/mnt/data/Antigravity_OFR/projects/007-waveletmask/Paper/applsci/review/round1/artifacts/t20_cross_domain_frequency/formal/shards/shard_00/examples/example_03_026_00000098.png`
- `/mnt/data/Antigravity_OFR/projects/007-waveletmask/Paper/applsci/review/round1/artifacts/t20_cross_domain_frequency/formal/shards/shard_00/examples/example_04_005_00000092.png`
- `/mnt/data/Antigravity_OFR/projects/007-waveletmask/Paper/applsci/review/round1/artifacts/t20_cross_domain_frequency/formal/shards/shard_00/examples/example_05_022_00000011.png`
- `/mnt/data/Antigravity_OFR/projects/007-waveletmask/Paper/applsci/review/round1/artifacts/t20_cross_domain_frequency/formal/shards/shard_00/examples/example_06_011_00000041.png`
- `/mnt/data/Antigravity_OFR/projects/007-waveletmask/Paper/applsci/review/round1/artifacts/t20_cross_domain_frequency/formal/shards/shard_01/examples/example_01_026_00000026.png`
- `/mnt/data/Antigravity_OFR/projects/007-waveletmask/Paper/applsci/review/round1/artifacts/t20_cross_domain_frequency/formal/shards/shard_01/examples/example_02_022_00000006.png`
- `/mnt/data/Antigravity_OFR/projects/007-waveletmask/Paper/applsci/review/round1/artifacts/t20_cross_domain_frequency/formal/shards/shard_01/examples/example_03_026_00000098.png`
- `/mnt/data/Antigravity_OFR/projects/007-waveletmask/Paper/applsci/review/round1/artifacts/t20_cross_domain_frequency/formal/shards/shard_01/examples/example_04_005_00000092.png`
- `/mnt/data/Antigravity_OFR/projects/007-waveletmask/Paper/applsci/review/round1/artifacts/t20_cross_domain_frequency/formal/shards/shard_01/examples/example_05_022_00000011.png`
- `/mnt/data/Antigravity_OFR/projects/007-waveletmask/Paper/applsci/review/round1/artifacts/t20_cross_domain_frequency/formal/shards/shard_01/examples/example_06_011_00000041.png`
- `/mnt/data/Antigravity_OFR/projects/007-waveletmask/Paper/applsci/review/round1/artifacts/t20_cross_domain_frequency/formal/shards/shard_02/examples/example_01_026_00000026.png`
- `/mnt/data/Antigravity_OFR/projects/007-waveletmask/Paper/applsci/review/round1/artifacts/t20_cross_domain_frequency/formal/shards/shard_02/examples/example_02_022_00000006.png`
- `/mnt/data/Antigravity_OFR/projects/007-waveletmask/Paper/applsci/review/round1/artifacts/t20_cross_domain_frequency/formal/shards/shard_02/examples/example_03_026_00000098.png`
- `/mnt/data/Antigravity_OFR/projects/007-waveletmask/Paper/applsci/review/round1/artifacts/t20_cross_domain_frequency/formal/shards/shard_02/examples/example_04_005_00000092.png`
- `/mnt/data/Antigravity_OFR/projects/007-waveletmask/Paper/applsci/review/round1/artifacts/t20_cross_domain_frequency/formal/shards/shard_02/examples/example_05_022_00000011.png`
- `/mnt/data/Antigravity_OFR/projects/007-waveletmask/Paper/applsci/review/round1/artifacts/t20_cross_domain_frequency/formal/shards/shard_02/examples/example_06_011_00000041.png`
- `/mnt/data/Antigravity_OFR/projects/007-waveletmask/Paper/applsci/review/round1/artifacts/t20_cross_domain_frequency/formal/shards/shard_03/examples/example_01_026_00000026.png`
- `/mnt/data/Antigravity_OFR/projects/007-waveletmask/Paper/applsci/review/round1/artifacts/t20_cross_domain_frequency/formal/shards/shard_03/examples/example_02_022_00000006.png`
- `/mnt/data/Antigravity_OFR/projects/007-waveletmask/Paper/applsci/review/round1/artifacts/t20_cross_domain_frequency/formal/shards/shard_03/examples/example_03_026_00000098.png`
- `/mnt/data/Antigravity_OFR/projects/007-waveletmask/Paper/applsci/review/round1/artifacts/t20_cross_domain_frequency/formal/shards/shard_03/examples/example_04_005_00000092.png`
- `/mnt/data/Antigravity_OFR/projects/007-waveletmask/Paper/applsci/review/round1/artifacts/t20_cross_domain_frequency/formal/shards/shard_03/examples/example_05_022_00000011.png`
- `/mnt/data/Antigravity_OFR/projects/007-waveletmask/Paper/applsci/review/round1/artifacts/t20_cross_domain_frequency/formal/shards/shard_03/examples/example_06_011_00000041.png`

AUPRC is the primary ranking metric because several deterministic operators reach their best F1 only at a near-all-positive threshold; raw best-F1 values remain in the CSV/JSON for audit.
Interpretation rule: deterministic Haar support should be based on the fused row's balanced AUPRC, Top-5 hit, and Top-5 false-positive trade-off against Sobel/Laplacian, Gaussian pyramid, Fourier temporal, and learned-shallow rows.
