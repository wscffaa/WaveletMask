# T11 Cue Complementarity Probe

The RTN spatial indicator is computed with the same flow-aligned forward mask contract used by the baseline implementation.
Residual proxy hits use paired LQ-GT residuals and are diagnostic rather than generator-side artifact labels.

- Clips: `12`
- Frames per clip: `7`
- Spatial size: `128`
- Metric size: `64`
- Top-k fraction: `0.05`
- Flow type: `raft`
- Checkpoint: `projects/007-waveletmask/Exp/checkpoints/waveletmask_ab/net_g_best.pth`

## Pairwise Cue Relation

| Pair | N | Pearson | Spearman | Top-5 overlap | Top-5 Jaccard |
|---|---:|---:|---:|---:|---:|
| spatial/local | 60 | 0.0717 | -0.0418 | 0.1641 | 0.0938 |
| spatial/flicker | 60 | 0.3763 | 0.4850 | 0.2300 | 0.1408 |
| local/flicker | 60 | -0.1635 | -0.3364 | 0.0024 | 0.0012 |

## Residual-Proxy Top-5% Hit

| Cue | N | Proxy area | Top-5 hit | Lift | Union area | Union hit | Union lift |
|---|---:|---:|---:|---:|---:|---:|---:|
| spatial | 60 | 0.2887 | 0.4236 | 2.2915 |  |  |  |
| local | 60 | 0.2887 | 0.3146 | 1.1583 |  |  |  |
| flicker | 60 | 0.2887 | 0.4276 | 1.7982 |  |  |  |
| wavelet | 60 | 0.2887 | 0.4276 | 1.7982 |  |  |  |
| top5_union_spatial_local_flicker | 60 | 0.2887 |  |  | 0.1304 | 0.3792 | 1.6830 |

## Qualitative Example Asset Directories

- `projects/007-waveletmask/Paper/applsci/review/round1/artifacts/t11_cue_complementarity/formal/examples/000_01`
- `projects/007-waveletmask/Paper/applsci/review/round1/artifacts/t11_cue_complementarity/formal/examples/000_02`
- `projects/007-waveletmask/Paper/applsci/review/round1/artifacts/t11_cue_complementarity/formal/examples/000_03`
- `projects/007-waveletmask/Paper/applsci/review/round1/artifacts/t11_cue_complementarity/formal/examples/000_04`
- `projects/007-waveletmask/Paper/applsci/review/round1/artifacts/t11_cue_complementarity/formal/examples/000_05`
- `projects/007-waveletmask/Paper/applsci/review/round1/artifacts/t11_cue_complementarity/formal/examples/001_01`
