# T8 Mask Interpretability and Residual-Proxy Overlap

The SRWOV/REDS synthetic files used here do not expose generator-side pixel annotations.
Overlap is therefore measured against residual-derived `LQ-GT` proxy regions, not true artifact masks.

- Clips: `30`
- Frames per clip: `20`
- Analysis size: `256`
- Metric size: `128`
- Residual threshold: `0.1`
- Flicker low-frequency threshold: `0.03`
- Top-k fraction: `0.05`
- Checkpoint: `projects/007-waveletmask/Exp/checkpoints/waveletmask_ab/net_g_best.pth`

| Branch | Proxy area | AUPRC | Best F1 | Best IoU | Precision | Recall | Inside/outside | Top-5% hit | Random hit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| local | 0.0820 | 0.0552 | 0.1515 | 0.0820 | 0.0820 | 1.0000 | 0.6103 | 0.0412 | 0.0820 |
| flicker | 0.2858 | 0.4888 | 0.4949 | 0.3288 | 0.4087 | 0.6271 | 3.5714 | 0.6854 | 0.2858 |
| wavelet | 0.2907 | 0.4943 | 0.4996 | 0.3330 | 0.4128 | 0.6328 | 3.5638 | 0.6908 | 0.2907 |

## Qualitative Example Asset Directories

- `projects/007-waveletmask/Paper/applsci/review/round1/artifacts/t8_mask_interpretability/formal/examples/example_01_026_00000026`
- `projects/007-waveletmask/Paper/applsci/review/round1/artifacts/t8_mask_interpretability/formal/examples/example_02_022_00000006`
- `projects/007-waveletmask/Paper/applsci/review/round1/artifacts/t8_mask_interpretability/formal/examples/example_03_026_00000098`
- `projects/007-waveletmask/Paper/applsci/review/round1/artifacts/t8_mask_interpretability/formal/examples/example_04_005_00000092`
- `projects/007-waveletmask/Paper/applsci/review/round1/artifacts/t8_mask_interpretability/formal/examples/example_05_011_00000041`
- `projects/007-waveletmask/Paper/applsci/review/round1/artifacts/t8_mask_interpretability/formal/examples/example_06_022_00000011`

AUPRC and best-F1 thresholds are diagnostic because the proxy labels are derived from residuals and were not used for training.
