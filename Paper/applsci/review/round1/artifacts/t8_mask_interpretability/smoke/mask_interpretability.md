# T8 Mask Interpretability and Residual-Proxy Overlap

The SRWOV/REDS synthetic files used here do not expose generator-side pixel annotations.
Overlap is therefore measured against residual-derived `LQ-GT` proxy regions, not true artifact masks.

- Clips: `2`
- Frames per clip: `3`
- Analysis size: `128`
- Metric size: `64`
- Residual threshold: `0.1`
- Flicker low-frequency threshold: `0.03`
- Top-k fraction: `0.05`
- Checkpoint: `projects/007-waveletmask/Exp/checkpoints/waveletmask_ab/net_g_best.pth`

| Branch | Proxy area | AUPRC | Best F1 | Best IoU | Precision | Recall | Inside/outside | Top-5% hit | Random hit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| local | 0.0660 | 0.0490 | 0.1240 | 0.0661 | 0.0661 | 1.0000 | 0.6778 | 0.0504 | 0.0660 |
| flicker | 0.2145 | 0.2815 | 0.3863 | 0.2394 | 0.2495 | 0.8552 | 1.3737 | 0.3417 | 0.2145 |
| wavelet | 0.2166 | 0.2836 | 0.3892 | 0.2416 | 0.2520 | 0.8552 | 1.3723 | 0.3426 | 0.2166 |

## Qualitative Examples

- `projects/007-waveletmask/Paper/applsci/review/round1/artifacts/t8_mask_interpretability/smoke/examples/example_01_001_00000001.png`
- `projects/007-waveletmask/Paper/applsci/review/round1/artifacts/t8_mask_interpretability/smoke/examples/example_02_000_00000049.png`

AUPRC and best-F1 thresholds are diagnostic because the proxy labels are derived from residuals and were not used for training.
