# T20 Cross-Domain Frequency Decomposition Ablation

Residual proxy labels are derived from paired `LQ-GT` frames; they are diagnostic labels, not generator-side artifact annotations.

- GT root: `/data/cff/Code/Antigravity_OFR/datasets/OldFilmRestoration/val/REDS/GT/val_sharp`
- LQ root: `/data/cff/Code/Antigravity_OFR/datasets/OldFilmRestoration/val/REDS/LQ/val_sharp_degraded`
- Clips: `30`
- Frames per clip: `20`
- Analysis size: `256`
- Metric size: `128`
- Residual threshold: `0.1`
- Top-k fraction: `0.05`

| Method | Family | Status | Fold | Proxy area | AUPRC | Best F1 | Top-5 hit | Top-5 FPR | Lift | Inside/outside | Note |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| learned_shallow_frequency | learned_shallow | done | 2 | 0.3487 | 0.3952 | 0.5171 | 0.5262 | 0.4738 | 1.5090 | 1.0077 | shallow conv frequency probe; val_clips=6; final_loss=1.062051 |

Interpretation rule: deterministic Haar support should be based on the fused row's balanced AUPRC, Top-5 hit, and Top-5 false-positive trade-off against Sobel/Laplacian, Gaussian pyramid, Fourier temporal, and learned-shallow rows.
