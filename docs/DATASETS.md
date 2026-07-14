## Datasets

### REDS (Training)
- **Source**: [REDS Dataset](https://seungjunnah.github.io/Datasets/reds.html)
- **Usage**: Clean frames are degraded on-the-fly with the RTN synthetic pipeline.
- **Path**: `datasets/train/REDS/train_sharp`

### SRWOV (Evaluation)
- **Source**: [SRWOV Benchmark](https://github.com/antigravity-ofr/SRWOV)
- **Synthetic split**: 30 clips with paired ground truth (PSNR, SSIM, LPIPS, DISTS).
- **Real-world split**: 216 archival clips without ground truth (qualitative + no-reference metrics).
- **Path**: `datasets/test/SRWOV`

The repository does not redistribute REDS or SRWOV. Download datasets from their official sources and arrange them according to the paths expected in the training/testing configs.
