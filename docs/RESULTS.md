## Results

Full metric tables with source attribution are available in the paper. Key SRWOV synthetic benchmark results:

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | DISTS↓ | Source |
|--------|-------|-------|--------|--------|--------|
| ShiftNet | 21.48 | 0.8234 | 0.2030 | 0.1303 | Reported |
| VRT | 22.39 | 0.8631 | 0.1750 | 0.1087 | Reported |
| RVRT | 22.43 | 0.8710 | 0.1655 | 0.1078 | Reported |
| RRTN | 24.15 | 0.9006 | 0.1082 | 0.0668 | Reported |
| DeepRemaster | 25.06 | 0.9185 | 0.1108 | 0.0895 | Reported |
| RTN | 26.16 | 0.8917 | 0.1034 | 0.0953 | Ours |
| MambaOFR | 25.99 | 0.9059 | 0.0815 | 0.0822 | Reported |
| **WaveletMask** | **26.60** | 0.8951 | 0.0891 | 0.0831 | Ours |

The "Ours" label indicates re-training under the unified WaveletMask protocol; "Reported" indicates values copied from SRWOV/MambaOFR benchmark tables or original papers.
