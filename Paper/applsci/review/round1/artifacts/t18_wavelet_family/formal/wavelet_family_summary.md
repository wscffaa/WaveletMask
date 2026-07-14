# T18 Wavelet-Family Summary

This artifact isolates the wavelet-family evidence already available in the current review workspace.

## Reference

- Main WaveletMask seed-2021 metrics: `PSNR 26.6100`, `SSIM 0.8954`, `LPIPS 0.0886`, `DISTS 0.0827`.

## Tested Follow-Up Variants

| Variant | PSNR | SSIM | LPIPS | DISTS | Delta PSNR | Delta SSIM | Delta LPIPS | Delta DISTS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| haar_2level | 26.5200 | 0.9011 | 0.0847 | 0.0760 | -0.0900 | 0.0057 | 0.0039 | 0.0067 |
| db2 | 17.9200 | 0.6473 | 0.4136 | 0.3154 | -8.6900 | -0.2481 | -0.3250 | -0.2327 |
| sym2 | 24.2100 | 0.8509 | 0.1507 | 0.1226 | -2.4000 | -0.0445 | -0.0621 | -0.0399 |
| coif1 | 25.8533 | 0.8874 | 0.1026 | 0.0893 | -0.7567 | -0.0080 | -0.0140 | -0.0066 |
| dtcwt | 26.4506 | 0.8992 | 0.0862 | 0.0780 | -0.1594 | 0.0038 | 0.0024 | 0.0047 |

## Missing Reviewer-Requested Variants

- None. All currently tracked reviewer-requested wavelet variants have local metrics.

## Decision

- All currently tracked wavelet-family reviewer variants now have local metrics; promote status only after manuscript/response text is updated from these results.

