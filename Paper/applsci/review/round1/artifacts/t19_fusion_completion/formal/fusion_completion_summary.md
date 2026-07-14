# T19 Fusion-Completion Summary

This artifact isolates the fusion-ablation evidence already available in the current review workspace.

## Reference

- Main max-fusion seed-2021 metrics: `PSNR 26.6100`, `SSIM 0.8954`, `LPIPS 0.0886`, `DISTS 0.0827`.

## Tested Follow-Up Variants

| Variant | PSNR | SSIM | LPIPS | DISTS | Delta PSNR | Delta SSIM | Delta LPIPS | Delta DISTS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| channel_attention | 23.2500 | 0.8281 | 0.1768 | 0.1359 | -3.3600 | -0.0673 | -0.0882 | -0.0532 |
| residual | 25.9800 | 0.8929 | 0.0966 | 0.0811 | -0.6300 | -0.0025 | -0.0080 | 0.0016 |
| bias0 | 26.5013 | 0.8981 | 0.0857 | 0.0781 | -0.1087 | 0.0027 | 0.0029 | 0.0046 |
| bias2 | 23.8376 | 0.8420 | 0.1598 | 0.1246 | -2.7724 | -0.0534 | -0.0712 | -0.0419 |
| scalar_gate | 25.2362 | 0.8911 | 0.0900 | 0.0795 | -1.3738 | -0.0043 | -0.0014 | 0.0032 |
| normalization_aware | 25.0074 | 0.8898 | 0.0898 | 0.0798 | -1.6026 | -0.0056 | -0.0012 | 0.0029 |

## Missing Stronger Variants

- None. All currently tracked stronger fusion variants have local metrics.

## Decision

- All currently tracked stronger fusion variants now have local metrics; promote status only after manuscript/response text is updated from these results.

