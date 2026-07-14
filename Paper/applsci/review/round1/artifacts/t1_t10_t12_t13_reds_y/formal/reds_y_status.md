# REDS Synthetic Y-Channel Submission Summary

Updated: 2026-06-08 CST

Protocol: `psnr.test_y_channel=true`, `ssim.test_y_channel=true`, `lpips.convert_to_grayscale=true`, `dists.convert_to_grayscale=true`.
This packaged summary retains only the metrics cited in the manuscript or response letter.

## Same-Protocol Main Comparison

| Variant | Seed | PSNR | SSIM | LPIPS | DISTS |
|---|---:|---:|---:|---:|---:|
| RTN baseline | 2021 | 26.1648 | 0.8923 | 0.1023 | 0.0946 |
| WaveletMask | 2021 | 26.6125 | 0.8954 | 0.0886 | 0.0827 |

## Three-Seed Robustness Check

| Variant | Seeds | Mean PSNR | Mean SSIM | Mean LPIPS | Mean DISTS |
|---|---|---:|---:|---:|---:|
| RTN baseline | 2021/2022/2023 | 24.5570 | 0.8808 | 0.1083 | 0.0922 |
| WaveletMask | 2021/2022/2023 | 26.2183 | 0.8983 | 0.0879 | 0.0831 |

## Follow-Up Ablations

| Variant | PSNR | SSIM | LPIPS | DISTS |
|---|---:|---:|---:|---:|
| Haar, two levels | 26.5175 | 0.9011 | 0.0847 | 0.0760 |
| Daubechies-2 | 17.9208 | 0.6473 | 0.4136 | 0.3154 |
| Symlet-2 | 24.2087 | 0.8509 | 0.1507 | 0.1226 |
| Attention fusion | 23.2479 | 0.8281 | 0.1768 | 0.1359 |
| Residual fusion | 25.9777 | 0.8929 | 0.0966 | 0.0811 |

## Flow-Compensated Pilot (Response-Side Exploratory Context Only)

All three flow-compensated seeds completed under the REDS Synthetic Y-channel protocol. These rows are retained as response-side exploratory material and are not used as decisive manuscript evidence.

| Variant | Seed | PSNR | SSIM | LPIPS | DISTS |
|---|---:|---:|---:|---:|---:|
| FC WaveletMask | 2021 | 23.5456 | 0.8511 | 0.1391 | 0.1061 |
| FC WaveletMask | 2022 | 22.2308 | 0.7970 | 0.2217 | 0.1731 |
| FC WaveletMask | 2023 | 26.2050 | 0.8954 | 0.0887 | 0.0782 |
| FC WaveletMask mean | 2021/2022/2023 | 23.9938 | 0.8478 | 0.1498 | 0.1191 |

### Local Evidence Index

| Variant | Seed | Log | Metrics | Test result |
|---|---:|---|---|---|
| FC WaveletMask | 2021 | `projects/007-waveletmask/Exp/evidence/reds_round1_y_logs/A100/test_round1_reds_y_t13_fc_ab_seed2021_20260608_103733.log` | `projects/007-waveletmask/Exp/evidence/reds_round1_y_logs/A100/metrics_round1_reds_y_t13_fc_ab_seed2021.yaml` | `projects/007-waveletmask/Exp/evidence/reds_round1_y_logs/A100/test_result_round1_reds_y_t13_fc_ab_seed2021.json` |
| FC WaveletMask | 2022 | `projects/007-waveletmask/Exp/evidence/reds_round1_y_logs/L40/test_round1_reds_y_t13_fc_ab_seed2022_20260607_203319.log` | `projects/007-waveletmask/Exp/evidence/reds_round1_y_logs/L40/metrics_round1_reds_y_t13_fc_ab_seed2022.yaml` | `projects/007-waveletmask/Exp/evidence/reds_round1_y_logs/L40/test_result_round1_reds_y_t13_fc_ab_seed2022.json` |
| FC WaveletMask | 2023 | `projects/007-waveletmask/Exp/evidence/reds_round1_y_logs/L40/test_round1_reds_y_t13_fc_ab_seed2023_20260607_203319.log` | `projects/007-waveletmask/Exp/evidence/reds_round1_y_logs/L40/metrics_round1_reds_y_t13_fc_ab_seed2023.yaml` | `projects/007-waveletmask/Exp/evidence/reds_round1_y_logs/L40/test_result_round1_reds_y_t13_fc_ab_seed2023.json` |
