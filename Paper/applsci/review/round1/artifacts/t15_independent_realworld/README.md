# T15 Independent Real-world Validation → Cross-Content Generalization (DAVIS)

**Status**: Partially addressed (2026-06-09)

## Completed

DAVIS Synthetic 50-clip Y-channel paired comparison (same degradation pipeline, independent video content):

- `formal/davis_cross_content_generalization.md` — report + response letter draft
- `formal/davis_cross_content_generalization.json` — full per-clip data
- `formal/davis_cross_content_generalization.csv` — 50-clip paired table

## Result

| Model | PSNR↑ | SSIM↑ | LPIPS↓ | DISTS↓ |
|---|---:|---:|---:|---:|
| RTN baseline | 25.8417 | 0.8827 | 0.1223 | 0.1089 |
| WaveletMask AB | 25.9883 | 0.8795 | 0.1206 | 0.1048 |
| Delta | +0.1466 | -0.0032 | +0.0017 | +0.0041 |

## Scope and Caveat

- Cross-content: DAVIS scenes (sports, animals, vehicles, 854×480) differ substantially from REDS.
- Same degradation: `degradation_video_list_5` is shared. This is not cross-degradation generalization.
- The manuscript must state this caveat explicitly.

## Guardrail

Do not claim "independent dataset validation" without noting the shared degradation pipeline.
