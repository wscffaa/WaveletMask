# T5 Persistent-Scratch Probe

Synthetic scratches are injected into clean REDS clips with controlled persistence length.
The probe measures detector response at the scratched center frame; it does not run the restoration backbone.

- Clips per condition: `2`
- Frames per clip: `7`
- Spatial size: `128`
- Lengths: `1,3`
- Modes: `fixed,drift`
- Opacity: `0.65`
- Checkpoint: `projects/007-waveletmask/Exp/checkpoints/waveletmask_ab/net_g_best.pth`

| Mode | L | N | Scratch area | Raw LHFM in | Raw rel. | Raw suppress | m_local in | m_local rel. | m_wavelet in | m_wavelet rel. | m_wavelet top-5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed | 1 | 2 | 0.0464 | 0.0926 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 0.1477 | 1.0000 | 0.6935 |
| fixed | 3 | 2 | 0.0464 | 0.0358 | 0.3864 | 0.6136 | 0.0001 | 2.2531 | 0.0014 | 0.0092 | 0.0000 |
| drift | 1 | 2 | 0.0464 | 0.0926 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 0.1477 | 1.0000 | 0.6935 |
| drift | 3 | 2 | 0.0464 | 0.0996 | 1.0760 | -0.0760 | 0.0000 | 0.8829 | 0.0053 | 0.0358 | 0.1453 |

## Qualitative Examples

- `projects/007-waveletmask/Paper/applsci/review/round1/artifacts/t5_persistent_scratch/smoke/examples/fixed_L1_000.png`
- `projects/007-waveletmask/Paper/applsci/review/round1/artifacts/t5_persistent_scratch/smoke/examples/drift_L1_000.png`
