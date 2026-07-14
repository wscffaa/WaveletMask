# T5 Persistent-Scratch Probe

Synthetic scratches are injected into clean REDS clips with controlled persistence length.
The probe measures detector response at the scratched center frame; it does not run the restoration backbone.

- Clips per condition: `12`
- Frames per clip: `7`
- Spatial size: `256`
- Lengths: `1,2,3,5`
- Modes: `fixed,drift`
- Opacity: `0.65`
- Checkpoint: `projects/007-waveletmask/Exp/checkpoints/waveletmask_ab/net_g_best.pth`

| Mode | L | N | Scratch area | Raw LHFM in | Raw rel. | Raw suppress | m_local in | m_local rel. | m_wavelet in | m_wavelet rel. | m_wavelet top-5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed | 1 | 12 | 0.0229 | 0.0862 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 0.1534 | 1.0000 | 0.3472 |
| fixed | 2 | 12 | 0.0229 | 0.0462 | 0.5355 | 0.4645 | 0.0001 | 1.7745 | 0.0026 | 0.0167 | 0.0117 |
| fixed | 3 | 12 | 0.0229 | 0.0302 | 0.3498 | 0.6502 | 0.0001 | 2.2068 | 0.0017 | 0.0108 | 0.0025 |
| fixed | 5 | 12 | 0.0229 | 0.0302 | 0.3498 | 0.6502 | 0.0001 | 2.2068 | 0.0017 | 0.0108 | 0.0025 |
| drift | 1 | 12 | 0.0229 | 0.0862 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 0.1534 | 1.0000 | 0.3472 |
| drift | 2 | 12 | 0.0229 | 0.0894 | 1.0364 | -0.0364 | 0.0000 | 0.9424 | 0.0209 | 0.1363 | 0.2104 |
| drift | 3 | 12 | 0.0229 | 0.0957 | 1.1097 | -0.1097 | 0.0000 | 0.9196 | 0.0064 | 0.0419 | 0.0747 |
| drift | 5 | 12 | 0.0229 | 0.0957 | 1.1097 | -0.1097 | 0.0000 | 0.9196 | 0.0064 | 0.0419 | 0.0747 |

## Qualitative Example Asset Directories

- `projects/007-waveletmask/Paper/applsci/review/round1/artifacts/t5_persistent_scratch/formal/examples/fixed_L1_002`
- `projects/007-waveletmask/Paper/applsci/review/round1/artifacts/t5_persistent_scratch/formal/examples/drift_L1_002`
- `projects/007-waveletmask/Paper/applsci/review/round1/artifacts/t5_persistent_scratch/formal/examples/fixed_L1_008`
- `projects/007-waveletmask/Paper/applsci/review/round1/artifacts/t5_persistent_scratch/formal/examples/drift_L1_008`
- `projects/007-waveletmask/Paper/applsci/review/round1/artifacts/t5_persistent_scratch/formal/examples/fixed_L1_011`
- `projects/007-waveletmask/Paper/applsci/review/round1/artifacts/t5_persistent_scratch/formal/examples/drift_L1_011`
