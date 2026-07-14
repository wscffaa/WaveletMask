# T7 GFM Failure-Mode Probe

Synthetic probe sequences are generated from held-out REDS GT clips. The script measures the trained GFM branch response only; it does not claim restoration-quality improvement.

- Clips per case: `12`
- Frames per probe: `7`
- Spatial size: `256`
- Input normalization: `[-1,1]`
- FPR threshold: per-clip clean GFM 99th percentile
- Cut guard threshold: `0.18`

| Case | N | Clean activation | Probe activation | Ratio | FPR | Guarded activation | Guarded FPR | Suppression |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| scene_cut | 12 | 0.0041 | 0.0066 | 2.3504 | 0.0505 | 0.0024 | 0.0199 | 0.6141 |
| flash | 12 | 0.0041 | 0.0257 | 10.4127 | 0.1574 | 0.0015 | 0.0036 | 0.9470 |
| illumination_ramp | 12 | 0.0041 | 0.0051 | 1.5218 | 0.0352 | 0.0049 | 0.0347 | 0.0247 |
