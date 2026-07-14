# T16 Motion-Sensitivity Summary

This artifact consolidates the existing scene-cut/flash probe, persistent-scratch probe, cue-complementarity probe, and FC pilot into one reviewer-facing motion-sensitivity packet.

## Gate Status

- `R1-7` evidence gate met: `True`
- `R2-2` evidence gate met: `True`

## Same-Protocol Reference

- Seed-2021 paired PSNR improvement: `0.4477` with 95% CI `[0.2955, 0.5981]`.

## Abrupt Temporal Probes

| Case | Activation ratio | FPR | Guarded FPR | Guard suppression |
|---|---:|---:|---:|---:|
| scene_cut | 2.3504 | 0.0505 | 0.0199 | 0.6141 |
| flash | 10.4127 | 0.1574 | 0.0036 | 0.9470 |
| illumination_ramp | 1.5218 | 0.0352 | 0.0347 | 0.0247 |

## Persistent-Scratch Probe

| Case | Raw LHFM suppression from L1 | Wavelet suppression from L1 | Wavelet top-5 hit |
|---|---:|---:|---:|
| fixed L2 | 0.4645 | 0.9833 | 0.0117 |
| fixed L3 | 0.6502 | 0.9892 | 0.0025 |
| drift L2 | -0.0364 | 0.8637 | 0.2104 |
| drift L3 | -0.1097 | 0.9581 | 0.0747 |

## Cue Complementarity

| Pair | Spearman | Top-5 overlap |
|---|---:|---:|
| spatial/flicker | 0.4850 | 0.2300 |
| local/flicker | -0.3364 | 0.0024 |

## Flow-Compensated Pilot

- FC pilot mean metrics: `PSNR 23.9953`, `SSIM 0.8478`, `LPIPS 0.1498`, `DISTS 0.1191`.
- Mean delta vs same-protocol WaveletMask seed-2021: `PSNR -2.6147`, `SSIM -0.0476`, `LPIPS -0.0612`, `DISTS -0.0364`.

## Interpretation

- The current evidence is strong enough to formalize `R1-7` as a reproducible failure-boundary diagnostic: scene cuts, flashes, and moving persistent scratches all have explicit probe evidence.
- `R2-2` is now covered by the motion/occlusion-binned benchmark (`motion_binned_benchmark.md`): PSNR improvement by motion tercile is low +0.085 dB, medium +0.477 dB, high +0.769 dB, so the restoration gain does not collapse under stronger motion.

## Notes

- R2-2 motion/occlusion-binned benchmark available at projects/007-waveletmask/Paper/applsci/review/round1/artifacts/t16_motion_binning/formal/motion_binned_benchmark.json.
- RAFT motion descriptors on GT frames; restoration deltas joined from same-protocol seed-2021 test logs; clip terciles rather than absolute thresholds.
- Caveats recorded in the benchmark protocol: camera-motion dominated content (29/30 clips) and residual-proxy clean-FP definition.
- The three-seed FC pilot mean trails the same-protocol WaveletMask seed-2021 result; the FC variant stays an ablated alternative.

