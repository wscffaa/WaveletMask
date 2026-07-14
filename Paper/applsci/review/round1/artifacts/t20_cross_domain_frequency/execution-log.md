# T20 Cross-Domain Frequency Execution Log

## Scope

Reviewer 3 requested an ablation across Haar DWT, Sobel/Laplacian, Gaussian pyramid, Fourier low/high temporal differences, and learned shallow frequency filters. This log records the execution evidence for the diagnostic experiment.

## Environment

| Server | Role | Path | GPU | Environment |
|---|---|---|---|---|
| L40_aliyun | deterministic shards | `/mnt/data/Antigravity_OFR` | 4 x NVIDIA L40 | `rtn`, Python 3.12.12 |
| A100_22_aliyun | learned folds | `/data/cff/Code/Antigravity_OFR` | 4 x NVIDIA A100 80GB PCIe | `rtn`, Python 3.12.12 |

Both servers had the canonical REDS validation paths:

- `datasets/OldFilmRestoration/val/REDS/GT/val_sharp`
- `datasets/OldFilmRestoration/val/REDS/LQ/val_sharp_degraded`

Remote worktrees were dirty before launch, so the script was copied directly with `scp` instead of running `git pull --rebase`.

## Completed Tasks

| ID | Evidence |
|---|---|
| P0 | Script copied to L40; `rtn` import check passed for `numpy`, `cv2`, and `torch`; synthetic deterministic smoke exited 0. |
| P1 | Script copied to A100; `rtn` import check passed for `numpy`, `cv2`, and `torch`; synthetic learned smoke exited 0. |
| P2 | L40 GPU0 REDS smoke wrote `smoke/cross_domain_frequency_ablation.{json,csv,md}` and `smoke/logs/p2_smoke_l40_gpu0.log`. |
| P3-P6 | L40 shards `shard_00` to `shard_03` each wrote `partial_deterministic.{json,csv,md}`; logs end with `EXIT:0`. |
| P7-P10 | A100 folds `fold_00` to `fold_03` each wrote `partial_learned.{json,csv,md}`; logs end with `EXIT:0`. |
| P11 | A100 `fold_04` wrote `partial_learned.{json,csv,md}`; JSON protocol records `device: cuda`; log ends with `EXIT:0`. |
| P12-P13 | Remote shards, learned folds, smoke output, and server logs were copied into this review artifact tree. |
| P14 | Local merge produced `formal/cross_domain_frequency_ablation.{json,csv,md}`. |
| P15 | Claim analysis saved as `formal/claim_analysis.md`. |
| P16 | Manuscript text updated in `Paper/applsci/sections/results.tex` and `Paper/applsci/sections/discussion.tex`; reviewer response saved as `response-letter-reviewer3.md`. |
| P17 | Final audit passed: merged CSV has 16 rows, five learned folds exist, nine formal server logs contain `EXIT:0`, and `latexmk -pdf -interaction=nonstopmode -halt-on-error waveletmask.tex` exited 0. |
| Post-review correction | Removed `Best F1` from displayed T20 tables because several deterministic operators reach best F1 only at a near-all-positive threshold; clarified that the Haar diagnostic row measures single-frame wavelet energy rather than the deployed LHFM/GFM temporal geometric-mean differencing. |

## Final Artifacts

- `formal/cross_domain_frequency_ablation.json`
- `formal/cross_domain_frequency_ablation.csv`
- `formal/cross_domain_frequency_ablation.md`
- `formal/claim_analysis.md`
- `formal/shards/`
- `formal/learned_folds/`
- `formal/logs_l40/`
- `formal/logs_a100/`
- `smoke/`
