# Platform Export Archive

This directory stores raw and processed exports from the live WaveletMask human-evaluation platform.

## Rules

- Keep raw participant exports under `raw/`.
- Use dated filenames, for example:
  - `20260608-waveletmask-preference.csv`
  - `20260608-waveletmask-expert.csv`
  - `20260608-waveletmask-admin-stats.json`
- Do not overwrite prior exports; append new dated snapshots.
- Do not place simulated data in this directory.

## Required Raw Inputs For Closure

- ordinary participant export CSV
- expert-rating export CSV
- admin stats JSON

## Export Path

The platform export scripts live under:

- `/Users/caifeifan/VibeScholar/Antigravity_OFR/papers/shuphd-ofr-latex/_workspace/06-user-study/scripts/export_collection.mjs`
- `/Users/caifeifan/VibeScholar/Antigravity_OFR/papers/shuphd-ofr-latex/_workspace/06-user-study/scripts/analyze_results.py`

Formal reviewer-facing summaries should be generated only after the raw files above are archived here.
