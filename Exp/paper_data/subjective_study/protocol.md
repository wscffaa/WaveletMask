# WaveletMask Subjective and Expert Review Protocol

Status: live collection plan. Submission-facing artifacts remain blocked until real platform exports are archived.

## Purpose

This study addresses the reviewers' request for subjective visual quality assessment, user preference study, and expert restoration evaluation on real-world old-film clips.

## Materials

- Clip set: 12 prepared SRWOV clips under `stimuli_videos/`.
- Per clip files: `lq.mp4`, `rtn.mp4`, `waveletmask.mp4`, and `gt.mp4` where available.
- Pairwise comparison: RTN versus WaveletMask, randomized A/B order.

## Ordinary User Study

- Minimum acceptance target: 15 completed participants.
- Stretch target: 24 completed participants.
- Task: choose A, B, or no preference.
- Questions:
  - Overall restoration preference.
  - Scratch/dust suppression.
  - Flicker or temporal stability.
  - Detail preservation.
- Aggregation:
  - Preference rate for WaveletMask.
  - Tie/no-preference rate.
  - Bootstrap 95 percent confidence interval by participant and clip.

## Expert Study

- Minimum acceptance target: 2 completed experts.
- Stretch target: 5 completed experts.
- Scale: 1 to 5, higher is better except over-restoration risk.
- Dimensions:
  - Artifact suppression.
  - Temporal coherence.
  - Detail preservation.
  - Archival plausibility.
  - Over-restoration risk.
- Aggregation:
  - Mean and standard deviation per method and dimension.
  - Short qualitative notes per expert.

## Evidence Handling

- Save raw platform exports under `platform_exports/raw/` with dated filenames.
- Derive submission-facing summaries only from real exports.
- Do not copy simulated values into the manuscript, response letter, supplementary package, or reviewer evidence pack.

## Placeholder Policy

Simulated summaries under `projects/007-waveletmask/Paper/applsci/review/round1/artifacts/simulated_placeholders/` are drafting-only references. They must be replaced by real platform exports before any submission-facing update.
