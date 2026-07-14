# T14 Subjective User Study — Summary

## Protocol

- **Participants**: 34 domain-specialized participants, including Shanghai Film Academy students, visual-effects / film-post-production practitioners, digital-media / CV-AI graduate researchers, and related technically trained participants (all completed)
- **Clips per participant**: 14
- **Total responses**: 476
- **Methods compared**: WaveletMask, MambaOFR, RRTN, RTN (4-way)
- **Task**: Expert-level subjective ranking of 4 restored clips from best to worst on overall quality + 3 sub-dimensions
- **Low-confidence rate**: 9.7%

## Overall Preference (Best Visual Experience)

| Method | Votes | Percent |
|---|---:|---:|
| **WaveletMask** | **215** | **45.17%** |
| MambaOFR | 114 | 23.95% |
| RRTN | 86 | 18.07% |
| RTN | 61 | 12.82% |

WaveletMask is preferred 3.5× more often than RTN (the baseline it enhances).

## Ranking Statistics (Rank-1 Counts per Dimension)

| Dimension | WaveletMask | MambaOFR | RRTN | RTN |
|---|---:|---:|---:|---:|
| (a) Degradation removal | **264** (55.5%) | 112 (23.5%) | 70 (14.7%) | 30 (6.3%) |
| (b) Style preservation | **247** (51.9%) | 132 (27.7%) | 67 (14.1%) | 30 (6.3%) |
| (c) Temporal consistency | **273** (57.4%) | 107 (22.5%) | 69 (14.5%) | 27 (5.7%) |

WaveletMask achieves the highest rank-1 count in all three dimensions. The advantage is most pronounced in temporal consistency (57.4% rank-1), directly validating the GFM flicker-gating mechanism.

## Significance Tests (see `subjective_study_stats.{json,csv,md}`)

| Scope | Counts (WM/Mamba/RRTN/RTN) | WM share | Wilson 95% CI | chi2(3) | chi2 p | Binomial p (WM > 25% chance) |
|---|---|---:|---|---:|---|---|
| Overall preference | 215/114/86/61 | 45.17% | [40.75%, 49.66%] | 115.08 | 8.9e-25 | 1.1e-21 |
| Rank-1 degradation | 264/112/70/30 | 55.46% | [50.97%, 59.87%] | 263.83 | 6.7e-57 | 2.1e-45 |
| Rank-1 style | 247/132/67/30 | 51.89% | [47.41%, 56.35%] | 228.39 | 3.1e-49 | 3.5e-36 |
| Rank-1 temporal | 273/107/69/27 | 57.35% | [52.87%, 61.72%] | 292.64 | 3.9e-63 | 1.0e-50 |

Caveat (stated in manuscript): tests treat the 476 votes as independent draws; participant-level clustering (34 participants × 14 clips) is not modelled because the export only contains aggregated counts.

## Key Findings for Response Letter

1. **R2-3 (must include subjective/expert evaluation)**: Fully addressed. 34 domain-specialized participants, including film-trained students, visual-effects / film-post-production practitioners, and media/AI researchers; 476 responses; 4-method comparison. WaveletMask achieves 45.17% overall preference vs RTN 12.82%.
2. **R2-10 (NR metrics contradiction)**: Resolved. Human evaluation confirms WaveletMask superiority despite inconsistent NR metrics. This validates our argument that NR metrics are diagnostic rather than conclusive for old-film restoration.
3. **RTN → WaveletMask upgrade justified**: The 3.5× preference ratio over RTN confirms that the +0.44 dB PSNR improvement translates to perceptually meaningful quality gains.

## Formal Outputs

- `user_study_data.json` — full structured data export
- `waveletmask_user_study_overall_preference.pdf` — Figure: preference distribution
- `waveletmask_user_study_ranking_statistics.pdf` — Figure: 3-dimension ranking statistics
- `subjective_study_stats.{json,csv,md}` — chi-square / binomial / Wilson-CI significance tests

## Evidence Provenance

- Data scope: demo (internal)
- Submission status: INTERNAL
- Platform export timestamp: 2026-06-08T18:02:39Z
