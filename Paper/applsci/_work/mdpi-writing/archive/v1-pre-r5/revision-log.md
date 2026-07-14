# WaveletMask Writing Revision Log

## 2026-05-09

- Initialized writing state for Applied Sciences migration.
- Set `Paper_1` as reference-only source; current 007 MDPI project remains canonical.
- Started migration boundary audit before rewriting paper sections.
- Open blockers: CLIPIQA naming, synthetic record discrepancy, and old visual asset reuse permission/provenance.
- Completed four migration audits and wrote the migration map under `docs/workflow/10-active/20260509-204416-007-waveletmask-paper-migration/`.
- Copied missing MDPI logo assets from the 001 Applied Sciences template so 007 can compile from a clean temporary checkout.
- Fresh LaTeX compile now exits successfully and produces a 21-page PDF; remaining warning is `wang2022basicsr` missing `journal`.
- Next blockers: stale RealWorld metrics, old synthetic discrepancy note, MambaOFR/RTN claim alignment, and figure provenance files.

## 2026-05-09 Submission Alignment

- Rewrote the Results evidence chain around frozen synthetic paper-data and removed unsupported sensitivity/runtime placeholders from the manuscript body.
- Inserted F2--F5 and added `figure_manifest.yaml`, `configs/visual_figures.yaml`, and `figures_review.md`.
- Downgraded RealWorld quantitative language to a provisional audit snapshot and removed final ranking claims from Abstract, Introduction, Discussion, and Conclusion.
- Fixed `CLIPIQA` naming and changed `wang2022basicsr` to `@misc` to eliminate the empty-journal bibliography warning.
- Corrected RTN/RRTN/DeepRemaster reference metadata and raised cited-reference DOI coverage to 35/46 (76.1%) using Crossref plus Chrome/DOI-page spot verification.
- Fresh `latexmk -g -pdf -interaction=nonstopmode -halt-on-error waveletmask.tex` exits 0 and produces a 19-page PDF.
- Remaining submission risks: F6 traceable RealWorld qualitative figure, final redraw of Paper_1-derived motivation assets, refreshed RealWorld logs, full-fusion checkpoint arbitration, and unused bib cleanup.

## 2026-05-10 MDPI Writing Polish

- Aligned Abstract, Introduction, Method, Results, Discussion, and Conclusion with the frozen evidence chain: WaveletMask-ab is now the submitted synthetic-best model.
- Demoted PixelWiseMaskFusion from a claimed core component to an ablated learned-fusion variant because the full-fusion checkpoint underperforms in the current synthetic snapshot.
- Corrected complexity language so the submitted detector reports about 0.5K extra parameters, while the 5.5K figure is reserved for learned-fusion variants.
- Remaining submission risks: refreshed RealWorld logs, traceable F6 qualitative output, final replacement/redraw of motivation assets, and release-grade data availability URL.

## 2026-05-12 MDPI Writing Skill Migration

- Migrated the project into the new `ofr-paper-mdpi-writing` motivation-first state machine.
- Added `step-00-motivation-anchor.md` with M1 selected from existing Idea/manuscript evidence: wavelet-domain degradation indicators replace pixel residual indicators.
- Added `step-01-figure-plan.md` and six figure contracts under `Paper/_work/mdpi-writing/contracts/`.
- Added dual-track writing guides under `Paper/_work/mdpi-writing/`.
- Updated `state.json` to `current_step=6`, `current_substep=6.blocked`; Figure1 is synced, while Figure2-Figure6 remain blocked pending final assets and caption sync.

## 2026-05-12 Fixed OFR Figure Skeleton Migration

- Upgraded the 007 figure plan to the fixed OFR skeleton required by the revised `ofr-paper-mdpi-writing` protocol.
- Preserved existing Figure1-Figure6 numbering to avoid breaking current manuscript references.
- Added Figure7 as `SOTA_SYNTHETIC_QUALITATIVE` and Figure8 as `ABLATION_QUALITATIVE`, closing the two fixed experiment-figure gaps in the prior six-figure plan.
- Added Figure9 as `DIFFERENCE_OR_ERROR_MAP` using the existing `_work/Fig9/Fig9b_error_map.*` assets as conditional discussion evidence.
- Upgraded Figure1-Figure6 contracts to schema v2 fields: `slot_id`, `fixed_level`, `drop_policy`, and `evidence_gate`.
- Updated `state.json` Step 6 blockers to include Figure7, Figure8, and Figure9; RealWorld ranking language remains blocked until refreshed frozen logs exist.

## 2026-05-12 Unified MDPI Work Root

- Merged `Paper/writing/`, `Paper/writing_guides/`, `Paper/applsci/figure_manifest.yaml`, `Paper/applsci/configs/visual_figures.yaml`, and `Paper/applsci/figures/{_contracts,_work,_archive}` into `Paper/_work/mdpi-writing/`.
- Updated state, contracts, figure manifest, visual figure plan, and figure metadata references to the unified work root.
- Kept `Paper/applsci/figures/` for final LaTeX-consumed figures only.

## 2026-05-12 Scheme B Figure Freeze

- User selected Scheme B: preserve the nine-figure main-text skeleton without renumbering.
- Promoted the existing architecture draft to `Paper/applsci/figures/Figure2.pdf` and synced the Introduction reference/caption.
- Promoted the WaveletNoiseMask module draft to `Paper/applsci/figures/Figure3.pdf` and synced the Method reference/caption.
- Promoted the motivation-only degradation examples asset to `Paper/applsci/figures/Figure4.pdf` and kept its caption explicitly outside the SOTA/quantitative evidence chain.
- Replaced the old Figure5 TikZ draft with `Figure5.v01.mask_path_schematic.pdf`, separating submitted max fusion from ablation-only learned fusion.
- Inserted `Paper/applsci/figures/Figure6.pdf` as traceable RealWorld qualitative evidence while keeping final RealWorld ranking blocked by missing refreshed logs.
- Promoted the existing Figure9 error-map export to `Paper/applsci/figures/Figure9.pdf` and inserted it into Discussion as synthetic GT-only analysis.
- Updated Figure2--Figure6 and Figure9 contracts, manifest entries, and `state.json`; remaining Step 6 figure blockers are Figure7--Figure8 plus refreshed RealWorld logs.

## 2026-05-13 Figure Governance Correction

- Reclassified the 2026-05-12 Figure2--Figure6 and Figure9 promotions as placeholders where their production route did not satisfy the owning figure skill.
- Restructured `Paper/_work/mdpi-writing/figures/` into `FigureX/source`, `FigureX/draft`, and `FigureX/export` directories.
- Moved legacy submission-only `FigX_*` assets from `Paper/applsci/figures/` into `_work/mdpi-writing/archive/20260512-noncompliant-promotion/submission-legacy/`.
- Archived the obsolete PixelWise fusion drafts under `_work/mdpi-writing/archive/20260512-noncompliant-promotion/obsolete-pixelwise-fusion/`.
- Reopened Step 1 as `revisit` and blocked Step 6 until Figure2/3 pass `ofr-paper-diagram` Route E and Figure6/7/8/9 pass the proper `ofr-paper-visual-studio` routes.
- Promoted Figure9 from conditional to fixed evidence for the difference/error-map slot; it remains blocked until final Figure 9 export confirmation.
