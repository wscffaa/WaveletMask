# Step 01 Figure Plan

This file migrates the existing 007 Applied Sciences draft into the
`ofr-paper-mdpi-writing` fixed OFR figure skeleton.

## Migration Decision

Assumption: 007 is a method paper for WaveletMask, not a dataset or
colorization paper. Therefore dataset synthesis, user study, and colorization
figures stay out of the fixed main-text plan.

The previous six-figure plan preserved useful motivation and Method assets but
missed two fixed evidence slots: synthetic SOTA qualitative comparison and
visual ablation. This v2 plan keeps existing figure numbers and adds Figure7
and Figure8 rather than renumbering the manuscript.

2026-05-12 user freeze decision: use Scheme B, a nine-figure main-text plan
that preserves Figure1--Figure9 numbering.

2026-05-13 governance correction: Figure2--Figure6 and Figure9 were demoted
from final/present to placeholder or blocked states where their production
route was not compliant. Figure2 and Figure3 must restart through
`ofr-paper-diagram` Route E. Figure6--Figure9 must be governed by
`ofr-paper-visual-studio` routes. Figure4 remains optional, and Figure9 is now
fixed rather than conditional.

## Fixed And Conditional Figure Plan

| Figure | Slot ID | Fixed level | Role | Archetype | Core conclusion | Status |
|--------|---------|-------------|------|-----------|-----------------|--------|
| Figure1 | `MECHANISM_ANALYSIS` | conditional | motivation/mechanism | `frequency_spectrum` | Wavelet-domain responses expose local high-frequency damage and low-frequency flicker cues that pixel residual indicators miss or confuse. | present |
| Figure2 | `METHOD_OVERVIEW` | fixed | method | `pipeline_overview` | WaveletMask replaces RTN's pixel residual indicator with a max-fused wavelet-domain degradation mask while preserving the recurrent restoration backbone. | blocked: needs Route E |
| Figure3 | `MODULE_DETAIL_WaveletNoiseMask` | fixed | method | `module_detail` | Local high-frequency and global flicker branches are separate wavelet-domain detectors before max fusion. | blocked: needs Route E |
| Figure4 | `PROBLEM_MOTIVATION` | optional | motivation | `qualitative_grid` | Old-film degradation examples motivate the failure modes but do not prove SOTA performance. | blocked: keep/drop decision |
| Figure5 | `MECHANISM_ANALYSIS_MASK_PATH` | conditional | results/discussion | `mask_overlay` | The local and flicker mask paths explain the submitted A+B max evidence chain and keep learned-fusion variants separate. | blocked: route decision |
| Figure6 | `SOTA_REALWORLD_QUALITATIVE` | fixed | results | `qualitative_grid` | Traceable RealWorld qualitative examples are required before final RealWorld visual claims. | blocked: Visual Studio regeneration |
| Figure7 | `SOTA_SYNTHETIC_QUALITATIVE` | fixed | results | `qualitative_grid` | Synthetic qualitative examples must align with the frozen full-reference metrics and include GT/ROI evidence. | blocked |
| Figure8 | `ABLATION_QUALITATIVE` | fixed | results | `ablation_table_visual` | Visual ablation must show how local high-frequency masking, global flicker masking, and max fusion affect restoration behavior. | blocked |
| Figure9 | `DIFFERENCE_OR_ERROR_MAP` | fixed | discussion | `heatmap_pair` | Error maps must show whether WaveletMask reduces reconstruction error on selected synthetic frames beyond isolated local artifacts. | blocked: finalization |

## Fixed Table Slots

| Table slot | Evidence path | Status |
|------------|---------------|--------|
| `SOTA_SYNTHETIC_TABLE` | `Exp/paper_data/main_results.yaml`, `Exp/paper_data/section_synth_8variants.yaml` | available |
| `SOTA_REALWORLD_TABLE` | `Exp/paper_data/main_results.yaml` | blocked: realworld metrics marked `stale_being_refreshed` |
| `ABLATION_TABLE` | `Idea/ablation_matrix.json`, `Exp/paper_data/section_synth_8variants.yaml`, `Exp/ablation/ablation_results.yaml` | available for synthetic; realworld still stale |

## Production Routes

| Figure | Route | Target |
|--------|-------|--------|
| Figure1 | manual redraw from real assets | `Paper/applsci/figures/Figure1.pdf` |
| Figure2 | `ofr-paper-diagram` Route E: ASCII -> layout -> approved final export | `Paper/applsci/figures/Figure2.pdf` |
| Figure3 | `ofr-paper-diagram` Route E: ASCII -> layout -> approved final export | `Paper/applsci/figures/Figure3.pdf` |
| Figure4 | optional `ofr-paper-visual-studio` traceable export or user-confirmed drop | `Paper/applsci/figures/Figure4.pdf` |
| Figure5 | `ofr-paper-visual-studio` real mask export or `ofr-paper-diagram` schematic route | `Paper/applsci/figures/Figure5.pdf` |
| Figure6 | `ofr-paper-visual-studio` Real-world/SOTA full-frame export | `Paper/applsci/figures/Figure6.pdf` |
| Figure7 | `ofr-paper-visual-studio` Synthetic/SOTA | `Paper/applsci/figures/Figure7.pdf` |
| Figure8 | `ofr-paper-visual-studio` Synthetic/Ablation | `Paper/applsci/figures/Figure8.pdf` |
| Figure9 | `ofr-paper-visual-studio` Figure 9 / error-map export | `Paper/applsci/figures/Figure9.pdf` |

## ASCII Sketches For New Fixed Gaps

### Figure7 — Synthetic SOTA Qualitative

```text
----------+----------+----------+----------+
| Input    | RTN      | SOTA ... | Ours     |
| ROI      | ROI      | ROI      | ROI      |
| GT       |          |          |          |
+----------+----------+----------+----------+
```

### Figure8 — Visual Ablation

```text
Input | Baseline | +Local | +Flicker | WaveletMask | GT
      | ROI      | ROI    | ROI      | ROI         |
```

### Figure9 — Difference / Error Map

```text
|Baseline - GT| |WaveletMask - GT| |Error reduction|
```

## Current Step 6 Blockers

- Figure2 and Figure3 are fixed but must pass `ofr-paper-diagram` Route E before final use.
- Figure4 needs a keep / supplement / drop decision.
- Figure5 needs a real-mask-vs-schematic route decision.
- Figure6 needs Visual Studio Real-world SOTA regeneration plus refreshed RealWorld logs.
- Figure7 target `figures/Figure7.pdf` is missing; Synthetic/SOTA scenes must be selected and exported.
- Figure8 target `figures/Figure8.pdf` is missing; Synthetic/Ablation scenes must be selected and exported.
- Figure9 is fixed, but the final error-map export must be confirmed through the Figure 9 route after the directory restructure.
- RealWorld no-reference metrics remain stale in `Exp/paper_data/main_results.yaml`; final RealWorld ranking language must stay blocked.

## User Confirmation Questions

Ask these before generating final PDFs:

1. What methods should be visible in Figure7 synthetic SOTA: RTN only plus WaveletMask, or the full SOTA set used in the table?
2. Should Figure8 use four variants (Baseline/+Local/+Flicker/WaveletMask) or include learned-fusion controls in a second row?
3. Should Figure3 remain one module-detail figure with Local/Flicker/Max subpanels, or be split?
4. Should Figure5 be a real mask visualization, a schematic, or moved to supplement?
5. Should Figure4 remain in the main text, move to supplement, or be removed?
6. Should Figure9 use the fixed three-column layout `|Baseline-GT| / |WaveletMask-GT| / error reduction`?
