# Step 00 Motivation Anchor

## Source Artifacts

| Artifact | Status | Notes |
|----------|--------|-------|
| `Idea/03-SCI-故事线.md` | present | Defines WaveletMask as A+B max fusion and removes harmful C from the submitted method. |
| `Idea/00-创新分析.md` | present | Identifies pixel-domain residual indicators as weak for scratches, dust, and flicker. |
| `Idea/01-baseline-decision.md` | present | Chooses RTN because `residual_indicator` is a narrow, compatible intervention point. |
| `Idea/diff_plan.json` | present | Keeps implementation project-local under WaveletMask variants. |
| `Idea/ablation_matrix.json` | present | Marks `waveletmask_ab` as the final synthetic-best variant. |

## Candidate Motivations

### M1. Frequency-Domain Degradation Indicator

- Motivation: Pixel-domain residual indicators confuse motion/noise and miss flicker, while wavelet sub-bands isolate local high-frequency damage and low-frequency brightness variation.
- Suitable figure archetypes: `frequency_spectrum`, `comparison_grid_3row`, `pipeline_overview`
- Required evidence: `Figure1.pdf`, `Figure2.pdf`, `Idea/04-框架对齐.md`
- Reviewer risk: The motivation figure must not be read as quantitative proof.

### M2. Simple Max Fusion Beats Learned Fusion

- Motivation: A simple max of local and flicker masks is more reliable than learned fusion for the current frozen synthetic evidence chain.
- Suitable figure archetypes: `ablation_table_visual`, `bar_comparison`, `mask_overlay`
- Required evidence: `Exp/ablation/ablation_results.yaml`, `Exp/paper_data/main_results.yaml`
- Reviewer risk: The story can overclaim if PixelWiseMaskFusion is still presented as a core component.

### M3. Real-World Evidence Boundary

- Motivation: The paper can be made submission-safe by separating traceable synthetic evidence from provisional RealWorld qualitative claims.
- Suitable figure archetypes: `qualitative_grid`, `failure_case`, `temporal_consistency_sequence`
- Required evidence: refreshed RealWorld logs and traceable `Figure6`
- Reviewer risk: Any final RealWorld ranking claim is unsafe until logs are refreshed.

## Selected Motivation

- Motivation ID: M1
- One-sentence paper claim: WaveletMask replaces RTN's pixel-domain residual indicator with a wavelet-domain degradation mask that separately captures local high-frequency damage and global flicker before max fusion.
- Why this is the main story: It matches the current manuscript, the accepted `WaveletMaskABNet` final architecture, and the strongest synthetic evidence. M2 is important, but it is an ablation conclusion rather than the opening motivation. M3 is a guardrail, not the central contribution.
- What this story must not claim: It must not claim final RealWorld superiority, and it must not present PixelWiseMaskFusion as part of the submitted WaveletMask model.

## 5-Slide Talk Outline

| Slide | Purpose | Content | Figure anchor |
|-------|---------|---------|---------------|
| 1 | Problem | Old-film restoration contains scratches, dust, flicker, and motion that interact in pixel space. | Figure1 |
| 2 | Existing failure | RTN's pixel residual indicator cannot separate degradation from motion and is weak for flicker. | Figure1 |
| 3 | Key insight | Haar sub-bands expose different degradation families: detail bands for local damage and LL for flicker. | Figure1 |
| 4 | Method | WaveletNoiseMask computes local/flicker masks and max-fuses them into RTN gated aggregation. | Figure2, Figure3 |
| 5 | Evidence | `waveletmask_ab` is synthetic-best; learned fusion variants underperform; RealWorld ranking remains provisional. | Figure5, Figure6 |

## Story Arc

1. Problem: Old-film degradation mixes local structural damage, temporal brightness variation, and motion.
2. Existing fail: Pixel residual indicators trigger on motion-like differences and are too weak for global flicker.
3. Our insight: Wavelet sub-bands give a compact frequency-domain view aligned with those degradation families.
4. Method: WaveletMask computes local high-frequency and global flicker masks, then max-fuses them as RTN's degradation indicator.
5. Evidence: Frozen synthetic metrics support the A+B max variant, while RealWorld visual evidence still needs traceable Figure6 and refreshed logs.

## Narrative Anchors

### Anchor A1: Pixel Residual Failure

- Claim supported: Pixel-domain residual indicators are an insufficient degradation signal for old-film artifacts.
- Required figure role: motivation
- Candidate archetypes: `frequency_spectrum`, `comparison_grid_3row`
- Evidence path: `projects/007-waveletmask/Paper/applsci/figures/Figure1.pdf`
- Reviewer risk: Must stay framed as motivation, not quantitative superiority.

### Anchor A2: WaveletMask Data Flow

- Claim supported: WaveletMask enters RTN through the `residual_indicator` interface without rewriting the restoration backbone.
- Required figure role: method
- Candidate archetypes: `pipeline_overview`, `module_detail`
- Evidence path: `projects/007-waveletmask/archs/waveletmask_ab_arch.py`
- Reviewer risk: Figure labels must not imply PixelWiseMaskFusion is in the submitted path.

### Anchor A3: Conservative Evidence Boundary

- Claim supported: The paper's defensible evidence is synthetic-best `waveletmask_ab` plus provisional RealWorld qualitative analysis.
- Required figure role: results
- Candidate archetypes: `ablation_table_visual`, `qualitative_grid`, `failure_case`
- Evidence path: `projects/007-waveletmask/Exp/paper_data/main_results.yaml`
- Reviewer risk: Figure6 is still missing from the manuscript and must be traceable before submission.

## Step 1 Handoff

- Required figure count: 6
- Mandatory figure roles:
  - motivation visualization
  - pipeline overview
  - module detail
  - degradation examples / motivation-only qualitative asset
  - mask interpretation / ablation explanation
  - RealWorld qualitative comparison
- Locked terms:
  - Method name: WaveletMask
  - Final architecture: WaveletMaskABNet
  - Submitted variant: `waveletmask_ab`
  - Archived variant: PixelWiseMaskFusion / C
- Gate decision: migrated from existing manuscript and Idea artifacts on 2026-05-12; user can override M1 before new prose rewriting.
