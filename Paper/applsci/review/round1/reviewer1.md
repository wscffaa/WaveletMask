# Reviewer 1 — Round 1 Comments

The manuscript proposes a wavelet-domain degradation detection framework (WaveletMask) for recurrent restoration of old films by integrating high- and low-frequency wavelet cues into the gating mechanism of RTN. The topic is relevant to Applied Sciences and the idea of incorporating wavelet-domain information into degradation detection is potentially interesting. However, the current manuscript contains several methodological, experimental, and validation-related weaknesses that prevent acceptance in its present form. The paper is generally well written and organized, but the experimental evidence is not yet sufficiently strong to support the claimed effectiveness and general applicability of the proposed approach.

## 1. Lack of Statistical Validation

The most serious weakness of the manuscript is the absence of statistical validation. The authors explicitly state that all reported results are based on single-run experiments and that no multi-seed evaluation, confidence intervals, or significance testing were performed. This is particularly problematic because the reported improvement over RTN is only +0.44 dB PSNR. Without multiple random seeds, standard deviations, confidence intervals, paired statistical significance tests it is impossible to determine whether the reported gain is statistically meaningful or simply due to training randomness. The authors themselves acknowledge this limitation, but it remains a major methodological deficiency.

## 2. Comparison Protocol Is Not Scientifically Fair

Table 2 combines retrained RTN and retrained WaveletMask with results from other methods reported in the literature. This introduces a substantial protocol mismatch because training datasets may differ, degradation synthesis procedures may differ, optimization settings may differ and evaluation scripts may differ. Therefore, claims that WaveletMask outperforms methods such as MambaOFR, RVRT, or RRTN are not scientifically justified.

## 3. Insufficient Real-World Validation

The real-world evaluation relies entirely on NIQE, BRISQUE, CLIPIQA and qualitative figures. The authors correctly note that these metrics are content-sensitive and not conclusive. The authors must include subjective visual quality assessment, user preference study and expert restoration evaluation.

## 4. Weak Theoretical Justification of LHFM and GFM

The proposed LHFM and GFM modules are largely heuristic. The manuscript states that geometric-mean differencing highlights transient defects and minimum temporal difference highlights flicker. However, no theoretical analysis is provided explaining why these specific operators are optimal or superior to alternatives.

## 5. No Artifact-Type Breakdown

The paper claims improvements for scratches, dust, and flicker, yet no separate evaluation is reported for each degradation category.

## 6. Wavelet Choice Is Not Justified

The entire framework relies on a single-level Haar wavelet decomposition. No comparison is provided against Daubechies wavelets, Symlets, Coiflets and multi-level wavelet decompositions.

## 7. Limited Analysis of Failure Cases

The manuscript repeatedly mentions that scene cuts remain problematic, difficult motion can generate false responses and persistent scratches may be missed. However, no dedicated experiments quantify these failure modes.

## 8. Ablation Study Remains Incomplete

The ablation study only examines LHFM, GFM and fusion strategies.

## Table Review

**Table 2**: Results originate from different experimental protocols. Direct ranking is therefore questionable.

**Table 3**: The no-reference metrics are inconsistent — LHFM achieves best BRISQUE, GFM achieves best NIQE and WaveletMask is not best on any metric. This weakens claims regarding real-world superiority. The authors should discuss this contradiction in greater depth.
