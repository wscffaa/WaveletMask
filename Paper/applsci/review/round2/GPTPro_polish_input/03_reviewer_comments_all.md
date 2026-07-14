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
# Reviewer 2 — Round 1 Comments

Dear authors,

I think the work is interesting but some comments are provided:

## 1. Persistent Scratches

The method assumes that scratches and dust are transient artifacts. However, many real archival films contain persistent scratches spanning multiple consecutive frames. Please include a dedicated experiment quantifying performance on persistent damage and analyze whether the geometric-mean criterion in LHFM suppresses such defects.

## 2. Motion-Aware Analysis

The wavelet-domain differences are computed without explicit motion compensation, whereas the RTN spatial indicator relies on optical-flow alignment. Please investigate the impact of camera motion, object motion, and occlusions on LHFM and GFM responses, and provide a motion-aware variant or an ablation using flow-compensated wavelet differences.

## 3. Wavelet Choice Ablation

The choice of the Haar wavelet is justified mainly by simplicity and efficiency. However, the paper does not demonstrate that Haar is the most suitable decomposition for old-film artifacts. An ablation comparing Haar against Daubechies, Symlets, or dual-tree complex wavelets would significantly strengthen the technical contribution.

## 4. Scene Cut Evaluation

The GFM branch is intended to capture brightness flicker, but the proposed minimum-difference criterion may also respond strongly to scene cuts, flashes, and abrupt illumination changes. Please provide a dedicated evaluation on clips containing scene transitions and discuss potential mitigation strategies such as cut detection or temporal consistency constraints.

## 5. Response Visualization and Correlation Analysis

The paper claims that wavelet-domain cues provide complementary information to pixel-domain cues, but this claim remains largely qualitative. It would be valuable to include response visualizations and quantitative correlation analyses between the RTN spatial indicator, LHFM output, and GFM output to demonstrate their complementarity more rigorously.

## 6. Stronger Fusion Alternatives

The learned fusion experiments appear limited to a single CNN architecture and initialization strategy. Since the conclusion that max fusion is preferable depends heavily on this design choice, please evaluate stronger alternatives such as attention-based fusion, uncertainty-aware fusion, normalization strategies, or residual fusion mechanisms before drawing broader conclusions.

## 7. Missing Reference

When introducing wavelets, please refer to https://link.springer.com/article/10.1007/s40534-025-00393-5

## 8. Independent Real-World Dataset

The quantitative evaluation relies almost entirely on the SRWOV synthetic benchmark. Given that the target application is real archival film restoration, additional experiments on independent real-world datasets or publicly available restoration benchmarks would be important to assess generalization beyond the SRWOV degradation model.

## 9. Computational Efficiency Metrics

The computational analysis reports parameters and FLOPs but omits runtime, latency, memory consumption, and throughput. Since one motivation of the method is its lightweight design, practical deployment metrics on representative hardware should be reported and compared against RTN and other baselines.

## 10. Mask Interpretability

The detector masks are trained only through restoration losses and receive no explicit supervision. As a result, it is unclear whether the generated masks actually correspond to degradation locations. Please provide qualitative mask visualizations, overlap analyses with synthetic degradation annotations, or other interpretability studies to validate that the learned masks behave as intended.
# Reviewer 3 — Round 1 Comments

The manuscript proposes WaveletMask, a wavelet-domain degradation detection module for recurrent old-film restoration. The core idea is to augment the RTN-style pixel-domain degradation indicator with frequency-specific masks: LHFM uses Haar high-frequency detail sub-bands to emphasize transient scratches and dust, while GFM uses the low-frequency LL sub-band to capture frame-level brightness flicker. These masks are fused by element-wise maximum and then combined with the original RTN spatial indicator to form the recurrent gate mask.

## 1. Empirical Validation Is Not Strong Enough

The principal weakness is that the empirical validation is not yet strong enough. The main quantitative claim rests on SRWOV synthetic data, with RTN and WaveletMask reported as single training runs and without multi-seed confidence intervals or paired per-clip statistical tests.

## 2. Comparison Is Only Partially Controlled

The comparison with state-of-the-art methods is only partially controlled. RTN and WaveletMask are retrained under the authors' protocol, whereas several competing methods are taken from published benchmark values. The manuscript acknowledges this caveat, but the table still visually positions WaveletMask as the best PSNR method, which may overstate the strength of the cross-method comparison.

## 3. Technical Limitations in the Detector Design

There are also technical limitations in the detector design. The wavelet-domain differences are computed across neighboring frames without explicit motion compensation, so motion boundaries, occlusions, camera movement, and scene cuts can produce false mask responses. The authors acknowledge that the downstream RTN flow alignment may absorb part of this issue, but the mask itself is not motion-invariant.

## 4. Suggested Frequency-Operator Ablation

A useful addition would be an ablation table containing: Haar DWT, Sobel/Laplacian high-frequency masks, Gaussian-pyramid low-frequency flicker masks, Fourier low-pass/high-pass temporal differences, and learned shallow frequency filters. This would clarify whether the contribution is genuinely wavelet-specific.
