Dear Reviewer 1,

Thank you for your thorough and constructive evaluation. We have carefully addressed each concern with additional experiments, revised wording, and clearer manuscript positioning. Below we provide point-by-point responses in the same order as the review report.

General note on evidence placement in this revision: to follow the journal's main-text conventions, the revised Results section is consolidated to four tables and the main text to seven figures, while all evidence added during this revision is retained in full. Statistical-reliability details appear in Appendix A, robustness and cross-content generalization in Appendix B, detector diagnostic probes in Appendix C, and complexity details with full ablation sweeps in Appendix D, all within the same manuscript PDF; the expert-level subjective study is reported in full in the Supplementary Materials (Tables S1--S2 and Figure S4), following the placement convention of recent dedicated old-film-restoration work (MambaOFR, CVPR 2025), whose user study likewise appears in its supplementary materials. Every item relocated to the appendices keeps its original label, is explicitly referenced from the corresponding Results subsection, and its key numbers remain quoted in the main text.

Comments 1: The most serious weakness of the manuscript is the absence of statistical validation. The authors explicitly state that all reported results are based on single-run experiments and that no multi-seed evaluation, confidence intervals, or significance testing were performed. This is particularly problematic because the reported improvement over RTN is only +0.44 dB PSNR. Without multiple random seeds, standard deviations, confidence intervals, paired statistical significance tests it is impossible to determine whether the reported gain is statistically meaningful or simply due to training randomness.

Response 1: Thank you for pointing out this important methodological weakness. We addressed it by training both the RTN and WaveletMask AB across three independent random seeds (2021, 2022, and 2023). We also added paired clip-level statistical testing on the seed-2021 same-protocol comparison, using 30 SRWOV/REDS synthetic clips, 20,000 bootstrap draws, and 20,000 paired permutation draws.

The formal seed-2021 paired statistics are:

| Metric | Mean improvement | 95% bootstrap CI | Permutation p |
|---|---:|---:|---:|
| PSNR | +0.4477 dB | [0.2955, 0.5981] | 5 × 10^−5^ |
| SSIM | +0.0031 | [0.0014, 0.0050] | 0.0015 |
| LPIPS | +0.0137 | [0.0108, 0.0166] | 5 × 10^−5^ |
| DISTS | +0.0119 | [0.0103, 0.0136] | 5 × 10^−5^ |

All four metrics show statistically significant improvement, and all confidence intervals exclude zero.

The three-seed summary (Table A2) further reports mean and standard deviation over seeds 2021/2022/2023: the RTN reaches 24.56 ± 1.67 dB PSNR versus 26.22 ± 0.37 dB for WaveletMask (mean difference +1.66 dB), with consistent improvements in SSIM (0.8808 ± 0.0129 vs 0.8983 ± 0.0045), LPIPS, and DISTS. WaveletMask is also markedly more stable across seeds, with a 4.5× smaller PSNR standard deviation. We deliberately conduct the formal paired test on seed 2021, where the RTN attains its strongest result among the three seeds; the reported +0.4477 dB is therefore a conservative estimate of the same-protocol improvement rather than a favorable selection.

Revision made: We added the three-seed summary table (Table A2) and paired clip-level statistics table (Table A1) in Appendix A, referenced from the Results section, and revised the text in the "Comparison with State-of-the-Art" subsection to quote the key statistics explicitly. The revised subsection now reads: "WaveletMask improves PSNR by +0.45 dB, all four paired clip-level gains are statistically significant under two-sided permutation tests (p ≤ 1.5 × 10^−3^), and a three-seed re-test confirms that the advantage persists with markedly higher stability than the baseline (26.22 ± 0.37 vs 24.56 ± 1.67 dB PSNR; Appendix A, Tables A1-A2)."

Comments 2: Table 2 combines retrained RTN and retrained WaveletMask with results from other methods reported in the literature. This introduces a substantial protocol mismatch because training datasets may differ, degradation synthesis procedures may differ, optimization settings may differ and evaluation scripts may differ. Therefore, claims that WaveletMask outperforms methods such as MambaOFR, RVRT, or RRTN are not scientifically justified.

Response 2: We agree with this concern. We re-checked the provenance of every external entry and revised the comparison protocol to separate controlled evidence from broader context. The strict same-protocol claim is now limited to the RTN versus WaveletMask under the same data, degradation pipeline, training recipe, and metric scripts. All external methods were re-trained and re-evaluated by us on the same SRWOV synthetic split under a unified benchmark pipeline; they are placed in a separate context table whose caption states this provenance explicitly and notes that the unified pipeline differs from the strict paired protocol in training snapshots and metric configuration, so the two tables are not merged into a single ranking.

Accordingly, we removed cross-protocol superiority language. The manuscript no longer claims that WaveletMask fairly outranks MambaOFR, RVRT, RRTN, or the other externally re-trained methods under an identical protocol. Instead, the direct claim is restricted to the strict paired RTN/WaveletMask comparison.

Revision made: The original comparison table was split into the "Same-protocol seed-2021 comparison" table (Table A1) and the "Unified re-training benchmark" context table (Table 3), whose caption now states that all listed methods were re-trained and re-evaluated by us under a unified benchmark pipeline. We also added an explicit protocol-scope statement in the Results section and Discussion.

Comments 3: The real-world evaluation relies entirely on NIQE, BRISQUE, CLIPIQA and qualitative figures. The authors correctly note that these metrics are content-sensitive and not conclusive. The authors must include subjective visual quality assessment, user preference study and expert restoration evaluation.

Response 3: Thank you for this important suggestion. We added an internal expert subjective quality evaluation with 34 domain-specialized internal raters. Each rater ranked four methods (WaveletMask, MambaOFR, RRTN, and RTN) on 14 real old-film clips across three perceptual dimensions and overall best visual experience.

The overall preference results are:

| Method | Votes | Percent |
|---|---:|---:|
| WaveletMask | 215 | 45.17% |
| MambaOFR | 114 | 23.95% |
| RRTN | 86 | 18.07% |
| RTN | 61 | 12.82% |

The rank-1 counts by perceptual dimension also favor WaveletMask:

| Dimension | WaveletMask | MambaOFR | RRTN | RTN |
|---|---:|---:|---:|---:|
| Degradation removal | 264 (55.5%) | 112 | 70 | 30 |
| Style preservation | 247 (51.9%) | 132 | 67 | 30 |
| Temporal consistency | 273 (57.4%) | 107 | 69 | 27 |

WaveletMask is preferred 3.5× more often than RTN and achieves the highest rank-1 count in all three perceptual dimensions. These results are statistically significant: the overall preference distribution deviates strongly from the uniform chance distribution (χ²(3) = 115.1, p = 8.9 × 10^−25^), and the WaveletMask share of 45.17% (Wilson 95% CI [40.8%, 49.7%]) is significantly above the 25% chance level (one-sided exact binomial test, p = 1.1 × 10^−21^); the per-dimension rank-1 distributions are likewise significantly non-uniform (all χ²(3) ≥ 228.4, p < 1 × 10^−48^). We state in the manuscript that these tests treat the 476 votes as independent draws without modelling rater-level clustering. Because the internal panel consists of domain-specialized raters and records only method-preference scores for restored images, this study serves as an expert-level subjective visual assessment rather than a general observer test. We acknowledge that a dedicated panel of professional film restorers would further strengthen this evidence; the revised Discussion lists broader validation on real archival collections as future work. In the revised manuscript, the no-reference metrics and qualitative comparisons constitute the real-world evidence in the main text, and the complete internal expert-level subjective study is reported in the Supplementary Materials as convergent human confirmation of the same conclusion.

Revision made: We have completed the requested internal expert visual-preference study (34 internal expert raters × 14 clips × 3 perceptual dimensions, 476 responses; WaveletMask ranks first with 45.2% overall preference against the 25% chance level, one-sided exact binomial p = 1.1 × 10^−21^). The complete protocol, result tables, figures, and statistical tests are provided as Supplementary Tables S1--S2 and Figure S4, following the main-text convention of dedicated old-film-restoration work in this area (MambaOFR, CVPR 2025), whose user study likewise appears in its supplementary materials rather than in the main text. We also revised the Discussion to explain why NIQE, BRISQUE, and CLIPIQA should be interpreted as content-sensitive diagnostics that are read jointly rather than individually. The main-text anchor in the Results section now reads: "As an additional perceptual assessment on the real-world clips, an expert-level subjective evaluation is reported in the Supplementary Materials (Tables S1 and S2 and Figure S4)."

Comments 4: The proposed LHFM and GFM modules are largely heuristic. The manuscript states that geometric-mean differencing highlights transient defects and minimum temporal difference highlights flicker. However, no theoretical analysis is provided explaining why these specific operators are optimal or superior to alternatives.

Response 4: We revised the manuscript to avoid implying that LHFM and GFM are theoretically optimal operators. Instead, we now present them as lightweight, frequency-domain heuristics with explicit operating assumptions and boundary conditions. We added a "Response Analysis" subsection that analyzes canonical temporal cases for LHFM and GFM.

For LHFM, the geometric mean behaves as a bilateral agreement rule: a transient scratch or dust particle at the current frame produces strong disagreement with both temporal neighbors, whereas one-sided residuals are attenuated. The same analysis also explains the limitation for persistent image-plane-fixed scratches, where both temporal differences can vanish.

For GFM, the minimum rule emphasizes low-frequency deviations that are present relative to both temporal neighbors, which is consistent with flicker-like frame-level brightness deviations. We also explicitly state that scene cuts and flashes can trigger false responses and quantify this behavior in the failure-mode probe.

Revision made: We revised the Method section by adding the "Response Analysis" subsection (Section 3.6) and cross-referencing the persistent-scratch probe (Table A7) and GFM failure-mode probe (Table A6), both reported in Appendix C. We also revised the Discussion to describe the operators as scoped, interpretable heuristics rather than optimal detectors.

Comments 5: The paper claims improvements for scratches, dust, and flicker, yet no separate evaluation is reported for each degradation category.

Response 5: We added an artifact-dominant synthetic breakdown by stratifying the 30 SRWOV/REDS synthetic clips using paired input-ground-truth residual descriptors. These descriptors identify scratch-dominant, dust-dominant, flicker-dominant, and mixed-residual clip strata. The results show that WaveletMask improves over RTN in all four strata:

| Stratum | Clips | ΔPSNR | ΔSSIM | ΔLPIPS | ΔDISTS |
|---|---:|---:|---:|---:|---:|
| Scratch-dominant | 5 | +0.64 | +0.0020 | +0.0110 | +0.0116 |
| Dust-dominant | 8 | +0.45 | +0.0049 | +0.0196 | +0.0120 |
| Flicker-dominant | 6 | +0.47 | +0.0055 | +0.0176 | +0.0153 |
| Mixed residual | 11 | +0.34 | +0.0017 | +0.0102 | +0.0109 |

The manuscript explicitly states that these are residual-derived clip strata rather than pixel-level generator labels.

Revision made: We added the "Artifact-Dominant Synthetic Breakdown" subsection and the artifact-breakdown table (Table A3) in Appendix B, referenced from the Results section with a summary conclusion sentence, and the wording limits the claim to residual-derived clip-level strata.

Comments 6: The entire framework relies on a single-level Haar wavelet decomposition. No comparison is provided against Daubechies wavelets, Symlets, Coiflets and multi-level wavelet decompositions.

Response 6: We appreciate this suggestion and agree that the choice of single-level Haar should be justified empirically rather than asserted. We added a wavelet-family ablation covering the requested alternatives, including two-level Haar, DTCWT, Coiflet-1, Symlet-2, and Daubechies-2:

| Variant | PSNR | SSIM | LPIPS | DISTS |
|---|---:|---:|---:|---:|
| Haar 1-level | 26.6125 | 0.8954 | 0.0886 | 0.0827 |
| Haar 2-level | 26.5175 | 0.9011 | 0.0847 | 0.0760 |
| DTCWT | 26.4506 | 0.8992 | 0.0862 | 0.0780 |
| Coiflet-1 | 25.8533 | 0.8874 | 0.1026 | 0.0893 |
| Symlet-2 | 24.2087 | 0.8509 | 0.1507 | 0.1226 |
| Daubechies-2 | 17.9208 | 0.6473 | 0.4136 | 0.3154 |

Single-level Haar gives the best PSNR under the implemented restoration-gating recipe. Haar 2-level and DTCWT remain competitive and improve several perceptual metrics at a small PSNR cost, whereas longer-support wavelets are less stable in the current lightweight detector. We retain single-level Haar as the default because PSNR is the primary fidelity criterion of the same-protocol comparison and the one-level transform is the cheapest and most spatially localized option; the revised Discussion (Section 5.1) explicitly notes that two-level Haar is a reasonable alternative when perceptual metrics are prioritized. We now state this as a scoped empirical conclusion, not as a universal claim that Haar is always superior.

Revision made: We added the wavelet-family rows in the follow-up ablation table (Table A12, full sweep in Appendix D), with the key wavelet-family contrasts retained in the main ablation table (Table 5, panel (c)), and revised the wavelet-family analysis in the Discussion (Section 5.1).

Comments 7: The manuscript repeatedly mentions that scene cuts remain problematic, difficult motion can generate false responses and persistent scratches may be missed. However, no dedicated experiments quantify these failure modes.

Response 7: We added dedicated analyses for the main failure modes.

For scene cuts, flashes, and illumination changes, the GFM failure-mode probe reports:

| Condition | Activation ratio vs clean | Guard suppression |
|---|---:|---:|
| Scene cut | 2.35× | 61.4% |
| Exposure flash | 10.41× | 94.7% |
| Illumination ramp | 1.52× | 2.5% |

For persistent scratches, fixed-position scratches with persistence length L ≥ 2 are suppressed by 46-65% in the raw LHFM response, and the final wavelet mask response drops to approximately 1.7% of the single-frame response. Drifting scratches are not suppressed in the same way. We now explicitly identify persistent fixed scratches as a boundary condition.

For motion-related behavior, we added a flow-compensated WaveletMask pilot. Across three seeds, the flow-compensated variant is unstable and trails the standard WaveletMask model by approximately 2.6 dB on average. We interpret this as evidence that optical-flow errors can be amplified after wavelet decomposition into high-frequency sub-band artifacts. We therefore avoid claiming motion invariance and retain complex motion and occlusion as limitations.

Revision made: We added the GFM failure-mode probe (Appendix C.1, Table A6) and the persistent-scratch response probe (Appendix C.2, Table A7) in Appendix C, referenced from the Results section, together with the probe visualization overview (Supplementary Figure S6), and revised the Discussion limitations subsection (Section 5.3).

Comments 8: The ablation study only examines LHFM, GFM and fusion strategies.

Response 8: We expanded the ablation and diagnostic evidence in three directions.

First, we added mask interpretability analysis against paired residual-proxy regions. The flicker and fused wavelet masks align strongly with residual-proxy regions, with the fused wavelet mask reaching AUPRC 0.4943 and top-5% localization 0.6908 versus a 0.2907 random-area baseline. The local branch does not align well with this proxy, which we now explicitly acknowledge as a limitation of the current unsupervised mask design and residual-proxy evaluation.

Second, we added stronger fusion alternatives. Six learned fusion alternatives were tested, including attention-based, residual, scalar-gate, normalization-aware, and initialization-variant designs. All tested learned fusion variants underperform the parameter-free max operation; the closest alternative, bias-0 initialization, remains 0.11 dB below the proposed max fusion.

Third, we added a cross-domain frequency-operator diagnostic comparing Haar DWT, Sobel/Laplacian filters, Gaussian-pyramid responses, Fourier temporal responses, and learned shallow frequency filters. This diagnostic narrows the claim: Haar is not presented as a universally strongest frozen detector; instead, it is a lightweight, stable wavelet choice for the end-to-end restoration-gating recipe.

Revision made: We added the mask interpretability subsection and table (Table A8), the cue-complementarity subsection and table (Table A9), and the cross-domain frequency diagnostic table (Table A10) in Appendix C, and the expanded fusion and wavelet-family ablations in the follow-up ablation table (Table A12) in Appendix D; all are referenced from the Results section, whose main ablation table retains the key fusion and wavelet-family contrasts.

Comments 9: Table 2: Results originate from different experimental protocols. Direct ranking is therefore questionable.

Response 9: We agree. This table-level concern is addressed together with Comment 2. The revised manuscript separates the strict paired RTN/WaveletMask comparison from the unified re-training benchmark context. Direct ranking is now only made within the same-protocol table, while the context table is explicitly labeled as a unified re-training benchmark whose training snapshots and metric configuration are not identical to the paired protocol.

Revision made: We split the original comparison into the same-protocol table (Table A1) and the unified re-training benchmark context table (Table 3) and revised the Results and Discussion text to prevent cross-protocol ranking claims.

Comments 10: Table 3: The no-reference metrics are inconsistent: LHFM achieves best BRISQUE, GFM achieves best NIQE and WaveletMask is not best on any metric. This weakens claims regarding real-world superiority. The authors should discuss this contradiction in greater depth.

Response 10: Thank you for identifying this ambiguity. We revised the real-world evaluation interpretation. The manuscript now explains that BRISQUE, NIQE, and CLIPIQA are based on different priors and can disagree in old-film restoration because restoration changes grain, contrast, local sharpness, and semantic plausibility simultaneously. We therefore no longer treat the no-reference metrics as decisive evidence of real-world superiority.

The ambiguity is further resolved by the convergent evidence of the expert-level subjective evaluation reported in the Supplementary Materials (Tables S1--S2 and Figure S4): WaveletMask obtains 45.17% overall preference, compared with 23.95% for MambaOFR, 18.07% for RRTN, and 12.82% for RTN, and also ranks first in degradation removal, style preservation, and temporal consistency. The no-reference metrics are now described as content-sensitive diagnostics that are interpreted jointly with the qualitative comparisons rather than individually.

Revision made: We revised the no-reference evaluation subsection: the unified main-text table now reports all ten methods together with a degraded-input anchor (all values measured by us, Table 4), and the RTN/LHFM/GFM variant panel is reported separately under the strict paired protocol in Appendix D (Table A14). We also added the expert-level subjective study as Supplementary Tables S1--S2 and Figure S4, and revised the Discussion paragraph on the joint interpretation of no-reference metrics.

We appreciate the reviewer's rigorous evaluation. The revised manuscript now includes statistical validation, protocol-fair comparisons, expert-level subjective evaluation, artifact-specific and failure-mode diagnostics, and a more conservative interpretation of WaveletMask's scope.

Sincerely,
The Authors
