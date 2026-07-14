Dear Reviewer 2,

Thank you for your detailed and insightful comments. We have addressed each point with new experiments, additional analyses, and more explicit manuscript revisions. Below we provide point-by-point responses in the same order as the review report.

General note on evidence placement in this revision: to follow the journal's main-text conventions, the revised Results section is consolidated to four tables and the main text to seven figures, while all evidence added during this revision is retained in full. Statistical-reliability details appear in Appendix A, robustness and cross-content generalization in Appendix B, detector diagnostic probes in Appendix C, and complexity details with full ablation sweeps in Appendix D, all within the same manuscript PDF; the expert-level subjective study is reported in full in the Supplementary Materials (Tables S1--S2 and Figure S4). Every item relocated to the appendices keeps its original label, is explicitly referenced from the corresponding Results subsection, and its key numbers remain quoted in the main text.

Comments 1: The method assumes that scratches and dust are transient artifacts. However, many real archival films contain persistent scratches spanning multiple consecutive frames. Please include a dedicated experiment quantifying performance on persistent damage and analyze whether the geometric-mean criterion in LHFM suppresses such defects.

Response 1: Thank you for raising this important boundary case. We added a controlled detector-response probe with synthetic scratches of persistence length L in {1, 2, 3, 5}, evaluated under both fixed image-plane and drifting modes.

| Mode | L | Raw LHFM suppression | Wavelet-mask relative response |
|---|---:|---:|---:|
| Fixed | 1 | 0% baseline | 100% |
| Fixed | 2 | 46.5% | 1.7% |
| Fixed | 3 | 65.0% | 1.1% |
| Fixed | 5 | 65.0% | 1.1% |
| Drift | 2 | −3.6% no suppression | 13.6% |
| Drift | 3 | −11.0% slight increase | 4.2% |
| Drift | 5 | −11.0% slight increase | 4.2% |

Responses saturate beyond L = 3: the L = 5 rows match the L = 3 rows, indicating that the suppression reaches a stable asymptote rather than growing without bound. The manuscript table (Table A7) reports all four lengths.

The probe confirms the expected LHFM limitation. For image-plane-fixed scratches, geometric-mean differencing suppresses the response when the same scratch persists across adjacent frames. Drifting scratches do not exhibit this suppression because the bilateral differences remain nonzero. The revised manuscript therefore treats persistent fixed scratches as a boundary condition of the current detector, not as a solved case.

Revision made: We added the "Persistent-Scratch Response Probe" subsection (Appendix C.2) and the persistent-scratch table (Table A7) in Appendix C, referenced from the Results section, together with the corresponding canonical-case analysis in Method (Section 3.6) and a limitation statement in Discussion.

Comments 2: The wavelet-domain differences are computed without explicit motion compensation, whereas the RTN spatial indicator relies on optical-flow alignment. Please investigate the impact of camera motion, object motion, and occlusions on LHFM and GFM responses, and provide a motion-aware variant or an ablation using flow-compensated wavelet differences.

Response 2: We addressed this concern with a dedicated motion/occlusion-binned benchmark plus two complementary analyses.

First, we added a motion- and occlusion-binned evaluation on the 30 fixed synthetic clips. We computed RAFT optical-flow statistics on the clean ground-truth frames of each clip: mean flow magnitude, a camera-motion proxy (median flow vector), an object-motion proxy (residual flow after removing the median), and a forward-backward occlusion ratio. Clips were stratified into terciles, and we aggregated both the restoration metrics of the same-protocol RTN and WaveletMask runs and the detector's clean-region false-positive behaviour (fraction of top-5% activation outside the residual proxy):

| Group | Bin | Clips | dPSNR | LHFM clean-FP | GFM clean-FP |
|---|---|---:|---:|---:|---:|
| Motion | Low | 10 | +0.085 | 0.646 | 0.440 |
| Motion | Medium | 10 | +0.477 | 0.782 | 0.627 |
| Motion | High | 10 | +0.769 | 0.681 | 0.584 |
| Occlusion | Low | 10 | +0.186 | 0.716 | 0.515 |
| Occlusion | Medium | 10 | +0.498 | 0.703 | 0.541 |
| Occlusion | High | 10 | +0.647 | 0.690 | 0.594 |

The improvement does not collapse under stronger motion; it is largest in the high-motion and high-occlusion terciles (clip-level Spearman 0.58 between motion magnitude and PSNR improvement, 0.49 for occlusion), and SSIM/LPIPS/DISTS improve in every bin. At the detector level, LHFM clean-region false positives are essentially uncorrelated with motion magnitude (Spearman −0.04) and GFM increases only weakly (0.30). We state two caveats explicitly in the manuscript: the REDS-derived clips are camera-motion dominated (29 of 30 under the median-flow split), so object-dominant motion remains under-represented, and the bins are relative terciles rather than absolute thresholds.

Second, we added a flow-aligned cue complementarity diagnostic using RAFT optical flow. The wavelet cues are largely complementary to the RTN spatial cue rather than redundant with it. In particular, the local and flicker branches have Spearman correlation −0.3364 and top-5% overlap 0.0024, indicating nearly disjoint response regions.

Third, we implemented and trained the requested flow-compensated WaveletMask variant as a pilot across three seeds. The results are unstable: seed 2021 = 23.5 dB, seed 2022 = 22.2 dB, and seed 2023 = 26.2 dB. On average, the flow-compensated approach trails the standard WaveletMask by approximately 2.6 dB. This suggests that optical-flow errors can be amplified after wavelet decomposition, because small spatial misalignments in the warped frame become high-frequency boundary artifacts across sub-bands. The current uncompensated design avoids this noise path by applying DWT frame-wise, and the binned benchmark above confirms that the residual motion sensitivity of the detector is absorbed by the downstream flow-aligned recurrent propagation rather than degrading restoration quality.

Revision made: We added the "Motion- and Occlusion-Binned Analysis" subsection in Appendix B (Appendix B.2, Table A4), the cue-complementarity analysis in Appendix C (Appendix C.4, Table A9), and the flow-compensated pilot discussion in the same Appendix C block, all referenced from the Results section with summary conclusion sentences, and revised the Discussion and Limitations sections accordingly (the limitation now scopes the remaining gap to object-dominant motion and absolute-threshold characterization rather than to motion in general).

Comments 3: The choice of the Haar wavelet is justified mainly by simplicity and efficiency. However, the paper does not demonstrate that Haar is the most suitable decomposition for old-film artifacts. An ablation comparing Haar against Daubechies, Symlets, or dual-tree complex wavelets would significantly strengthen the technical contribution.

Response 3: We added a wavelet-family ablation covering the requested alternatives:

| Wavelet | PSNR | SSIM | LPIPS | DISTS |
|---|---:|---:|---:|---:|
| Haar 1-level | 26.6125 | 0.8954 | 0.0886 | 0.0827 |
| Haar 2-level | 26.5175 | 0.9011 | 0.0847 | 0.0760 |
| DTCWT | 26.4506 | 0.8992 | 0.0862 | 0.0780 |
| Coiflet-1 | 25.8533 | 0.8874 | 0.1026 | 0.0893 |
| Symlet-2 | 24.2087 | 0.8509 | 0.1507 | 0.1226 |
| Daubechies-2 | 17.9208 | 0.6473 | 0.4136 | 0.3154 |

Single-level Haar gives the best PSNR under the implemented restoration-gating recipe. DTCWT and Haar 2-level remain competitive and improve several perceptual metrics at a small PSNR cost, while Coiflet-1 degrades more clearly and longer-support wavelets suffer severe failure in the current lightweight detector. We retain single-level Haar as the default because PSNR is the primary fidelity criterion of the same-protocol comparison and the one-level transform is the cheapest and most spatially localized option; the revised Discussion (Section 5.1) explicitly notes that two-level Haar is a reasonable alternative when perceptual metrics are prioritized. We have revised the manuscript to state this as a scoped empirical result, not as a universal claim that Haar is always the best decomposition.

Revision made: We added the wavelet-family rows in the follow-up ablation table (Table A12, full sweep in Appendix D), with the key wavelet-family contrasts retained in the main ablation table (Table 5, panel (c)), and revised the wavelet-family analysis in the Discussion (Section 5.1).

Comments 4: The GFM branch is intended to capture brightness flicker, but the proposed minimum-difference criterion may also respond strongly to scene cuts, flashes, and abrupt illumination changes. Please provide a dedicated evaluation on clips containing scene transitions and discuss potential mitigation strategies such as cut detection or temporal consistency constraints.

Response 4: We added a GFM failure-mode probe using 12 seven-frame synthetic cases per condition:

| Condition | N | Activation ratio vs clean | Guard suppression |
|---|---:|---:|---:|
| Scene cut | 12 | 2.35× | 61.4% |
| Exposure flash | 12 | 10.41× | 94.7% |
| Illumination ramp | 12 | 1.52× | 2.5% |

A simple luminance-difference guard suppresses most flash responses and partially suppresses scene-cut responses. Gradual illumination changes produce much weaker false activation. We use controlled synthetic seven-frame transitions rather than in-the-wild footage because this isolates the GFM response from confounding content changes and allows exact activation ratios to be measured against a clean baseline of the same content. We do not incorporate the guard into the current model because the revision keeps the proposed method as a lightweight gating enhancement, but we now discuss cut-aware GFM as a practical mitigation direction.

Revision made: We added the "GFM Failure-Mode Probe" subsection (Appendix C.1) and the GFM failure-mode table (Table A6) in Appendix C, referenced from the Results section, together with the GFM canonical-case discussion in Method (Section 3.6) and the corresponding boundary-condition discussion in Discussion.

Comments 5: The paper claims that wavelet-domain cues provide complementary information to pixel-domain cues, but this claim remains largely qualitative. It would be valuable to include response visualizations and quantitative correlation analyses between the RTN spatial indicator, LHFM output, and GFM output to demonstrate their complementarity more rigorously.

Response 5: We added a flow-aligned cue complementarity diagnostic using RAFT optical flow:

| Cue pair | Spearman ρ | Top-5% overlap | Top-5% Jaccard |
|---|---:|---:|---:|
| Spatial / Local | −0.0418 | 0.1641 | 0.0938 |
| Spatial / Flicker | 0.4850 | 0.2300 | 0.1408 |
| Local / Flicker | −0.3364 | 0.0024 | 0.0012 |

The local and flicker branches respond to nearly disjoint regions, and the spatial indicator has only weak overlap with the local branch. We also added diagnostic visualizations showing persistent-scratch examples, residual-proxy overlap examples, and cue-complementarity heatmaps.

Revision made: We added the "Cue Complementarity Analysis" subsection (Appendix C.4) and the cue-complementarity table (Table A9) in Appendix C, referenced from the Results section, the probe visualization overview as Supplementary Figure S6, and the corresponding interpretation in Discussion (Section 5.1).

Comments 6: The learned fusion experiments appear limited to a single CNN architecture and initialization strategy. Since the conclusion that max fusion is preferable depends heavily on this design choice, please evaluate stronger alternatives such as attention-based fusion, uncertainty-aware fusion, normalization strategies, or residual fusion mechanisms before drawing broader conclusions.

Response 6: We appreciate this suggestion and agree that the preference for max fusion should be tested against stronger learned alternatives before drawing broader conclusions. We expanded the fusion analysis with six alternatives covering attention-based, residual, scalar-gate, normalization-aware, and initialization-variant designs:

| Fusion method | PSNR | Δ vs max |
|---|---:|---:|
| Max proposed | 26.6125 | 0.00 |
| Bias-0 initialization | 26.5000 | −0.11 |
| Residual fusion | 25.9777 | −0.63 |
| Scalar gate | 25.2400 | −1.37 |
| Normalization-aware | 25.0100 | −1.60 |
| Bias-2 initialization | 23.8400 | −2.77 |
| Channel attention | 23.2479 | −3.36 |

All six tested learned fusion alternatives underperform the parameter-free max operation. We also narrowed the conclusion: the manuscript now states that max fusion is the most reliable choice under the tested configurations and training recipe, not that learned fusion is generally inferior.

Regarding uncertainty-aware fusion specifically: we did not implement a fully probabilistic head that predicts per-pixel variance. The closest tested designs are the scalar-gate variant, which learns a confidence weight over the wavelet cue, and the normalization-aware variant, which calibrates cue scales before fusion; both underperform max fusion (−1.37 dB and −1.60 dB). Given the consistent direction across all six variants, we consider it unlikely that a variance-predicting head would reverse the conclusion under the same training recipe. We list uncertainty-calibrated learned fusion as a concrete future direction in the revised Discussion.

Revision made: We revised the fusion-strategy analysis (Appendix D.1), the fusion-comparison table (Table A11), and the follow-up ablation table (Table A12), now reported in full in Appendix D with the key fusion contrasts retained in the main ablation table (Table 5, panel (b)), and the corresponding Discussion paragraph.

Comments 7: When introducing wavelets, please refer to https://link.springer.com/article/10.1007/s40534-025-00393-5

Response 7: Thank you for the reference suggestion. We added the cited work by Bernardini, Bono, and Collina (Railway Engineering Science, 2025) in the wavelet-introduction paragraph and included the full BibTeX entry.

Revision made: We added the citation in the wavelet-related discussion in the Introduction/Related Work material and added the corresponding BibTeX entry `bernardini2025driveby` in `references.bib`. The added sentence in the Introduction reads: "More broadly, wavelet coefficients have been used as localized damage evidence in signal-analysis detection settings, including continuous-wavelet-transform features coupled with sparse autoencoders (Bernardini et al., 2025)."

Comments 8: The quantitative evaluation relies almost entirely on the SRWOV synthetic benchmark. Given that the target application is real archival film restoration, additional experiments on independent real-world datasets or publicly available restoration benchmarks would be important to assess generalization beyond the SRWOV degradation model.

Response 8: We agree that generalization beyond the SRWOV synthetic setting should be addressed carefully. To our knowledge, no publicly available paired old-film video restoration benchmark with clean ground truth exists beyond SRWOV/REDS-based protocols; real archival footage normally lacks clean references. We therefore added a DAVIS synthetic benchmark as a cross-content generalization test. DAVIS uses different video content, including outdoor sports, animals, and vehicles, while keeping the same degradation pipeline so that full-reference evaluation remains possible.

| Model | PSNR | SSIM | LPIPS | DISTS |
|---|---:|---:|---:|---:|
| RTN | 25.8417 | 0.8827 | 0.1223 | 0.1089 |
| WaveletMask | 25.9883 | 0.8795 | 0.1206 | 0.1048 |
| Improvement | +0.15 | −0.003 | +0.002 | +0.004 |

For LPIPS and DISTS, positive values in the improvement row denote error reduction. The result supports cross-content robustness under the same degradation pipeline. We explicitly state that it does not establish true cross-degradation generalization on independently degraded archival footage, which remains future work.

In addition to the DAVIS cross-content test, the revised manuscript adds an internal expert subjective evaluation on 14 real archival film clips (34 internal expert raters, 476 rankings; Supplementary Table S1), where WaveletMask obtains 45.17% overall preference versus 12.82% for RTN. The DAVIS test addresses quantitative cross-content generalization, while the real-footage study addresses perceived quality on genuinely degraded archival material; together they cover the two aspects of this concern that are measurable in the absence of paired ground truth.

Revision made: We added the "Cross-Content Generalization" subsection (Appendix B.3) and the DAVIS cross-content table (Table A5) in Appendix B, referenced from the Results section with a summary conclusion sentence, and a scoped limitation statement in Discussion.

Comments 9: The computational analysis reports parameters and FLOPs but omits runtime, latency, memory consumption, and throughput. Since one motivation of the method is its lightweight design, practical deployment metrics on representative hardware should be reported and compared against RTN and other baselines.

Response 9: We added a runtime benchmark covering all five compared methods on a single NVIDIA L40 GPU using PyTorch 2.7.0, CUDA 12.6, FP32 evaluation mode, seven-frame clips, three warm-up passes, and ten timed runs, all under one unified protocol (batch size 1, 256 × 256). Parameters and FLOPs cover the full executable forward pass and include the RAFT flow estimator for the flow-based methods; DeepRemaster uses no flow estimator.

| Method | Params M | FLOPs G | Latency ms/clip | Throughput fps | Peak memory GB |
|---|---:|---:|---:|---:|---:|
| DeepRemaster | 9.8781 | 914.0 | 17.9 | 391.6 | 0.35 |
| RTN | 5.7351 | 2184.7 | 169.5 | 41.3 | 0.24 |
| RRTN | 7.0503 | 2782.2 | 298.4 | 23.5 | 0.75 |
| MambaOFR | 8.2984 | 2832.8 | 404.8 | 17.3 | 0.60 |
| WaveletMask | 5.7360 | 2184.9 | 171.4 | 40.8 | 0.24 |

WaveletMask adds only 898 detector parameters and 0.1494 G FLOPs to the RTN, and leaves latency and peak memory essentially unchanged (171.4 versus 169.5 ms per clip, identical peak memory). Across methods, DeepRemaster is fastest because it uses no optical-flow estimator but carries the most parameters, whereas the recurrent flow-based methods (RTN, RRTN, MambaOFR, WaveletMask) share the RAFT cost; WaveletMask is the lightest of the flow-based group and adds negligible overhead to its RTN backbone. In the revised manuscript, the main-text cost table reports the batch-1 256 × 256 setting alongside parameters and FLOPs, the 720p batch-1 cost (about 5.4 s per seven-frame clip at 9.9 GB peak memory for both RTN and WaveletMask, hence not yet real-time at high resolution) is quoted in the accompanying text, and the remaining batch/resolution measurements are documented with the full protocol in Appendix D.

All five methods were re-trained in our unified benchmark pipeline, so the wall-clock figures above are measured on identical hardware and protocol rather than copied from heterogeneous sources; absolute latency therefore also reflects each method's implementation, and the deployment-relevant quantity for the proposed gate is the negligible overhead it adds to the RTN host network.

Revision made: We expanded the "Computational Complexity" subsection (Section 4.6) and consolidated the complexity comparison and the runtime benchmark into a single main-text cost table (Table 6), with the measurement protocol, the remaining batch/resolution settings, and the module-overhead table (Table A13) documented in Appendix D.

Comments 10: The detector masks are trained only through restoration losses and receive no explicit supervision. As a result, it is unclear whether the generated masks actually correspond to degradation locations. Please provide qualitative mask visualizations, overlap analyses with synthetic degradation annotations, or other interpretability studies to validate that the learned masks behave as intended.

Response 10: We added a mask interpretability probe using 30 clips and 20 sampled frames per clip. Because the fixed SRWOV/REDS synthetic split does not expose generator-side pixel annotations, we use paired input-ground-truth residual proxies rather than true artifact masks. The results are:

| Branch | AUPRC | Best F1 | Inside/outside ratio | Top-5% localization |
|---|---:|---:|---:|---:|
| Local | 0.055 | 0.152 | 0.61 | 0.041 |
| Flicker | 0.489 | 0.495 | 3.57 | 0.685 |
| Wavelet fused | 0.494 | 0.500 | 3.56 | 0.691 |

The flicker and fused wavelet masks are well concentrated in residual-proxy regions, with top-5% hit rate around 0.69 versus a random-area baseline around 0.29. The local branch does not align well with this high-pass residual proxy. We now explicitly acknowledge this as a limitation and state that generator-side or hand-labeled artifact masks would be needed for a stronger pixel-level localization study. We also added qualitative heatmap visualizations.

Revision made: We added the "Mask Interpretability and Residual-Proxy Overlap" subsection (Appendix C.3) and the mask-interpretability table (Table A8) in Appendix C, referenced from the Results section, the probe visualization overview as Supplementary Figure S6, and the corresponding limitation discussion.

We sincerely thank Reviewer 2 for the detailed and constructive feedback. The revised manuscript addresses the raised concerns with dedicated experiments, quantitative diagnostics, and explicit acknowledgment of remaining limitations.

Sincerely,
The Authors
