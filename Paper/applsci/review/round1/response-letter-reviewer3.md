Dear Reviewer 3,

Thank you for the detailed and technically specific assessment. We have revised the empirical evidence and the wording of the claims to make the scope of WaveletMask clearer. Below we provide point-by-point responses in the same order as the review report.

General note on evidence placement in this revision: to follow the journal's main-text conventions, the revised Results section is consolidated to four tables and the main text to seven figures, while all evidence added during this revision is retained in full. Statistical-reliability details appear in Appendix A, robustness and cross-content generalization in Appendix B, detector diagnostic probes in Appendix C, and complexity details with full ablation sweeps in Appendix D, all within the same manuscript PDF. The unified re-training benchmark (Table 3) is the main-text comparison table, while the strict paired same-protocol comparison (Table A1) is reported separately in Appendix A; the two therefore remain distinct tables, exactly as committed in this letter. Every relocated item keeps its original label, is explicitly referenced from the corresponding Results subsection, and its key numbers remain quoted in the main text.

Comments 1: The principal weakness is that the empirical validation is not yet strong enough. The main quantitative claim rests on SRWOV synthetic data, with RTN and WaveletMask reported as single training runs and without multi-seed confidence intervals or paired per-clip statistical tests.

Response 1: Thank you for identifying this key weakness. We added a three-seed same-protocol re-test and paired clip-level statistical analysis. Under the seed-2021 same-protocol comparison, WaveletMask improves RTN by +0.4477 dB PSNR with a 95% confidence interval of [0.2955, 0.5981] and a paired permutation test of p = 5 × 10^−5^. SSIM, LPIPS, and DISTS also show statistically significant paired improvements:

| Metric | Mean improvement | 95% bootstrap CI | Permutation p |
|---|---:|---:|---:|
| PSNR | +0.4477 dB | [0.2955, 0.5981] | 5 × 10^−5^ |
| SSIM | +0.0031 | [0.0014, 0.0050] | 0.0015 |
| LPIPS | +0.0137 | [0.0108, 0.0166] | 5 × 10^−5^ |
| DISTS | +0.0119 | [0.0103, 0.0136] | 5 × 10^−5^ |

We also added a three-seed summary (Table A2) reporting mean and standard deviation over seeds 2021/2022/2023: the RTN reaches 24.56 ± 1.67 dB PSNR versus 26.22 ± 0.37 dB for WaveletMask (mean difference +1.66 dB), with consistent improvements in SSIM, LPIPS, and DISTS. WaveletMask is also markedly more stable across seeds, with a 4.5× smaller PSNR standard deviation. We deliberately conduct the formal paired test on seed 2021, where the RTN attains its strongest result among the three seeds; the reported +0.4477 dB is therefore a conservative estimate of the same-protocol improvement rather than a favorable selection.

Revision made: We added the three-seed same-protocol summary table (Table A2) and paired clip-level statistics table (Table A1) in Appendix A, referenced from the Results section, whose comparison subsection quotes the key statistics explicitly. The revised subsection now reads: "WaveletMask improves PSNR by +0.45 dB, all four paired clip-level gains are statistically significant under two-sided permutation tests (p ≤ 1.5 × 10^−3^), and a three-seed re-test confirms that the advantage persists with markedly higher stability than the baseline (26.22 ± 0.37 vs 24.56 ± 1.67 dB PSNR; Appendix A, Tables A1-A2)."

Comments 2: The comparison with state-of-the-art methods is only partially controlled. RTN and WaveletMask are retrained under the authors' protocol, whereas several competing methods are taken from published benchmark values. The manuscript acknowledges this caveat, but the table still visually positions WaveletMask as the best PSNR method, which may overstate the strength of the cross-method comparison.

Response 2: We agree with this concern and revised the comparison structure. The controlled claim is now limited to the same-protocol RTN versus WaveletMask. Literature-reported methods are placed in a separate context table and are explicitly described as non-identical protocol results. We removed language that could imply a fully controlled cross-method ranking.

The revised manuscript now distinguishes two evidential roles: the same-protocol table supports the direct RTN/WaveletMask claim, while the literature-context table only situates the result relative to prior reported values.

Revision made: We split the original comparison into the same-protocol table (Table A1) and the literature-context table (Table 3) and revised the Results and Discussion wording to avoid cross-protocol superiority claims.

Comments 3: There are also technical limitations in the detector design. The wavelet-domain differences are computed across neighboring frames without explicit motion compensation, so motion boundaries, occlusions, camera movement, and scene cuts can produce false mask responses. The authors acknowledge that the downstream RTN flow alignment may absorb part of this issue, but the mask itself is not motion-invariant.

Response 3: We agree and revised the manuscript to state this as a boundary condition, now backed by a direct quantification. We added failure-mode probes for scene cuts, exposure flashes, illumination ramps, and persistent scratches, and a new motion- and occlusion-binned benchmark on the 30 fixed synthetic clips using RAFT flow statistics (mean magnitude, camera/object split via the median flow vector, and forward-backward occlusion ratio). The binned result shows that the restoration improvement of WaveletMask over the same-protocol RTN does not collapse under stronger motion: +0.085 dB in the low-motion tercile, +0.477 dB in the medium tercile, and +0.769 dB in the high tercile (Spearman 0.58 between clip motion magnitude and PSNR improvement; occlusion terciles +0.186/+0.498/+0.647 dB). Detector-level clean-region false positives are essentially uncorrelated with motion for LHFM (Spearman −0.04) and only weakly correlated for GFM (0.30).

We also trained the flow-compensated WaveletMask pilot. The pilot was unstable and trailed the standard WaveletMask model by approximately 2.6 dB on average across three seeds, suggesting that pre-compensating the wavelet detector can amplify optical-flow errors into high-frequency sub-band artifacts.

The revised text therefore avoids claiming motion invariance. It states that the mask itself is not motion-invariant, but that on the present benchmark the downstream flow-aligned recurrent propagation absorbs the motion-induced detector noise, with abrupt transitions and stationary image-plane defects remaining the clearest boundary cases. We also state the caveats explicitly: the benchmark is camera-motion dominated (29 of 30 clips), and the bins are relative terciles.

Revision made: We added the motion/occlusion-binned analysis in Appendix B (Appendix B.2, Table A4), and the GFM failure-mode probe (Appendix C.1, Table A6), persistent-scratch probe (Appendix C.2, Table A7), and cue-complementarity/motion-related analysis (Appendix C.4) in Appendix C, all referenced from the Results section with summary conclusion sentences, and revised the Discussion and Limitations sections to state these boundary conditions explicitly.

Comments 4: A useful addition would be an ablation table containing: Haar DWT, Sobel/Laplacian high-frequency masks, Gaussian-pyramid low-frequency flicker masks, Fourier low-pass/high-pass temporal differences, and learned shallow frequency filters. This would clarify whether the contribution is genuinely wavelet-specific.

Response 4: Thank you for this precise suggestion. We added a new cross-domain frequency-operator diagnostic on the SRWOV/REDS residual-proxy setting. The experiment uses the same 30 clips and 20 sampled frames per clip as the mask interpretability analysis. Because generator-side artifact masks are unavailable, labels are derived from paired LQ-GT residual proxies, and the results are treated as diagnostic rather than as supervised artifact segmentation.

The Haar row in this diagnostic measures single-frame wavelet sub-band energy rather than the temporal geometric-mean differencing used by the deployed LHFM/GFM modules. Therefore, this diagnostic isolates frequency-operator behavior from the full restoration-gating pipeline. The end-to-end restoration ablation remains the primary evidence that Haar-based temporal gating is effective in the proposed model.

| Operator | AUPRC | Top-5 hit | Top-5 non-proxy |
|---|---:|---:|---:|
| Haar fused | 0.3166 | 0.3685 | 0.6315 |
| Sobel | 0.3062 | 0.3470 | 0.6530 |
| Laplacian | 0.2993 | 0.3352 | 0.6648 |
| Gaussian fused | 0.3270 | 0.3815 | 0.6185 |
| Fourier temporal fused | 0.3885 | 0.4384 | 0.5616 |
| Learned shallow frequency, five-fold mean | 0.4265 | 0.5278 | 0.4722 |

The added result narrows the claim. Haar is competitive with Sobel and Laplacian and close to the Gaussian-pyramid diagnostic, but Fourier temporal responses and the proxy-supervised learned shallow probe align more strongly with residual-proxy labels. We therefore revised the manuscript to avoid claiming that Haar is a universally strongest frequency detector. Our claim is now that Haar is a lightweight, zero-parameter, locally supported wavelet choice that integrates stably into the current restoration-gating recipe, while Fourier-style and supervised learned frequency probes are valuable alternatives for future detector designs.

Revision made: We added the cross-domain frequency diagnostic table (Table A10) in Appendix C, referenced from the ablation study in the Results section, and revised the wavelet-family analysis in the Discussion (Section 5.1) to state the narrower conclusion.

We appreciate the reviewer's rigorous technical suggestions. The revised manuscript now provides stronger statistical validation, clearer comparison boundaries, explicit motion/failure-mode limitations, and the requested cross-domain frequency-operator diagnostic.

Sincerely,
The Authors
