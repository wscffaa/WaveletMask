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
