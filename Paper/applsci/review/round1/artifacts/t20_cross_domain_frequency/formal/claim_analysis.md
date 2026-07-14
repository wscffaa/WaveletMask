# T20 Claim Analysis

## Evidence

- Experiment: cross-domain frequency decomposition diagnostic on SRWOV/REDS residual proxies.
- Data: 30 clips, 20 sampled non-boundary frames per clip.
- Labels: residual-derived `LQ-GT` proxy regions; not generator-side artifact annotations.
- Deterministic operators: Haar DWT, Sobel, Laplacian, Gaussian pyramid, Fourier temporal.
- Learned operator: shallow frequency probe, five-fold clip-level split, proxy-supervised.
- Final table: `cross_domain_frequency_ablation.csv`.

## Key Results

| Operator | AUPRC | Top-5 hit | Top-5 non-proxy |
|---|---:|---:|---:|
| Haar fused | 0.3166 | 0.3685 | 0.6315 |
| Sobel | 0.3062 | 0.3470 | 0.6530 |
| Laplacian | 0.2993 | 0.3352 | 0.6648 |
| Gaussian fused | 0.3270 | 0.3815 | 0.6185 |
| Fourier temporal fused | 0.3885 | 0.4384 | 0.5616 |
| Learned shallow frequency, five-fold mean | 0.4265 | 0.5278 | 0.4722 |

Best F1 is retained in the raw CSV/JSON but is not used as the display or ranking metric. For several deterministic operators, the best-F1 operating point degenerates to a near-all-positive threshold, so AUPRC and Top-5 localization are the safer diagnostic metrics.

## Interpretation

The result does not support a broad claim that Haar DWT is the strongest frozen residual-proxy detector across all frequency operators. Fourier temporal responses outperform Haar fused on AUPRC, Top-5 hit, and Top-5 non-proxy rate. The proxy-supervised learned shallow filter is stronger still.

The Haar fused row measures single-frame wavelet sub-band energy. It does not implement the temporal geometric-mean differencing used by the deployed LHFM/GFM modules. The diagnostic therefore isolates the frequency-operator choice; the end-to-end restoration ablation remains the evidence for the full Haar-based temporal gating pipeline.

The defensible claim is narrower:

- Haar is competitive with Sobel/Laplacian and close to Gaussian pyramid in this frozen diagnostic.
- Haar remains the best single-level wavelet choice in the end-to-end restoration ablation.
- Haar has the intended engineering properties for the deployed module: zero parameters, local support, low overhead, no proxy-label supervision, and stable integration into the RTN-style gate.
- Fourier and learned shallow filters should be framed as useful future detector alternatives, not as failed baselines.

## Response Strategy

Use a moderate, evidence-first tone. State that we added the requested cross-domain diagnostic and revised the manuscript to avoid overclaiming wavelet specificity. Do not claim that Haar is universally superior. Emphasize that the paper's main claim is restoration-gating effectiveness under a lightweight zero-parameter frequency prior, not supervised residual-proxy segmentation.

## Manuscript Changes

- Added Table `cross_domain_frequency_ablation` to `Paper/applsci/sections/results.tex`.
- Updated `Paper/applsci/sections/discussion.tex` to explicitly state that Fourier temporal and proxy-supervised learned probes can align more strongly with residual proxies.
