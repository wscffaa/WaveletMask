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
