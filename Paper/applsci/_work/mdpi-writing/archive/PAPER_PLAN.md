# PAPER PLAN — WaveletMask OFR (修订版)

> Project: 007-waveletmask
> Target: Applied Sciences (MDPI)
> Status: Draft (revision in progress)
> Revised: 2026-04-23
> Revision reason: L40 权威数据确认 AB 为最优，C 模块有害，论文从 3→2 创新

## Story Arc

**Task**: Old Film Restoration (OFR) — restoring vintage footage degraded by scratches, dust, flicker, and noise.

**Challenge**: Recurrent video restoration networks (e.g., RTN) rely on pixel-level frame differencing as degradation indicator, which (1) confuses motion with film-specific noise (scratches, dust), and (2) cannot detect global brightness flicker.

**Insight**: Haar wavelet transform naturally decomposes frames into frequency sub-bands — high-frequency sub-bands (LH/HL/HH) isolate local structural damage, while the low-frequency sub-band (LL) captures global brightness drift. Geometric-mean temporal filtering naturally suppresses motion while preserving transient noise.

**Method**: WaveletMask — two complementary modules: (A) Local High-Frequency Mask via geometric-mean differencing on HF sub-bands, (B) Global Flicker Mask via min-strategy on LL sub-bands, fused by element-wise max.

**Result**: +0.44 dB PSNR, -0.0143 LPIPS, -0.0122 DISTS on synthetic benchmarks, with zero additional trainable parameters. Super-additive complementarity: A+B > A + B individually.

## Claims–Evidence Matrix

| # | Claim | Evidence | Section |
|---|-------|----------|---------|
| C1 | Pixel-domain differencing confuses motion with film noise | Spatial mask responds to both scratches and motion equally | §1, §3.1 |
| C2 | Wavelet HF sub-bands with geometric mean effectively detect local noise | local (26.38) > base (26.16) = +0.22 dB; geometric mean filters motion | §3.2, §4.3 |
| C3 | LL sub-band with min strategy detects flicker while excluding scene changes | flicker SSIM (0.8980) > base SSIM (0.8917) = +0.0063 | §3.3, §4.3 |
| C4 | A and B exhibit super-additive complementarity | ab (+0.44) > local (+0.22) + flicker (-0.06) = +0.16 | §4.3 |
| C5 | Simple max fusion outperforms all learned fusion variants | ab (26.60) > gated (26.35) > local_fusion (25.99) > flicker_fusion (25.95) > full (24.98) | §4.3, §5 |
| C6 | WaveletMask is plug-and-play with zero extra trainable parameters | Only replaces indicator signal, no backbone modification | §3.4 |
| C7 | Computational overhead is negligible | ~5.5K params in lightweight CNN, ~0.16% extra FLOPs | §4.4 |

## Section Plan

### §1 Introduction (~1.5 pages)
- P1: OFR task importance + degradation diversity
- P2: Current approach (RTN pixel differencing) + two blind spots (noise-motion confusion, flicker invisibility)
- P3: Wavelet domain insight + our approach overview (A local + B global + max fusion)
- P4: Contribution list (4 items)

### §2 Related Work (~1 page)
- §2.1: Old Film Restoration methods (RTN, RRTN, DeepRemaster, MambaOFR)
- §2.2: Wavelet transforms in image/video processing
- §2.3: Degradation-aware mechanisms in restoration

### §3 Materials and Methods (~2 pages)
- §3.1: Problem Formulation + RTN baseline review + spatial mask limitations
- §3.2: Local High-Frequency Mask (Innovation A) — Haar DWT → HF sub-bands → geometric mean → lightweight CNN → sigmoid
- §3.3: Global Flicker Mask (Innovation B) — LL sub-band → bidirectional min → lightweight CNN → sigmoid
- §3.4: Mask Fusion and Integration — max fusion rationale + initialization strategy + architecture diagram

### §4 Results (~2 pages)
- §4.1: Experimental setup (datasets, metrics, training details)
- §4.2: Main results (synthetic quantitative + real-world qualitative)
- §4.3: Ablation study — Main table (4 variants) + Fusion strategy comparison (5 variants)
- §4.4: Computational complexity analysis

### §5 Discussion (~0.5 page)
- Super-additive complementarity analysis
- "Learned fusion is not always better" — gradient analysis of why CNN alpha harms performance
- Limitations and future work (more backbones, real-world paired data)

### §6 Conclusions (~0.5 page)

## Figure Plan

| # | Type | Content | Status |
|---|------|---------|--------|
| F1 | Motivation | Wavelet vs pixel-domain degradation sensing (Figure 1) | ✅ Final |
| F2 | Pipeline | WaveletMask architecture overview (Figure 2) | ✅ Final |
| F3 | Module | WaveletNoiseMask module detail: HaarDWT2d → high_freq_conv/low_freq_conv → max fusion (Figure 3) | ✅ Final |
| F4 | Comparison | Qualitative synthetic SOTA: RTN/RVRT/MambaOFR/WaveletMask-ab (Figure 4) | ✅ Final |
| F5 | Comparison | Qualitative real-world SOTA (Figure 5) | ✅ Final |
| F6 | Ablation | Ablation visual comparison: base/local/flicker/ab (Figure 6) | ✅ Final |

## Table Plan

| # | Type | Content | Data Source |
|---|------|---------|-------------|
| T1 | Main results | Synthetic PSNR/SSIM/LPIPS/DISTS (4 variants) | ablation_results.yaml |
| T2 | Main results | Real-world NIQE/BRISQUE/CLIPIQA | ablation_results.yaml |
| T3 | Ablation | Fusion strategy comparison (max vs 4 learned variants) | ablation_results.yaml |
| T4 | Complexity | Parameters + FLOPs + inference time | To measure |
