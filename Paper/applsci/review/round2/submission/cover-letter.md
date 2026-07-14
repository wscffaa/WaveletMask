# Cover Letter

**Manuscript Title:** WaveletMask: Wavelet-Domain Mask-Guided Degradation Detection for Old Film Restoration

**Journal:** Applied Sciences (MDPI)

**Submission type:** Resubmission after major revision (Reject & Resubmit, Round 1)

**Previous Manuscript ID:** applsci-4348902

---

Dear Editor,

We thank you and the three reviewers for the constructive and detailed evaluation of our manuscript. This submission is a comprehensively revised version of our previous manuscript applsci-4348902, which received a "Reject and Encourage Resubmission" decision; we kindly ask that it be linked to the original submission. We have carefully addressed every comment and substantially strengthened the experimental evidence, the statistical rigor, and the scope of our claims. A point-by-point response to each reviewer is provided in the accompanying response documents, and all changes are marked in the highlighted version of the manuscript. Below we summarize the principal revisions.

## Principal Revisions

1. **Statistical reliability (Reviewer 1, Comment 1; Reviewer 3).** The original single-run results have been replaced by a strict paired controlled comparison against the shared-backbone RTN baseline, reported with clip-level bootstrap confidence intervals and two-sided paired permutation tests, together with a three-seed re-test. The proposed gate improves PSNR by +0.45 dB with p = 5×10⁻⁵, and the advantage persists across all three seeds with markedly lower variance than the baseline (Appendix A).

2. **Fair comparison protocol (Reviewer 1, Comments 2 and 9; Reviewer 3).** The original mixed-protocol table has been split into two distinct tables: a strict same-protocol RTN-versus-WaveletMask comparison (the controlled evidence for the proposed gate) and a unified re-training benchmark in which all ten methods were re-trained and re-evaluated by us under one pipeline. Direct ranking claims are confined to the same-protocol table; the unified benchmark is presented as comparative context, with its protocol differences disclosed in the caption.

3. **Real-world validation via an internal expert subjective study (Reviewer 1, Comment 3).** We added an internal expert-level subjective evaluation with 34 domain-specialized internal raters over 14 real archival clips. WaveletMask receives 45.17% of overall preferences and ranks first in all three perceptual dimensions, with significance confirmed by exact binomial and chi-square tests; the complete study is reported in the Supplementary Materials (Tables S1--S2 and Figure S4).

4. **Artifact-type breakdown (Reviewer 1, Comment 5).** We added a degradation-stratified analysis showing that WaveletMask improves over RTN across scratch-, dust-, flicker-dominant, and mixed strata (Appendix B).

5. **Wavelet-choice justification (Reviewer 1, Comment 6).** We added a wavelet-family sweep (Haar single/two-level, DTCWT, Coiflet, Symlet, Daubechies) and now present single-level Haar as a scoped empirical choice rather than a universal claim (Appendix D).

6. **Failure-mode and robustness experiments (Reviewer 1, Comment 7; Reviewer 2; Reviewer 3).** We added dedicated diagnostic probes quantifying the GFM scene-cut/flash failure mode, the LHFM persistent-scratch limitation, mask interpretability against residual proxies, cue complementarity, motion/occlusion-binned robustness, a cross-content DAVIS generalization test, and a cross-domain frequency-operator diagnostic (Appendices B and C).

7. **Theoretical and design justification of LHFM and GFM (Reviewer 1, Comment 4; Reviewer 2).** We expanded the methodological motivation for the geometric-mean and minimum-difference operators and the parameter-free maximum fusion rule, supported by a full fusion-strategy ablation (Appendix D).

8. **No-reference metric discussion (Reviewer 1, Table Review).** We added an explicit discussion of why BRISQUE, NIQE, and CLIPIQA+ can disagree on real footage, and we report a degraded-input anchor so that restoration gains over the unrestored input are explicit.

To follow the journal's main-text conventions, the revised Results section is consolidated to a compact set of main tables and figures, while all evidence added during this revision is retained in full in Appendices A–D within the same manuscript PDF and in the Supplementary Materials. Every relocated item keeps its original label and is referenced from the corresponding Results subsection.

We believe these revisions resolve the reviewers' concerns and significantly improve the rigor and completeness of the manuscript. We hope it is now suitable for publication in Applied Sciences and thank you again for the opportunity to revise.

## Ethics Statement

The expert subjective evaluation added in this revision was an internal visual-quality scoring of restored images by internal expert raters. All algorithmic experiments were conducted on publicly available video datasets for old-film restoration evaluation. The internal scoring did not recruit external human subjects, collect human biological samples, record identifiable personal or private information, or involve any intervention on the human body. In accordance with Article 2 of the *Measures for Scientific and Technological Ethics Review (Trial)* (Guo Ke Fa Jian [2023] No. 167; jointly issued by the Ministry of Science and Technology and nine other Chinese authorities), this work does not fall within the scope requiring mandatory ethics review, and no ethics approval or exemption document is required.

## Declarations

We confirm that neither the manuscript nor any parts of its content are currently under consideration for publication with, or published in, another journal. All authors have approved the manuscript and agree with its submission to Applied Sciences. We acknowledge that this work was previously submitted to Applied Sciences (Manuscript ID: applsci-4348902), and the present resubmission addresses all reviewer and editor comments from that round.

Sincerely,

The Authors
