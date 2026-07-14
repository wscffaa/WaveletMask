Dear Reviewer 1,

We thank Reviewer 1 for the thorough re-evaluation and the recommendation of "Accept with Minor Revisions." All four remaining points are editorial or disclosure issues; no new experiments are required. We have added the requested transparency notes and qualifications to the revised manuscript, and we detail each change below.

Comments 1: All ten methods were re-trained by the authors — commendable. But re-training raises a symmetric fairness question: were the external methods (e.g., MambaOFR, RVRT, VRT) trained with their officially recommended configurations and budgets? One appendix sentence (or a pointer to the per-method configs in the released evidence package) stating that official configurations/iteration budgets were used would close this loop.

Response 1: We agree that symmetric transparency strengthens the comparison. We added a sentence in Appendix D.1 (Complexity and Ablation Details) stating that every external method in the unified benchmark was trained with its officially recommended configuration and the same iteration budget as the shared-backbone RTN baseline, and that the per-method YAML configuration files are included in the released evidence package.

Revision made: Appendix D.1 now reads: "All external methods in the unified benchmark (Table 3) were trained with their officially recommended configurations under a common iteration budget; the corresponding per-method configuration files are provided in the released evidence package."

Comments 2: My round-1 comment asked the authors to "include" subjective assessment. The study was completed and is statistically convincing, but the main text now contains zero pointer to it; it survives only as back-matter Supplementary declarations. I accept the journal-convention argument, but a single anchor sentence in Section 4.4.2 ("An expert-level subjective study with 34 trained participants, reported in Supplementary Tables S1–S2, corroborates this ranking") would cost two lines and remove any impression that the requested evidence was buried.

Response 2: Thank you for this practical suggestion. We added a neutral anchor sentence in Section 4.4.2 that points readers to the Supplementary Materials without reintroducing the subjective percentages into the main text.

Revision made: The last paragraph of Section 4.4.2 now ends with: "An expert-level subjective evaluation with 34 trained raters is reported in the Supplementary Materials (Tables S1–S2 and Figure S4) and corroborates the perceptual ranking inferred from the no-reference metrics."

Comments 3: In Table 4, WaveletMask's second place on CLIPIQA+ (0.4356) leads MambaOFR (0.4351) by 0.0005. The conclusion's "the only method placing within the top two on all three no-reference metrics" is technically true but hinges on a margin far below metric noise. Suggest qualifying with "(narrowly on CLIPIQA+)" or rounding-consistent wording.

Response 3: We agree. We qualified the conclusion so that the narrow CLIPIQA+ margin is explicit.

Revision made: The Conclusions section now reads: "WaveletMask is the only method placing within the top two on all three no-reference metrics (narrowly on CLIPIQA+, where it leads the next-best method by 0.0005)."

Comments 4: The caption of Appendix Table A1 discloses that WaveletMask's 26.6125 (paired) vs 26.6017 (unified) differ across protocols. The same discrepancy is larger for RTN — 26.1648 (paired/ablation, Table 5) vs 25.8514 (unified, Table 3) — yet is only implicitly covered. Add one explicit sentence to prevent a careless reader from reading Table 3's RTN row as the paired baseline.

Response 4: We agree that this asymmetry could mislead readers. We added an explicit sentence in Section 4.4.1 noting that the RTN values in the paired protocol (Table A1 / Table 5) and the unified benchmark (Table 3) come from different training snapshots and metric configurations.

Revision made: Section 4.4.1 now includes: "As with WaveletMask, the RTN paired snapshot (26.1648 dB in Table A1 and Table 5) and the unified benchmark snapshot (25.8514 dB in Table 3) differ because the two protocols use different training snapshots and metric configurations; the paired value is the controlled baseline for the gate comparison."

We sincerely thank Reviewer 1 for the constructive and rigorous evaluation, which has substantially improved the manuscript. A marked-up version with all changes highlighted has been uploaded together with the revised manuscript.

Sincerely,
The Authors
