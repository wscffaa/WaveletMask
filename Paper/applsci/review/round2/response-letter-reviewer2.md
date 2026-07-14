Dear Reviewer 2,

We thank Reviewer 2 for the positive re-evaluation and the recommendation of "Accept." The three remaining notes are editorial; we have implemented the two actionable suggestions and confirmed that the third is already handled in the revised manuscript.

Comments 1: C10's residual tension deserves one sentence of reinforcement: LHFM's top-5% hit (0.041) is below its random-area baseline (0.082), meaning the local branch is "anti-localized" on this proxy, yet the end-to-end model still gains +0.21 dB. The current explanation is acceptable, but adding "the high-pass residual proxy itself cannot distinguish true scratches from texture edges" in Appendix C would make the argument more complete.

Response 1: Thank you for this precise suggestion. We added the proposed sentence to Appendix C.3 to make the proxy limitation explicit.

Revision made: Appendix C.3 now includes: "The high-pass residual proxy itself cannot distinguish true scratches from texture edges, which partly explains why the local branch's top-5% hit rate on this proxy falls below its random-area baseline while still contributing to end-to-end restoration."

Comments 2: Appendix D runtime protocol states "batch size 4 at 720p exceeded the available memory for both models in this session" — "in this session" is colloquial. Suggest "under the 48 GB L40 configuration used here" and include the memory ceiling.

Response 2: We agree. We replaced the colloquial phrasing and added the GPU memory ceiling.

Revision made: The sentence in Appendix D now reads: "Batch size 4 at 720p exceeded the available 48 GB L40 GPU memory for both models under the configuration used here."

Comments 3: Table 4's RTN (unified retrain snapshot) BRISQUE 19.00 versus Appendix D Table's (paired snapshot) 29.31 difference is already disclosed and supports the concern about NR-metric sensitivity. This is handled appropriately.

Response 3: We thank the reviewer for confirming that this discrepancy is adequately disclosed. We retained the existing disclosure in Section 4.4.2 and Appendix D without further change.

Revision made: No manuscript change required.

We sincerely thank Reviewer 2 for the detailed and constructive feedback, which has substantially improved the manuscript. A marked-up version with all changes highlighted has been uploaded together with the revised manuscript.

Sincerely,
The Authors
