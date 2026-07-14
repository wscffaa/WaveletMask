# 007-WaveletMask Round-2 Response Letter Diagnosis

## Current Situation

- Round-1 response letters exist at `review/round1/response-letter-reviewer{1,2,3}.md` and have been converted to docx in `review/round2/submission/`.
- Simulated round-2 reviews are available at `review/round2/simulated-review/reviewer{1,2,3}-round2.md`.
- No dedicated round-2 response letters have been written yet.

## Non-Standard Elements in the Existing Round-1 Response Letters

1. **Missing MDPI template sections.** The letters jump directly to point-by-point responses without the standard MDPI structure: (1) Summary, (2) General Evaluation, (3) Point-by-point response, (4) English language response, (5) Additional clarifications.
2. **Non-template preface.** Each letter opens with a "General note on evidence placement" paragraph that belongs in the Summary section or in the cover letter, not as a standalone preface to the response letter.
3. **Excessive length for some points.** Several individual responses exceed the recommended half-page limit and could be tightened while preserving evidence.
4. **Inconsistent action subsections.** The `Revision made` subsections are not uniformly present across all three letters.
5. **Formatting of mathematical notation.** Although already corrected in a later pass, the source files must continue to use proper Unicode notation (×, ², ±, ≥, ≤, ρ, Δ) rather than ASCII equivalents.

## Round-2 Task

Write concise, MDPI-template-compliant response letters for the simulated round-2 reviews:

- **Reviewer 1:** Accept with Minor Revisions → address N1–N4.
- **Reviewer 2:** Accept → address E1–E3 (E3 is already handled, so mainly E1–E2).
- **Reviewer 3:** Accept → address optional N1 (no mandatory change required, but acknowledge and restate future-work placement).

All responses should use the `Comments N / Response N / Revision made` structure within the MDPI template, avoid internal LaTeX labels, and keep each point concise.
