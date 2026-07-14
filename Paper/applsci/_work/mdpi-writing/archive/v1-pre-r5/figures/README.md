# Figure Work Directory Contract

All non-submission figure work for the 007 MDPI draft stays under this
directory. `Paper/applsci/figures/FigureX.pdf` is only the LaTeX-consumed
asset path; finality is controlled by `../figure_manifest.yaml` and the
per-figure contract.

Each `FigureX/` directory uses the same compact layout:

- `source/`: trace records, selections, metadata, layout JSON, and raw asset
  references.
- `draft/`: non-final candidates, legacy promotions, prompts, sketches, and
  temporary exports.
- `export/`: outputs from the owning figure skill that are candidates for
  promotion to `Paper/applsci/figures/FigureX.pdf`.

Route ownership:

- Figure2 and Figure3: `ofr-paper-diagram` Route E.
- Figure6, Figure7, Figure8, and Figure9: `ofr-paper-visual-studio`.
- Figure5: must be explicitly routed as either real mask visualization or
  schematic before final export.
- Figure4: optional; keep, supplement, or drop must be recorded before final
  submission.
