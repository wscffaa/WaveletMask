# Goal: Rewrite `draw_clean_evidence` to match HTML Variant C

## Task

Rewrite the `draw_clean_evidence` function in `projects/007-waveletmask/Paper/applsci/scripts/extract_fig1_motivation_from_l40.py` to exactly match the HTML layout at `projects/007-waveletmask/Paper/_work/mdpi-writing/figures/fig1_layout_options.html` (Variant C). The HTML is the ground truth.

## Layout (2×5 horizontal grid)

```
Row 1 (Scratch):
  Col 0: Input (curr.png cropped to ROI)
  Col 1: Pixel cue (pixel_indicator.png cropped to ROI, Blues cmap, percentile normalized)
  Col 2: LH / HL / HH vertical stack — each cell = coefficient map (top) + center-cropped input (bottom)
  Col 3: m_local (local.png cropped to ROI, inferno cmap)
  Col 4: Fused (fused.png cropped to ROI, plasma cmap)

Row 2 (Flicker):
  Col 0: Input (curr.png, full image)
  Col 1: Pixel cue (pixel_indicator.png, Blues cmap, percentile normalized)
  Col 2: LL coefficient map (top) + center-cropped input (bottom)
  Col 3: m_flicker (flicker.png, viridis cmap)
  Col 4: Fused (fused.png, plasma cmap)
```

## Canvas

`MDPI_TEXTWIDTH_IN` (5.46") × 2.83"

## Grid

- 5 equal-width columns, 2 rows
- Column gap: 0.012
- Row gap: 0.06
- Margins: left=0.045, right=0.015, bottom=0.06, top=0.08

## Colormaps (MUST match existing functions)

| Panel | Colormap | Reference |
|-------|----------|-----------|
| LH/HL/HH coefficients | `"magma"` | `response_triptych` line 547 |
| LL coefficient | `"viridis"` | `ll_response_panel` line 595 |
| Pixel cue | `"Blues"`, percentile normalized via `normalize01` | |
| m_local | `"inferno"` | |
| m_flicker | `"viridis"` | |
| Fused | `"plasma"` | |
| Input panels | plain RGB via `image_array()` | |

## Wavelet Cell Structure (Col 2)

- **Row 1**: 3 subcells stacked vertically (LH top, HL middle, HH bottom). Each subcell: coefficient map (top 52%) + center-cropped input (bottom 48%).
- **Row 2**: 1 cell. LL coefficient (top 52%) + center-cropped input (bottom 48%).

## Center-crop Logic

Crop the center of the input image (`curr.png`) to match coefficient image dimensions:

```python
def center_crop(image: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    h, w = image.shape[:2]
    th, tw = target_hw
    y0 = (h - th) // 2
    x0 = (w - tw) // 2
    return image[y0:y0 + th, x0:x0 + tw]
```

- Synthetic: crop 312×176 center from 624×352 input
- Real: crop 232×176 center from 464×352 input

## Labels

Each panel has a badge label in the top-left corner:
- 9pt bold, white text
- Semi-transparent black background: `bbox=dict(boxstyle="round,pad=0.18", facecolor=(0,0,0,0.55), edgecolor="none")`
- Labels: "Input", "Pixel cue", "LH", "HL", "HH", "LL", "m_local", "m_flicker", "fused"

## NO Decorations

No header rows, no (a)/(b) row labels, no rotated text, no colored borders, no accent bars, no flow arrows.

## Existing Helper Functions (reuse, DO NOT redefine)

- `draw_cell(ax, image, ...)` — imshow + border
- `fit_panel(rgb, size, mode, pad_color)` — resize/pad image
- `panel_rgb(array, cmap, fixed_vmax)` — apply colormap to 2D array
- `normalize01(array)` — percentile normalization
- `image_array(pil_image)` — PIL to float32 RGB
- `crop_arr(array, roi)` — crop numpy array to ROI
- `crop_image(pil_image, roi)` — crop PIL image to ROI
- `axis_panel_size(fig_w, fig_h, ax_w, ax_h)` — calculate pixel size
- `MDPI_TEXTWIDTH_IN` — 5.46 inches

## Data Inputs (in `synth` and `real` dicts)

- `synth["images"]["curr"]` — PIL Image 624×352
- `synth["maps"]["pixel_indicator"]` — numpy 624×352
- `synth["maps"]["lh_diff"]`, `["hl_diff"]`, `["hh_diff"]` — numpy 312×176
- `synth["maps"]["local"]` — numpy 624×352
- `synth["maps"]["fused"]` — numpy 624×352
- `synth["roi"]` — (336, 192, 432, 288)
- `synth["maps"]["lh_diff"]` shape after `crop_arr` with `synth_half_roi` — 312×176
- `real["images"]["curr"]` — PIL Image 464×352
- `real["maps"]["pixel_indicator"]` — numpy 464×352
- `real["maps"]["ll_diff"]` — numpy 232×176
- `real["maps"]["flicker"]` — numpy 464×352
- `real["maps"]["fused"]` — numpy 464×352

## Scope

Only modify:
1. `draw_clean_evidence` function (current lines 925-1023)
2. `metadata["panel_grid"]` cols → 5
3. `metadata["layout_version"]` → `"mdpi_v6_horizontal_2x5"`

Do NOT touch any other function.

## Validation

After rewriting, run:
```bash
ssh L40_aliyun 'cd /mnt/data/Antigravity_OFR && source ~/miniconda3/etc/profile.d/conda.sh && conda activate rtn && python projects/007-waveletmask/Paper/applsci/scripts/extract_fig1_motivation_from_l40.py --skip-candidates 2>&1'
```

Then open the HTML preview in browser alongside the generated PDF. Visual layout, colormaps, label positions, and image content must match.

## Report Format

```
Changed files:
- <file>:<line_range> — <what changed>

Validation results:
- <command>: <output snippet>

Residual risks:
- <risk>

Blocked items:
- <item or "none">
```
