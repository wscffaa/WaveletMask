#!/usr/bin/env python3
"""Build MDPI-ready diagnostic composite figures for T5/T8/T11 panels."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont, JpegImagePlugin  # noqa: F401


DPI = 600
CANVAS_W = 4200
MARGIN_X = 100
TOP_MARGIN = 120
BOTTOM_MARGIN = 120
MAIN_ROW_LABEL_W = 0
SUPP_ROW_LABEL_W = 280
COL_GAP = 18
ROW_GAP = 24
GROUP_GAP = 110
GROUP_HEADER_H = 104
COL_LABEL_H = 110
BORDER = (176, 184, 196)
TEXT = (18, 24, 33)
MUTED = (74, 85, 104)
BG = (255, 255, 255)
HEADER_BG = (244, 247, 250)


@dataclass(frozen=True)
class FigureSet:
    key: str
    panel_label: str
    title: str
    examples_dir: Path
    cases: tuple[str, ...]
    main_cases: tuple[str, ...]
    columns: tuple[tuple[str, str], ...]


def paper_dir() -> Path:
    return Path(__file__).resolve().parents[3]


def round_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
    )
    for name in names:
        path = Path(name)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


GROUP_FONT = font(92, bold=True)
LABEL_FONT = font(76, bold=True)
ROW_FONT = font(72)


def text_size(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.multiline_textbbox((0, 0), text, font=fnt, spacing=4, align="center")
    return box[2] - box[0], box[3] - box[1]


def center_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    fnt: ImageFont.ImageFont,
    fill: tuple[int, int, int] = TEXT,
    spacing: int = 4,
) -> None:
    w, h = text_size(draw, text, fnt)
    x0, y0, x1, y1 = box
    draw.multiline_text(
        (x0 + (x1 - x0 - w) // 2, y0 + (y1 - y0 - h) // 2),
        text,
        font=fnt,
        fill=fill,
        align="center",
        spacing=spacing,
    )


def left_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    fnt: ImageFont.ImageFont,
    fill: tuple[int, int, int] = TEXT,
) -> None:
    draw.text(xy, text, font=fnt, fill=fill)


def tile_size(row_label_w: int) -> int:
    return (CANVAS_W - 2 * MARGIN_X - row_label_w - 5 * COL_GAP) // 6


def load_tile(path: Path, size: int) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(path)
    with Image.open(path) as im:
        return im.convert("RGB").resize((size, size), Image.Resampling.LANCZOS)


def case_label(key: str, case: str, row_idx: int) -> str:
    if key == "t5":
        parts = case.split("_")
        return f"{parts[0]}\n{parts[1]}\n{parts[2]}"
    if key == "t8":
        return f"ex.\n{row_idx + 1}"
    return case.replace("_", "\n")


def draw_columns(
    draw: ImageDraw.ImageDraw,
    fig: FigureSet,
    y: int,
    size: int,
    row_label_w: int,
) -> None:
    x0 = MARGIN_X + row_label_w
    for idx, (_, label) in enumerate(fig.columns):
        x = x0 + idx * (size + COL_GAP)
        center_text(draw, (x, y, x + size, y + COL_LABEL_H), label, LABEL_FONT, fill=MUTED)


def draw_case_row(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    fig: FigureSet,
    case: str,
    row_label: str,
    y: int,
    size: int,
    row_label_w: int,
) -> list[str]:
    case_dir = fig.examples_dir / case
    if not case_dir.is_dir():
        raise FileNotFoundError(case_dir)

    if row_label_w:
        center_text(
            draw,
            (MARGIN_X, y, MARGIN_X + row_label_w - 24, y + size),
            row_label,
            ROW_FONT,
            fill=MUTED,
            spacing=8,
        )

    x0 = MARGIN_X + row_label_w
    used: list[str] = []
    for idx, (filename, _) in enumerate(fig.columns):
        x = x0 + idx * (size + COL_GAP)
        src = case_dir / filename
        tile = load_tile(src, size)
        canvas.paste(tile, (x, y))
        draw.rectangle((x, y, x + size - 1, y + size - 1), outline=BORDER, width=2)
        used.append(str(src))
    return used


def verify_cases(fig: FigureSet) -> None:
    for case in fig.cases:
        case_dir = fig.examples_dir / case
        if not case_dir.is_dir():
            raise FileNotFoundError(case_dir)
        files = sorted(p.name for p in case_dir.glob("*.png"))
        expected = sorted(filename for filename, _ in fig.columns)
        if files != expected:
            raise RuntimeError(
                f"{case_dir} has panels {files}, expected exactly {expected}"
            )


def save_outputs(canvas: Image.Image, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(stem.with_suffix(".png"), dpi=(DPI, DPI))
    canvas.save(stem.with_suffix(".pdf"), resolution=DPI)


def draw_main(figs: Iterable[FigureSet], out_stem: Path) -> dict[str, list[str]]:
    figs = tuple(figs)
    size = tile_size(MAIN_ROW_LABEL_W)
    group_h = sum(
        GROUP_HEADER_H + COL_LABEL_H + len(fig.main_cases) * size
        + max(0, len(fig.main_cases) - 1) * ROW_GAP
        for fig in figs
    )
    canvas_h = TOP_MARGIN + group_h + (len(figs) - 1) * GROUP_GAP + BOTTOM_MARGIN
    canvas = Image.new("RGB", (CANVAS_W, canvas_h), BG)
    draw = ImageDraw.Draw(canvas)
    sources: dict[str, list[str]] = {}

    y = TOP_MARGIN
    for fig in figs:
        left_text(draw, (MARGIN_X, y), f"{fig.panel_label} {fig.title}", GROUP_FONT)
        y += GROUP_HEADER_H
        draw.rectangle(
            (MARGIN_X, y, CANVAS_W - MARGIN_X, y + COL_LABEL_H),
            fill=HEADER_BG,
        )
        draw_columns(draw, fig, y, size, MAIN_ROW_LABEL_W)
        y += COL_LABEL_H

        used: list[str] = []
        for row_idx, case in enumerate(fig.main_cases):
            if row_idx:
                y += ROW_GAP
            used.extend(draw_case_row(canvas, draw, fig, case, "", y, size, MAIN_ROW_LABEL_W))
            y += size
        sources[fig.key] = used
        y += GROUP_GAP

    save_outputs(canvas, out_stem)
    return sources


def draw_supplement(fig: FigureSet, out_stem: Path) -> list[str]:
    size = tile_size(SUPP_ROW_LABEL_W)
    canvas_h = TOP_MARGIN + GROUP_HEADER_H + COL_LABEL_H + len(fig.cases) * size
    canvas_h += (len(fig.cases) - 1) * ROW_GAP + BOTTOM_MARGIN
    canvas = Image.new("RGB", (CANVAS_W, canvas_h), BG)
    draw = ImageDraw.Draw(canvas)

    y = TOP_MARGIN
    left_text(draw, (MARGIN_X, y), f"{fig.panel_label} {fig.title}", GROUP_FONT)
    y += GROUP_HEADER_H
    draw.rectangle((MARGIN_X, y, CANVAS_W - MARGIN_X, y + COL_LABEL_H), fill=HEADER_BG)
    draw_columns(draw, fig, y, size, SUPP_ROW_LABEL_W)
    y += COL_LABEL_H

    used: list[str] = []
    for row_idx, case in enumerate(fig.cases):
        if row_idx:
            y += ROW_GAP
        used.extend(
            draw_case_row(
                canvas,
                draw,
                fig,
                case,
                case_label(fig.key, case, row_idx),
                y,
                size,
                SUPP_ROW_LABEL_W,
            )
        )
        y += size

    save_outputs(canvas, out_stem)
    return used


def build_manifest(
    out_dir: Path,
    main_sources: dict[str, list[str]],
    supp_sources: dict[str, list[str]],
) -> None:
    manifest = {
        "dpi": DPI,
        "canvas_width_px": CANVAS_W,
        "main_tile_size_px": tile_size(MAIN_ROW_LABEL_W),
        "supplementary_tile_size_px": tile_size(SUPP_ROW_LABEL_W),
        "main": {
            "outputs": [
                str(out_dir / f"Figure9_DiagnosticVisualization.{ext}")
                for ext in ("png", "pdf")
            ],
            "sources": main_sources,
        },
        "supplementary": {
            key: {
                "outputs": [
                    str(out_dir / f"{stem}.{ext}") for ext in ("png", "pdf")
                ],
                "sources": sources,
            }
            for key, (stem, sources) in supp_sources.items()
        },
    }
    formal = round_dir() / "artifacts" / "diagnostic_figures" / "formal"
    formal.mkdir(parents=True, exist_ok=True)
    (formal / "diagnostic_figures_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


def figure_sets() -> tuple[FigureSet, FigureSet, FigureSet]:
    root = round_dir() / "artifacts"
    t5_dir = root / "t5_persistent_scratch" / "formal" / "examples"
    t8_dir = root / "t8_mask_interpretability" / "formal" / "examples"
    t11_dir = root / "t11_cue_complementarity" / "formal" / "examples"
    return (
        FigureSet(
            key="t5",
            panel_label="(a)",
            title="Scratch response",
            examples_dir=t5_dir,
            cases=(
                "fixed_L1_002",
                "fixed_L1_008",
                "fixed_L1_011",
                "drift_L1_002",
                "drift_L1_008",
                "drift_L1_011",
            ),
            main_cases=("fixed_L1_002",),
            columns=(
                ("01_clean.png", "Clean"),
                ("02_scratch.png", "Scratch"),
                ("03_scratch_mask.png", "Mask"),
                ("04_raw_lhfm_heatmap.png", "Raw LHFM"),
                ("05_m_local_heatmap.png", "Local"),
                ("06_m_wavelet_heatmap.png", "Wavelet"),
            ),
        ),
        FigureSet(
            key="t8",
            panel_label="(b)",
            title="Residual-proxy overlap",
            examples_dir=t8_dir,
            cases=(
                "example_01_026_00000026",
                "example_02_022_00000006",
                "example_03_026_00000098",
                "example_04_005_00000092",
                "example_05_011_00000041",
                "example_06_022_00000011",
            ),
            main_cases=("example_01_026_00000026",),
            columns=(
                ("01_lq.png", "LQ"),
                ("02_gt.png", "GT"),
                ("03_residual_proxy.png", "Proxy"),
                ("04_m_local_heatmap.png", "Local"),
                ("05_m_flicker_heatmap.png", "Flicker"),
                ("06_m_wavelet_heatmap.png", "Wavelet"),
            ),
        ),
        FigureSet(
            key="t11",
            panel_label="(c)",
            title="Cue complementarity",
            examples_dir=t11_dir,
            cases=("000_01", "000_02", "000_03", "000_04", "000_05", "001_01"),
            main_cases=("000_01",),
            columns=(
                ("01_lq.png", "LQ"),
                ("02_residual_proxy.png", "Proxy"),
                ("03_spatial_heatmap.png", "Spatial"),
                ("04_local_heatmap.png", "Local"),
                ("05_flicker_heatmap.png", "Flicker"),
                ("06_wavelet_heatmap.png", "Wavelet"),
            ),
        ),
    )


def main() -> None:
    figs = figure_sets()
    for fig in figs:
        verify_cases(fig)

    out_dir = paper_dir() / "figures"
    main_sources = draw_main(figs, out_dir / "Figure9_DiagnosticVisualization")
    supp_sources = {
        "s1": (
            "FigureS1_PersistentScratchDiagnostic",
            draw_supplement(figs[0], out_dir / "FigureS1_PersistentScratchDiagnostic"),
        ),
        "s2": (
            "FigureS2_MaskInterpretabilityDiagnostic",
            draw_supplement(figs[1], out_dir / "FigureS2_MaskInterpretabilityDiagnostic"),
        ),
        "s3": (
            "FigureS3_CueComplementarityDiagnostic",
            draw_supplement(figs[2], out_dir / "FigureS3_CueComplementarityDiagnostic"),
        ),
    }
    build_manifest(out_dir, main_sources, supp_sources)

    print(f"wrote {out_dir / 'Figure9_DiagnosticVisualization.pdf'}")
    print(f"wrote {out_dir / 'FigureS1_PersistentScratchDiagnostic.pdf'}")
    print(f"wrote {out_dir / 'FigureS2_MaskInterpretabilityDiagnostic.pdf'}")
    print(f"wrote {out_dir / 'FigureS3_CueComplementarityDiagnostic.pdf'}")
    print(
        "wrote "
        + str(
            round_dir()
            / "artifacts"
            / "diagnostic_figures"
            / "formal"
            / "diagnostic_figures_manifest.json"
        )
    )


if __name__ == "__main__":
    main()
