#!/usr/bin/env python3
"""Generate script-derived draft figures from real old-film damage frames.

This script creates work-in-progress assets under Paper/_work/mdpi-writing.
It does not overwrite manuscript figures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont


DEFAULT_SOURCE = Path("/Users/caifeifan/VibeScholar/Paper_1/Images/Framework/GT")
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[3]
    / "Paper" / "_work" / "mdpi-writing" / "figures" / "real_damage_extract"
)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Helvetica.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


TITLE = font(34, True)
HEAD = font(24, True)
BODY = font(18)
SMALL = font(15)


def read_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def gray_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("L"), dtype=np.float32) / 255.0


def to_u8(x: np.ndarray, percentile: tuple[float, float] = (1.0, 99.0)) -> Image.Image:
    lo, hi = np.percentile(x, percentile)
    if hi <= lo:
        hi = lo + 1e-6
    y = np.clip((x - lo) / (hi - lo), 0, 1)
    return Image.fromarray((y * 255).astype(np.uint8), "L")


def fit(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    out = Image.new("RGB", size, "white")
    im = image.copy()
    im.thumbnail(size, Image.Resampling.LANCZOS)
    out.paste(im, ((size[0] - im.width) // 2, (size[1] - im.height) // 2))
    return out


def haar(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h, w = gray.shape
    gray = gray[: h - h % 2, : w - w % 2]
    a = gray[0::2, 0::2]
    b = gray[0::2, 1::2]
    c = gray[1::2, 0::2]
    d = gray[1::2, 1::2]
    ll = (a + b + c + d) / 4.0
    lh = np.abs((a - b + c - d) / 4.0)
    hl = np.abs((a + b - c - d) / 4.0)
    hh = np.abs((a - b - c + d) / 4.0)
    return ll, lh, hl, hh


def diff_map(current: Image.Image, previous: Image.Image, next_image: Image.Image) -> Image.Image:
    c = gray_array(current)
    p = gray_array(previous)
    n = gray_array(next_image)
    # Geometric mean suppresses one-sided motion while retaining transient defects.
    diff = np.sqrt(np.abs(c - p) * np.abs(c - n) + 1e-6)
    return to_u8(diff, (2, 99.5))


def highpass_mask(image: Image.Image) -> np.ndarray:
    gray = gray_array(image)
    blur = np.asarray(image.convert("L").filter(ImageFilter.GaussianBlur(8)), dtype=np.float32) / 255.0
    residual = np.abs(gray - blur)
    bright = np.maximum(gray - np.percentile(gray, 92), 0)
    score = residual + 0.8 * bright
    return score


def best_roi(score: np.ndarray, box: int = 260, step: int = 48) -> tuple[int, int, int, int]:
    h, w = score.shape
    best = (0.0, 0, 0)
    for y in range(0, max(1, h - box), step):
        for x in range(0, max(1, w - box), step):
            patch = score[y : y + box, x : x + box]
            val = float(patch.mean() + 0.5 * patch.max())
            if val > best[0]:
                best = (val, x, y)
    _, x, y = best
    return x, y, min(x + box, w), min(y + box, h)


def overlay_mask(image: Image.Image, score: np.ndarray, color: tuple[int, int, int]) -> Image.Image:
    base = image.convert("RGBA")
    threshold = np.percentile(score, 97.5)
    mask = (score >= threshold).astype(np.uint8) * 150
    color_layer = Image.new("RGBA", image.size, (*color, 0))
    color_layer.putalpha(Image.fromarray(mask, "L"))
    return Image.alpha_composite(base, color_layer).convert("RGB")


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fill=(30, 42, 58), fnt=BODY) -> None:
    draw.text(xy, text, fill=fill, font=fnt)


def draw_panel_border(draw: ImageDraw.ImageDraw, xy: tuple[int, int, int, int]) -> None:
    draw.rounded_rectangle(xy, radius=6, outline=(180, 188, 198), width=2)


def make_fig2(source: Path, output: Path) -> dict:
    paths = {i: source / f"{i}.png" for i in range(1, 6)}
    current = read_rgb(paths[3])
    previous = read_rgb(paths[2])
    next_image = read_rgb(paths[4])

    canvas = Image.new("RGB", (1800, 1050), "white")
    draw = ImageDraw.Draw(canvas)
    draw_label(draw, (45, 35), "Fig2. Wavelet-domain motivation", fnt=TITLE)

    frame = fit(current, (420, 360))
    pixel = fit(ImageOpsColor(diff_map(current, previous, next_image), "blue"), (420, 360))
    ll, lh, hl, hh = haar(gray_array(current))
    tiles = [
        ("LL", to_u8(ll, (1, 99))),
        ("LH", to_u8(lh, (5, 99.8))),
        ("HL", to_u8(hl, (5, 99.8))),
        ("HH", to_u8(hh, (5, 99.8))),
    ]

    x0, y0 = 70, 170
    x1, x2 = 570, 1070
    canvas.paste(frame, (x0, y0))
    canvas.paste(pixel, (x1, y0))
    draw_panel_border(draw, (x0, y0, x0 + 420, y0 + 360))
    draw_panel_border(draw, (x1, y0, x1 + 420, y0 + 360))
    draw_label(draw, (x0, y0 - 38), "Degraded frame", fnt=HEAD)
    draw_label(draw, (x1, y0 - 38), "Transient pixel difference", fnt=HEAD)

    grid_x, grid_y = x2, y0
    tile_w, tile_h = 250, 170
    for idx, (name, tile) in enumerate(tiles):
        tx = grid_x + (idx % 2) * (tile_w + 28)
        ty = grid_y + (idx // 2) * (tile_h + 46)
        rgb = fit(tile.convert("RGB"), (tile_w, tile_h))
        if name != "LL":
            rgb = ImageEnhance.Contrast(rgb).enhance(1.6)
        canvas.paste(rgb, (tx, ty))
        draw_panel_border(draw, (tx, ty, tx + tile_w, ty + tile_h))
        draw_label(draw, (tx + 8, ty - 28), name, fnt=HEAD)
    draw_label(draw, (grid_x, grid_y - 38), "Haar DWT sub-bands", fnt=HEAD)

    arrow(draw, (500, 350), (555, 350))
    arrow(draw, (1000, 350), (1055, 350))

    draw_label(draw, (85, 575), "Real damaged frame from Paper_1/Images/Framework/GT.", fnt=BODY)
    draw_label(draw, (585, 575), "Frame difference mixes defect, motion, and exposure change.", fnt=BODY)
    draw_label(draw, (1085, 575), "LL captures low-frequency brightness; LH/HL/HH expose local damage.", fnt=BODY)

    callouts = [
        (95, 720, "Use as motivation, not metric evidence."),
        (95, 765, "Script source: 2.png, 3.png, 4.png"),
        (95, 810, "Replace with cleaner PPT redraw before final submission."),
    ]
    for x, y, text in callouts:
        draw_label(draw, (x, y), text, fnt=SMALL)

    path = output / "Fig2_wavelet_frequency_motivation_script.png"
    canvas.save(path)
    return {"figure": str(path), "source_frames": [str(paths[2]), str(paths[3]), str(paths[4])]}


def ImageOpsColor(gray: Image.Image, palette: str) -> Image.Image:
    arr = np.asarray(gray, dtype=np.float32) / 255.0
    if palette == "blue":
        rgb = np.stack([20 + 20 * arr, 45 + 80 * arr, 75 + 180 * arr], axis=-1)
    else:
        rgb = np.stack([arr * 255, arr * 255, arr * 255], axis=-1)
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), "RGB")


def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int]) -> None:
    draw.line([start, end], fill=(25, 84, 135), width=7)
    x, y = end
    draw.polygon([(x, y), (x - 18, y - 12), (x - 18, y + 12)], fill=(25, 84, 135))


def make_fig4(source: Path, output: Path) -> dict:
    rows = [
        ("Sparse defects", source / "1.png", (0, 150, 170)),
        ("Dense blotches", source / "4.png", (34, 115, 210)),
        ("Frame variation", source / "5.png", (218, 146, 0)),
    ]
    canvas = Image.new("RGB", (1800, 1180), "white")
    draw = ImageDraw.Draw(canvas)
    draw_label(draw, (45, 35), "Fig4. WaveletMask-specific degradation examples", fnt=TITLE)

    headers = ["Degraded frame", "Detected damage overlay", "Zoom crop", "Wavelet response cue"]
    xs = [235, 610, 985, 1360]
    for x, h in zip(xs, headers):
        draw_label(draw, (x, 110), h, fnt=HEAD)

    metadata = {"figure": str(output / "Fig4_degradation_examples_script.png"), "rows": []}
    for idx, (label, path, color) in enumerate(rows):
        y = 170 + idx * 310
        image = read_rgb(path)
        score = highpass_mask(image)
        roi = best_roi(score)
        overlay = overlay_mask(image, score, color)
        crop = image.crop(roi)
        cue = to_u8(score, (5, 99.7)).crop(roi)
        cue = ImageEnhance.Contrast(cue.convert("RGB")).enhance(1.8)

        draw_label(draw, (50, y + 105), label, fnt=HEAD)
        panels = [
            fit(image, (310, 220)),
            fit(overlay, (310, 220)),
            fit(crop, (310, 220)),
            fit(cue, (310, 220)),
        ]
        for x, panel in zip(xs, panels):
            canvas.paste(panel, (x, y))
            draw_panel_border(draw, (x, y, x + 310, y + 220))
        draw.rectangle((xs[0] + int(roi[0] * 310 / image.width), y + int(roi[1] * 220 / image.height),
                        xs[0] + int(roi[2] * 310 / image.width), y + int(roi[3] * 220 / image.height)),
                       outline=color, width=3)
        metadata["rows"].append({"label": label, "source": str(path), "roi_xyxy": roi})

    draw_label(draw, (55, 1118), "Script-derived draft from real old-film frames. Use for PPT redraw; do not claim quantitative evidence.", fnt=SMALL)
    path = output / "Fig4_degradation_examples_script.png"
    canvas.save(path)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    meta = {
        "source_dir": str(args.source),
        "outputs": [
            make_fig2(args.source, args.output),
            make_fig4(args.source, args.output),
        ],
        "f6_status": "not_generated: requires paired RTN and WaveletMask restoration outputs; input frames alone are insufficient.",
    }
    (args.output / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
