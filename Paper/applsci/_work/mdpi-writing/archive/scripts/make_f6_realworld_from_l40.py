#!/usr/bin/env python3
"""Build Fig6 from L40 real-world inference outputs.

Inputs are copied from:
projects/007-waveletmask/Exp/tmp_f6_realworld_mini on L40.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "figures" / "_work" / "f6_realworld_l40" / "raw"
OUT = ROOT / "figures" / "_work" / "f6_realworld_l40"


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
HEAD = font(23, True)
BODY = font(17)
SMALL = font(14)


SCENES = [
    ("scene_gRaos_1586", "Scene A"),
    ("scene_rJl_720", "Scene B"),
    ("scene_uqj_86", "Scene C"),
]


def read(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def fit(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    out = Image.new("RGB", size, "white")
    im = image.copy()
    im.thumbnail(size, Image.Resampling.LANCZOS)
    out.paste(im, ((size[0] - im.width) // 2, (size[1] - im.height) // 2))
    return out


def gray(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("L"), dtype=np.float32) / 255.0


def find_roi(input_img: Image.Image, base_img: Image.Image, wave_img: Image.Image, box: int = 110) -> tuple[int, int, int, int]:
    inp = gray(input_img)
    base = gray(base_img)
    wave = gray(wave_img)
    blur = np.asarray(input_img.convert("L").filter(ImageFilter.GaussianBlur(5)), dtype=np.float32) / 255.0
    high = np.abs(inp - blur)
    score = np.abs(inp - base) + np.abs(base - wave) + 0.7 * high
    h, w = score.shape
    best = (-1.0, 0, 0)
    for y in range(0, max(1, h - box), 20):
        for x in range(0, max(1, w - box), 20):
            patch = score[y : y + box, x : x + box]
            val = float(patch.mean() + 0.25 * patch.max())
            if val > best[0]:
                best = (val, x, y)
    _, x, y = best
    return x, y, x + box, y + box


def draw_border(draw: ImageDraw.ImageDraw, xy: tuple[int, int, int, int], color=(180, 188, 198), width: int = 2) -> None:
    draw.rounded_rectangle(xy, radius=5, outline=color, width=width)


def locate(scene: str, kind: str) -> Path:
    if kind == "input":
        return RAW / "input" / scene / "00015.jpg"
    if kind == "base":
        return RAW / "base" / scene / "outputs" / "00015_f6_waveletmask_base_realworld_mini.png"
    if kind == "ab":
        return RAW / "ab" / scene / "outputs" / "00015_f6_waveletmask_ab_realworld_mini.png"
    raise ValueError(kind)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    canvas = Image.new("RGB", (1820, 1280), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((45, 35), "Fig6. Real-world qualitative comparison from L40 mini inference", fill=(30, 42, 58), font=TITLE)

    headers = ["Degraded input", "RTN-style base", "WaveletMask-ab", "Zoom crop"]
    xs = [210, 575, 940, 1305]
    row_y0 = 170
    panel = (320, 240)
    for x, header in zip(xs, headers):
        draw.text((x, 115), header, fill=(30, 42, 58), font=HEAD)

    metadata = {
        "source": "L40_aliyun:/mnt/data/Antigravity_OFR/projects/007-waveletmask/Exp/tmp_f6_realworld_mini",
        "models": {
            "base": "projects/007-waveletmask/Exp/checkpoints/waveletmask_base/net_g_best.pth",
            "waveletmask_ab": "projects/007-waveletmask/Exp/checkpoints/waveletmask_ab/net_g_best.pth",
        },
        "commands": [
            "CUDA_VISIBLE_DEVICES=2 python test.py -opt projects/007-waveletmask/Exp/tmp_f6_realworld_mini/configs/f6_waveletmask_base_realworld_mini.yml",
            "CUDA_VISIBLE_DEVICES=3 python test.py -opt projects/007-waveletmask/Exp/tmp_f6_realworld_mini/configs/f6_waveletmask_ab_realworld_mini.yml",
        ],
        "scenes": [],
    }

    for row_idx, (scene, label) in enumerate(SCENES):
        y = row_y0 + row_idx * 335
        images = {
            "input": read(locate(scene, "input")),
            "base": read(locate(scene, "base")),
            "ab": read(locate(scene, "ab")),
        }
        roi = find_roi(images["input"], images["base"], images["ab"])
        draw.text((55, y + 94), label, fill=(30, 42, 58), font=HEAD)

        for col, kind in enumerate(["input", "base", "ab"]):
            x = xs[col]
            thumb = fit(images[kind], panel)
            canvas.paste(thumb, (x, y))
            draw_border(draw, (x, y, x + panel[0], y + panel[1]))
            sx, sy = panel[0] / images[kind].width, panel[1] / images[kind].height
            rx = (x + int(roi[0] * sx), y + int(roi[1] * sy), x + int(roi[2] * sx), y + int(roi[3] * sy))
            draw.rectangle(rx, outline=(0, 137, 150), width=3)

        crop_x = xs[3]
        crop_w, crop_h = 105, 105
        for i, kind in enumerate(["input", "base", "ab"]):
            crop = images[kind].crop(roi).resize((crop_w, crop_h), Image.Resampling.LANCZOS)
            cx = crop_x + i * (crop_w + 18)
            canvas.paste(crop, (cx, y + 38))
            draw_border(draw, (cx, y + 38, cx + crop_w, y + 38 + crop_h), color=(0, 137, 150), width=2)
            draw.text((cx + 18, y + 150), ["Input", "Base", "WM"][i], fill=(30, 42, 58), font=SMALL)

        metadata["scenes"].append(
            {
                "scene": scene,
                "frame": "00015",
                "roi_xyxy": roi,
                "input": str(locate(scene, "input")),
                "base": str(locate(scene, "base")),
                "waveletmask_ab": str(locate(scene, "ab")),
            }
        )

    draw.text(
        (55, 1225),
        "Script-derived from real L40 inference. Mini-set qualitative evidence only; do not use as refreshed RealWorld ranking.",
        fill=(30, 42, 58),
        font=BODY,
    )
    figure = OUT / "Fig6_realworld_comparison_l40_script.png"
    meta_path = OUT / "Fig6_realworld_comparison_l40_metadata.json"
    canvas.save(figure)
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(figure)
    print(meta_path)


if __name__ == "__main__":
    main()
