#!/usr/bin/env python3
"""Build WaveletMask architecture draft figures.

The figures are intentionally generated from functional blocks, not from
decorative image generation, so the draft can be refined in PPT without
changing the method semantics.
"""

from __future__ import annotations

import json
import math
from io import BytesIO
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont, ImageOps
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas as pdf_canvas


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "figures" / "_work" / "architecture_drafts"
GT_FRAME_DIR = Path("/Users/caifeifan/VibeScholar/Paper_1/Images/Framework/GT")

W, H = 2200, 1300
SCALE = 2

INK = "#1f2937"
MUTED = "#64748b"
GRID = "#cbd5e1"
BLUE = "#2563eb"
BLUE_LIGHT = "#dbeafe"
TEAL = "#0f766e"
TEAL_LIGHT = "#ccfbf1"
GREEN = "#15803d"
GREEN_LIGHT = "#dcfce7"
ORANGE = "#c2410c"
ORANGE_LIGHT = "#ffedd5"
PURPLE = "#7c3aed"
PURPLE_LIGHT = "#ede9fe"
RED = "#b91c1c"
RED_LIGHT = "#fee2e2"
GRAY_LIGHT = "#f8fafc"
WHITE = "#ffffff"


def font_path() -> str:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return path
    return ""


FONT = font_path()


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    if bold:
        bold_candidates = [
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/Library/Fonts/Arial Bold.ttf",
        ]
        for path in bold_candidates:
            if Path(path).exists():
                return ImageFont.truetype(path, size)
    if FONT:
        return ImageFont.truetype(FONT, size)
    return ImageFont.load_default()


@dataclass
class Block:
    key: str
    x: int
    y: int
    w: int
    h: int
    label: str
    sublabel: str = ""
    fill: str = WHITE
    outline: str = INK
    title: bool = False

    @property
    def center(self) -> tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h // 2)

    @property
    def left(self) -> tuple[int, int]:
        return (self.x, self.y + self.h // 2)

    @property
    def right(self) -> tuple[int, int]:
        return (self.x + self.w, self.y + self.h // 2)

    @property
    def top(self) -> tuple[int, int]:
        return (self.x + self.w // 2, self.y)

    @property
    def bottom(self) -> tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h)


class Svg:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.items: list[str] = []

    def rect(self, x: int, y: int, w: int, h: int, fill: str, stroke: str, sw: int = 4, rx: int = 18, dash: bool = False):
        dash_attr = ' stroke-dasharray="18 14"' if dash else ""
        self.items.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw}"{dash_attr}/>'
        )

    def text(self, x: int, y: int, text: str, size: int = 34, fill: str = INK, anchor: str = "middle", weight: str = "400"):
        escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        self.items.append(
            f'<text x="{x}" y="{y}" font-family="Arial, Helvetica, sans-serif" font-size="{size}" '
            f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{escaped}</text>'
        )

    def line(self, p1: tuple[int, int], p2: tuple[int, int], color: str = INK, sw: int = 5, dash: bool = False):
        dash_attr = ' stroke-dasharray="14 12"' if dash else ""
        self.items.append(
            f'<line x1="{p1[0]}" y1="{p1[1]}" x2="{p2[0]}" y2="{p2[1]}" stroke="{color}" '
            f'stroke-width="{sw}" stroke-linecap="round" marker-end="url(#arrow)"{dash_attr}/>'
        )

    def polyline(self, pts: Iterable[tuple[int, int]], color: str = INK, sw: int = 5, dash: bool = False):
        point_text = " ".join(f"{x},{y}" for x, y in pts)
        dash_attr = ' stroke-dasharray="14 12"' if dash else ""
        self.items.append(
            f'<polyline points="{point_text}" fill="none" stroke="{color}" stroke-width="{sw}" '
            f'stroke-linecap="round" stroke-linejoin="round" marker-end="url(#arrow)"{dash_attr}/>'
        )

    def circle(self, x: int, y: int, r: int, fill: str, stroke: str, sw: int = 4):
        self.items.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

    def save(self, path: Path):
        path.write_text(
            "\n".join(
                [
                    f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" viewBox="0 0 {self.width} {self.height}">',
                    "<defs>",
                    '<marker id="arrow" markerWidth="14" markerHeight="14" refX="11" refY="5" orient="auto" markerUnits="strokeWidth">',
                    '<path d="M 0 0 L 12 5 L 0 10 z" fill="#1f2937"/>',
                    "</marker>",
                    "</defs>",
                    f'<rect x="0" y="0" width="{self.width}" height="{self.height}" fill="white"/>',
                    *self.items,
                    "</svg>",
                ]
            ),
            encoding="utf-8",
        )


class Canvas:
    def __init__(self, title: str):
        self.image = Image.new("RGB", (W * SCALE, H * SCALE), WHITE)
        self.draw = ImageDraw.Draw(self.image)
        self.svg = Svg(W, H)
        self.blocks: dict[str, Block] = {}
        if title:
            self.title(title)

    def xy(self, p: tuple[int, int]) -> tuple[int, int]:
        return (p[0] * SCALE, p[1] * SCALE)

    def title(self, text: str):
        font = load_font(46 * SCALE, bold=True)
        self.draw.text(self.xy((W // 2, 62)), text, fill=INK, anchor="mm", font=font)
        self.svg.text(W // 2, 76, text, size=46, fill=INK, weight="700")

    def add_group(self, x: int, y: int, w: int, h: int, label: str, color: str):
        self.draw.rounded_rectangle(
            [x * SCALE, y * SCALE, (x + w) * SCALE, (y + h) * SCALE],
            radius=24 * SCALE,
            fill=GRAY_LIGHT,
            outline=color,
            width=3 * SCALE,
        )
        self.svg.rect(x, y, w, h, GRAY_LIGHT, color, sw=3, rx=24)
        font = load_font(30 * SCALE, bold=True)
        self.draw.text(self.xy((x + 28, y + 34)), label, fill=color, anchor="lm", font=font)
        self.svg.text(x + 28, y + 44, label, size=30, fill=color, anchor="start", weight="700")

    def add_block(self, block: Block):
        self.blocks[block.key] = block
        width = 4 if not block.title else 6
        self.draw.rounded_rectangle(
            [block.x * SCALE, block.y * SCALE, (block.x + block.w) * SCALE, (block.y + block.h) * SCALE],
            radius=18 * SCALE,
            fill=block.fill,
            outline=block.outline,
            width=width * SCALE,
        )
        self.svg.rect(block.x, block.y, block.w, block.h, block.fill, block.outline, sw=width, rx=18)
        self._center_text(block)

    def _center_text(self, block: Block):
        title_font = load_font((27 if not block.title else 30) * SCALE, bold=True)
        sub_font = load_font(21 * SCALE)
        lines = []
        for part in block.label.split("\n"):
            lines.extend(wrap_text(part, title_font, block.w * SCALE - 40 * SCALE))
        sublines = []
        if block.sublabel:
            for part in block.sublabel.split("\n"):
                sublines.extend(wrap_text(part, sub_font, block.w * SCALE - 44 * SCALE))
        line_h = 34 if not block.title else 38
        sub_h = 26
        total = len(lines) * line_h + len(sublines) * sub_h + (12 if sublines else 0)
        y = block.y + block.h // 2 - total // 2 + line_h // 2
        for line in lines:
            self.draw.text(self.xy((block.x + block.w // 2, y)), line, fill=INK, anchor="mm", font=title_font)
            self.svg.text(block.x + block.w // 2, y + 11, line, size=30 if not block.title else 34, fill=INK, weight="700")
            y += line_h
        if sublines:
            y += 12
        for line in sublines:
            self.draw.text(self.xy((block.x + block.w // 2, y)), line, fill=MUTED, anchor="mm", font=sub_font)
            self.svg.text(block.x + block.w // 2, y + 9, line, size=24, fill=MUTED)
            y += sub_h

    def note(self, x: int, y: int, text: str, color: str = MUTED, size: int = 25, anchor: str = "mm"):
        font = load_font(size * SCALE)
        self.draw.text(self.xy((x, y)), text, fill=color, anchor=anchor, font=font)
        svg_anchor = "middle" if "m" in anchor else "start"
        self.svg.text(x, y + 8, text, size=size, fill=color, anchor=svg_anchor)

    def arrow(self, start: tuple[int, int], end: tuple[int, int], color: str = INK, width: int = 5, dash: bool = False):
        p1 = self.xy(start)
        p2 = self.xy(end)
        if dash:
            dashed_line(self.draw, p1, p2, fill=color, width=width * SCALE, dash=22 * SCALE)
        else:
            self.draw.line([p1, p2], fill=color, width=width * SCALE)
        draw_arrow_head(self.draw, p1, p2, color, width * SCALE)
        self.svg.line(start, end, color=color, sw=width, dash=dash)

    def elbow(self, pts: list[tuple[int, int]], color: str = INK, width: int = 5, dash: bool = False):
        scaled = [self.xy(p) for p in pts]
        if dash:
            for a, b in zip(scaled, scaled[1:]):
                dashed_line(self.draw, a, b, fill=color, width=width * SCALE, dash=22 * SCALE)
        else:
            self.draw.line(scaled, fill=color, width=width * SCALE, joint="curve")
        draw_arrow_head(self.draw, scaled[-2], scaled[-1], color, width * SCALE)
        self.svg.polyline(pts, color=color, sw=width, dash=dash)

    def junction(self, x: int, y: int, label: str, fill: str, outline: str):
        r = 45
        self.draw.ellipse([(x - r) * SCALE, (y - r) * SCALE, (x + r) * SCALE, (y + r) * SCALE], fill=fill, outline=outline, width=4 * SCALE)
        self.svg.circle(x, y, r, fill, outline, sw=4)
        font = load_font(26 * SCALE, bold=True)
        self.draw.text(self.xy((x, y)), label, fill=INK, anchor="mm", font=font)
        self.svg.text(x, y + 9, label, size=26, fill=INK, weight="700")

    def frame_stack(self, x: int, y: int, w: int, h: int, label: str, color: str = INK):
        thumb = load_stack_thumbnail(w * SCALE, h * SCALE)
        for i in range(5):
            dx = (4 - i) * 14
            dy = (4 - i) * 10
            shade = 235 - i * 18
            fill = f"#{shade:02x}{shade:02x}{shade:02x}"
            box = [(x + dx) * SCALE, (y + dy) * SCALE, (x + dx + w) * SCALE, (y + dy + h) * SCALE]
            if thumb is not None:
                self.image.paste(thumb, (box[0][0], box[0][1]))
                self.draw.rectangle(box, outline=color, width=3 * SCALE)
            else:
                self.draw.rectangle(box, fill=fill, outline=color, width=3 * SCALE)
            self.svg.rect(x + dx, y + dy, w, h, fill, color, sw=2, rx=0)
        font = load_font(28 * SCALE, bold=True)
        self.draw.text(self.xy((x + w // 2 + 28, y + h + 100)), label, fill=color, anchor="mm", font=font)
        self.svg.text(x + w // 2 + 28, y + h + 110, label, size=28, fill=color, weight="700")

    def save(self, stem: str):
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        png = OUT_DIR / f"{stem}.png"
        pdf = OUT_DIR / f"{stem}.pdf"
        svg = OUT_DIR / f"{stem}.svg"
        final = self.image.resize((W, H), Image.Resampling.LANCZOS)
        final.save(png, dpi=(300, 300))
        save_pdf(final, pdf)
        self.svg.save(svg)
        return {"png": str(png), "pdf": str(pdf), "svg": str(svg)}


def wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    if not text:
        return []
    words = text.split()
    lines: list[str] = []
    line = ""
    for word in words:
        trial = word if not line else f"{line} {word}"
        if font.getbbox(trial)[2] <= max_width:
            line = trial
            continue
        if line:
            lines.append(line)
        line = word
    if line:
        lines.append(line)
    return lines


def draw_arrow_head(draw: ImageDraw.ImageDraw, p1: tuple[int, int], p2: tuple[int, int], color: str, width: int):
    angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
    length = max(18, width * 4)
    spread = math.pi / 7
    points = [p2]
    for delta in (math.pi - spread, math.pi + spread):
        points.append((int(p2[0] + length * math.cos(angle + delta)), int(p2[1] + length * math.sin(angle + delta))))
    draw.polygon(points, fill=color)


def load_stack_thumbnail(width: int, height: int) -> Image.Image | None:
    if not GT_FRAME_DIR.exists():
        return None
    candidates = sorted(GT_FRAME_DIR.glob("*.*"))
    if not candidates:
        return None
    try:
        image = Image.open(candidates[0]).convert("L").convert("RGB")
    except OSError:
        return None
    return ImageOps.fit(image, (width, height), method=Image.Resampling.LANCZOS)


def dashed_line(draw: ImageDraw.ImageDraw, p1: tuple[int, int], p2: tuple[int, int], fill: str, width: int, dash: int):
    x1, y1 = p1
    x2, y2 = p2
    dist = math.hypot(x2 - x1, y2 - y1)
    if dist == 0:
        return
    steps = int(dist // dash)
    for i in range(steps + 1):
        if i % 2:
            continue
        t0 = i / max(steps, 1)
        t1 = min((i + 0.65) / max(steps, 1), 1.0)
        a = (int(x1 + (x2 - x1) * t0), int(y1 + (y2 - y1) * t0))
        b = (int(x1 + (x2 - x1) * t1), int(y1 + (y2 - y1) * t1))
        draw.line([a, b], fill=fill, width=width)


def save_pdf(image: Image.Image, path: Path) -> None:
    """Embed the rendered diagram in a PDF without relying on Pillow's JPEG codec."""
    buf = BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    pdf = pdf_canvas.Canvas(str(path), pagesize=(W, H))
    pdf.drawImage(ImageReader(buf), 0, 0, width=W, height=H, preserveAspectRatio=False, mask="auto")
    pdf.showPage()
    pdf.save()


def draw_figure1() -> dict[str, str]:
    c = Canvas("")
    c.frame_stack(80, 495, 190, 145, "degraded clip")
    c.frame_stack(1900, 500, 145, 120, "restored clip")
    c.add_block(Block("flow", 330, 500, 270, 135, "Bidirectional\noptical flow", "RAFT", BLUE_LIGHT, BLUE, True))
    c.add_group(650, 130, 870, 285, "forward recurrent pass", BLUE)
    c.add_group(650, 570, 870, 285, "backward recurrent pass", RED)
    c.add_block(Block("wavelet", 820, 440, 360, 110, "WaveletNoiseMask", "Local HF + Flicker -> max", ORANGE_LIGHT, ORANGE, True))

    top = [
        Block("fw_warp", 700, 230, 180, 95, "Flow warp", "s[t-1->t]", BLUE_LIGHT, BLUE),
        Block("fw_gate", 935, 220, 260, 115, "Wavelet-guided\ngate", "", ORANGE_LIGHT, ORANGE, True),
        Block("fw_swin", 1255, 230, 220, 95, "Swin spatial\nrefinement", "", BLUE_LIGHT, BLUE),
    ]
    bottom = [
        Block("bw_warp", 700, 670, 180, 95, "Flow warp", "s[t+1->t]", BLUE_LIGHT, BLUE),
        Block("bw_gate", 935, 660, 260, 115, "Wavelet-guided\ngate", "", ORANGE_LIGHT, ORANGE, True),
        Block("bw_swin", 1255, 670, 220, 95, "Swin spatial\nrefinement", "", BLUE_LIGHT, BLUE),
    ]
    for b in top + bottom:
        c.add_block(b)
    c.add_block(Block("concat", 1560, 500, 165, 100, "Concat", "f_f + f_b", WHITE, INK, True))
    c.add_block(Block("recon", 1745, 500, 190, 100, "Reconstruction", "residual tail", WHITE, INK, True))

    c.arrow((270, 585), c.blocks["flow"].left, INK)
    c.elbow([c.blocks["flow"].right, (635, 568), (635, 278), c.blocks["fw_warp"].left], INK)
    c.elbow([c.blocks["flow"].right, (635, 568), (635, 718), c.blocks["bw_warp"].left], INK)
    c.arrow(c.blocks["fw_warp"].right, c.blocks["fw_gate"].left, INK)
    c.arrow(c.blocks["fw_gate"].right, c.blocks["fw_swin"].left, INK)
    c.arrow(c.blocks["bw_warp"].right, c.blocks["bw_gate"].left, INK)
    c.arrow(c.blocks["bw_gate"].right, c.blocks["bw_swin"].left, INK)
    c.elbow([c.blocks["fw_swin"].right, (1535, 278), (1535, 535), c.blocks["concat"].left], INK)
    c.elbow([c.blocks["bw_swin"].right, (1535, 718), (1535, 565), c.blocks["concat"].left], INK)
    c.arrow(c.blocks["concat"].right, c.blocks["recon"].left, INK)
    c.arrow(c.blocks["recon"].right, (1900, 565), INK)
    c.elbow([(270, 500), (500, 420), c.blocks["wavelet"].left], ORANGE)
    c.arrow(c.blocks["wavelet"].top, c.blocks["fw_gate"].bottom, ORANGE, dash=True)
    c.arrow(c.blocks["wavelet"].bottom, c.blocks["bw_gate"].top, ORANGE, dash=True)
    return c.save("Fig1_waveletmask_overall_architecture")


def draw_figure2() -> dict[str, str]:
    c = Canvas("")
    c.add_group(80, 145, 530, 970, "wavelet decomposition", TEAL)
    c.add_group(690, 145, 1410, 970, "noise mask estimation", BLUE)
    c.frame_stack(170, 270, 165, 110, "I[t-1], I[t], I[t+1]")
    blocks = [
        Block("gray", 175, 535, 315, 100, "Gray conversion", "luminance channel", GRAY_LIGHT, INK),
        Block("dwt", 175, 755, 315, 115, "Haar DWT", "LL / LH / HL / HH", TEAL_LIGHT, TEAL, True),
        Block("hf", 760, 255, 265, 105, "HF bands", "LH, HL, HH", TEAL_LIGHT, TEAL),
        Block("hfdiff", 1115, 255, 315, 105, "Temporal HF diff", "sqrt(|c-p| |c-n|)", TEAL_LIGHT, TEAL),
        Block("hfconv", 1515, 255, 275, 105, "Local mask head", "Conv 3-16-1", TEAL_LIGHT, TEAL),
        Block("mlocal", 1835, 255, 180, 105, "m_local", "upsampled", WHITE, TEAL, True),
        Block("ll", 760, 640, 265, 105, "LL band", "low frequency", GREEN_LIGHT, GREEN),
        Block("lldiff", 1115, 640, 315, 105, "Temporal LL diff", "min(|c-p|, |c-n|)", GREEN_LIGHT, GREEN),
        Block("llconv", 1515, 640, 275, 105, "Flicker mask head", "Conv 1-16-1", GREEN_LIGHT, GREEN),
        Block("mflicker", 1835, 640, 180, 105, "m_flicker", "upsampled", WHITE, GREEN, True),
        Block("out", 1725, 905, 330, 120, "Wavelet noise mask", "m_w = max(local, flicker)", ORANGE_LIGHT, ORANGE, True),
    ]
    for b in blocks:
        c.add_block(b)
    c.junction(2050, 500, "max", ORANGE_LIGHT, ORANGE)

    c.arrow((280, 490), c.blocks["gray"].top, INK)
    c.arrow(c.blocks["gray"].bottom, c.blocks["dwt"].top, INK)
    c.elbow([c.blocks["dwt"].right, (650, 812), (650, 307), c.blocks["hf"].left], TEAL)
    c.arrow(c.blocks["hf"].right, c.blocks["hfdiff"].left, TEAL)
    c.arrow(c.blocks["hfdiff"].right, c.blocks["hfconv"].left, TEAL)
    c.arrow(c.blocks["hfconv"].right, c.blocks["mlocal"].left, TEAL)
    c.elbow([c.blocks["dwt"].right, (650, 812), (650, 692), c.blocks["ll"].left], GREEN)
    c.arrow(c.blocks["ll"].right, c.blocks["lldiff"].left, GREEN)
    c.arrow(c.blocks["lldiff"].right, c.blocks["llconv"].left, GREEN)
    c.arrow(c.blocks["llconv"].right, c.blocks["mflicker"].left, GREEN)
    c.elbow([c.blocks["mlocal"].right, (2050, 307), (2050, 455)], TEAL)
    c.elbow([c.blocks["mflicker"].right, (2050, 692), (2050, 545)], GREEN)
    c.arrow((2050, 545), c.blocks["out"].top, ORANGE)
    return c.save("Fig2_wavelet_noise_mask_module")


def draw_figure3() -> dict[str, str]:
    c = Canvas("")
    c.add_group(90, 150, 700, 930, "gated aggregation inputs", BLUE)
    c.add_group(880, 150, 1220, 930, "wavelet-guided feature update", ORANGE)
    blocks = [
        Block("hidden", 190, 265, 420, 115, "Warped hidden state", "flow-aligned h", BLUE_LIGHT, BLUE, True),
        Block("frame", 190, 520, 420, 115, "Current frame", "projection head phi(x_t)", BLUE_LIGHT, BLUE, True),
        Block("mask", 190, 775, 420, 115, "Wavelet indicator", "m_w from Fig. 2", TEAL_LIGHT, TEAL, True),
        Block("concat", 945, 420, 330, 115, "Concatenate", "h, phi, m_w", ORANGE_LIGHT, ORANGE),
        Block("conv1", 1355, 330, 290, 105, "Gate conv", "(2C + 1) -> C/2", ORANGE_LIGHT, ORANGE),
        Block("conv2", 1355, 555, 290, 105, "Sigmoid gate", "C/2 -> 1", ORANGE_LIGHT, ORANGE),
        Block("blend", 1725, 420, 300, 130, "Feature blend", "g h + (1-g) phi", BLUE_LIGHT, BLUE, True),
        Block("swin", 1725, 725, 300, 115, "Swin spatial block", "updated feature", BLUE_LIGHT, BLUE),
    ]
    for b in blocks:
        c.add_block(b)

    c.arrow(c.blocks["hidden"].right, c.blocks["concat"].left, BLUE)
    c.arrow(c.blocks["frame"].right, c.blocks["concat"].left, BLUE)
    c.arrow(c.blocks["mask"].right, c.blocks["concat"].left, TEAL)
    c.arrow(c.blocks["concat"].right, c.blocks["conv1"].left, ORANGE)
    c.arrow(c.blocks["conv1"].bottom, c.blocks["conv2"].top, ORANGE)
    c.arrow(c.blocks["conv2"].right, c.blocks["blend"].left, ORANGE)
    c.arrow(c.blocks["blend"].bottom, c.blocks["swin"].top, BLUE)
    return c.save("Fig3_wavelet_guided_gated_aggregation_module")


def write_ascii_sidecars() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "Fig1_waveletmask_overall_architecture.ascii.txt").write_text(
        """degraded clip -> bidirectional optical flow -> forward/backward flow warp
degraded clip -> WaveletNoiseMask -> m_t^w --\\
flow warp + x_t + m_t^w -> Wavelet-guided gate -> Swin spatial refinement -> f_t^f / f_t^b
f_t^f + f_t^b -> Concat -> Reconstruction -> restored clip
""",
        encoding="utf-8",
    )
    (OUT_DIR / "Fig2_wavelet_noise_mask_module.ascii.txt").write_text(
        """Frame triplet -> gray conversion -> Haar DWT
  LH/HL/HH -> geometric mean difference -> Conv 3->16->1 -> upsample -> local mask --\\
  LL       -> bidirectional min difference -> Conv 1->16->1 -> upsample -> flicker mask -> max -> wavelet noise mask
""",
        encoding="utf-8",
    )
    (OUT_DIR / "Fig3_wavelet_guided_gated_aggregation_module.ascii.txt").write_text(
        """Warped hidden state --\\
Current frame feature -> Concatenate -> Gate conv -> Sigmoid gate -> Feature blend -> Swin spatial block
Wavelet indicator ----/
""",
        encoding="utf-8",
    )


def main() -> None:
    outputs = {
        "fig1": draw_figure1(),
        "fig2": draw_figure2(),
        "fig3": draw_figure3(),
    }
    write_ascii_sidecars()
    (OUT_DIR / "architecture_manifest.json").write_text(json.dumps(outputs, indent=2), encoding="utf-8")
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
