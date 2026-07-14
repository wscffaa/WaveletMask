#!/usr/bin/env python3
"""Persistent-scratch response probe for WaveletMask round-1 revision."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F


PROBES = ("raw_lhfm", "m_local", "m_wavelet")
MODES = ("fixed", "drift")


@dataclass
class ProbeRow:
    mode: str
    length: int
    clip: str
    scratch_area: float
    raw_lhfm_inside: float
    raw_lhfm_outside: float
    raw_lhfm_ratio: float
    raw_lhfm_top5: float
    m_local_inside: float
    m_local_outside: float
    m_local_ratio: float
    m_local_top5: float
    m_wavelet_inside: float
    m_wavelet_outside: float
    m_wavelet_ratio: float
    m_wavelet_top5: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".", type=Path)
    parser.add_argument("--gt-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--clips", default=12, type=int)
    parser.add_argument("--frames", default=7, type=int)
    parser.add_argument("--size", default=256, type=int)
    parser.add_argument("--lengths", default="1,2,3,5")
    parser.add_argument("--modes", default="fixed,drift")
    parser.add_argument("--width-min", default=2, type=int)
    parser.add_argument("--width-max", default=5, type=int)
    parser.add_argument("--opacity", default=0.65, type=float)
    parser.add_argument("--seed", default=20260606, type=int)
    parser.add_argument("--top-fraction", default=0.05, type=float)
    parser.add_argument("--visual-examples", default=6, type=int)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--no-normalize-tanh", dest="normalize_tanh", action="store_false")
    parser.set_defaults(normalize_tanh=True)
    return parser.parse_args()


def parse_csv_ints(raw: str) -> List[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one integer")
    return values


def parse_csv_modes(raw: str) -> List[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    bad = sorted(set(values) - set(MODES))
    if bad:
        raise ValueError(f"Unsupported modes: {bad}")
    if not values:
        raise ValueError("Expected at least one mode")
    return values


def load_wavelet_mask(repo_root: Path, checkpoint: Path, device: torch.device):
    sys.path.insert(0, str(repo_root / "projects/007-waveletmask/archs"))
    from components.wavelet_mask import WaveletNoiseMask  # type: ignore

    model = WaveletNoiseMask().to(device)
    payload = torch.load(checkpoint, map_location="cpu")
    state = payload.get("params", payload)
    prefix = "wavelet_mask."
    sub_state = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
    if not sub_state:
        raise RuntimeError(f"No keys with prefix {prefix!r} found in {checkpoint}")
    missing, unexpected = model.load_state_dict(sub_state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Bad wavelet state: missing={missing}, unexpected={unexpected}")
    model.eval()
    return model


def list_clips(gt_root: Path, count: int) -> List[Path]:
    clips = sorted(path for path in gt_root.iterdir() if path.is_dir())
    if len(clips) < count:
        raise ValueError(f"Need {count} clips, found {len(clips)} in {gt_root}")
    return clips[:count]


def read_frame(path: Path, size: int) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    img = img[y0:y0 + side, x0:x0 + side]
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255.0


def load_sequence(clip: Path, frames: int, size: int) -> np.ndarray:
    files = sorted(clip.glob("*.png"))
    if len(files) < frames:
        raise ValueError(f"Need {frames} frames in {clip}, found {len(files)}")
    center = len(files) // 2
    start = max(0, center - frames // 2)
    selected = files[start:start + frames]
    if len(selected) < frames:
        selected = files[:frames]
    return np.stack([read_frame(path, size) for path in selected], axis=0)


def make_base_scratch_mask(size: int, width: int, x: int, angle: float) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.uint8)
    length = int(round(size * 1.5))
    center = np.array([x, size // 2], dtype=np.float32)
    direction = np.array([math.sin(angle), math.cos(angle)], dtype=np.float32)
    p1 = center - direction * length
    p2 = center + direction * length
    cv2.line(
        mask,
        (int(round(p1[0])), int(round(p1[1]))),
        (int(round(p2[0])), int(round(p2[1]))),
        255,
        width,
        lineType=cv2.LINE_AA,
    )
    return mask.astype(np.float32) / 255.0


def active_indices(frames: int, length: int) -> List[int]:
    center = frames // 2
    start = center - (length - 1) // 2
    end = start + length
    if end > frames:
        end = frames
        start = end - length
    start = max(0, start)
    return list(range(start, end))


def shift_mask(mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
    mat = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(mask, mat, (mask.shape[1], mask.shape[0]), flags=cv2.INTER_LINEAR, borderValue=0)


def inject_scratch(
    clean: np.ndarray,
    mode: str,
    length: int,
    width: int,
    x: int,
    angle: float,
    opacity: float,
    bright: bool,
) -> tuple[np.ndarray, np.ndarray]:
    out = clean.copy()
    frames, size = clean.shape[0], clean.shape[1]
    base_mask = make_base_scratch_mask(size, width, x, angle)
    masks = np.zeros((frames, size, size), dtype=np.float32)
    center = frames // 2
    target = 1.0 if bright else 0.0
    for idx in active_indices(frames, length):
        if mode == "drift":
            offset = idx - center
            mask = shift_mask(base_mask, dx=offset * 2, dy=0)
        else:
            mask = base_mask
        masks[idx] = mask
        alpha = (mask * opacity)[..., None]
        out[idx] = np.clip(out[idx] * (1.0 - alpha) + target * alpha, 0.0, 1.0)
    return out, masks


def to_tensor(seq: np.ndarray, device: torch.device, normalize_tanh: bool) -> torch.Tensor:
    tensor = torch.from_numpy(seq).permute(0, 3, 1, 2).unsqueeze(0).to(device)
    if normalize_tanh:
        tensor = tensor * 2.0 - 1.0
    return tensor.float()


def raw_lhfm_response(model, tensor: torch.Tensor) -> torch.Tensor:
    _b, t, c, h, w = tensor.shape
    if c == 3:
        gray = 0.299 * tensor[:, :, 0] + 0.587 * tensor[:, :, 1] + 0.114 * tensor[:, :, 2]
    elif c == 1:
        gray = tensor[:, :, 0]
    else:
        gray = tensor.mean(dim=2)
    responses = []
    for idx in range(t):
        if idx == 0 or idx == t - 1:
            responses.append(torch.zeros(tensor.shape[0], 1, h, w, device=tensor.device))
            continue
        curr = model.dwt(gray[:, idx])
        prev = model.dwt(gray[:, idx - 1])
        next_ = model.dwt(gray[:, idx + 1])
        diff = model._compute_high_freq_diff(curr, prev, next_)
        score = diff.mean(dim=1, keepdim=True)
        score = F.interpolate(score, size=(h, w), mode="bilinear", align_corners=False)
        responses.append(score)
    return torch.stack(responses, dim=1)


def response_stats(score: np.ndarray, mask: np.ndarray, top_fraction: float) -> tuple[float, float, float, float]:
    score_flat = score.reshape(-1).astype(np.float32)
    mask_flat = mask.reshape(-1) > 0.25
    inside = float(score_flat[mask_flat].mean()) if mask_flat.any() else 0.0
    outside = float(score_flat[~mask_flat].mean()) if (~mask_flat).any() else 0.0
    ratio = inside / max(outside, 1e-12)
    k = max(1, int(round(top_fraction * score_flat.size)))
    if k >= score_flat.size:
        top_idx = np.arange(score_flat.size)
    else:
        top_idx = np.argpartition(score_flat, -k)[-k:]
    top_hit = float(mask_flat[top_idx].mean())
    return inside, outside, ratio, top_hit


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return float(sum(vals) / len(vals)) if vals else float("nan")


def aggregate(rows: List[ProbeRow], lengths: Sequence[int], modes: Sequence[str]) -> List[Dict[str, object]]:
    output: List[Dict[str, object]] = []
    baselines: Dict[tuple[str, str], float] = {}
    for mode in modes:
        one = [row for row in rows if row.mode == mode and row.length == 1]
        for probe in PROBES:
            baselines[(mode, probe)] = mean(getattr(row, f"{probe}_inside") for row in one)

    for mode in modes:
        for length in lengths:
            selected = [row for row in rows if row.mode == mode and row.length == length]
            if not selected:
                continue
            item: Dict[str, object] = {
                "mode": mode,
                "length": length,
                "n": len(selected),
                "scratch_area": mean(row.scratch_area for row in selected),
            }
            for probe in PROBES:
                inside = mean(getattr(row, f"{probe}_inside") for row in selected)
                outside = mean(getattr(row, f"{probe}_outside") for row in selected)
                ratio = mean(getattr(row, f"{probe}_ratio") for row in selected)
                top5 = mean(getattr(row, f"{probe}_top5") for row in selected)
                base = baselines[(mode, probe)]
                item[f"{probe}_inside"] = inside
                item[f"{probe}_outside"] = outside
                item[f"{probe}_ratio"] = ratio
                item[f"{probe}_top5"] = top5
                item[f"{probe}_relative_to_l1"] = inside / max(base, 1e-12)
                item[f"{probe}_suppression_from_l1"] = 1.0 - inside / max(base, 1e-12)
            output.append(item)
    return output


def normalized_u8(arr: np.ndarray) -> np.ndarray:
    hi = float(np.percentile(arr, 99.0))
    if hi <= 1e-12:
        return np.zeros(arr.shape, dtype=np.uint8)
    return np.clip(arr / hi * 255.0, 0, 255).astype(np.uint8)


def write_panel_assets(path: Path, panels: Sequence[tuple[str, np.ndarray]]) -> List[Path]:
    asset_dir = path.with_suffix("")
    if path.exists():
        path.unlink()
    asset_dir.mkdir(parents=True, exist_ok=True)
    outputs: List[Path] = []
    for idx, (name, panel) in enumerate(panels, start=1):
        out_path = asset_dir / f"{idx:02d}_{name}.png"
        cv2.imwrite(str(out_path), panel)
        outputs.append(out_path)
    return outputs


def write_visual(path: Path, clean: np.ndarray, degraded: np.ndarray, mask: np.ndarray, scores: Dict[str, np.ndarray]) -> List[Path]:
    panels = [
        ("clean", cv2.cvtColor(np.clip(clean * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)),
        ("scratch", cv2.cvtColor(np.clip(degraded * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)),
        ("scratch_mask", (mask > 0.25).astype(np.uint8) * 255),
    ]
    for key in PROBES:
        panels.append((f"{key}_heatmap", cv2.applyColorMap(normalized_u8(scores[key]), cv2.COLORMAP_TURBO)))
    return write_panel_assets(path, panels)


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fields = ["mode", "length", "n", "scratch_area"]
    for probe in PROBES:
        fields.extend([
            f"{probe}_inside",
            f"{probe}_outside",
            f"{probe}_ratio",
            f"{probe}_top5",
            f"{probe}_relative_to_l1",
            f"{probe}_suppression_from_l1",
        ])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: object, digits: int = 4) -> str:
    if not isinstance(value, (float, int)) or not math.isfinite(float(value)):
        return ""
    return f"{float(value):.{digits}f}"


def write_markdown(path: Path, rows: List[Dict[str, object]], args: argparse.Namespace, examples: List[Path]) -> None:
    lines = [
        "# T5 Persistent-Scratch Probe",
        "",
        "Synthetic scratches are injected into clean REDS clips with controlled persistence length.",
        "The probe measures detector response at the scratched center frame; it does not run the restoration backbone.",
        "",
        f"- Clips per condition: `{args.clips}`",
        f"- Frames per clip: `{args.frames}`",
        f"- Spatial size: `{args.size}`",
        f"- Lengths: `{args.lengths}`",
        f"- Modes: `{args.modes}`",
        f"- Opacity: `{args.opacity}`",
        f"- Checkpoint: `{args.checkpoint}`",
        "",
        "| Mode | L | N | Scratch area | Raw LHFM in | Raw rel. | Raw suppress | m_local in | m_local rel. | m_wavelet in | m_wavelet rel. | m_wavelet top-5 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['mode']} | {row['length']} | {row['n']} | {fmt(row['scratch_area'])} | "
            f"{fmt(row['raw_lhfm_inside'])} | {fmt(row['raw_lhfm_relative_to_l1'])} | "
            f"{fmt(row['raw_lhfm_suppression_from_l1'])} | {fmt(row['m_local_inside'])} | "
            f"{fmt(row['m_local_relative_to_l1'])} | {fmt(row['m_wavelet_inside'])} | "
            f"{fmt(row['m_wavelet_relative_to_l1'])} | {fmt(row['m_wavelet_top5'])} |"
        )
    if examples:
        lines.extend(["", "## Qualitative Example Asset Directories", ""])
        for example in examples:
            lines.append(f"- `{example}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    lengths = parse_csv_ints(args.lengths)
    modes = parse_csv_modes(args.modes)
    if max(lengths) > args.frames:
        raise ValueError("Persistence length cannot exceed frame count")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    model = load_wavelet_mask(args.repo_root, args.checkpoint, device)
    clips = list_clips(args.gt_root, args.clips)
    center = args.frames // 2

    rows: List[ProbeRow] = []
    visual_items: List[tuple[float, Path, np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]] = []
    with torch.inference_mode():
        for clip_idx, clip in enumerate(clips):
            clean = load_sequence(clip, args.frames, args.size)
            width = random.randint(args.width_min, args.width_max)
            x = random.randint(args.size // 5, args.size - args.size // 5)
            angle = random.uniform(-0.12, 0.12)
            bright = (clip_idx % 2) == 0
            for mode in modes:
                for length in lengths:
                    degraded, scratch_masks = inject_scratch(
                        clean,
                        mode,
                        length,
                        width,
                        x,
                        angle,
                        args.opacity,
                        bright,
                    )
                    tensor = to_tensor(degraded, device, args.normalize_tanh)
                    raw = raw_lhfm_response(model, tensor)[0, center, 0].detach().float().cpu().numpy()
                    local = model.compute_local_mask(tensor)[0, center, 0].detach().float().cpu().numpy()
                    flicker = model.compute_flicker_mask(tensor)[0, center, 0].detach().float().cpu().numpy()
                    wavelet = np.maximum(local, flicker)
                    mask = scratch_masks[center]
                    raw_inside, raw_outside, raw_ratio, raw_top5 = response_stats(raw, mask, args.top_fraction)
                    local_inside, local_outside, local_ratio, local_top5 = response_stats(local, mask, args.top_fraction)
                    wavelet_inside, wavelet_outside, wavelet_ratio, wavelet_top5 = response_stats(wavelet, mask, args.top_fraction)
                    rows.append(
                        ProbeRow(
                            mode=mode,
                            length=length,
                            clip=clip.name,
                            scratch_area=float((mask > 0.25).mean()),
                            raw_lhfm_inside=raw_inside,
                            raw_lhfm_outside=raw_outside,
                            raw_lhfm_ratio=raw_ratio,
                            raw_lhfm_top5=raw_top5,
                            m_local_inside=local_inside,
                            m_local_outside=local_outside,
                            m_local_ratio=local_ratio,
                            m_local_top5=local_top5,
                            m_wavelet_inside=wavelet_inside,
                            m_wavelet_outside=wavelet_outside,
                            m_wavelet_ratio=wavelet_ratio,
                            m_wavelet_top5=wavelet_top5,
                        )
                    )
                    if len(visual_items) < args.visual_examples or length in (1, max(lengths)):
                        score = raw_inside if length == 1 else -raw_inside
                        scores = {"raw_lhfm": raw, "m_local": local, "m_wavelet": wavelet}
                        path = args.output_dir / "examples" / f"{mode}_L{length}_{clip.name}.png"
                        visual_items.append((score, path, clean[center], degraded[center], mask, scores))
                        visual_items = sorted(visual_items, key=lambda item: item[0], reverse=True)[:args.visual_examples]

    summary = aggregate(rows, lengths, modes)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    examples: List[Path] = []
    example_assets: Dict[str, List[str]] = {}
    for _score, path, clean_frame, degraded_frame, mask, scores in visual_items:
        assets = write_visual(path, clean_frame, degraded_frame, mask, scores)
        example_dir = path.with_suffix("")
        examples.append(example_dir)
        example_assets[str(example_dir)] = [str(asset) for asset in assets]

    payload = {
        "protocol": {
            "gt_root": str(args.gt_root),
            "checkpoint": str(args.checkpoint),
            "clips": args.clips,
            "frames": args.frames,
            "size": args.size,
            "lengths": lengths,
            "modes": modes,
            "width_min": args.width_min,
            "width_max": args.width_max,
            "opacity": args.opacity,
            "seed": args.seed,
            "normalize_tanh": args.normalize_tanh,
            "top_fraction": args.top_fraction,
            "note": "Detector-response probe on synthetic persistent scratches; restoration output is not evaluated.",
        },
        "rows": [row.__dict__ for row in rows],
        "summary": summary,
        "examples": [str(path) for path in examples],
        "example_assets": example_assets,
    }
    (args.output_dir / "persistent_scratch_probe.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_csv(args.output_dir / "persistent_scratch_probe.csv", summary)
    write_markdown(args.output_dir / "persistent_scratch_probe.md", summary, args, examples)

    for row in summary:
        print(
            row["mode"],
            row["length"],
            row["n"],
            fmt(row["raw_lhfm_inside"]),
            fmt(row["raw_lhfm_relative_to_l1"]),
            fmt(row["raw_lhfm_suppression_from_l1"]),
            fmt(row["m_local_inside"]),
            fmt(row["m_local_relative_to_l1"]),
            fmt(row["m_wavelet_inside"]),
            fmt(row["m_wavelet_relative_to_l1"]),
        )


if __name__ == "__main__":
    main()
