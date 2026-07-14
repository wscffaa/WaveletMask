#!/usr/bin/env python3
"""Probe GFM responses on scene-cut, flash, and illumination-ramp cases."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import cv2
import numpy as np
import torch


@dataclass
class ProbeResult:
    case: str
    clip: str
    clean_activation: float
    activation: float
    activation_ratio: float
    clean_p99_threshold: float
    false_positive_rate: float
    guarded_activation: float
    guarded_false_positive_rate: float
    cut_guard_ratio: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".", type=Path)
    parser.add_argument("--gt-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--clips", default=12, type=int)
    parser.add_argument("--frames", default=7, type=int)
    parser.add_argument("--size", default=256, type=int)
    parser.add_argument("--seed", default=20260606, type=int)
    parser.add_argument("--cut-threshold", default=0.18, type=float)
    parser.add_argument("--no-normalize-tanh", dest="normalize_tanh", action="store_false")
    parser.set_defaults(normalize_tanh=True)
    return parser.parse_args()


def load_gfm(repo_root: Path, checkpoint: Path, device: torch.device):
    sys.path.insert(0, str(repo_root / "projects/007-waveletmask/archs"))
    from components.wavelet_mask import WaveletNoiseMask  # type: ignore

    model = WaveletNoiseMask().to(device)

    payload = torch.load(checkpoint, map_location="cpu")
    state = payload.get("params", payload)
    prefix = "wavelet_mask."
    sub_state = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
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


def make_cases(seq_a: np.ndarray, seq_b: np.ndarray) -> Dict[str, np.ndarray]:
    frames = seq_a.shape[0]
    center = frames // 2

    scene_cut = seq_a.copy()
    scene_cut[center:] = seq_b[center:]

    flash = seq_a.copy()
    flash[center] = np.clip(flash[center] * 1.35 + 0.12, 0.0, 1.0)

    ramp = seq_a.copy()
    gains = np.linspace(0.80, 1.20, frames, dtype=np.float32)
    offsets = np.linspace(-0.05, 0.05, frames, dtype=np.float32)
    for idx in range(frames):
        ramp[idx] = np.clip(ramp[idx] * gains[idx] + offsets[idx], 0.0, 1.0)

    return {"scene_cut": scene_cut, "flash": flash, "illumination_ramp": ramp}


def rgb_to_tensor(seq: np.ndarray, device: torch.device, normalize_tanh: bool) -> torch.Tensor:
    tensor = torch.from_numpy(seq).permute(0, 3, 1, 2).unsqueeze(0).to(device)
    if normalize_tanh:
        tensor = tensor * 2.0 - 1.0
    return tensor.float()


def cut_guard(seq: np.ndarray, threshold: float) -> np.ndarray:
    gray = np.dot(seq[..., :3], np.array([0.299, 0.587, 0.114], dtype=np.float32))
    diffs = np.mean(np.abs(gray[1:] - gray[:-1]), axis=(1, 2))
    frame_flags = np.zeros(seq.shape[0], dtype=bool)
    for idx, value in enumerate(diffs):
        if value > threshold:
            frame_flags[idx] = True
            frame_flags[idx + 1] = True
    return frame_flags


def clean_threshold(clean_mask: np.ndarray) -> tuple[float, float]:
    valid = clean_mask[1:-1]
    activation = float(valid.mean()) if valid.size else 0.0
    threshold = float(np.quantile(valid, 0.99)) if valid.size else 0.0
    return activation, threshold


def summarize_mask(mask: np.ndarray, guard: np.ndarray, threshold: float) -> tuple[float, float, float, float]:
    # mask: [T, H, W]
    valid = mask[1:-1]
    if valid.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    activation = float(valid.mean())
    false_positive_rate = float((valid > threshold).mean())
    guarded = mask.copy()
    guarded[guard] = 0.0
    guarded_valid = guarded[1:-1]
    guarded_activation = float(guarded_valid.mean())
    guarded_fpr = float((guarded_valid > threshold).mean())
    return activation, false_positive_rate, guarded_activation, guarded_fpr


def mean(rows: Iterable[float]) -> float:
    values = list(rows)
    return float(sum(values) / len(values)) if values else float("nan")


def aggregate(results: List[ProbeResult]) -> List[Dict[str, float | str | int]]:
    output: List[Dict[str, float | str | int]] = []
    for case in ("scene_cut", "flash", "illumination_ramp"):
        rows = [row for row in results if row.case == case]
        output.append({
            "case": case,
            "n": len(rows),
            "clean_activation": mean(row.clean_activation for row in rows),
            "activation": mean(row.activation for row in rows),
            "activation_ratio": mean(row.activation_ratio for row in rows),
            "clean_p99_threshold": mean(row.clean_p99_threshold for row in rows),
            "false_positive_rate": mean(row.false_positive_rate for row in rows),
            "guarded_activation": mean(row.guarded_activation for row in rows),
            "guarded_false_positive_rate": mean(row.guarded_false_positive_rate for row in rows),
            "cut_guard_ratio": mean(row.cut_guard_ratio for row in rows),
        })
    return output


def write_csv(path: Path, rows: List[Dict[str, float | str | int]]) -> None:
    fields = [
        "case",
        "n",
        "clean_activation",
        "activation",
        "activation_ratio",
        "clean_p99_threshold",
        "false_positive_rate",
        "guarded_activation",
        "guarded_false_positive_rate",
        "cut_guard_ratio",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float) -> str:
    if not math.isfinite(value):
        return ""
    return f"{value:.4f}"


def write_markdown(path: Path, rows: List[Dict[str, float | str | int]], args: argparse.Namespace) -> None:
    lines = [
        "# T7 GFM Failure-Mode Probe",
        "",
        "Synthetic probe sequences are generated from held-out REDS GT clips. The script measures the trained GFM branch response only; it does not claim restoration-quality improvement.",
        "",
        f"- Clips per case: `{args.clips}`",
        f"- Frames per probe: `{args.frames}`",
        f"- Spatial size: `{args.size}`",
        f"- Input normalization: `{'[-1,1]' if args.normalize_tanh else '[0,1]'}`",
        "- FPR threshold: per-clip clean GFM 99th percentile",
        f"- Cut guard threshold: `{args.cut_threshold}`",
        "",
        "| Case | N | Clean activation | Probe activation | Ratio | FPR | Guarded activation | Guarded FPR | Suppression |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['case']} | {row['n']} | {fmt(float(row['clean_activation']))} | "
            f"{fmt(float(row['activation']))} | {fmt(float(row['activation_ratio']))} | "
            f"{fmt(float(row['false_positive_rate']))} | {fmt(float(row['guarded_activation']))} | "
            f"{fmt(float(row['guarded_false_positive_rate']))} | {fmt(float(row['cut_guard_ratio']))} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_gfm(args.repo_root, args.checkpoint, device)
    clips = list_clips(args.gt_root, args.clips + 1)

    results: List[ProbeResult] = []
    with torch.inference_mode():
        for idx, clip in enumerate(clips[:args.clips]):
            seq_a = load_sequence(clip, args.frames, args.size)
            seq_b = load_sequence(clips[idx + 1], args.frames, args.size)
            clean_tensor = rgb_to_tensor(seq_a, device, args.normalize_tanh)
            clean_mask = model.compute_flicker_mask(clean_tensor)[0, :, 0].detach().float().cpu().numpy()
            base_activation, threshold = clean_threshold(clean_mask)
            for case, seq in make_cases(seq_a, seq_b).items():
                tensor = rgb_to_tensor(seq, device, args.normalize_tanh)
                gfm = model.compute_flicker_mask(tensor)[0, :, 0].detach().float().cpu().numpy()
                guard = cut_guard(seq, args.cut_threshold)
                activation, fpr, guarded_activation, guarded_fpr = summarize_mask(gfm, guard, threshold)
                activation_ratio = activation / max(base_activation, 1e-12)
                suppression = 0.0 if activation <= 1e-12 else 1.0 - guarded_activation / activation
                results.append(
                    ProbeResult(
                        case,
                        clip.name,
                        base_activation,
                        activation,
                        activation_ratio,
                        threshold,
                        fpr,
                        guarded_activation,
                        guarded_fpr,
                        suppression,
                    )
                )

    rows = aggregate(results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": {
            "gt_root": str(args.gt_root),
            "checkpoint": str(args.checkpoint),
            "clips": args.clips,
            "frames": args.frames,
            "size": args.size,
            "seed": args.seed,
            "normalize_tanh": args.normalize_tanh,
            "threshold": "per_clip_clean_p99",
            "cut_threshold": args.cut_threshold,
            "note": "Probe measures trained GFM response on synthetic cases; no restoration model output is evaluated.",
        },
        "rows": [row.__dict__ for row in results],
        "summary": rows,
    }
    (args.output_dir / "gfm_failure_probe.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    write_csv(args.output_dir / "gfm_failure_probe.csv", rows)
    write_markdown(args.output_dir / "gfm_failure_probe.md", rows, args)
    for row in rows:
        print(
            row["case"],
            row["n"],
            fmt(float(row["clean_activation"])),
            fmt(float(row["activation"])),
            fmt(float(row["activation_ratio"])),
            fmt(float(row["false_positive_rate"])),
            fmt(float(row["guarded_activation"])),
            fmt(float(row["guarded_false_positive_rate"])),
            fmt(float(row["cut_guard_ratio"])),
        )


if __name__ == "__main__":
    main()
