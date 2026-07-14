#!/usr/bin/env python3
"""Artifact-dominant breakdown for WaveletMask round-1 revision.

The SRWOV synthetic split does not ship per-artifact labels. This script derives
clip-level strata from paired input/GT residual descriptors, then aggregates the
per-clip metrics already logged by the RTN baseline and WaveletMask tests.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import cv2
import numpy as np


METRICS = ("psnr", "ssim", "lpips", "dists")
HIGHER_IS_BETTER = {"psnr": True, "ssim": True, "lpips": False, "dists": False}


@dataclass
class ClipFeatures:
    clip: str
    scratch_area: float
    dust_count_mpx: float
    flicker_shift: float
    lowfreq_ratio: float
    residual_area: float
    residual_mean: float
    label: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-root", required=True, type=Path)
    parser.add_argument("--lq-root", required=True, type=Path)
    parser.add_argument("--base-log", required=True, type=Path)
    parser.add_argument("--wavelet-log", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--frames-per-clip", default=20, type=int)
    parser.add_argument("--residual-threshold", default=0.10, type=float)
    parser.add_argument("--max-side", default=360, type=int)
    return parser.parse_args()


def parse_clip_metrics(log_path: Path) -> Dict[str, Dict[str, float]]:
    pattern = re.compile(
        r"->\s+(?P<clip>[^ ]+)\s+metrics:\s+"
        r"psnr:\s+(?P<psnr>[0-9.]+),\s+"
        r"ssim:\s+(?P<ssim>[0-9.]+),\s+"
        r"lpips:\s+(?P<lpips>[0-9.]+),\s+"
        r"dists:\s+(?P<dists>[0-9.]+)"
    )
    metrics: Dict[str, Dict[str, float]] = {}
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.search(line)
        if not match:
            continue
        clip = match.group("clip")
        metrics[clip] = {name: float(match.group(name)) for name in METRICS}
    if not metrics:
        raise ValueError(f"No per-clip metrics found in {log_path}")
    return metrics


def sample_files(files: Sequence[Path], count: int) -> List[Path]:
    if count <= 0 or count >= len(files):
        return list(files)
    idx = np.linspace(0, len(files) - 1, count, dtype=int)
    return [files[int(i)] for i in idx]


def read_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img.astype(np.float32) / 255.0


def resize_for_features(arr: np.ndarray, max_side: int) -> np.ndarray:
    h, w = arr.shape[:2]
    side = max(h, w)
    if max_side <= 0 or side <= max_side:
        return arr
    scale = max_side / float(side)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def component_features(mask: np.ndarray) -> tuple[float, float]:
    n_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
    h_img, w_img = mask.shape
    denom = float(h_img * w_img)
    scratch_area = 0.0
    dust_count = 0
    for idx in range(1, n_labels):
        _x, _y, width, height, area = stats[idx]
        if area < 3:
            continue
        aspect = max(width, height) / max(1, min(width, height))
        if area >= 20 and aspect >= 6.0 and max(width, height) >= 18:
            scratch_area += area / denom
        elif area <= 80 and aspect <= 3.0:
            dust_count += 1
    return scratch_area, dust_count / denom * 1_000_000.0


def compute_clip_features(
    clip: str,
    gt_root: Path,
    lq_root: Path,
    frames_per_clip: int,
    residual_threshold: float,
    max_side: int,
) -> ClipFeatures:
    gt_files = sorted((gt_root / clip).glob("*.png"))
    if not gt_files:
        raise FileNotFoundError(f"No GT frames for clip {clip}")

    scratch_values: List[float] = []
    dust_values: List[float] = []
    flicker_values: List[float] = []
    lowfreq_values: List[float] = []
    area_values: List[float] = []
    mean_values: List[float] = []

    for gt_path in sample_files(gt_files, frames_per_clip):
        lq_path = lq_root / clip / gt_path.name
        gt = read_gray(gt_path)
        lq = read_gray(lq_path)
        if gt.shape != lq.shape:
            lq = cv2.resize(lq, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_AREA)

        signed = resize_for_features(lq - gt, max_side)
        residual = np.abs(signed)
        mask = (residual > residual_threshold).astype(np.uint8)
        scratch_area, dust_count_mpx = component_features(mask)

        low = cv2.GaussianBlur(signed, (0, 0), 11)
        lowfreq_ratio = float(np.mean(low * low) / (np.mean(signed * signed) + 1e-8))

        scratch_values.append(scratch_area)
        dust_values.append(dust_count_mpx)
        flicker_values.append(float(abs(np.mean(signed))))
        lowfreq_values.append(lowfreq_ratio)
        area_values.append(float(np.mean(mask)))
        mean_values.append(float(np.mean(residual)))

    return ClipFeatures(
        clip=clip,
        scratch_area=float(np.mean(scratch_values)),
        dust_count_mpx=float(np.mean(dust_values)),
        flicker_shift=float(np.mean(flicker_values)),
        lowfreq_ratio=float(np.mean(lowfreq_values)),
        residual_area=float(np.mean(area_values)),
        residual_mean=float(np.mean(mean_values)),
    )


def robust_z(values: Iterable[float]) -> List[float]:
    arr = np.array(list(values), dtype=np.float64)
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    scale = mad if mad > 1e-12 else float(np.std(arr) + 1e-12)
    return [float((value - median) / scale) for value in arr]


def assign_labels(features: List[ClipFeatures]) -> None:
    z_scratch = robust_z(f.scratch_area for f in features)
    z_dust = robust_z(f.dust_count_mpx for f in features)
    z_flicker = robust_z(f.flicker_shift for f in features)
    z_lowfreq = robust_z(f.lowfreq_ratio for f in features)

    for idx, feature in enumerate(features):
        scores = {
            "scratch": z_scratch[idx],
            "dust": z_dust[idx],
            "flicker": 0.6 * z_flicker[idx] + 0.4 * z_lowfreq[idx],
        }
        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        positives = [name for name, value in scores.items() if value > 0.75]
        if len(positives) >= 2 or ordered[0][1] <= 0.0:
            feature.label = "mixed"
        else:
            feature.label = ordered[0][0]


def mean_metric(rows: List[Dict[str, float]], metric: str) -> float:
    return float(sum(row[metric] for row in rows) / len(rows))


def aggregate(
    features: List[ClipFeatures],
    base_metrics: Dict[str, Dict[str, float]],
    wavelet_metrics: Dict[str, Dict[str, float]],
) -> List[Dict[str, object]]:
    output: List[Dict[str, object]] = []
    for label in ("scratch", "dust", "flicker", "mixed"):
        clips = [feature.clip for feature in features if feature.label == label]
        if not clips:
            continue
        base_rows = [base_metrics[clip] for clip in clips]
        wavelet_rows = [wavelet_metrics[clip] for clip in clips]
        row: Dict[str, object] = {"label": label, "clips": clips, "n": len(clips)}
        for metric in METRICS:
            base_mean = mean_metric(base_rows, metric)
            wavelet_mean = mean_metric(wavelet_rows, metric)
            delta = wavelet_mean - base_mean
            row[f"base_{metric}"] = base_mean
            row[f"wavelet_{metric}"] = wavelet_mean
            row[f"delta_{metric}"] = delta if HIGHER_IS_BETTER[metric] else -delta
        output.append(row)
    return output


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fields = ["label", "n", "clips"]
    for metric in METRICS:
        fields.extend([f"base_{metric}", f"wavelet_{metric}", f"delta_{metric}"])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            serialised = dict(row)
            serialised["clips"] = " ".join(row["clips"])  # type: ignore[index]
            writer.writerow(serialised)


def fmt(value: object, digits: int) -> str:
    if not isinstance(value, (float, int)) or not math.isfinite(float(value)):
        return ""
    return f"{float(value):.{digits}f}"


def write_markdown(path: Path, rows: List[Dict[str, object]], args: argparse.Namespace) -> None:
    lines = [
        "# T4 Artifact-Dominant Breakdown",
        "",
        "Clip strata are derived from paired `LQ-GT` residual descriptors on the fixed SRWOV/REDS synthetic split.",
        "They are not pixel-level generator labels.",
        "",
        f"- Frames sampled per clip: `{args.frames_per_clip}`",
        f"- Residual threshold: `{args.residual_threshold}`",
        f"- Feature max side: `{args.max_side}`",
        f"- Base log: `{args.base_log}`",
        f"- WaveletMask log: `{args.wavelet_log}`",
        "",
        "| Stratum | Clips | RTN PSNR | WaveletMask PSNR | Delta PSNR | Delta SSIM | Delta LPIPS | Delta DISTS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {label} | {n} | {base_psnr} | {wavelet_psnr} | {delta_psnr} | "
            "{delta_ssim} | {delta_lpips} | {delta_dists} |".format(
                label=row["label"],
                n=row["n"],
                base_psnr=fmt(row["base_psnr"], 2),
                wavelet_psnr=fmt(row["wavelet_psnr"], 2),
                delta_psnr=fmt(row["delta_psnr"], 2),
                delta_ssim=fmt(row["delta_ssim"], 4),
                delta_lpips=fmt(row["delta_lpips"], 4),
                delta_dists=fmt(row["delta_dists"], 4),
            )
        )
    lines.append("")
    lines.append("Positive deltas mean WaveletMask improves over the RTN baseline; for LPIPS/DISTS this is reported as the reduction in error.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base_metrics = parse_clip_metrics(args.base_log)
    wavelet_metrics = parse_clip_metrics(args.wavelet_log)
    clips = sorted(set(base_metrics) & set(wavelet_metrics))
    if not clips:
        raise ValueError("No overlapping clips between base and WaveletMask logs")

    features = [
        compute_clip_features(
            clip,
            args.gt_root,
            args.lq_root,
            args.frames_per_clip,
            args.residual_threshold,
            args.max_side,
        )
        for clip in clips
    ]
    assign_labels(features)
    rows = aggregate(features, base_metrics, wavelet_metrics)

    payload = {
        "protocol": {
            "gt_root": str(args.gt_root),
            "lq_root": str(args.lq_root),
            "base_log": str(args.base_log),
            "wavelet_log": str(args.wavelet_log),
            "frames_per_clip": args.frames_per_clip,
            "residual_threshold": args.residual_threshold,
            "max_side": args.max_side,
            "label_note": "Clip strata are residual-derived descriptors, not pixel-level generator labels.",
        },
        "features": [feature.__dict__ for feature in features],
        "summary": rows,
    }
    (args.output_dir / "artifact_breakdown.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_csv(args.output_dir / "artifact_breakdown.csv", rows)
    write_markdown(args.output_dir / "artifact_breakdown.md", rows, args)

    for row in rows:
        print(
            row["label"],
            row["n"],
            fmt(row["delta_psnr"], 3),
            fmt(row["delta_ssim"], 4),
            fmt(row["delta_lpips"], 4),
            fmt(row["delta_dists"], 4),
        )


if __name__ == "__main__":
    main()
