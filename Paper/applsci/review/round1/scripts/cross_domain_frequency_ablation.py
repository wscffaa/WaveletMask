#!/usr/bin/env python3
"""Cross-domain frequency decomposition ablation for Reviewer 3.

This is a diagnostic experiment, not a restoration training run. It compares
frequency decomposition score maps against residual-derived LQ-GT proxy regions
on REDS Synthetic. The learned row is a shallow frequency-filter probe trained
only on the proxy labels so that the paper can separate deterministic design
choice evidence from supervised oracle-like evidence.
"""

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

try:
    import cv2  # type: ignore
except ImportError as exc:  # pragma: no cover - exercised on env failures
    cv2 = None  # type: ignore
    CV2_ERROR = str(exc)
else:
    CV2_ERROR = ""

try:
    import numpy as np  # type: ignore
except ImportError as exc:  # pragma: no cover - exercised on env failures
    np = None  # type: ignore
    NUMPY_ERROR = str(exc)
else:
    NUMPY_ERROR = ""


DETERMINISTIC_SHARDS: Sequence[Sequence[str]] = (
    ("haar_low", "haar_high", "haar_fused"),
    ("sobel", "laplacian"),
    ("gaussian_low", "gaussian_high", "gaussian_fused"),
    ("fourier_temporal_low", "fourier_temporal_high", "fourier_temporal_fused"),
)
DETERMINISTIC_METHODS = tuple(method for shard in DETERMINISTIC_SHARDS for method in shard)
LEARNED_METHOD = "learned_shallow_frequency"
SUMMARY_FIELDS = [
    "method",
    "family",
    "status",
    "fold",
    "pixels",
    "positives",
    "proxy_area",
    "auprc",
    "best_f1",
    "best_iou",
    "best_precision",
    "best_recall",
    "best_threshold",
    "inside_score",
    "outside_score",
    "inside_outside_ratio",
    "top5_localization",
    "top5_false_positive",
    "top5_lift",
    "random_localization",
    "score_mean",
    "note",
]


@dataclass
class FrameSample:
    clip: str
    frame: str
    prev_lq: object
    curr_lq: object
    next_lq: object
    curr_gt: object
    label: object


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        required=True,
        choices=("all", "deterministic-shard", "learned-fold", "merge"),
    )
    parser.add_argument("--repo-root", default=".", type=Path)
    parser.add_argument(
        "--gt-root",
        default=Path("datasets/OldFilmRestoration/val/REDS/GT/val_sharp"),
        type=Path,
    )
    parser.add_argument(
        "--lq-root",
        default=Path("datasets/OldFilmRestoration/val/REDS/LQ/val_sharp_degraded"),
        type=Path,
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--clips", default=30, type=int, help="Number of clips; <=0 means all clips.")
    parser.add_argument("--frames-per-clip", default=20, type=int)
    parser.add_argument("--size", default=256, type=int)
    parser.add_argument("--metric-size", default=128, type=int)
    parser.add_argument("--residual-threshold", default=0.10, type=float)
    parser.add_argument("--top-fraction", default=0.05, type=float)
    parser.add_argument("--visual-examples", default=6, type=int)
    parser.add_argument("--shard-index", default=0, type=int)
    parser.add_argument("--num-shards", default=4, type=int)
    parser.add_argument("--fold-index", default=0, type=int)
    parser.add_argument("--num-folds", default=5, type=int)
    parser.add_argument("--seed", default=20260610, type=int)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--batch-size", default=12, type=int)
    parser.add_argument("--learning-rate", default=0.001, type=float)
    parser.add_argument("--no-learned", action="store_true")
    parser.add_argument(
        "--synthetic-smoke",
        action="store_true",
        help="Use generated paired frames when dataset paths are absent; for local script validation only.",
    )
    return parser.parse_args()


def need_array_stack(output_dir: Path) -> bool:
    if NUMPY_ERROR or CV2_ERROR:
        output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "status": "blocked-by-env",
            "reason": "Required array/image dependencies are missing.",
            "errors": {"numpy": NUMPY_ERROR, "cv2": CV2_ERROR},
        }
        (output_dir / "environment_blocked.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return False
    return True


def resolve_path(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def list_clips(gt_root: Path, clips: int) -> List[Path]:
    paths = sorted(path for path in gt_root.iterdir() if path.is_dir())
    if not paths:
        raise FileNotFoundError(f"No clip directories under {gt_root}")
    return paths if clips <= 0 else paths[:clips]


def sample_indices(frame_count: int, count: int) -> List[int]:
    if frame_count < 3:
        return []
    valid = np.arange(1, frame_count - 1)
    if count <= 0 or count >= len(valid):
        return [int(i) for i in valid]
    return [int(i) for i in np.linspace(valid[0], valid[-1], count, dtype=int)]


def read_rgb(path: Path, size: int) -> object:
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


def rgb_to_gray(img: object) -> object:
    arr = img  # keep annotations runtime-light when numpy import is optional
    return (
        0.299 * arr[..., 0]
        + 0.587 * arr[..., 1]
        + 0.114 * arr[..., 2]
    ).astype(np.float32)


def resize_score(arr: object, size: int) -> object:
    if arr.shape[0] == size and arr.shape[1] == size:
        return arr.astype(np.float32, copy=False)
    return cv2.resize(arr.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)


def resize_label(arr: object, size: int) -> object:
    if arr.shape[0] == size and arr.shape[1] == size:
        return arr.astype(np.bool_, copy=False)
    out = cv2.resize(arr.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)
    return out.astype(np.bool_)


def normalize_score(arr: object) -> object:
    arr = arr.astype(np.float32, copy=False)
    lo = float(np.percentile(arr, 1.0))
    hi = float(np.percentile(arr, 99.0))
    if hi <= lo + 1e-12:
        return np.zeros(arr.shape, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def residual_proxy(lq: object, gt: object, threshold: float, metric_size: int) -> object:
    residual = np.mean(np.abs(lq - gt), axis=2)
    label = residual > threshold
    return resize_label(label, metric_size)


def haar_scores(gray: object) -> Dict[str, object]:
    h, w = gray.shape
    h2 = h - (h % 2)
    w2 = w - (w % 2)
    base = gray[:h2, :w2]
    a = base[0::2, 0::2]
    b = base[0::2, 1::2]
    c = base[1::2, 0::2]
    d = base[1::2, 1::2]
    ll = (a + b + c + d) * 0.25
    lh = (a - b + c - d) * 0.25
    hl = (a + b - c - d) * 0.25
    hh = (a - b - c + d) * 0.25
    low = np.abs(ll - cv2.GaussianBlur(ll, (0, 0), 3.0))
    high = np.abs(lh) + np.abs(hl) + np.abs(hh)
    low = cv2.resize(low, (w, h), interpolation=cv2.INTER_LINEAR)
    high = cv2.resize(high, (w, h), interpolation=cv2.INTER_LINEAR)
    low = normalize_score(low)
    high = normalize_score(high)
    return {
        "haar_low": low,
        "haar_high": high,
        "haar_fused": np.maximum(low, high),
    }


def fft_band_scores(arr: object, radius_fraction: float = 0.18) -> tuple[object, object]:
    rows, cols = arr.shape
    freq = np.fft.fftshift(np.fft.fft2(arr.astype(np.float32)))
    yy, xx = np.ogrid[:rows, :cols]
    cy = rows // 2
    cx = cols // 2
    radius = max(1, int(min(rows, cols) * radius_fraction))
    low_mask = ((yy - cy) ** 2 + (xx - cx) ** 2) <= radius * radius
    low_freq = np.zeros_like(freq)
    high_freq = np.zeros_like(freq)
    low_freq[low_mask] = freq[low_mask]
    high_freq[~low_mask] = freq[~low_mask]
    low = np.abs(np.fft.ifft2(np.fft.ifftshift(low_freq))).astype(np.float32)
    high = np.abs(np.fft.ifft2(np.fft.ifftshift(high_freq))).astype(np.float32)
    return normalize_score(low), normalize_score(high)


def deterministic_maps(sample: FrameSample, methods: Sequence[str], metric_size: int) -> Dict[str, object]:
    curr = rgb_to_gray(sample.curr_lq)
    prev = rgb_to_gray(sample.prev_lq)
    nxt = rgb_to_gray(sample.next_lq)
    maps: Dict[str, object] = {}

    needed = set(methods)
    if needed & {"haar_low", "haar_high", "haar_fused"}:
        maps.update(haar_scores(curr))
    if "sobel" in needed:
        sx = cv2.Sobel(curr, cv2.CV_32F, 1, 0, ksize=3)
        sy = cv2.Sobel(curr, cv2.CV_32F, 0, 1, ksize=3)
        maps["sobel"] = normalize_score(np.sqrt(sx * sx + sy * sy))
    if "laplacian" in needed:
        maps["laplacian"] = normalize_score(np.abs(cv2.Laplacian(curr, cv2.CV_32F, ksize=3)))
    if needed & {"gaussian_low", "gaussian_high", "gaussian_fused"}:
        low = cv2.GaussianBlur(curr, (0, 0), 5.0)
        low_score = normalize_score(np.abs(low - cv2.GaussianBlur(low, (0, 0), 11.0)))
        high_score = normalize_score(np.abs(curr - low))
        maps["gaussian_low"] = low_score
        maps["gaussian_high"] = high_score
        maps["gaussian_fused"] = np.maximum(low_score, high_score)
    if needed & {"fourier_temporal_low", "fourier_temporal_high", "fourier_temporal_fused"}:
        temporal = 0.5 * (np.abs(curr - prev) + np.abs(nxt - curr))
        low_band, high_band = fft_band_scores(temporal)
        maps["fourier_temporal_low"] = low_band
        maps["fourier_temporal_high"] = high_band
        maps["fourier_temporal_fused"] = np.maximum(low_band, high_band)

    return {method: resize_score(maps[method], metric_size) for method in methods}


def synthetic_samples(args: argparse.Namespace) -> List[FrameSample]:
    rng = np.random.default_rng(args.seed)
    samples: List[FrameSample] = []
    clip_count = args.clips if args.clips > 0 else 2
    frame_count = args.frames_per_clip if args.frames_per_clip > 0 else 4
    for clip_idx in range(max(1, clip_count)):
        for frame_idx in range(max(1, frame_count)):
            gt = rng.random((args.size, args.size, 3), dtype=np.float32)
            lq = gt.copy()
            x0 = int(args.size * (0.2 + 0.1 * ((clip_idx + frame_idx) % 4)))
            lq[:, x0:x0 + 3, :] = np.clip(lq[:, x0:x0 + 3, :] + 0.35, 0.0, 1.0)
            lq = np.clip(lq + rng.normal(0, 0.015, lq.shape).astype(np.float32), 0.0, 1.0)
            prev_lq = np.roll(lq, shift=-1, axis=1)
            next_lq = np.roll(lq, shift=1, axis=1)
            samples.append(
                FrameSample(
                    clip=f"synthetic_{clip_idx:02d}",
                    frame=f"{frame_idx:08d}.png",
                    prev_lq=prev_lq,
                    curr_lq=lq,
                    next_lq=next_lq,
                    curr_gt=gt,
                    label=residual_proxy(lq, gt, args.residual_threshold, args.metric_size),
                )
            )
    return samples


def collect_samples(args: argparse.Namespace) -> List[FrameSample]:
    gt_root = resolve_path(args.repo_root, args.gt_root)
    lq_root = resolve_path(args.repo_root, args.lq_root)
    if args.synthetic_smoke:
        return synthetic_samples(args)
    clips = list_clips(gt_root, args.clips)
    samples: List[FrameSample] = []
    for clip in clips:
        gt_files = sorted(clip.glob("*.png"))
        indices = sample_indices(len(gt_files), args.frames_per_clip)
        for idx in indices:
            frame_paths = [gt_files[idx - 1], gt_files[idx], gt_files[idx + 1]]
            lq_paths = [lq_root / clip.name / path.name for path in frame_paths]
            prev_lq = read_rgb(lq_paths[0], args.size)
            curr_lq = read_rgb(lq_paths[1], args.size)
            next_lq = read_rgb(lq_paths[2], args.size)
            curr_gt = read_rgb(gt_files[idx], args.size)
            label = residual_proxy(curr_lq, curr_gt, args.residual_threshold, args.metric_size)
            samples.append(
                FrameSample(
                    clip=clip.name,
                    frame=gt_files[idx].name,
                    prev_lq=prev_lq,
                    curr_lq=curr_lq,
                    next_lq=next_lq,
                    curr_gt=curr_gt,
                    label=label,
                )
            )
    if not samples:
        raise RuntimeError("No paired samples collected.")
    return samples


def summarize_scores(
    method: str,
    family: str,
    scores: object,
    labels: object,
    top_fraction: float,
    fold: object = "",
    status: str = "done",
    note: str = "",
) -> Dict[str, object]:
    scores = scores.astype(np.float32, copy=False).reshape(-1)
    labels = labels.astype(np.bool_, copy=False).reshape(-1)
    n = int(scores.size)
    positives = int(labels.sum())
    negatives = n - positives
    if n == 0:
        raise ValueError("Empty score array")
    proxy_area = positives / n
    inside = float(scores[labels].mean()) if positives else 0.0
    outside = float(scores[~labels].mean()) if negatives else 0.0
    ratio = inside / max(outside, 1e-12)
    score_mean = float(scores.mean())
    k_top = max(1, int(round(top_fraction * n)))

    if positives == 0:
        return {
            "method": method,
            "family": family,
            "status": status,
            "fold": fold,
            "pixels": float(n),
            "positives": positives,
            "proxy_area": proxy_area,
            "auprc": float("nan"),
            "best_f1": float("nan"),
            "best_iou": float("nan"),
            "best_precision": float("nan"),
            "best_recall": float("nan"),
            "best_threshold": float("nan"),
            "inside_score": inside,
            "outside_score": outside,
            "inside_outside_ratio": ratio,
            "top5_localization": 0.0,
            "top5_false_positive": 1.0,
            "top5_lift": float("nan"),
            "random_localization": proxy_area,
            "score_mean": score_mean,
            "note": note,
        }

    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    top_hit = float(sorted_labels[:k_top].mean())
    tp = np.cumsum(sorted_labels, dtype=np.float64)
    rank = np.arange(1, n + 1, dtype=np.float64)
    precision = tp / rank
    recall = tp / float(positives)
    auprc = float(precision[sorted_labels].sum() / float(positives))
    denom = precision + recall
    f1 = np.divide(2.0 * precision * recall, denom, out=np.zeros_like(precision), where=denom > 0)
    best_idx = int(np.argmax(f1))
    best_tp = float(tp[best_idx])
    best_rank = float(rank[best_idx])
    best_fp = best_rank - best_tp
    best_fn = float(positives) - best_tp
    best_iou = best_tp / max(best_tp + best_fp + best_fn, 1e-12)
    return {
        "method": method,
        "family": family,
        "status": status,
        "fold": fold,
        "pixels": float(n),
        "positives": positives,
        "proxy_area": proxy_area,
        "auprc": auprc,
        "best_f1": float(f1[best_idx]),
        "best_iou": float(best_iou),
        "best_precision": float(precision[best_idx]),
        "best_recall": float(recall[best_idx]),
        "best_threshold": float(scores[order[best_idx]]),
        "inside_score": inside,
        "outside_score": outside,
        "inside_outside_ratio": ratio,
        "top5_localization": top_hit,
        "top5_false_positive": 1.0 - top_hit,
        "top5_lift": top_hit / max(proxy_area, 1e-12),
        "random_localization": proxy_area,
        "score_mean": score_mean,
        "note": note,
    }


def method_family(method: str) -> str:
    if method.startswith("haar"):
        return "haar_dwt"
    if method in {"sobel", "laplacian"}:
        return "gradient_edge"
    if method.startswith("gaussian"):
        return "gaussian_pyramid"
    if method.startswith("fourier"):
        return "fourier_temporal"
    if method.startswith("learned"):
        return "learned_shallow"
    return "unknown"


def fmt(value: object, digits: int = 4) -> str:
    if not isinstance(value, (float, int)) or not math.isfinite(float(value)):
        return ""
    return f"{float(value):.{digits}f}"


def normalized_u8(arr: object) -> object:
    hi = float(np.percentile(arr, 99.0))
    if hi <= 1e-12:
        return np.zeros(arr.shape, dtype=np.uint8)
    return np.clip(arr / hi * 255.0, 0, 255).astype(np.uint8)


def write_panel_assets(path: Path, panels: Sequence[tuple[str, object]]) -> List[Path]:
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


def rgb_panel(arr: object) -> object:
    return cv2.cvtColor(np.clip(arr * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def heatmap(arr: object) -> object:
    return cv2.applyColorMap(normalized_u8(arr), cv2.COLORMAP_TURBO)


def binary_panel(arr: object) -> object:
    panel = np.zeros((*arr.shape, 3), dtype=np.uint8)
    panel[arr] = (0, 220, 0)
    return panel


def write_examples(output_dir: Path, samples: Sequence[FrameSample], methods: Sequence[str]) -> List[Path]:
    example_paths: List[Path] = []
    if not samples:
        return example_paths
    ranked = sorted(samples, key=lambda sample: float(np.mean(sample.label)), reverse=True)
    for idx, sample in enumerate(ranked[: max(0, min(len(ranked), 6))], start=1):
        maps = deterministic_maps(sample, methods, sample.label.shape[0])
        panels: List[tuple[str, object]] = [
            ("lq", rgb_panel(sample.curr_lq)),
            ("gt", rgb_panel(sample.curr_gt)),
            ("residual_proxy", binary_panel(sample.label)),
        ]
        for method in methods[:4]:
            panels.append((f"{method}_heatmap", heatmap(maps[method])))
        path = output_dir / "examples" / f"example_{idx:02d}_{sample.clip}_{sample.frame}.png"
        assets = write_panel_assets(path, panels)
        if assets:
            example_paths.append(path.with_suffix(""))
    return example_paths


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in SUMMARY_FIELDS} for row in rows])


def write_markdown(path: Path, rows: List[Dict[str, object]], protocol: Dict[str, object], examples: Sequence[str]) -> None:
    lines = [
        "# T20 Cross-Domain Frequency Decomposition Ablation",
        "",
        "Residual proxy labels are derived from paired `LQ-GT` frames; they are diagnostic labels, not generator-side artifact annotations.",
        "",
        f"- GT root: `{protocol.get('gt_root', '')}`",
        f"- LQ root: `{protocol.get('lq_root', '')}`",
        f"- Clips: `{protocol.get('clips', '')}`",
        f"- Frames per clip: `{protocol.get('frames_per_clip', '')}`",
        f"- Analysis size: `{protocol.get('size', '')}`",
        f"- Metric size: `{protocol.get('metric_size', '')}`",
        f"- Residual threshold: `{protocol.get('residual_threshold', '')}`",
        f"- Top-k fraction: `{protocol.get('top_fraction', '')}`",
        "",
        "| Method | Family | Status | Fold | Proxy area | AUPRC | Top-5 hit | Top-5 FPR | Lift | Inside/outside | Note |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('method', '')} | {row.get('family', '')} | {row.get('status', '')} | "
            f"{row.get('fold', '')} | {fmt(row.get('proxy_area'))} | {fmt(row.get('auprc'))} | "
            f"{fmt(row.get('top5_localization'))} | {fmt(row.get('top5_false_positive'))} | {fmt(row.get('top5_lift'))} | "
            f"{fmt(row.get('inside_outside_ratio'))} | {row.get('note', '')} |"
        )
    if examples:
        lines.extend(["", "## Qualitative Example Asset Directories", ""])
        for example in examples:
            lines.append(f"- `{example}`")
    lines.extend(
        [
            "",
            "AUPRC is the primary ranking metric because several deterministic operators reach their best F1 only at a near-all-positive threshold; raw best-F1 values remain in the CSV/JSON for audit.",
            "Interpretation rule: deterministic Haar support should be based on the fused row's balanced AUPRC, Top-5 hit, and Top-5 false-positive trade-off against Sobel/Laplacian, Gaussian pyramid, Fourier temporal, and learned-shallow rows.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def protocol_payload(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "mode": args.mode,
        "gt_root": str(resolve_path(args.repo_root, args.gt_root)),
        "lq_root": str(resolve_path(args.repo_root, args.lq_root)),
        "clips": args.clips,
        "frames_per_clip": args.frames_per_clip,
        "size": args.size,
        "metric_size": args.metric_size,
        "residual_threshold": args.residual_threshold,
        "top_fraction": args.top_fraction,
        "seed": args.seed,
        "label_note": "Residual-derived proxy regions from paired LQ-GT frames; not generator-side artifact annotations.",
    }


def run_deterministic(args: argparse.Namespace, methods: Sequence[str], partial_name: str) -> Dict[str, object]:
    samples = collect_samples(args)
    score_chunks: Dict[str, List[object]] = {method: [] for method in methods}
    label_chunks: Dict[str, List[object]] = {method: [] for method in methods}
    for sample in samples:
        maps = deterministic_maps(sample, methods, args.metric_size)
        for method in methods:
            score_chunks[method].append(maps[method].reshape(-1).astype(np.float32, copy=False))
            label_chunks[method].append(sample.label.reshape(-1).astype(np.bool_, copy=False))

    rows = []
    for method in methods:
        scores = np.concatenate(score_chunks[method], axis=0)
        labels = np.concatenate(label_chunks[method], axis=0)
        rows.append(
            summarize_scores(
                method,
                method_family(method),
                scores,
                labels,
                args.top_fraction,
                note="deterministic frequency operator",
            )
        )

    examples = [str(path) for path in write_examples(args.output_dir, samples, methods[:4])]
    payload = {
        "status": "done",
        "protocol": protocol_payload(args),
        "summary": rows,
        "examples": examples,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / partial_name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(args.output_dir / partial_name.replace(".json", ".csv"), rows)
    write_markdown(args.output_dir / partial_name.replace(".json", ".md"), rows, payload["protocol"], examples)
    return payload


def import_torch_or_block(output_dir: Path, args: argparse.Namespace) -> object:
    try:
        import torch  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised on env failures
        output_dir.mkdir(parents=True, exist_ok=True)
        row = {
            "method": LEARNED_METHOD,
            "family": method_family(LEARNED_METHOD),
            "status": "blocked-by-env",
            "fold": args.fold_index,
            "note": f"PyTorch import failed: {exc}",
        }
        payload = {
            "status": "blocked-by-env",
            "protocol": protocol_payload(args),
            "summary": [row],
            "examples": [],
        }
        (output_dir / "partial_learned.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        write_csv(output_dir / "partial_learned.csv", [row])
        write_markdown(output_dir / "partial_learned.md", [row], payload["protocol"], [])
        return None
    return torch


def learned_features(sample: FrameSample, metric_size: int) -> object:
    curr = resize_score(rgb_to_gray(sample.curr_lq), metric_size)
    prev = resize_score(rgb_to_gray(sample.prev_lq), metric_size)
    nxt = resize_score(rgb_to_gray(sample.next_lq), metric_size)
    low = cv2.GaussianBlur(curr, (0, 0), 5.0)
    high = normalize_score(np.abs(curr - low))
    low_score = normalize_score(np.abs(low - cv2.GaussianBlur(low, (0, 0), 11.0)))
    lap = normalize_score(np.abs(cv2.Laplacian(curr, cv2.CV_32F, ksize=3)))
    sx = cv2.Sobel(curr, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(curr, cv2.CV_32F, 0, 1, ksize=3)
    sobel = normalize_score(np.sqrt(sx * sx + sy * sy))
    temporal = normalize_score(0.5 * (np.abs(curr - prev) + np.abs(nxt - curr)))
    return np.stack([curr, high, low_score, lap, sobel, temporal], axis=0).astype(np.float32)


def split_fold(samples: Sequence[FrameSample], fold_index: int, num_folds: int) -> tuple[List[FrameSample], List[FrameSample]]:
    clips = sorted({sample.clip for sample in samples})
    fold_index = fold_index % max(1, num_folds)
    val_clips = {clip for idx, clip in enumerate(clips) if idx % max(1, num_folds) == fold_index}
    if not val_clips:
        val_clips = {clips[fold_index % len(clips)]}
    train = [sample for sample in samples if sample.clip not in val_clips]
    val = [sample for sample in samples if sample.clip in val_clips]
    if not train:
        train = val
    return train, val


def run_learned_fold(args: argparse.Namespace) -> Dict[str, object]:
    torch = import_torch_or_block(args.output_dir, args)
    if torch is None:
        return json.loads((args.output_dir / "partial_learned.json").read_text(encoding="utf-8"))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    samples = collect_samples(args)
    train_samples, val_samples = split_fold(samples, args.fold_index, args.num_folds)
    x_train = np.stack([learned_features(sample, args.metric_size) for sample in train_samples], axis=0)
    y_train = np.stack([sample.label.astype(np.float32) for sample in train_samples], axis=0)[:, None, :, :]
    x_val = np.stack([learned_features(sample, args.metric_size) for sample in val_samples], axis=0)
    y_val = np.stack([sample.label.astype(np.float32) for sample in val_samples], axis=0)[:, None, :, :]

    model = torch.nn.Conv2d(x_train.shape[1], 1, kernel_size=3, padding=1, bias=True).to(device)
    positives = float(y_train.sum())
    negatives = float(y_train.size - positives)
    pos_weight = torch.tensor([negatives / max(positives, 1.0)], dtype=torch.float32, device=device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    train_x = torch.from_numpy(x_train).to(device)
    train_y = torch.from_numpy(y_train).to(device)

    order = list(range(train_x.shape[0]))
    loss_trace: List[float] = []
    for _epoch in range(max(1, args.epochs)):
        random.shuffle(order)
        for start in range(0, len(order), max(1, args.batch_size)):
            idx = order[start:start + max(1, args.batch_size)]
            batch_x = train_x[idx]
            batch_y = train_y[idx]
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()
            loss_trace.append(float(loss.detach().cpu()))

    with torch.inference_mode():
        val_x = torch.from_numpy(x_val).to(device)
        logits = model(val_x)
        probs = torch.sigmoid(logits).detach().cpu().numpy()

    scores = probs.reshape(-1).astype(np.float32)
    labels = y_val.reshape(-1).astype(np.bool_)
    row = summarize_scores(
        LEARNED_METHOD,
        method_family(LEARNED_METHOD),
        scores,
        labels,
        args.top_fraction,
        fold=args.fold_index,
        note=f"shallow conv frequency probe; val_clips={len({s.clip for s in val_samples})}; final_loss={loss_trace[-1]:.6f}",
    )
    payload = {
        "status": "done",
        "protocol": {
            **protocol_payload(args),
            "fold_index": args.fold_index,
            "num_folds": args.num_folds,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "device": str(device),
            "train_frames": len(train_samples),
            "val_frames": len(val_samples),
            "loss_trace": loss_trace,
        },
        "summary": [row],
        "examples": [],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "partial_learned.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output_dir / "partial_learned.csv", [row])
    write_markdown(args.output_dir / "partial_learned.md", [row], payload["protocol"], [])
    return payload


def read_jsons(paths: Iterable[Path]) -> List[Dict[str, object]]:
    payloads: List[Dict[str, object]] = []
    for path in paths:
        try:
            payloads.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Bad JSON in {path}: {exc}") from exc
    return payloads


def merge_payloads(args: argparse.Namespace) -> Dict[str, object]:
    shard_root = args.output_dir / "shards"
    learned_root = args.output_dir / "learned_folds"
    shard_paths = sorted(shard_root.glob("shard_*/partial_deterministic.json"))
    learned_paths = sorted(learned_root.glob("fold_*/partial_learned.json"))
    if not shard_paths:
        raise FileNotFoundError(f"No deterministic shard JSONs under {shard_root}")

    rows: List[Dict[str, object]] = []
    sources: List[str] = []
    examples: List[str] = []
    for payload in read_jsons(shard_paths + learned_paths):
        sources.append(str(payload.get("protocol", {}).get("mode", "")))
        rows.extend(payload.get("summary", []))
        examples.extend(str(path) for path in payload.get("examples", []))

    method_seen: Dict[tuple[str, object], bool] = {}
    unique_rows: List[Dict[str, object]] = []
    for row in rows:
        key = (str(row.get("method", "")), row.get("fold", ""))
        if key in method_seen:
            continue
        method_seen[key] = True
        unique_rows.append(row)

    final = {
        "status": "done",
        "protocol": {
            **protocol_payload(args),
            "deterministic_shards": [str(path) for path in shard_paths],
            "learned_folds": [str(path) for path in learned_paths],
            "source_modes": sources,
        },
        "summary": unique_rows,
        "examples": examples,
    }
    final_json = args.output_dir / "cross_domain_frequency_ablation.json"
    final_csv = args.output_dir / "cross_domain_frequency_ablation.csv"
    final_md = args.output_dir / "cross_domain_frequency_ablation.md"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_json.write_text(json.dumps(final, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(final_csv, unique_rows)
    write_markdown(final_md, unique_rows, final["protocol"], examples)
    return final


def main() -> None:
    args = parse_args()
    args.repo_root = args.repo_root.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode != "merge" and not need_array_stack(args.output_dir):
        sys.exit(2)

    if args.mode == "deterministic-shard":
        if args.num_shards != len(DETERMINISTIC_SHARDS):
            raise ValueError(f"--num-shards must be {len(DETERMINISTIC_SHARDS)} for this script")
        methods = DETERMINISTIC_SHARDS[args.shard_index % len(DETERMINISTIC_SHARDS)]
        payload = run_deterministic(args, methods, "partial_deterministic.json")
    elif args.mode == "learned-fold":
        payload = run_learned_fold(args)
    elif args.mode == "merge":
        payload = merge_payloads(args)
    elif args.mode == "all":
        payload = run_deterministic(args, DETERMINISTIC_METHODS, "cross_domain_frequency_ablation.json")
        write_csv(args.output_dir / "cross_domain_frequency_ablation.csv", payload["summary"])
        write_markdown(
            args.output_dir / "cross_domain_frequency_ablation.md",
            payload["summary"],
            payload["protocol"],
            payload.get("examples", []),
        )
        if not args.no_learned:
            learned = run_learned_fold(args)
            payload["summary"].extend(learned.get("summary", []))
            (args.output_dir / "cross_domain_frequency_ablation.json").write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            write_csv(args.output_dir / "cross_domain_frequency_ablation.csv", payload["summary"])
            write_markdown(
                args.output_dir / "cross_domain_frequency_ablation.md",
                payload["summary"],
                payload["protocol"],
                payload.get("examples", []),
            )
    else:  # pragma: no cover
        raise AssertionError(args.mode)

    for row in payload.get("summary", []):
        print(
            row.get("method", ""),
            row.get("status", ""),
            fmt(row.get("auprc")),
            fmt(row.get("best_f1")),
            fmt(row.get("top5_localization")),
            fmt(row.get("top5_false_positive")),
            row.get("note", ""),
        )


if __name__ == "__main__":
    main()
