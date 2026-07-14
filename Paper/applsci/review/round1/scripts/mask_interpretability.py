#!/usr/bin/env python3
"""Mask interpretability probe for WaveletMask round-1 revision.

The SRWOV/REDS synthetic split used here does not expose generator-side
artifact masks. This script therefore builds residual-derived proxy regions
from paired LQ/GT frames and measures whether the trained WaveletNoiseMask
branches concentrate activation in those proxy regions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import cv2
import numpy as np
import torch


BRANCHES = ("local", "flicker", "wavelet")


@dataclass
class FrameRow:
    clip: str
    frame: str
    branch: str
    proxy_area: float
    score_mean: float
    inside_score: float
    outside_score: float
    inside_outside_ratio: float
    top5_localization: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".", type=Path)
    parser.add_argument("--gt-root", required=True, type=Path)
    parser.add_argument("--lq-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--clips", default=30, type=int, help="Number of clips; <=0 means all clips.")
    parser.add_argument("--frames-per-clip", default=20, type=int)
    parser.add_argument("--size", default=256, type=int)
    parser.add_argument("--metric-size", default=128, type=int)
    parser.add_argument("--residual-threshold", default=0.10, type=float)
    parser.add_argument("--local-highpass-quantile", default=0.75, type=float)
    parser.add_argument("--flicker-threshold", default=0.03, type=float)
    parser.add_argument("--top-fraction", default=0.05, type=float)
    parser.add_argument("--visual-examples", default=6, type=int)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--no-normalize-tanh", dest="normalize_tanh", action="store_false")
    parser.set_defaults(normalize_tanh=True)
    return parser.parse_args()


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


def read_rgb(path: Path, size: int) -> np.ndarray:
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


def rgb_to_gray(img: np.ndarray) -> np.ndarray:
    return (
        0.299 * img[..., 0]
        + 0.587 * img[..., 1]
        + 0.114 * img[..., 2]
    ).astype(np.float32)


def make_proxies(
    lq: np.ndarray,
    gt: np.ndarray,
    residual_threshold: float,
    local_highpass_quantile: float,
    flicker_threshold: float,
) -> Dict[str, np.ndarray]:
    residual = np.mean(np.abs(lq - gt), axis=2)
    signed_gray = rgb_to_gray(lq) - rgb_to_gray(gt)

    highpass = np.abs(signed_gray - cv2.GaussianBlur(signed_gray, (0, 0), 3.0))
    high_cut = float(np.quantile(highpass, local_highpass_quantile))
    local = (residual > residual_threshold) & (highpass >= high_cut)

    low = np.abs(cv2.GaussianBlur(signed_gray, (0, 0), 11.0))
    flicker = (residual > residual_threshold) & (low > flicker_threshold)

    wavelet = local | flicker
    return {
        "local": local.astype(np.bool_),
        "flicker": flicker.astype(np.bool_),
        "wavelet": wavelet.astype(np.bool_),
    }


def to_tensor(frames: Sequence[np.ndarray], device: torch.device, normalize_tanh: bool) -> torch.Tensor:
    arr = np.stack(frames, axis=0)
    tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).unsqueeze(0).to(device)
    if normalize_tanh:
        tensor = tensor * 2.0 - 1.0
    return tensor.float()


def resize_score(arr: np.ndarray, size: int) -> np.ndarray:
    if arr.shape[0] == size and arr.shape[1] == size:
        return arr.astype(np.float32, copy=False)
    return cv2.resize(arr.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)


def resize_label(arr: np.ndarray, size: int) -> np.ndarray:
    if arr.shape[0] == size and arr.shape[1] == size:
        return arr.astype(np.bool_, copy=False)
    out = cv2.resize(arr.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)
    return out.astype(np.bool_)


def frame_stats(score: np.ndarray, label: np.ndarray, top_fraction: float) -> tuple[float, float, float, float]:
    flat_score = score.reshape(-1).astype(np.float32)
    flat_label = label.reshape(-1).astype(np.bool_)
    inside = float(flat_score[flat_label].mean()) if flat_label.any() else 0.0
    outside = float(flat_score[~flat_label].mean()) if (~flat_label).any() else 0.0
    ratio = inside / max(outside, 1e-12)
    k = max(1, int(round(top_fraction * flat_score.size)))
    if k >= flat_score.size:
        top_idx = np.arange(flat_score.size)
    else:
        top_idx = np.argpartition(flat_score, -k)[-k:]
    top_rate = float(flat_label[top_idx].mean())
    return inside, outside, ratio, top_rate


def summarize_branch(scores: np.ndarray, labels: np.ndarray, top_fraction: float) -> Dict[str, float]:
    scores = scores.astype(np.float32, copy=False).reshape(-1)
    labels = labels.astype(np.bool_, copy=False).reshape(-1)
    n = int(scores.size)
    positives = int(labels.sum())
    negatives = n - positives
    if n == 0:
        raise ValueError("Empty score array")

    inside = float(scores[labels].mean()) if positives else 0.0
    outside = float(scores[~labels].mean()) if negatives else 0.0
    ratio = inside / max(outside, 1e-12)

    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    k_top = max(1, int(round(top_fraction * n)))
    top_rate = float(sorted_labels[:k_top].mean())
    proxy_area = positives / n

    if positives == 0:
        return {
            "pixels": float(n),
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
            "top5_localization": top_rate,
            "random_localization": proxy_area,
            "score_mean": float(scores.mean()),
        }

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
        "pixels": float(n),
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
        "top5_localization": top_rate,
        "random_localization": proxy_area,
        "score_mean": float(scores.mean()),
    }


def normalized_u8(arr: np.ndarray, p: float = 99.0) -> np.ndarray:
    hi = float(np.percentile(arr, p))
    if hi <= 1e-12:
        return np.zeros(arr.shape, dtype=np.uint8)
    return np.clip(arr / hi * 255.0, 0, 255).astype(np.uint8)


def heatmap(arr: np.ndarray) -> np.ndarray:
    colored = cv2.applyColorMap(normalized_u8(arr), cv2.COLORMAP_TURBO)
    return colored


def binary_panel(arr: np.ndarray) -> np.ndarray:
    panel = np.zeros((*arr.shape, 3), dtype=np.uint8)
    panel[arr] = (0, 220, 0)
    return panel


def rgb_panel(arr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.clip(arr * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


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


def write_visual(path: Path, item: Dict[str, object]) -> List[Path]:
    lq = item["lq"]  # type: ignore[assignment]
    gt = item["gt"]  # type: ignore[assignment]
    proxies = item["proxies"]  # type: ignore[assignment]
    masks = item["masks"]  # type: ignore[assignment]
    assert isinstance(lq, np.ndarray)
    assert isinstance(gt, np.ndarray)
    assert isinstance(proxies, dict)
    assert isinstance(masks, dict)

    panels = [
        ("lq", rgb_panel(lq)),
        ("gt", rgb_panel(gt)),
        ("residual_proxy", binary_panel(proxies["wavelet"])),
        ("m_local_heatmap", heatmap(masks["local"])),
        ("m_flicker_heatmap", heatmap(masks["flicker"])),
        ("m_wavelet_heatmap", heatmap(masks["wavelet"])),
    ]
    return write_panel_assets(path, panels)


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fields = [
        "branch",
        "pixels",
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
        "random_localization",
        "score_mean",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: object, digits: int = 4) -> str:
    if not isinstance(value, (float, int)) or not math.isfinite(float(value)):
        return ""
    return f"{float(value):.{digits}f}"


def write_markdown(path: Path, rows: List[Dict[str, object]], args: argparse.Namespace, example_paths: List[Path]) -> None:
    lines = [
        "# T8 Mask Interpretability and Residual-Proxy Overlap",
        "",
        "The SRWOV/REDS synthetic files used here do not expose generator-side pixel annotations.",
        "Overlap is therefore measured against residual-derived `LQ-GT` proxy regions, not true artifact masks.",
        "",
        f"- Clips: `{args.clips}`",
        f"- Frames per clip: `{args.frames_per_clip}`",
        f"- Analysis size: `{args.size}`",
        f"- Metric size: `{args.metric_size}`",
        f"- Residual threshold: `{args.residual_threshold}`",
        f"- Flicker low-frequency threshold: `{args.flicker_threshold}`",
        f"- Top-k fraction: `{args.top_fraction}`",
        f"- Checkpoint: `{args.checkpoint}`",
        "",
        "| Branch | Proxy area | AUPRC | Best F1 | Best IoU | Precision | Recall | Inside/outside | Top-5% hit | Random hit |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['branch']} | {fmt(row['proxy_area'])} | {fmt(row['auprc'])} | "
            f"{fmt(row['best_f1'])} | {fmt(row['best_iou'])} | {fmt(row['best_precision'])} | "
            f"{fmt(row['best_recall'])} | {fmt(row['inside_outside_ratio'])} | "
            f"{fmt(row['top5_localization'])} | {fmt(row['random_localization'])} |"
        )
    if example_paths:
        lines.extend(["", "## Qualitative Example Asset Directories", ""])
        for example in example_paths:
            lines.append(f"- `{example}`")
    lines.append("")
    lines.append("AUPRC and best-F1 thresholds are diagnostic because the proxy labels are derived from residuals and were not used for training.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    model = load_wavelet_mask(args.repo_root, args.checkpoint, device)
    clips = list_clips(args.gt_root, args.clips)

    score_chunks: Dict[str, List[np.ndarray]] = {branch: [] for branch in BRANCHES}
    label_chunks: Dict[str, List[np.ndarray]] = {branch: [] for branch in BRANCHES}
    frame_rows: List[FrameRow] = []
    visual_candidates: List[Dict[str, object]] = []

    with torch.inference_mode():
        for clip in clips:
            gt_files = sorted(clip.glob("*.png"))
            indices = sample_indices(len(gt_files), args.frames_per_clip)
            if not indices:
                continue
            for idx in indices:
                frame_paths = [gt_files[idx - 1], gt_files[idx], gt_files[idx + 1]]
                lq_paths = [args.lq_root / clip.name / path.name for path in frame_paths]
                lq_frames = [read_rgb(path, args.size) for path in lq_paths]
                gt_curr = read_rgb(gt_files[idx], args.size)
                lq_curr = lq_frames[1]

                tensor = to_tensor(lq_frames, device, args.normalize_tanh)
                local = model.compute_local_mask(tensor)[0, 1, 0].detach().float().cpu().numpy()
                flicker = model.compute_flicker_mask(tensor)[0, 1, 0].detach().float().cpu().numpy()
                masks = {
                    "local": local,
                    "flicker": flicker,
                    "wavelet": np.maximum(local, flicker),
                }
                proxies = make_proxies(
                    lq_curr,
                    gt_curr,
                    args.residual_threshold,
                    args.local_highpass_quantile,
                    args.flicker_threshold,
                )

                metric_scores = {
                    branch: resize_score(masks[branch], args.metric_size).reshape(-1)
                    for branch in BRANCHES
                }
                metric_labels = {
                    branch: resize_label(proxies[branch], args.metric_size).reshape(-1)
                    for branch in BRANCHES
                }
                for branch in BRANCHES:
                    score = metric_scores[branch]
                    label = metric_labels[branch]
                    score_chunks[branch].append(score.astype(np.float32, copy=False))
                    label_chunks[branch].append(label.astype(np.bool_, copy=False))
                    inside, outside, ratio, top_rate = frame_stats(score, label, args.top_fraction)
                    frame_rows.append(
                        FrameRow(
                            clip=clip.name,
                            frame=gt_files[idx].name,
                            branch=branch,
                            proxy_area=float(label.mean()),
                            score_mean=float(score.mean()),
                            inside_score=inside,
                            outside_score=outside,
                            inside_outside_ratio=ratio,
                            top5_localization=top_rate,
                        )
                    )

                candidate = {
                    "rank_score": float(proxies["wavelet"].mean()),
                    "clip": clip.name,
                    "frame": gt_files[idx].name,
                    "lq": lq_curr,
                    "gt": gt_curr,
                    "proxies": proxies,
                    "masks": masks,
                }
                visual_candidates.append(candidate)
                visual_candidates = sorted(
                    visual_candidates,
                    key=lambda item: float(item["rank_score"]),
                    reverse=True,
                )[: max(args.visual_examples, 0)]

    summary_rows: List[Dict[str, object]] = []
    for branch in BRANCHES:
        if not score_chunks[branch]:
            raise RuntimeError(f"No samples collected for branch {branch}")
        scores = np.concatenate(score_chunks[branch], axis=0)
        labels = np.concatenate(label_chunks[branch], axis=0)
        metrics = summarize_branch(scores, labels, args.top_fraction)
        metrics["branch"] = branch
        summary_rows.append(metrics)

    examples_dir = args.output_dir / "examples"
    example_paths: List[Path] = []
    example_assets: Dict[str, List[str]] = {}
    for idx, item in enumerate(visual_candidates, start=1):
        path = examples_dir / f"example_{idx:02d}_{item['clip']}_{item['frame']}"
        path = path.with_suffix(".png")
        assets = write_visual(path, item)
        example_dir = path.with_suffix("")
        example_paths.append(example_dir)
        example_assets[str(example_dir)] = [str(asset) for asset in assets]

    payload = {
        "protocol": {
            "gt_root": str(args.gt_root),
            "lq_root": str(args.lq_root),
            "checkpoint": str(args.checkpoint),
            "clips": args.clips,
            "frames_per_clip": args.frames_per_clip,
            "size": args.size,
            "metric_size": args.metric_size,
            "normalize_tanh": args.normalize_tanh,
            "residual_threshold": args.residual_threshold,
            "local_highpass_quantile": args.local_highpass_quantile,
            "flicker_threshold": args.flicker_threshold,
            "top_fraction": args.top_fraction,
            "label_note": "Residual-derived proxy regions from paired LQ-GT frames; not generator-side artifact annotations.",
        },
        "summary": summary_rows,
        "frame_rows": [row.__dict__ for row in frame_rows],
        "examples": [str(path) for path in example_paths],
        "example_assets": example_assets,
    }
    (args.output_dir / "mask_interpretability.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_csv(args.output_dir / "mask_interpretability.csv", summary_rows)
    write_markdown(args.output_dir / "mask_interpretability.md", summary_rows, args, example_paths)

    for row in summary_rows:
        print(
            row["branch"],
            fmt(row["proxy_area"]),
            fmt(row["auprc"]),
            fmt(row["best_f1"]),
            fmt(row["best_iou"]),
            fmt(row["inside_outside_ratio"]),
            fmt(row["top5_localization"]),
            fmt(row["random_localization"]),
        )


if __name__ == "__main__":
    main()
