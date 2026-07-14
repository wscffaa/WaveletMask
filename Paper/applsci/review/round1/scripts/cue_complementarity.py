#!/usr/bin/env python3
"""Cue complementarity probe for WaveletMask round-1 revision."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import cv2
import numpy as np
import torch


CUES = ("spatial", "local", "flicker", "wavelet")
PAIR_CUES = ("spatial", "local", "flicker")
PAIRS = (("spatial", "local"), ("spatial", "flicker"), ("local", "flicker"))


@dataclass
class PairFrameRow:
    clip: str
    frame_index: int
    cue_a: str
    cue_b: str
    pearson: float
    spearman: float
    topk_overlap: float
    topk_jaccard: float


@dataclass
class CueFrameRow:
    clip: str
    frame_index: int
    cue: str
    proxy_area: float
    topk_hit: float
    topk_lift: float
    score_mean: float


@dataclass
class UnionFrameRow:
    clip: str
    frame_index: int
    proxy_area: float
    union_area: float
    union_hit: float
    union_lift: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".", type=Path)
    parser.add_argument("--lq-root", required=True, type=Path)
    parser.add_argument("--gt-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--clips", default=12, type=int)
    parser.add_argument("--frames", default=7, type=int)
    parser.add_argument("--size", default=128, type=int)
    parser.add_argument("--metric-size", default=64, type=int)
    parser.add_argument("--residual-threshold", default=0.10, type=float)
    parser.add_argument("--top-fraction", default=0.05, type=float)
    parser.add_argument("--visual-examples", default=6, type=int)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--flow-type", default="raft", choices=("raft", "spynet"))
    parser.add_argument("--no-normalize-tanh", dest="normalize_tanh", action="store_false")
    parser.set_defaults(normalize_tanh=True)
    return parser.parse_args()


def load_models(repo_root: Path, checkpoint: Path, device: torch.device, flow_type: str):
    sys.path.insert(0, str(repo_root / "framework"))
    sys.path.insert(0, str(repo_root / "projects/007-waveletmask/archs"))
    from basicofr.archs.components.arch_util import flow_warp  # type: ignore
    from basicofr.archs.flow import FlowEstimator  # type: ignore
    from components.wavelet_mask import WaveletNoiseMask  # type: ignore

    flow = FlowEstimator(estimator_type=flow_type, device=device, normalization="tanh")
    wavelet = WaveletNoiseMask().to(device)
    payload = torch.load(checkpoint, map_location="cpu")
    state = payload.get("params", payload)
    prefix = "wavelet_mask."
    sub_state = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
    if not sub_state:
        raise RuntimeError(f"No keys with prefix {prefix!r} found in {checkpoint}")
    missing, unexpected = wavelet.load_state_dict(sub_state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Bad wavelet state: missing={missing}, unexpected={unexpected}")
    wavelet.eval()
    return flow, flow_warp, wavelet


def list_clips(root: Path, count: int) -> List[Path]:
    clips = sorted(path for path in root.iterdir() if path.is_dir())
    if len(clips) < count:
        raise ValueError(f"Need {count} clips, found {len(clips)} in {root}")
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


def load_sequence(root: Path, clip_name: str, frames: int, size: int) -> tuple[np.ndarray, List[str]]:
    files = sorted((root / clip_name).glob("*.png"))
    if len(files) < frames:
        raise ValueError(f"Need {frames} frames in {root / clip_name}, found {len(files)}")
    center = len(files) // 2
    start = max(0, center - frames // 2)
    selected = files[start:start + frames]
    if len(selected) < frames:
        selected = files[:frames]
    return np.stack([read_frame(path, size) for path in selected], axis=0), [path.name for path in selected]


def to_tensor(seq: np.ndarray, device: torch.device, normalize_tanh: bool) -> torch.Tensor:
    tensor = torch.from_numpy(seq).permute(0, 3, 1, 2).unsqueeze(0).to(device)
    if normalize_tanh:
        tensor = tensor * 2.0 - 1.0
    return tensor.float()


def spatial_indicator(flow_estimator, flow_warp, tensor: torch.Tensor) -> torch.Tensor:
    n, t, _c, h, w = tensor.shape
    forward_flow, _backward_flow = flow_estimator.compute_flow(tensor)
    masks = [tensor.new_zeros(n, 1, h, w)]
    for idx in range(1, t):
        flow = forward_flow[:, idx - 1]
        pixel_prop = flow_warp(tensor[:, idx - 1], flow.permute(0, 2, 3, 1))
        curr = tensor[:, idx]
        masks.append(torch.abs(pixel_prop[:, :1] - curr[:, :1]))
    return torch.stack(masks, dim=1)


def proxy_mask(lq: np.ndarray, gt: np.ndarray, threshold: float, size: int) -> np.ndarray:
    residual = np.mean(np.abs(lq - gt), axis=2)
    if residual.shape[0] != size or residual.shape[1] != size:
        residual = cv2.resize(residual, (size, size), interpolation=cv2.INTER_AREA)
    return residual > threshold


def resize_score(arr: np.ndarray, size: int) -> np.ndarray:
    if arr.shape[0] == size and arr.shape[1] == size:
        return arr.astype(np.float32, copy=False)
    return cv2.resize(arr.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)


def resize_label(arr: np.ndarray, size: int) -> np.ndarray:
    if arr.shape[0] == size and arr.shape[1] == size:
        return arr.astype(np.bool_, copy=False)
    out = cv2.resize(arr.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)
    return out.astype(np.bool_)


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(np.float64, copy=False)
    y = y.astype(np.float64, copy=False)
    x = x - float(x.mean())
    y = y - float(y.mean())
    denom = math.sqrt(float(np.dot(x, x)) * float(np.dot(y, y)))
    if denom <= 1e-20:
        return float("nan")
    return float(np.dot(x, y) / denom)


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.shape[0]:
        end = start + 1
        while end < values.shape[0] and sorted_values[end] == sorted_values[start]:
            end += 1
        rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = rank
        start = end
    return ranks


def topk_mask(values: np.ndarray, fraction: float) -> np.ndarray:
    flat = values.reshape(-1).astype(np.float32)
    k = max(1, int(round(fraction * flat.size)))
    if k >= flat.size:
        out = np.ones(flat.size, dtype=np.bool_)
    else:
        idx = np.argpartition(flat, -k)[-k:]
        out = np.zeros(flat.size, dtype=np.bool_)
        out[idx] = True
    return out


def mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def aggregate_pairs(rows: List[PairFrameRow]) -> List[Dict[str, object]]:
    output: List[Dict[str, object]] = []
    for cue_a, cue_b in PAIRS:
        selected = [row for row in rows if row.cue_a == cue_a and row.cue_b == cue_b]
        output.append({
            "pair": f"{cue_a}/{cue_b}",
            "n": len(selected),
            "pearson": mean(row.pearson for row in selected),
            "spearman": mean(row.spearman for row in selected),
            "topk_overlap": mean(row.topk_overlap for row in selected),
            "topk_jaccard": mean(row.topk_jaccard for row in selected),
        })
    return output


def aggregate_cues(rows: List[CueFrameRow], union_rows: List[UnionFrameRow]) -> List[Dict[str, object]]:
    output: List[Dict[str, object]] = []
    for cue in CUES:
        selected = [row for row in rows if row.cue == cue]
        output.append({
            "cue": cue,
            "n": len(selected),
            "proxy_area": mean(row.proxy_area for row in selected),
            "topk_hit": mean(row.topk_hit for row in selected),
            "topk_lift": mean(row.topk_lift for row in selected),
            "score_mean": mean(row.score_mean for row in selected),
            "union_area": float("nan"),
            "union_hit": float("nan"),
            "union_lift": float("nan"),
        })
    output.append({
        "cue": "top5_union_spatial_local_flicker",
        "n": len(union_rows),
        "proxy_area": mean(row.proxy_area for row in union_rows),
        "topk_hit": float("nan"),
        "topk_lift": float("nan"),
        "score_mean": float("nan"),
        "union_area": mean(row.union_area for row in union_rows),
        "union_hit": mean(row.union_hit for row in union_rows),
        "union_lift": mean(row.union_lift for row in union_rows),
    })
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


def write_visual(path: Path, lq: np.ndarray, proxy: np.ndarray, scores: Dict[str, np.ndarray]) -> List[Path]:
    proxy_panel = np.zeros((*proxy.shape, 3), dtype=np.uint8)
    proxy_panel[proxy] = (0, 220, 0)
    panels = [
        ("lq", cv2.cvtColor(np.clip(lq * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)),
        ("residual_proxy", proxy_panel),
    ]
    for cue in CUES:
        panels.append((f"{cue}_heatmap", cv2.applyColorMap(normalized_u8(scores[cue]), cv2.COLORMAP_TURBO)))
    return write_panel_assets(path, panels)


def write_csv(path: Path, rows: List[Dict[str, object]], fields: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: object, digits: int = 4) -> str:
    if not isinstance(value, (float, int)) or not math.isfinite(float(value)):
        return ""
    return f"{float(value):.{digits}f}"


def write_markdown(
    path: Path,
    pair_rows: List[Dict[str, object]],
    cue_rows: List[Dict[str, object]],
    args: argparse.Namespace,
    examples: List[Path],
) -> None:
    lines = [
        "# T11 Cue Complementarity Probe",
        "",
        "The RTN spatial indicator is computed with the same flow-aligned forward mask contract used by the baseline implementation.",
        "Residual proxy hits use paired LQ-GT residuals and are diagnostic rather than generator-side artifact labels.",
        "",
        f"- Clips: `{args.clips}`",
        f"- Frames per clip: `{args.frames}`",
        f"- Spatial size: `{args.size}`",
        f"- Metric size: `{args.metric_size}`",
        f"- Top-k fraction: `{args.top_fraction}`",
        f"- Flow type: `{args.flow_type}`",
        f"- Checkpoint: `{args.checkpoint}`",
        "",
        "## Pairwise Cue Relation",
        "",
        "| Pair | N | Pearson | Spearman | Top-5 overlap | Top-5 Jaccard |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in pair_rows:
        lines.append(
            f"| {row['pair']} | {row['n']} | {fmt(row['pearson'])} | {fmt(row['spearman'])} | "
            f"{fmt(row['topk_overlap'])} | {fmt(row['topk_jaccard'])} |"
        )
    lines.extend([
        "",
        "## Residual-Proxy Top-5% Hit",
        "",
        "| Cue | N | Proxy area | Top-5 hit | Lift | Union area | Union hit | Union lift |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in cue_rows:
        lines.append(
            f"| {row['cue']} | {row['n']} | {fmt(row['proxy_area'])} | {fmt(row['topk_hit'])} | "
            f"{fmt(row['topk_lift'])} | {fmt(row['union_area'])} | {fmt(row['union_hit'])} | "
            f"{fmt(row['union_lift'])} |"
        )
    if examples:
        lines.extend(["", "## Qualitative Example Asset Directories", ""])
        for example in examples:
            lines.append(f"- `{example}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    flow, flow_warp, wavelet = load_models(args.repo_root, args.checkpoint, device, args.flow_type)
    clips = list_clips(args.lq_root, args.clips)

    pair_rows: List[PairFrameRow] = []
    cue_rows: List[CueFrameRow] = []
    union_rows: List[UnionFrameRow] = []
    visual_items: List[tuple[float, Path, np.ndarray, np.ndarray, Dict[str, np.ndarray]]] = []

    with torch.inference_mode():
        for clip in clips:
            lq_seq, _names = load_sequence(args.lq_root, clip.name, args.frames, args.size)
            gt_seq, _ = load_sequence(args.gt_root, clip.name, args.frames, args.size)
            tensor = to_tensor(lq_seq, device, args.normalize_tanh)
            spatial = spatial_indicator(flow, flow_warp, tensor)[0, :, 0].detach().float().cpu().numpy()
            local = wavelet.compute_local_mask(tensor)[0, :, 0].detach().float().cpu().numpy()
            flicker = wavelet.compute_flicker_mask(tensor)[0, :, 0].detach().float().cpu().numpy()
            masks = {
                "spatial": spatial,
                "local": local,
                "flicker": flicker,
                "wavelet": np.maximum(local, flicker),
            }

            for frame_idx in range(1, args.frames - 1):
                scores = {cue: resize_score(masks[cue][frame_idx], args.metric_size) for cue in CUES}
                flat = {cue: scores[cue].reshape(-1).astype(np.float32) for cue in CUES}
                top_sets = {cue: topk_mask(flat[cue], args.top_fraction) for cue in CUES}

                for cue_a, cue_b in PAIRS:
                    a, b = flat[cue_a], flat[cue_b]
                    top_a, top_b = top_sets[cue_a], top_sets[cue_b]
                    intersection = int(np.logical_and(top_a, top_b).sum())
                    union = int(np.logical_or(top_a, top_b).sum())
                    pair_rows.append(
                        PairFrameRow(
                            clip=clip.name,
                            frame_index=frame_idx,
                            cue_a=cue_a,
                            cue_b=cue_b,
                            pearson=pearson(a, b),
                            spearman=pearson(rankdata(a), rankdata(b)),
                            topk_overlap=intersection / max(1, int(top_a.sum())),
                            topk_jaccard=intersection / max(1, union),
                        )
                    )

                proxy = resize_label(
                    proxy_mask(lq_seq[frame_idx], gt_seq[frame_idx], args.residual_threshold, args.size),
                    args.metric_size,
                ).reshape(-1)
                proxy_area = float(proxy.mean())
                for cue in CUES:
                    top = top_sets[cue]
                    top_hit = float(proxy[top].mean())
                    cue_rows.append(
                        CueFrameRow(
                            clip=clip.name,
                            frame_index=frame_idx,
                            cue=cue,
                            proxy_area=proxy_area,
                            topk_hit=top_hit,
                            topk_lift=top_hit / max(proxy_area, 1e-12),
                            score_mean=float(flat[cue].mean()),
                        )
                    )

                cue_union = np.logical_or.reduce([top_sets[cue] for cue in PAIR_CUES])
                union_hit = float(proxy[cue_union].mean())
                union_rows.append(
                    UnionFrameRow(
                        clip=clip.name,
                        frame_index=frame_idx,
                        proxy_area=proxy_area,
                        union_area=float(cue_union.mean()),
                        union_hit=union_hit,
                        union_lift=union_hit / max(proxy_area, 1e-12),
                    )
                )

                if len(visual_items) < args.visual_examples:
                    proxy_img = proxy.reshape(args.metric_size, args.metric_size)
                    score_img = {cue: scores[cue] for cue in CUES}
                    path = args.output_dir / "examples" / f"{clip.name}_{frame_idx:02d}.png"
                    visual_items.append((proxy_area, path, lq_seq[frame_idx], proxy_img, score_img))

    pair_summary = aggregate_pairs(pair_rows)
    cue_summary = aggregate_cues(cue_rows, union_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    examples: List[Path] = []
    example_assets: Dict[str, List[str]] = {}
    for _score, path, lq, proxy, scores in visual_items:
        proxy_full = cv2.resize(proxy.astype(np.uint8), (args.size, args.size), interpolation=cv2.INTER_NEAREST).astype(bool)
        scores_full = {cue: resize_score(scores[cue], args.size) for cue in CUES}
        assets = write_visual(path, lq, proxy_full, scores_full)
        example_dir = path.with_suffix("")
        examples.append(example_dir)
        example_assets[str(example_dir)] = [str(asset) for asset in assets]

    payload = {
        "protocol": {
            "lq_root": str(args.lq_root),
            "gt_root": str(args.gt_root),
            "checkpoint": str(args.checkpoint),
            "clips": args.clips,
            "frames": args.frames,
            "size": args.size,
            "metric_size": args.metric_size,
            "residual_threshold": args.residual_threshold,
            "top_fraction": args.top_fraction,
            "flow_type": args.flow_type,
            "normalize_tanh": args.normalize_tanh,
            "note": "Flow-aligned RTN spatial indicator versus trained wavelet masks; residual proxy is LQ-GT derived.",
        },
        "pair_summary": pair_summary,
        "cue_summary": cue_summary,
        "pair_rows": [row.__dict__ for row in pair_rows],
        "cue_rows": [row.__dict__ for row in cue_rows],
        "union_rows": [row.__dict__ for row in union_rows],
        "examples": [str(path) for path in examples],
        "example_assets": example_assets,
    }
    (args.output_dir / "cue_complementarity.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_csv(
        args.output_dir / "cue_complementarity_pairs.csv",
        pair_summary,
        ("pair", "n", "pearson", "spearman", "topk_overlap", "topk_jaccard"),
    )
    write_csv(
        args.output_dir / "cue_complementarity_proxy.csv",
        cue_summary,
        ("cue", "n", "proxy_area", "topk_hit", "topk_lift", "score_mean", "union_area", "union_hit", "union_lift"),
    )
    write_markdown(args.output_dir / "cue_complementarity.md", pair_summary, cue_summary, args, examples)

    for row in pair_summary:
        print(
            "pair",
            row["pair"],
            row["n"],
            fmt(row["pearson"]),
            fmt(row["spearman"]),
            fmt(row["topk_overlap"]),
            fmt(row["topk_jaccard"]),
        )
    for row in cue_summary:
        print(
            "cue",
            row["cue"],
            row["n"],
            fmt(row["proxy_area"]),
            fmt(row["topk_hit"]),
            fmt(row["topk_lift"]),
            fmt(row["union_area"]),
            fmt(row["union_hit"]),
            fmt(row["union_lift"]),
        )


if __name__ == "__main__":
    main()
