#!/usr/bin/env python3
"""Motion/occlusion-binned benchmark for WaveletMask round-1 revision (R2-C2 / T6).

Computes RAFT-based motion descriptors (overall magnitude, camera/object split,
forward-backward occlusion ratio) on the GT frames of the 30 fixed SRWOV/REDS
synthetic clips, measures detector clean-region false-positive behaviour on the
degraded inputs, joins the per-clip restoration metrics already logged by the
re-trained RTN baseline and WaveletMask tests, and reports everything stratified
by motion/occlusion bins.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Sequence

import cv2
import numpy as np
import torch


METRICS = ("psnr", "ssim", "lpips", "dists")
HIGHER_IS_BETTER = {"psnr": True, "ssim": True, "lpips": False, "dists": False}
CUES = ("spatial", "local", "flicker", "wavelet")


@dataclass
class ClipRow:
    clip: str
    motion_mag: float
    camera_mag: float
    object_mag: float
    occlusion_ratio: float
    base_psnr: float
    wavelet_psnr: float
    delta_psnr: float
    delta_ssim: float
    delta_lpips: float
    delta_dists: float
    spatial_top5_clean: float
    local_top5_clean: float
    flicker_top5_clean: float
    wavelet_top5_clean: float
    motion_bin: str = ""
    occlusion_bin: str = ""
    dominance: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".", type=Path)
    parser.add_argument("--gt-root", default=Path("framework/datasets/val/REDS/GT/val_sharp"), type=Path)
    parser.add_argument("--lq-root", default=Path("framework/datasets/val/REDS/LQ/val_sharp_degraded"), type=Path)
    parser.add_argument(
        "--base-log",
        default=Path(
            "projects/007-waveletmask/Exp/results/waveletmask_base_reds_synthetic_best/"
            "test_waveletmask_base_reds_synthetic_best_20260415_174118.log"
        ),
        type=Path,
    )
    parser.add_argument(
        "--wavelet-log",
        default=Path(
            "projects/007-waveletmask/Exp/results/waveletmask_ab_reds_synthetic_best/"
            "test_waveletmask_ab_reds_synthetic_best_20260415_191427.log"
        ),
        type=Path,
    )
    parser.add_argument(
        "--checkpoint",
        default=Path("projects/007-waveletmask/Exp/checkpoints/waveletmask_ab/net_g_best.pth"),
        type=Path,
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--clips", default=30, type=int)
    parser.add_argument("--frames", default=7, type=int)
    parser.add_argument("--size", default=128, type=int)
    parser.add_argument("--metric-size", default=64, type=int)
    parser.add_argument("--residual-threshold", default=0.10, type=float)
    parser.add_argument("--top-fraction", default=0.05, type=float)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--flow-type", default="raft", choices=("raft", "spynet"))
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
        metrics[match.group("clip")] = {name: float(match.group(name)) for name in METRICS}
    if not metrics:
        raise ValueError(f"No per-clip metrics found in {log_path}")
    return metrics


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


def load_sequence(root: Path, clip_name: str, frames: int, size: int) -> np.ndarray:
    files = sorted((root / clip_name).glob("*.png"))
    if len(files) < frames:
        raise ValueError(f"Need {frames} frames in {root / clip_name}, found {len(files)}")
    center = len(files) // 2
    start = max(0, center - frames // 2)
    selected = files[start:start + frames]
    if len(selected) < frames:
        selected = files[:frames]
    return np.stack([read_frame(path, size) for path in selected], axis=0)


def to_tensor(seq: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(seq).permute(0, 3, 1, 2).unsqueeze(0).to(device)
    return (tensor * 2.0 - 1.0).float()


def spatial_indicator(flow_estimator, flow_warp, tensor: torch.Tensor) -> torch.Tensor:
    n, t, _c, h, w = tensor.shape
    forward_flow, _ = flow_estimator.compute_flow(tensor)
    masks = [tensor.new_zeros(n, 1, h, w)]
    for idx in range(1, t):
        flow = forward_flow[:, idx - 1]
        pixel_prop = flow_warp(tensor[:, idx - 1], flow.permute(0, 2, 3, 1))
        curr = tensor[:, idx]
        masks.append(torch.abs(pixel_prop[:, :1] - curr[:, :1]))
    return torch.stack(masks, dim=1)


def motion_descriptors(flow_estimator, flow_warp, gt_tensor: torch.Tensor) -> Dict[str, float]:
    """RAFT-based per-clip motion statistics computed on clean GT frames."""
    forward_flow, backward_flow = flow_estimator.compute_flow(gt_tensor)
    fwd = forward_flow[0]  # (t-1, 2, h, w)
    bwd = backward_flow[0]

    mags: List[float] = []
    camera: List[float] = []
    objects: List[float] = []
    occlusion: List[float] = []
    for idx in range(fwd.shape[0]):
        f = fwd[idx]
        mag = torch.sqrt(f[0] ** 2 + f[1] ** 2)
        mags.append(float(mag.mean()))

        med_u = float(f[0].median())
        med_v = float(f[1].median())
        camera.append(math.sqrt(med_u ** 2 + med_v ** 2))
        residual = torch.sqrt((f[0] - med_u) ** 2 + (f[1] - med_v) ** 2)
        objects.append(float(residual.mean()))

        b = bwd[idx].unsqueeze(0)
        b_warped = flow_warp(b, f.unsqueeze(0).permute(0, 2, 3, 1))[0]
        diff_sq = (f[0] + b_warped[0]) ** 2 + (f[1] + b_warped[1]) ** 2
        f_sq = f[0] ** 2 + f[1] ** 2
        b_sq = b_warped[0] ** 2 + b_warped[1] ** 2
        occluded = diff_sq > 0.01 * (f_sq + b_sq) + 0.5
        occlusion.append(float(occluded.float().mean()))

    return {
        "motion_mag": float(np.mean(mags)),
        "camera_mag": float(np.mean(camera)),
        "object_mag": float(np.mean(objects)),
        "occlusion_ratio": float(np.mean(occlusion)),
    }


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


def topk_mask(values: np.ndarray, fraction: float) -> np.ndarray:
    flat = values.reshape(-1).astype(np.float32)
    k = max(1, int(round(fraction * flat.size)))
    if k >= flat.size:
        return np.ones(flat.size, dtype=np.bool_)
    idx = np.argpartition(flat, -k)[-k:]
    out = np.zeros(flat.size, dtype=np.bool_)
    out[idx] = True
    return out


def clean_fp_rates(
    lq_seq: np.ndarray,
    gt_seq: np.ndarray,
    masks: Dict[str, np.ndarray],
    frames: int,
    metric_size: int,
    threshold: float,
    top_fraction: float,
    size: int,
) -> Dict[str, float]:
    """Fraction of each cue's top-k activation that falls outside the residual proxy."""
    per_cue: Dict[str, List[float]] = {cue: [] for cue in CUES}
    for frame_idx in range(1, frames - 1):
        proxy = resize_label(
            proxy_mask(lq_seq[frame_idx], gt_seq[frame_idx], threshold, size), metric_size
        ).reshape(-1)
        for cue in CUES:
            score = resize_score(masks[cue][frame_idx], metric_size).reshape(-1)
            top = topk_mask(score, top_fraction)
            per_cue[cue].append(float((~proxy[top]).mean()))
    return {cue: float(np.mean(vals)) for cue, vals in per_cue.items()}


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    def rankdata(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty(values.shape[0], dtype=np.float64)
        sorted_values = values[order]
        start = 0
        while start < values.shape[0]:
            end = start + 1
            while end < values.shape[0] and sorted_values[end] == sorted_values[start]:
                end += 1
            ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
            start = end
        return ranks

    a = rankdata(np.asarray(x, dtype=np.float64))
    b = rankdata(np.asarray(y, dtype=np.float64))
    a = a - a.mean()
    b = b - b.mean()
    denom = math.sqrt(float(np.dot(a, a)) * float(np.dot(b, b)))
    if denom <= 1e-20:
        return float("nan")
    return float(np.dot(a, b) / denom)


def tercile_bins(rows: List[ClipRow], key: str, attr: str, labels: Sequence[str]) -> None:
    ordered = sorted(rows, key=lambda row: getattr(row, key))
    n = len(ordered)
    cut1, cut2 = n // 3, (2 * n) // 3
    for idx, row in enumerate(ordered):
        if idx < cut1:
            setattr(row, attr, labels[0])
        elif idx < cut2:
            setattr(row, attr, labels[1])
        else:
            setattr(row, attr, labels[2])


def aggregate(rows: List[ClipRow], attr: str, label: str) -> Dict[str, object]:
    out: Dict[str, object] = {
        "group": attr,
        "bin": label,
        "n_clips": len(rows),
        "motion_mag": float(np.mean([row.motion_mag for row in rows])),
        "occlusion_ratio": float(np.mean([row.occlusion_ratio for row in rows])),
        "base_psnr": float(np.mean([row.base_psnr for row in rows])),
        "wavelet_psnr": float(np.mean([row.wavelet_psnr for row in rows])),
    }
    for metric in METRICS:
        out[f"delta_{metric}"] = float(np.mean([getattr(row, f"delta_{metric}") for row in rows]))
    for cue in CUES:
        out[f"{cue}_top5_clean"] = float(np.mean([getattr(row, f"{cue}_top5_clean") for row in rows]))
    return out


def fmt(value: object, digits: int = 4) -> str:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return ""
    return f"{float(value):.{digits}f}"


def write_markdown(path: Path, args: argparse.Namespace, bins: List[Dict[str, object]],
                   correlations: Dict[str, float], rows: List[ClipRow]) -> None:
    lines = [
        "# T6 Motion/Occlusion-Binned Benchmark",
        "",
        "Motion descriptors are RAFT statistics on clean GT frames (camera proxy = median",
        "flow vector; object proxy = residual after removing the median; occlusion =",
        "forward-backward inconsistency). Restoration metrics join the per-clip values",
        "already logged by the same-protocol seed-2021 RTN baseline and WaveletMask tests.",
        "Clean-FP is the fraction of each detector's top-5% activation that falls outside",
        "the LQ-GT residual proxy (lower is better).",
        "",
        f"- Clips: `{len(rows)}`",
        f"- Frames per clip: `{args.frames}`",
        f"- Spatial size: `{args.size}`",
        f"- Flow type: `{args.flow_type}`",
        f"- Checkpoint: `{args.checkpoint}`",
        "",
        "## Binned Restoration Improvement (WaveletMask - RTN)",
        "",
        "| Group | Bin | N | Motion | Occl. | dPSNR | dSSIM | dLPIPS | dDISTS |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in bins:
        lines.append(
            f"| {row['group']} | {row['bin']} | {row['n_clips']} | {fmt(row['motion_mag'])} | "
            f"{fmt(row['occlusion_ratio'])} | {fmt(row['delta_psnr'])} | {fmt(row['delta_ssim'])} | "
            f"{fmt(row['delta_lpips'])} | {fmt(row['delta_dists'])} |"
        )
    lines.extend([
        "",
        "## Binned Detector Clean-FP (top-5% outside residual proxy)",
        "",
        "| Group | Bin | N | Spatial | LHFM | GFM | Wavelet(max) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for row in bins:
        lines.append(
            f"| {row['group']} | {row['bin']} | {row['n_clips']} | {fmt(row['spatial_top5_clean'])} | "
            f"{fmt(row['local_top5_clean'])} | {fmt(row['flicker_top5_clean'])} | "
            f"{fmt(row['wavelet_top5_clean'])} |"
        )
    lines.extend(["", "## Clip-Level Spearman Correlations", ""])
    lines.append("| Pair | Spearman |")
    lines.append("|---|---:|")
    for name, value in correlations.items():
        lines.append(f"| {name} | {fmt(value)} |")
    lines.extend(["", "## Per-Clip Rows", ""])
    lines.append(
        "| Clip | Motion | Camera | Object | Occl. | Bin | Dom. | dPSNR | LHFM FP | GFM FP |"
    )
    lines.append("|---|---:|---:|---:|---:|---|---|---:|---:|---:|")
    for row in sorted(rows, key=lambda r: r.motion_mag):
        lines.append(
            f"| {row.clip} | {fmt(row.motion_mag)} | {fmt(row.camera_mag)} | {fmt(row.object_mag)} | "
            f"{fmt(row.occlusion_ratio)} | {row.motion_bin} | {row.dominance} | {fmt(row.delta_psnr)} | "
            f"{fmt(row.local_top5_clean)} | {fmt(row.flicker_top5_clean)} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    flow, flow_warp, wavelet = load_models(args.repo_root, args.checkpoint, device, args.flow_type)
    base_metrics = parse_clip_metrics(args.base_log)
    wavelet_metrics = parse_clip_metrics(args.wavelet_log)
    clips = list_clips(args.lq_root, args.clips)

    rows: List[ClipRow] = []
    with torch.inference_mode():
        for clip in clips:
            name = clip.name
            if name not in base_metrics or name not in wavelet_metrics:
                raise KeyError(f"Clip {name} missing from test logs")
            lq_seq = load_sequence(args.lq_root, name, args.frames, args.size)
            gt_seq = load_sequence(args.gt_root, name, args.frames, args.size)
            lq_tensor = to_tensor(lq_seq, device)
            gt_tensor = to_tensor(gt_seq, device)

            desc = motion_descriptors(flow, flow_warp, gt_tensor)
            spatial = spatial_indicator(flow, flow_warp, lq_tensor)[0, :, 0].detach().float().cpu().numpy()
            local = wavelet.compute_local_mask(lq_tensor)[0, :, 0].detach().float().cpu().numpy()
            flicker = wavelet.compute_flicker_mask(lq_tensor)[0, :, 0].detach().float().cpu().numpy()
            masks = {
                "spatial": spatial,
                "local": local,
                "flicker": flicker,
                "wavelet": np.maximum(local, flicker),
            }
            fp = clean_fp_rates(
                lq_seq, gt_seq, masks, args.frames, args.metric_size,
                args.residual_threshold, args.top_fraction, args.size,
            )

            deltas: Dict[str, float] = {}
            for metric in METRICS:
                direction = 1.0 if HIGHER_IS_BETTER[metric] else -1.0
                deltas[metric] = direction * (wavelet_metrics[name][metric] - base_metrics[name][metric])

            rows.append(ClipRow(
                clip=name,
                motion_mag=desc["motion_mag"],
                camera_mag=desc["camera_mag"],
                object_mag=desc["object_mag"],
                occlusion_ratio=desc["occlusion_ratio"],
                base_psnr=base_metrics[name]["psnr"],
                wavelet_psnr=wavelet_metrics[name]["psnr"],
                delta_psnr=deltas["psnr"],
                delta_ssim=deltas["ssim"],
                delta_lpips=deltas["lpips"],
                delta_dists=deltas["dists"],
                spatial_top5_clean=fp["spatial"],
                local_top5_clean=fp["local"],
                flicker_top5_clean=fp["flicker"],
                wavelet_top5_clean=fp["wavelet"],
            ))
            print(
                f"clip {name} motion {desc['motion_mag']:.3f} cam {desc['camera_mag']:.3f} "
                f"obj {desc['object_mag']:.3f} occ {desc['occlusion_ratio']:.4f} "
                f"dpsnr {deltas['psnr']:+.3f} lhfm_fp {fp['local']:.3f} gfm_fp {fp['flicker']:.3f}",
                flush=True,
            )

    tercile_bins(rows, "motion_mag", "motion_bin", ("low", "medium", "high"))
    tercile_bins(rows, "occlusion_ratio", "occlusion_bin", ("low", "medium", "high"))
    for row in rows:
        row.dominance = "camera" if row.camera_mag >= row.object_mag else "object"

    bins: List[Dict[str, object]] = []
    for label in ("low", "medium", "high"):
        bins.append(aggregate([r for r in rows if r.motion_bin == label], "motion", label))
    for label in ("low", "medium", "high"):
        bins.append(aggregate([r for r in rows if r.occlusion_bin == label], "occlusion", label))
    for label in ("camera", "object"):
        selected = [r for r in rows if r.dominance == label]
        if selected:
            bins.append(aggregate(selected, "dominance", label))

    correlations = {
        "motion_mag vs delta_psnr": spearman([r.motion_mag for r in rows], [r.delta_psnr for r in rows]),
        "occlusion_ratio vs delta_psnr": spearman([r.occlusion_ratio for r in rows], [r.delta_psnr for r in rows]),
        "motion_mag vs lhfm_top5_clean": spearman([r.motion_mag for r in rows], [r.local_top5_clean for r in rows]),
        "motion_mag vs gfm_top5_clean": spearman([r.motion_mag for r in rows], [r.flicker_top5_clean for r in rows]),
        "motion_mag vs spatial_top5_clean": spearman([r.motion_mag for r in rows], [r.spatial_top5_clean for r in rows]),
        "occlusion_ratio vs lhfm_top5_clean": spearman([r.occlusion_ratio for r in rows], [r.local_top5_clean for r in rows]),
        "occlusion_ratio vs gfm_top5_clean": spearman([r.occlusion_ratio for r in rows], [r.flicker_top5_clean for r in rows]),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": {
            "gt_root": str(args.gt_root),
            "lq_root": str(args.lq_root),
            "base_log": str(args.base_log),
            "wavelet_log": str(args.wavelet_log),
            "checkpoint": str(args.checkpoint),
            "clips": len(rows),
            "frames": args.frames,
            "size": args.size,
            "metric_size": args.metric_size,
            "residual_threshold": args.residual_threshold,
            "top_fraction": args.top_fraction,
            "flow_type": args.flow_type,
            "note": (
                "Motion descriptors from RAFT on GT frames; clip terciles, not absolute "
                "thresholds. Clean-FP uses the LQ-GT residual proxy, which is diagnostic "
                "rather than a generator-side artifact label."
            ),
        },
        "clip_rows": [asdict(row) for row in rows],
        "bins": bins,
        "correlations": correlations,
    }
    (args.output_dir / "motion_binned_benchmark.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    fields = list(bins[0].keys())
    with (args.output_dir / "motion_binned_benchmark.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(bins)
    write_markdown(args.output_dir / "motion_binned_benchmark.md", args, bins, correlations, rows)

    for row in bins:
        print(
            "bin", row["group"], row["bin"], row["n_clips"],
            fmt(row["delta_psnr"]), fmt(row["local_top5_clean"]), fmt(row["flicker_top5_clean"]),
        )
    for name, value in correlations.items():
        print("corr", name, fmt(value))


if __name__ == "__main__":
    main()
