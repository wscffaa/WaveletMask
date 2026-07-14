#!/usr/bin/env python3
"""Build Figure 1 motivation from real L40 data and WaveletMask checkpoint.

The script is intended to run on L40_aliyun from the repository root. It reads
real validation frames, loads the submitted WaveletMask-ab checkpoint, exports
raw intermediate tensors, and writes the manuscript Figure 1 asset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np
import torch
from PIL import Image, ImageFilter


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
EXPECTED_CHECKPOINT_SHA256 = "662bf4ae334a8f41fd69f39c3698b1ca8b3d0f2607fa148bd8bdab300744a873"

APPLSCI_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT = Path(__file__).resolve().parents[5]
FRAMEWORK_ROOT = REPO_ROOT / "framework"
WORK_ROOT = PROJECT_ROOT / "Paper" / "_work" / "mdpi-writing"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["TeX Gyre Pagella", "Palatino", "P052", "DejaVu Serif"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)

TEXT = "#111827"
MUTED = "#4b5563"
BORDER = "#9aa4b2"
GRID = "#e5e7eb"
ROW_A = "#d55e00"
ROW_B = "#0072b2"
PIXEL_BLUE = "#56b4e9"
WAVELET_GREEN = "#009e73"
MASK_PURPLE = "#7e57c2"
MDPI_TEXTWIDTH_IN = 13.86 / 2.54


@dataclass(frozen=True)
class Sequence:
    name: str
    frames: tuple[Path, ...]


@dataclass(frozen=True)
class CandidateRef:
    kind: str
    sequence: str
    frame_index: int
    frame_path: Path
    cheap_score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "Exp/checkpoints/waveletmask_ab/net_g_best.pth")
    parser.add_argument("--synthetic-lq-root", type=Path, default=FRAMEWORK_ROOT / "datasets/val/REDS/LQ/val_sharp_degraded")
    parser.add_argument("--synthetic-gt-root", type=Path, default=FRAMEWORK_ROOT / "datasets/val/REDS/GT/val_sharp")
    parser.add_argument(
        "--real-root",
        type=Path,
        default=FRAMEWORK_ROOT / "datasets/OldFilmRestoration/val/RealFilm/MambaOFR_SRWOV",
    )
    parser.add_argument("--output-dir", type=Path, default=WORK_ROOT / "figures/Fig1_real_motivation")
    parser.add_argument("--figure-pdf", type=Path, default=WORK_ROOT / "figures/Fig1_real_motivation/Fig2_wavelet_frequency_motivation.pdf")
    parser.add_argument("--figure-svg", type=Path, default=WORK_ROOT / "figures/Fig1_real_motivation/Fig2_wavelet_frequency_motivation.svg")
    parser.add_argument("--figure-png", type=Path, default=WORK_ROOT / "figures/Fig1_real_motivation/Fig2_wavelet_frequency_motivation.png")
    parser.add_argument("--short-side", type=int, default=352)
    parser.add_argument("--align", type=int, default=16)
    parser.add_argument("--max-synthetic-sequences", type=int, default=30)
    parser.add_argument("--max-real-sequences", type=int, default=24)
    parser.add_argument("--max-frames-per-sequence", type=int, default=18)
    parser.add_argument("--preselect", type=int, default=24)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-candidates", action="store_true")
    return parser.parse_args()


def add_import_paths() -> None:
    for path in (FRAMEWORK_ROOT,):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_inputs(args: argparse.Namespace) -> None:
    required = [args.checkpoint, args.synthetic_lq_root, args.synthetic_gt_root, args.real_root]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs: " + ", ".join(missing))
    actual = sha256(args.checkpoint)
    if actual != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(f"Checkpoint sha256 mismatch: {actual}")


def image_files_in_dir(path: Path) -> tuple[Path, ...]:
    files = [p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    return tuple(sorted(files))


def find_sequences(root: Path) -> list[Sequence]:
    sequences: list[Sequence] = []
    for directory in sorted([root, *[p for p in root.rglob("*") if p.is_dir()]]):
        frames = image_files_in_dir(directory)
        if len(frames) >= 3:
            sequences.append(Sequence(name=str(directory.relative_to(root)), frames=frames))
    return sequences


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def resize_align(image: Image.Image, short_side: int, align: int) -> Image.Image:
    w, h = image.size
    if min(w, h) != short_side:
        if w < h:
            new_w = short_side
            new_h = round(short_side * h / w)
        else:
            new_h = short_side
            new_w = round(short_side * w / h)
    else:
        new_w, new_h = w, h
    new_w = max(align * 4, (new_w // align) * align)
    new_h = max(align * 4, (new_h // align) * align)
    return image.resize((new_w, new_h), Image.Resampling.BICUBIC)


def read_triplet(paths: tuple[Path, ...], idx: int, short_side: int, align: int) -> tuple[Image.Image, Image.Image, Image.Image]:
    return tuple(resize_align(load_rgb(paths[i]), short_side, align) for i in (idx - 1, idx, idx + 1))  # type: ignore[return-value]


def gray_np(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("L"), dtype=np.float32) / 255.0


def pil_to_tensor(images: Iterable[Image.Image], device: torch.device) -> torch.Tensor:
    arrays = [np.asarray(im, dtype=np.float32) / 255.0 for im in images]
    stacked = np.stack(arrays, axis=0)
    tensor = torch.from_numpy(stacked).permute(0, 3, 1, 2).unsqueeze(0)
    return tensor.mul(2.0).sub(1.0).to(device)


def normalize01(x: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    lo, hi = np.percentile(x, [lo_pct, hi_pct])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(x.min()), float(x.max())
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def heat_png(path: Path, array: np.ndarray, cmap: str) -> None:
    arr = (plt.get_cmap(cmap)(normalize01(array))[:, :, :3] * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def high_residual_map(damage: np.ndarray) -> np.ndarray:
    blur = np.asarray(
        Image.fromarray((np.clip(damage, 0, 1) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(5)),
        dtype=np.float32,
    ) / 255.0
    return np.abs(damage - blur)


def cheap_synthetic_score(lq_path: Path, synthetic_lq_root: Path, synthetic_gt_root: Path) -> float:
    gt_path = synthetic_gt_root / lq_path.relative_to(synthetic_lq_root)
    lq = gray_np(load_rgb(lq_path))
    if gt_path.exists():
        gt = gray_np(load_rgb(gt_path))
        damage = np.abs(lq - gt)
    else:
        damage = np.abs(lq - np.asarray(load_rgb(lq_path).filter(ImageFilter.GaussianBlur(5)).convert("L"), dtype=np.float32) / 255.0)
    high = high_residual_map(damage)
    return float(np.percentile(high, 99.5) + 0.25 * np.mean(damage))


def cheap_real_score(paths: tuple[Path, ...], idx: int) -> float:
    prev, curr, next_img = [gray_np(load_rgb(paths[i])) for i in (idx - 1, idx, idx + 1)]
    mean_jump = min(abs(float(curr.mean() - prev.mean())), abs(float(curr.mean() - next_img.mean())))
    texture = float(np.percentile(np.abs(curr - curr.mean()), 95))
    return mean_jump + 0.05 * texture


def preselect_candidates(
    kind: str,
    sequences: list[Sequence],
    args: argparse.Namespace,
    max_sequences: int,
) -> list[CandidateRef]:
    refs: list[CandidateRef] = []
    for seq in sequences[:max_sequences]:
        usable = range(1, len(seq.frames) - 1)
        stride = max(1, math.ceil(len(seq.frames) / max(args.max_frames_per_sequence, 1)))
        for idx in list(usable)[::stride]:
            if kind == "synthetic":
                score = cheap_synthetic_score(seq.frames[idx], args.synthetic_lq_root, args.synthetic_gt_root)
            else:
                score = cheap_real_score(seq.frames, idx)
            refs.append(CandidateRef(kind=kind, sequence=seq.name, frame_index=idx, frame_path=seq.frames[idx], cheap_score=score))
    refs.sort(key=lambda r: (-r.cheap_score, r.sequence, r.frame_index))
    return refs[: args.preselect]


def load_model(checkpoint: Path, device: torch.device):
    add_import_paths()
    import basicofr.archs  # noqa: F401 - registers project architectures.
    from basicsr.utils.registry import ARCH_REGISTRY

    WaveletMaskABNet = ARCH_REGISTRY.get("WaveletMaskABNet")
    model = WaveletMaskABNet(num_feat=16, num_block=6, flow_type="raft").to(device).eval()
    obj = torch.load(checkpoint, map_location="cpu")
    state = obj["params"] if isinstance(obj, dict) and "params" in obj else obj
    result = model.load_state_dict(state, strict=False)
    return model, {"missing_keys": result.missing_keys, "unexpected_keys": result.unexpected_keys}


def tensor_to_np(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().float().cpu().squeeze().numpy()


def model_maps(model, lqs: torch.Tensor) -> dict[str, np.ndarray | float]:
    from basicofr.archs.components.arch_util import flow_warp

    with torch.no_grad():
        local = model.wavelet_mask.compute_local_mask(lqs)
        flicker = model.wavelet_mask.compute_flicker_mask(lqs)
        fused = model.wavelet_mask(lqs)
        max_fused = torch.max(local, flicker)
        max_error = float((fused - max_fused).abs().max().item())

        forward_flow, backward_flow = model.comp_flow(lqs)
        prev_warp = flow_warp(lqs[:, 0], forward_flow[:, 0].permute(0, 2, 3, 1))
        next_warp = flow_warp(lqs[:, 2], backward_flow[:, 1].permute(0, 2, 3, 1))
        curr = lqs[:, 1]
        pixel_indicator = 0.5 * (torch.abs(prev_warp - curr)[:, 0:1] + torch.abs(next_warp - curr)[:, 0:1])

        lqs_float = lqs.float()
        gray = 0.299 * lqs_float[:, :, 0] + 0.587 * lqs_float[:, :, 1] + 0.114 * lqs_float[:, :, 2]
        prev_c = model.wavelet_mask.dwt(gray[:, 0])
        curr_c = model.wavelet_mask.dwt(gray[:, 1])
        next_c = model.wavelet_mask.dwt(gray[:, 2])
        high_diff = model.wavelet_mask._compute_high_freq_diff(curr_c, prev_c, next_c)
        low_diff = model.wavelet_mask._compute_low_freq_diff(curr_c, prev_c, next_c)

    return {
        "pixel_indicator": tensor_to_np(pixel_indicator),
        "local": tensor_to_np(local[:, 1]),
        "flicker": tensor_to_np(flicker[:, 1]),
        "fused": tensor_to_np(fused[:, 1]),
        "lh_diff": tensor_to_np(high_diff[:, 0]),
        "hl_diff": tensor_to_np(high_diff[:, 1]),
        "hh_diff": tensor_to_np(high_diff[:, 2]),
        "ll_diff": tensor_to_np(low_diff[:, 0]),
        "ll": tensor_to_np(curr_c["LL"][:, 0]),
        "lh": tensor_to_np(curr_c["LH"][:, 0]),
        "hl": tensor_to_np(curr_c["HL"][:, 0]),
        "hh": tensor_to_np(curr_c["HH"][:, 0]),
        "max_fusion_error": max_error,
    }


def best_roi(score: np.ndarray, box: int) -> tuple[int, int, int, int]:
    h, w = score.shape
    box = min(box, h, w)
    stride = max(8, box // 4)
    best = (-1.0, 0, 0)
    for y in range(0, max(1, h - box + 1), stride):
        for x in range(0, max(1, w - box + 1), stride):
            patch = score[y : y + box, x : x + box]
            val = float(patch.mean() + 0.5 * np.percentile(patch, 99))
            if val > best[0]:
                best = (val, x, y)
    _, x, y = best
    return x, y, x + box, y + box


def crop_arr(arr: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = roi
    return arr[y0:y1, x0:x1]


def crop_image(image: Image.Image, roi: tuple[int, int, int, int]) -> Image.Image:
    return image.crop(roi)


def scaled_roi(roi: tuple[int, int, int, int], target: np.ndarray, source_shape: tuple[int, int]) -> tuple[int, int, int, int]:
    src_h, src_w = source_shape
    dst_h, dst_w = target.shape[:2]
    sx = dst_w / max(src_w, 1)
    sy = dst_h / max(src_h, 1)
    x0, y0, x1, y1 = roi
    out = (
        max(0, min(dst_w - 1, int(round(x0 * sx)))),
        max(0, min(dst_h - 1, int(round(y0 * sy)))),
        max(1, min(dst_w, int(round(x1 * sx)))),
        max(1, min(dst_h, int(round(y1 * sy)))),
    )
    if out[2] <= out[0] or out[3] <= out[1]:
        return (0, 0, dst_w, dst_h)
    return out


def evaluate_candidate(
    model,
    ref: CandidateRef,
    sequence: Sequence,
    args: argparse.Namespace,
    device: torch.device,
    locked_roi: tuple[int, int, int, int] | None = None,
) -> dict:
    prev_img, curr_img, next_img = read_triplet(sequence.frames, ref.frame_index, args.short_side, args.align)
    lqs = pil_to_tensor((prev_img, curr_img, next_img), device)
    maps = model_maps(model, lqs)
    local = maps["local"]
    flicker = maps["flicker"]
    pixel = maps["pixel_indicator"]
    assert isinstance(local, np.ndarray) and isinstance(flicker, np.ndarray) and isinstance(pixel, np.ndarray)

    if ref.kind == "synthetic":
        gt_path = args.synthetic_gt_root / ref.frame_path.relative_to(args.synthetic_lq_root)
        if gt_path.exists():
            gt_img = resize_align(load_rgb(gt_path), args.short_side, args.align)
            damage = np.abs(gray_np(curr_img) - gray_np(gt_img))
        else:
            damage = np.abs(gray_np(curr_img) - np.asarray(curr_img.filter(ImageFilter.GaussianBlur(5)).convert("L"), dtype=np.float32) / 255.0)
        high_damage = high_residual_map(damage)
        maps["gt_damage"] = damage
        maps["gt_high_damage"] = high_damage
        mask_signal = float(np.percentile(high_damage, 99.5) - np.percentile(high_damage, 50))
        pixel_signal = float(np.percentile(pixel, 99.5) - np.percentile(pixel, 50))
        score = mask_signal / (pixel_signal + 0.05)
        roi = locked_roi if locked_roi is not None else best_roi(high_damage, box=max(96, min(high_damage.shape) // 4))
    else:
        mask_signal = float(flicker.mean() + np.percentile(flicker, 95))
        pixel_signal = float(pixel.std())
        score = mask_signal - 0.25 * pixel_signal
        h, w = flicker.shape
        roi = locked_roi if locked_roi is not None else (0, 0, w, h)

    return {
        "ref": ref,
        "score": score,
        "mask_signal": mask_signal,
        "pixel_signal": pixel_signal,
        "roi": roi,
        "images": {"prev": prev_img, "curr": curr_img, "next": next_img},
        "maps": maps,
        "resized_size": curr_img.size,
    }


def choose_candidate(
    kind: str,
    model,
    args: argparse.Namespace,
    device: torch.device,
    locked: dict | None = None,
) -> tuple[dict, list[dict]]:
    root = args.synthetic_lq_root if kind == "synthetic" else args.real_root
    sequences = find_sequences(root)
    seq_by_name = {seq.name: seq for seq in sequences}

    if locked is not None:
        sequence = seq_by_name[locked["sequence"]]
        frame_index = int(locked["frame_index"])
        ref = CandidateRef(
            kind=kind,
            sequence=locked["sequence"],
            frame_index=frame_index,
            frame_path=sequence.frames[frame_index],
            cheap_score=float(locked.get("cheap_score", 0.0)),
        )
        item = evaluate_candidate(model, ref, sequence, args, device, tuple(locked["roi_xyxy"]))
        summary = [{
            "kind": kind,
            "sequence": ref.sequence,
            "frame_index": frame_index,
            "frame_path": str(ref.frame_path),
            "cheap_score": ref.cheap_score,
            "model_score": item["score"],
            "mask_signal": item["mask_signal"],
            "pixel_signal": item["pixel_signal"],
            "roi_xyxy": item["roi"],
            "selection_mode": "locked_from_existing_metadata",
        }]
        return item, summary

    max_sequences = args.max_synthetic_sequences if kind == "synthetic" else args.max_real_sequences
    refs = preselect_candidates(kind, sequences, args, max_sequences)
    if not refs:
        raise RuntimeError(f"No {kind} candidates found under {root}")
    evaluated = [evaluate_candidate(model, ref, seq_by_name[ref.sequence], args, device) for ref in refs]
    evaluated.sort(key=lambda item: (-item["score"], item["ref"].sequence, item["ref"].frame_index))
    summary = [
        {
            "kind": item["ref"].kind,
            "sequence": item["ref"].sequence,
            "frame_index": item["ref"].frame_index,
            "frame_path": str(item["ref"].frame_path),
            "cheap_score": item["ref"].cheap_score,
            "model_score": item["score"],
            "mask_signal": item["mask_signal"],
            "pixel_signal": item["pixel_signal"],
            "roi_xyxy": item["roi"],
        }
        for item in evaluated
    ]
    return evaluated[0], summary


def save_raw(item: dict, out_dir: Path, prefix: str) -> dict:
    raw = out_dir / "raw" / prefix
    raw.mkdir(parents=True, exist_ok=True)
    for name, image in item["images"].items():
        image.save(raw / f"{name}.png")
    maps = item["maps"]
    np.savez_compressed(raw / "maps.npz", **{k: v for k, v in maps.items() if isinstance(v, np.ndarray)})
    for key, cmap in {
        "pixel_indicator": "Blues",
        "lh_diff": "magma",
        "hl_diff": "magma",
        "hh_diff": "magma",
        "ll_diff": "viridis",
        "local": "inferno",
        "flicker": "viridis",
        "fused": "plasma",
        "gt_damage": "gray",
        "gt_high_damage": "magma",
    }.items():
        if key in maps:
            heat_png(raw / f"{key}.png", maps[key], cmap)
    return {"raw_dir": str(raw), "maps_npz": str(raw / "maps.npz")}


def image_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image, dtype=np.float32) / 255.0


def triplet_strip(prev_img: Image.Image, curr_img: Image.Image, next_img: Image.Image) -> np.ndarray:
    h = 180
    panels = []
    for image in (prev_img, curr_img, next_img):
        im = image.copy()
        im.thumbnail((260, h), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (260, h), "white")
        canvas.paste(im, ((260 - im.width) // 2, (h - im.height) // 2))
        panels.append(np.asarray(canvas, dtype=np.float32) / 255.0)
    return np.concatenate(panels, axis=1)


def diff_triplet_tile(maps: dict[str, np.ndarray], roi: tuple[int, int, int, int] | None) -> np.ndarray:
    arrays = [maps["lh_diff"], maps["hl_diff"], maps["hh_diff"]]
    if roi is not None:
        arrays = [crop_arr(a, roi) for a in arrays]
    tiles = []
    for arr in arrays:
        rgb = plt.get_cmap("magma")(normalize01(arr))[:, :, :3]
        tiles.append(rgb)
    gap = np.ones((tiles[0].shape[0], max(3, tiles[0].shape[1] // 40), 3), dtype=np.float32)
    return np.concatenate([tiles[0], gap, tiles[1], gap, tiles[2]], axis=1)


def fixed_indicator_display(array: np.ndarray, vmax: float = 1.0) -> np.ndarray:
    return np.clip(np.asarray(array, dtype=np.float32) / vmax, 0.0, 1.0)


def panel_rgb(array: np.ndarray, cmap: str | None = None, fixed_vmax: float | None = None) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 3:
        if arr.max() > 1.0:
            arr = arr / 255.0
        return np.clip(arr, 0.0, 1.0)
    if fixed_vmax is not None:
        norm = fixed_indicator_display(arr, fixed_vmax)
    else:
        norm = normalize01(arr)
    return plt.get_cmap(cmap or "viridis")(norm)[:, :, :3]


def fit_panel(
    rgb: np.ndarray,
    size: tuple[int, int] = (360, 250),
    *,
    mode: str = "contain",
    pad_color: str = "white",
) -> np.ndarray:
    width, height = size
    image = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    if mode not in {"contain", "cover"}:
        raise ValueError(f"Unknown fit mode: {mode}")
    scale_fn = max if mode == "cover" else min
    scale = scale_fn(width / max(image.width, 1), height / max(image.height, 1))
    resized = (max(1, int(round(image.width * scale))), max(1, int(round(image.height * scale))))
    image = image.resize(resized, Image.Resampling.BICUBIC)
    if mode == "cover":
        x0 = max(0, (image.width - width) // 2)
        y0 = max(0, (image.height - height) // 2)
        image = image.crop((x0, y0, x0 + width, y0 + height))
    canvas = Image.new("RGB", (width, height), pad_color)
    canvas.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
    return np.asarray(canvas, dtype=np.float32) / 255.0


def response_triptych(
    maps: dict[str, np.ndarray],
    roi: tuple[int, int, int, int],
    size: tuple[int, int] = (360, 250),
) -> np.ndarray:
    width, height = size
    gap = max(5, int(round(width * 0.025)))
    tile_w = max(24, (width - 2 * gap) // 3)
    panels = []
    for key in ("lh_diff", "hl_diff", "hh_diff"):
        rgb = fit_panel(panel_rgb(crop_arr(maps[key], roi), "magma"), size=(tile_w, height), mode="cover", pad_color="#f8fafc")
        panels.append(rgb)
    gap_rgb = np.ones((height, gap, 3), dtype=np.float32)
    return np.concatenate([panels[0], gap_rgb, panels[1], gap_rgb, panels[2]], axis=1)


def triplet_panel(
    prev_img: Image.Image,
    curr_img: Image.Image,
    next_img: Image.Image,
    size: tuple[int, int] = (360, 250),
) -> np.ndarray:
    width, height = size
    gap = max(5, int(round(width * 0.025)))
    tile_w = max(24, (width - 2 * gap) // 3)
    panels = []
    for image in (prev_img, curr_img, next_img):
        panels.append(fit_panel(image_array(image), size=(tile_w, height), mode="cover", pad_color="#f8fafc"))
    gap_rgb = np.ones((height, gap, 3), dtype=np.float32)
    return np.concatenate([panels[0], gap_rgb, panels[1], gap_rgb, panels[2]], axis=1)


def mask_pair_panel(
    branch_rgb: np.ndarray,
    fused_rgb: np.ndarray,
    size: tuple[int, int] = (420, 250),
) -> np.ndarray:
    width, height = size
    gap = max(6, int(round(width * 0.025)))
    band_h = max(24, int(round(height * 0.145)))
    branch_w = int(round((width - gap) * 0.47))
    fused_w = width - gap - branch_w
    branch = fit_panel(branch_rgb, size=(branch_w, height - band_h), mode="cover", pad_color="#f8fafc")
    fused = fit_panel(fused_rgb, size=(fused_w, height - band_h), mode="cover", pad_color="#f8fafc")
    gap_rgb = np.ones((height, gap, 3), dtype=np.float32)
    content = np.concatenate([branch, gap_rgb[band_h:], fused], axis=1)
    band = np.ones((band_h, content.shape[1], 3), dtype=np.float32)
    return np.concatenate([band, content], axis=0)


def ll_response_panel(array: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    return fit_panel(panel_rgb(array, "viridis"), size=(width, height), mode="cover", pad_color="#f8fafc")


def axis_panel_size(fig_w: float, fig_h: float, ax_w: float, ax_h: float, base_h: int = 260) -> tuple[int, int]:
    aspect = (fig_w * ax_w) / max(fig_h * ax_h, 0.001)
    return max(80, int(round(base_h * aspect))), base_h


def draw_cell(
    ax,
    image: np.ndarray,
    title: str | None = None,
    inner_labels: tuple[str, ...] = (),
    *,
    border_color: str = BORDER,
    label_y: float = 0.975,
    label_positions: tuple[float, ...] | None = None,
    label_fontsize: float = 5.2,
) -> None:
    ax.imshow(image, aspect="auto")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(border_color)
        spine.set_linewidth(0.45)
    if title:
        ax.set_title(title, fontsize=8.0, color=TEXT, pad=3)
    if inner_labels:
        if label_positions is not None:
            xs = np.asarray(label_positions, dtype=np.float32)
        elif len(inner_labels) == 1:
            xs = np.asarray([0.5], dtype=np.float32)
        else:
            xs = np.linspace(0.16, 0.84, len(inner_labels))
        for x, label in zip(xs, inner_labels):
            ax.text(
                x,
                label_y,
                label,
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=label_fontsize,
                fontfamily="DejaVu Sans",
                fontweight="normal",
                color=MUTED,
                bbox=dict(boxstyle="round,pad=0.08", facecolor="white", edgecolor="none", alpha=0.78),
            )


def add_flow_arrow(fig, x0: float, y0: float, x1: float, y1: float, color: str = "#9aa4b2") -> None:
    fig.add_artist(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=6,
            linewidth=0.45,
            color=color,
            shrinkA=0,
            shrinkB=0,
        )
    )


def save_figure(fig, pdf_path: Path, svg_path: Path, png_path: Path | None = None) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, dpi=600)
    fig.savefig(svg_path)
    if png_path is not None:
        png_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png_path, dpi=600)


def save_preview(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)


def draw_story_matrix(synth: dict, real: dict) -> plt.Figure:
    synth_roi = synth["roi"]
    synth_images = synth["images"]
    synth_maps = synth["maps"]
    synth_half_roi = scaled_roi(synth_roi, synth_maps["lh_diff"], synth_maps["local"].shape)
    real_images = real["images"]
    real_maps = real["maps"]

    fig_w, fig_h = MDPI_TEXTWIDTH_IN, 2.58
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    left, right, bottom, top = 0.087, 0.018, 0.055, 0.135
    col_gap, row_gap = 0.014, 0.092
    rel_w = np.asarray([1.08, 1.00, 1.50, 1.00, 1.02], dtype=np.float32)
    avail_w = 1.0 - left - right - col_gap * (len(rel_w) - 1)
    col_w = (avail_w * rel_w / rel_w.sum()).tolist()
    cell_h = (1.0 - bottom - top - row_gap) / 2
    row_y = [bottom + cell_h + row_gap, bottom]
    col_x: list[float] = []
    x = left
    for w in col_w:
        col_x.append(x)
        x += w + col_gap
    headers = [
        ("Input", TEXT),
        ("Pixel cue", PIXEL_BLUE),
        ("Wavelet cue", WAVELET_GREEN),
        ("Branch mask", MASK_PURPLE),
        ("Fused mask", MASK_PURPLE),
    ]

    main_w = col_x[-1] + col_w[-1] - left
    for y, accent in ((row_y[0], ROW_A), (row_y[1], ROW_B)):
        fig.add_artist(
            Rectangle(
                (left - 0.052, y - 0.014),
                main_w + 0.052,
                cell_h + 0.028,
                transform=fig.transFigure,
                facecolor="#fbfdff",
                edgecolor="none",
                zorder=-10,
            )
        )
        fig.add_artist(
            Rectangle(
                (left - 0.022, y + 0.002),
                0.005,
                cell_h - 0.004,
                transform=fig.transFigure,
                facecolor=accent,
                edgecolor="none",
                zorder=-5,
            )
        )

    axes = [[fig.add_axes([col_x[c], row_y[r], col_w[c], cell_h]) for c in range(5)] for r in range(2)]
    for c, (header, color) in enumerate(headers):
        fig.text(col_x[c] + col_w[c] / 2, 0.958, header, ha="center", va="top", fontsize=8.3, fontweight="normal", color=TEXT)
        fig.add_artist(
            Rectangle(
                (col_x[c], 0.925),
                col_w[c],
                0.004,
                transform=fig.transFigure,
                facecolor=color,
                edgecolor="none",
                alpha=0.72,
            )
        )

    fig.text(0.020, row_y[0] + cell_h - 0.010, "(a)", ha="left", va="top", fontsize=8.5, fontweight="bold", color=TEXT)
    fig.text(0.050, row_y[0] + cell_h / 2, "Sparse\nscratch", ha="center", va="center", fontsize=8.0, color=TEXT)
    fig.text(0.020, row_y[1] + cell_h - 0.010, "(b)", ha="left", va="top", fontsize=8.5, fontweight="bold", color=TEXT)
    fig.text(0.050, row_y[1] + cell_h / 2, "Brightness\nflicker", ha="center", va="center", fontsize=8.0, color=TEXT)

    sizes = [axis_panel_size(fig_w, fig_h, col_w[c], cell_h) for c in range(5)]
    draw_cell(axes[0][0], fit_panel(image_array(crop_image(synth_images["curr"], synth_roi)), size=sizes[0], mode="cover", pad_color="#f8fafc"))
    draw_cell(
        axes[0][1],
        fit_panel(panel_rgb(crop_arr(synth_maps["pixel_indicator"], synth_roi), "Blues", fixed_vmax=1.0), size=sizes[1], mode="cover", pad_color="#f8fafc"),
    )
    draw_cell(axes[0][2], response_triptych(synth_maps, synth_half_roi, size=sizes[2]), inner_labels=("LH", "HL", "HH"))
    draw_cell(axes[0][3], fit_panel(panel_rgb(crop_arr(synth_maps["local"], synth_roi), "inferno"), size=sizes[3], mode="cover", pad_color="#f8fafc"))
    draw_cell(axes[0][4], fit_panel(panel_rgb(crop_arr(synth_maps["fused"], synth_roi), "plasma"), size=sizes[4], mode="cover", pad_color="#f8fafc"))

    draw_cell(
        axes[1][0],
        triplet_panel(real_images["prev"], real_images["curr"], real_images["next"], size=sizes[0]),
        inner_labels=("t-1", "t", "t+1"),
    )
    draw_cell(axes[1][1], fit_panel(panel_rgb(real_maps["pixel_indicator"], "Blues", fixed_vmax=1.0), size=sizes[1], mode="cover", pad_color="#f8fafc"))
    draw_cell(axes[1][2], ll_response_panel(real_maps["ll_diff"], size=sizes[2]), inner_labels=("LL",))
    draw_cell(axes[1][3], fit_panel(panel_rgb(real_maps["flicker"], "viridis"), size=sizes[3], mode="cover", pad_color="#f8fafc"))
    draw_cell(axes[1][4], fit_panel(panel_rgb(real_maps["fused"], "plasma"), size=sizes[4], mode="cover", pad_color="#f8fafc"))

    for y in (row_y[0] + cell_h / 2, row_y[1] + cell_h / 2):
        for c in range(4):
            add_flow_arrow(fig, col_x[c] + col_w[c] + 0.004, y, col_x[c + 1] - 0.004, y)
    return fig


def draw_grouped_flow(synth: dict, real: dict) -> plt.Figure:
    synth_roi = synth["roi"]
    synth_images = synth["images"]
    synth_maps = synth["maps"]
    synth_half_roi = scaled_roi(synth_roi, synth_maps["lh_diff"], synth_maps["local"].shape)
    real_images = real["images"]
    real_maps = real["maps"]

    fig_w, fig_h = MDPI_TEXTWIDTH_IN, 2.72
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    left, right, bottom, top = 0.082, 0.018, 0.055, 0.155
    group_gap, row_gap = 0.026, 0.110
    group_rel = np.asarray([1.55, 1.18, 1.45], dtype=np.float32)
    avail_w = 1.0 - left - right - 2 * group_gap
    group_w = (avail_w * group_rel / group_rel.sum()).tolist()
    group_x: list[float] = []
    x = left
    for w in group_w:
        group_x.append(x)
        x += w + group_gap
    cell_h = (1.0 - bottom - top - row_gap) / 2
    row_y = [bottom + cell_h + row_gap, bottom]

    group_headers = [
        ("Pixel-domain observation", "#94a3b8"),
        ("Wavelet-domain separation", WAVELET_GREEN),
        ("Branch/fused evidence", MASK_PURPLE),
    ]
    for i, (label, color) in enumerate(group_headers):
        fig.text(group_x[i] + group_w[i] / 2, 0.957, label, ha="center", va="top", fontsize=8.0, fontweight="normal", color=TEXT)
        fig.add_artist(Rectangle((group_x[i], 0.925), group_w[i], 0.004, transform=fig.transFigure, facecolor=color, edgecolor="none", alpha=0.74))

    for y, tag, label, accent in (
        (row_y[0], "a", "Sparse scratch/dust", ROW_A),
        (row_y[1], "b", "Brightness flicker", ROW_B),
    ):
        fig.add_artist(Rectangle((left - 0.048, y - 0.014), 1.0 - left - right + 0.048, cell_h + 0.028, transform=fig.transFigure, facecolor="#fbfdff", edgecolor="none", zorder=-10))
        fig.add_artist(Rectangle((left - 0.021, y + 0.003), 0.005, cell_h - 0.006, transform=fig.transFigure, facecolor=accent, edgecolor="none", zorder=-5))
        fig.text(0.020, y + cell_h - 0.010, f"({tag})", ha="left", va="top", fontsize=8.5, fontweight="bold", color=TEXT)
        fig.text(0.047, y + cell_h / 2, label.replace(" ", "\n", 1), ha="center", va="center", fontsize=8.0, color=TEXT)

    cell_gap = 0.010
    axes = []
    for r, y in enumerate(row_y):
        g0_input_w = (group_w[0] - cell_gap) * 0.53
        g0_pixel_w = group_w[0] - cell_gap - g0_input_w
        g2_branch_w = (group_w[2] - cell_gap) * 0.47
        g2_fused_w = group_w[2] - cell_gap - g2_branch_w
        axes.append(
            {
                "input": fig.add_axes([group_x[0], y, g0_input_w, cell_h]),
                "pixel": fig.add_axes([group_x[0] + g0_input_w + cell_gap, y, g0_pixel_w, cell_h]),
                "wavelet": fig.add_axes([group_x[1], y, group_w[1], cell_h]),
                "branch": fig.add_axes([group_x[2], y, g2_branch_w, cell_h]),
                "fused": fig.add_axes([group_x[2] + g2_branch_w + cell_gap, y, g2_fused_w, cell_h]),
            }
        )

    def size(ax) -> tuple[int, int]:
        box = ax.get_position()
        return axis_panel_size(fig_w, fig_h, box.width, box.height)

    draw_cell(axes[0]["input"], fit_panel(image_array(crop_image(synth_images["curr"], synth_roi)), size=size(axes[0]["input"]), mode="cover", pad_color="#f8fafc"), "Input")
    draw_cell(axes[0]["pixel"], fit_panel(panel_rgb(crop_arr(synth_maps["pixel_indicator"], synth_roi), "Blues", fixed_vmax=1.0), size=size(axes[0]["pixel"]), mode="cover", pad_color="#f8fafc"), "Pixel cue")
    draw_cell(axes[0]["wavelet"], response_triptych(synth_maps, synth_half_roi, size=size(axes[0]["wavelet"])), "LH/HL/HH", inner_labels=("LH", "HL", "HH"))
    draw_cell(axes[0]["branch"], fit_panel(panel_rgb(crop_arr(synth_maps["local"], synth_roi), "inferno"), size=size(axes[0]["branch"]), mode="cover", pad_color="#f8fafc"), "m_local")
    draw_cell(axes[0]["fused"], fit_panel(panel_rgb(crop_arr(synth_maps["fused"], synth_roi), "plasma"), size=size(axes[0]["fused"]), mode="cover", pad_color="#f8fafc"), "max")

    draw_cell(axes[1]["input"], triplet_panel(real_images["prev"], real_images["curr"], real_images["next"], size=size(axes[1]["input"])), "Triplet", inner_labels=("t-1", "t", "t+1"))
    draw_cell(axes[1]["pixel"], fit_panel(panel_rgb(real_maps["pixel_indicator"], "Blues", fixed_vmax=1.0), size=size(axes[1]["pixel"]), mode="cover", pad_color="#f8fafc"), "Pixel cue")
    draw_cell(axes[1]["wavelet"], ll_response_panel(real_maps["ll_diff"], size=size(axes[1]["wavelet"])), "LL temporal", inner_labels=("LL",))
    draw_cell(axes[1]["branch"], fit_panel(panel_rgb(real_maps["flicker"], "viridis"), size=size(axes[1]["branch"]), mode="cover", pad_color="#f8fafc"), "m_flicker")
    draw_cell(axes[1]["fused"], fit_panel(panel_rgb(real_maps["fused"], "plasma"), size=size(axes[1]["fused"]), mode="cover", pad_color="#f8fafc"), "max")

    for y in (row_y[0] + cell_h / 2, row_y[1] + cell_h / 2):
        add_flow_arrow(fig, group_x[0] + group_w[0] + 0.006, y, group_x[1] - 0.006, y)
        add_flow_arrow(fig, group_x[1] + group_w[1] + 0.006, y, group_x[2] - 0.006, y)
    return fig


def draw_compact_flow(synth: dict, real: dict) -> plt.Figure:
    synth_roi = synth["roi"]
    synth_images = synth["images"]
    synth_maps = synth["maps"]
    synth_half_roi = scaled_roi(synth_roi, synth_maps["lh_diff"], synth_maps["local"].shape)
    real_images = real["images"]
    real_maps = real["maps"]

    fig_w, fig_h = MDPI_TEXTWIDTH_IN, 2.38
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    left, right, bottom, top = 0.084, 0.018, 0.055, 0.135
    col_gap, row_gap = 0.016, 0.090
    rel_w = np.asarray([1.10, 0.94, 1.34, 1.58], dtype=np.float32)
    avail_w = 1.0 - left - right - col_gap * (len(rel_w) - 1)
    col_w = (avail_w * rel_w / rel_w.sum()).tolist()
    cell_h = (1.0 - bottom - top - row_gap) / 2
    row_y = [bottom + cell_h + row_gap, bottom]
    col_x: list[float] = []
    x = left
    for w in col_w:
        col_x.append(x)
        x += w + col_gap

    headers = [("Input", TEXT), ("Pixel indicator", TEXT), ("Wavelet response", TEXT), ("Masks", TEXT)]
    for c, (header, color) in enumerate(headers):
        fig.text(col_x[c] + col_w[c] / 2, 0.958, header, ha="center", va="top", fontsize=8.5, fontweight="normal", color=TEXT)
        fig.add_artist(Rectangle((col_x[c], 0.925), col_w[c], 0.003, transform=fig.transFigure, facecolor=GRID, edgecolor="none", alpha=1.0))

    for y, tag, label, accent in (
        (row_y[0], "a", "Sparse\nscratch", ROW_A),
        (row_y[1], "b", "Brightness\nflicker", ROW_B),
    ):
        fig.add_artist(Rectangle((left - 0.048, y - 0.014), 1.0 - left - right + 0.048, cell_h + 0.028, transform=fig.transFigure, facecolor="#fbfdff", edgecolor="none", zorder=-10))
        fig.text(0.020, y + cell_h - 0.010, f"({tag})", ha="left", va="top", fontsize=8.4, fontweight="bold", color=TEXT)
        fig.text(0.056, y + cell_h / 2, label, ha="center", va="center", fontsize=7.4, color=TEXT)

    axes = [[fig.add_axes([col_x[c], row_y[r], col_w[c], cell_h]) for c in range(4)] for r in range(2)]
    sizes = [axis_panel_size(fig_w, fig_h, col_w[c], cell_h) for c in range(4)]
    draw_cell(axes[0][0], fit_panel(image_array(crop_image(synth_images["curr"], synth_roi)), size=sizes[0], mode="cover", pad_color="#f8fafc"))
    draw_cell(axes[0][1], fit_panel(panel_rgb(crop_arr(synth_maps["pixel_indicator"], synth_roi), "Blues", fixed_vmax=1.0), size=sizes[1], mode="cover", pad_color="#f8fafc"))
    draw_cell(axes[0][2], response_triptych(synth_maps, synth_half_roi, size=sizes[2]), inner_labels=("LH", "HL", "HH"))
    draw_cell(
        axes[0][3],
        mask_pair_panel(
            panel_rgb(crop_arr(synth_maps["local"], synth_roi), "inferno"),
            panel_rgb(crop_arr(synth_maps["fused"], synth_roi), "plasma"),
            size=sizes[3],
        ),
        inner_labels=("branch", "fused"),
        label_positions=(0.235, 0.745),
    )

    draw_cell(axes[1][0], triplet_panel(real_images["prev"], real_images["curr"], real_images["next"], size=sizes[0]), inner_labels=("t-1", "t", "t+1"))
    draw_cell(axes[1][1], fit_panel(panel_rgb(real_maps["pixel_indicator"], "Blues", fixed_vmax=1.0), size=sizes[1], mode="cover", pad_color="#f8fafc"))
    draw_cell(axes[1][2], ll_response_panel(real_maps["ll_diff"], size=sizes[2]), inner_labels=("LL",))
    draw_cell(
        axes[1][3],
        mask_pair_panel(panel_rgb(real_maps["flicker"], "viridis"), panel_rgb(real_maps["fused"], "plasma"), size=sizes[3]),
        inner_labels=("branch", "fused"),
        label_positions=(0.235, 0.745),
    )

    for y in (row_y[0] + cell_h / 2, row_y[1] + cell_h / 2):
        for c in range(3):
            add_flow_arrow(fig, col_x[c] + col_w[c] + 0.005, y, col_x[c + 1] - 0.005, y)
    return fig


def draw_clean_evidence(synth: dict, real: dict) -> plt.Figure:
    synth_roi = synth["roi"]
    synth_images = synth["images"]
    synth_maps = synth["maps"]
    synth_half_roi = scaled_roi(synth_roi, synth_maps["lh_diff"], synth_maps["local"].shape)
    real_images = real["images"]
    real_maps = real["maps"]

    fig_w, fig_h = MDPI_TEXTWIDTH_IN, 2.82
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    left, right, bottom, top = 0.090, 0.018, 0.055, 0.100
    col_gap, row_gap = 0.014, 0.062
    rel_w = np.asarray([1.05, 1.05, 1.35, 1.05, 1.05], dtype=np.float32)
    avail_w = 1.0 - left - right - 4 * col_gap
    col_widths = (avail_w * rel_w / rel_w.sum()).tolist()
    cell_h = (1.0 - bottom - top - row_gap) / 2
    row_y = [bottom + cell_h + row_gap, bottom]
    col_x: list[float] = []
    x = left
    for width in col_widths:
        col_x.append(x)
        x += width + col_gap

    headers = ["Input", "Pixel indicator", "Wavelet response", "Branch mask", "Fused mask"]
    for idx, header in enumerate(headers):
        x0 = col_x[idx]
        w = col_widths[idx]
        fig.text(
            x0 + w / 2,
            0.958,
            header,
            ha="center",
            va="top",
            fontsize=5.4,
            fontfamily="DejaVu Sans",
            color=TEXT,
        )
        fig.add_artist(
            Rectangle((x0, 0.932), w, 0.0018, transform=fig.transFigure, facecolor=GRID, edgecolor="none")
        )

    row_labels = [
        ("(a)", "Sparse\nscratch"),
        ("(b)", "Brightness\nflicker"),
    ]
    for row_idx, (tag, label) in enumerate(row_labels):
        y = row_y[row_idx]
        fig.text(
            0.023,
            y + cell_h - 0.004,
            tag,
            ha="left",
            va="top",
            fontsize=5.7,
            fontfamily="DejaVu Sans",
            fontweight="bold",
            color=TEXT,
        )
        fig.text(
            0.055,
            y + cell_h / 2,
            label,
            ha="center",
            va="center",
            fontsize=4.8,
            fontfamily="DejaVu Sans",
            color=MUTED,
        )
        fig.add_artist(
            Rectangle(
                (left - 0.032, y - 0.010),
                1.0 - left - right + 0.032,
                cell_h + 0.020,
                transform=fig.transFigure,
                facecolor="#fbfdff",
                edgecolor="none",
                zorder=-10,
            )
        )

    def draw_panel(bounds: list[float], image: np.ndarray, *, mode: str = "cover") -> None:
        ax = fig.add_axes(bounds)
        size = axis_panel_size(fig_w, fig_h, bounds[2], bounds[3])
        draw_cell(ax, fit_panel(image, size=size, mode=mode, pad_color="#f8fafc"))

    bounds = [[col_x[c], row_y[r], col_widths[c], cell_h] for r in range(2) for c in range(5)]
    top = bounds[0:5]
    bottom_row = bounds[5:10]

    draw_panel(top[0], image_array(crop_image(synth_images["curr"], synth_roi)))
    draw_panel(top[1], panel_rgb(crop_arr(synth_maps["pixel_indicator"], synth_roi), "Blues", fixed_vmax=1.0))
    ax = fig.add_axes(top[2])
    draw_cell(
        ax,
        response_triptych(synth_maps, synth_half_roi, size=axis_panel_size(fig_w, fig_h, top[2][2], top[2][3])),
        inner_labels=("LH", "HL", "HH"),
        label_fontsize=3.9,
    )
    draw_panel(top[3], panel_rgb(crop_arr(synth_maps["local"], synth_roi), "inferno"))
    draw_panel(top[4], panel_rgb(crop_arr(synth_maps["fused"], synth_roi), "plasma"))

    ax = fig.add_axes(bottom_row[0])
    draw_cell(
        ax,
        triplet_panel(real_images["prev"], real_images["curr"], real_images["next"], size=axis_panel_size(fig_w, fig_h, bottom_row[0][2], bottom_row[0][3])),
        inner_labels=("t-1", "t", "t+1"),
        label_fontsize=3.9,
    )
    draw_panel(bottom_row[1], panel_rgb(real_maps["pixel_indicator"], "Blues", fixed_vmax=1.0))
    ax = fig.add_axes(bottom_row[2])
    draw_cell(
        ax,
        ll_response_panel(real_maps["ll_diff"], size=axis_panel_size(fig_w, fig_h, bottom_row[2][2], bottom_row[2][3])),
        inner_labels=("LL",),
        label_fontsize=3.9,
    )
    draw_panel(bottom_row[3], panel_rgb(real_maps["flicker"], "viridis"))
    draw_panel(bottom_row[4], panel_rgb(real_maps["fused"], "plasma"))

    return fig


def draw_figure(synth: dict, real: dict, args: argparse.Namespace, metadata: dict) -> None:
    candidates_dir = args.output_dir / "candidates"
    candidates: list[dict[str, str]] = []

    preview = args.output_dir / "Fig2_wavelet_frequency_motivation_preview.png"
    fig = draw_story_matrix(synth, real)
    if not args.skip_candidates:
        pdf = candidates_dir / "Fig1_candidate_A_story_matrix.pdf"
        svg = candidates_dir / "Fig1_candidate_A_story_matrix.svg"
        png = candidates_dir / "Fig1_candidate_A_story_matrix.png"
        save_figure(fig, pdf, svg, png)
        candidates.append({"name": "A_story_matrix", "pdf": str(pdf), "svg": str(svg), "preview_png": str(png)})
    plt.close(fig)

    if not args.skip_candidates:
        for name, drawer in (("B_grouped_flow", draw_grouped_flow),):
            fig = drawer(synth, real)
            pdf = candidates_dir / f"Fig1_candidate_{name}.pdf"
            svg = candidates_dir / f"Fig1_candidate_{name}.svg"
            png = candidates_dir / f"Fig1_candidate_{name}.png"
            save_figure(fig, pdf, svg, png)
            candidates.append({"name": name, "pdf": str(pdf), "svg": str(svg), "preview_png": str(png)})
            plt.close(fig)

    fig = draw_clean_evidence(synth, real)
    save_figure(fig, args.figure_pdf, args.figure_svg, args.figure_png)
    save_preview(fig, preview)
    if not args.skip_candidates:
        pdf = candidates_dir / "Fig1_candidate_D_mdpi_clean.pdf"
        svg = candidates_dir / "Fig1_candidate_D_mdpi_clean.svg"
        png = candidates_dir / "Fig1_candidate_D_mdpi_clean.png"
        save_figure(fig, pdf, svg, png)
        candidates.append({"name": "D_mdpi_clean", "pdf": str(pdf), "svg": str(svg), "preview_png": str(png)})
    plt.close(fig)

    metadata["layout_version"] = "mdpi_v11_small_sans_labels_filled_response"
    metadata["layout_final_candidate"] = "D_mdpi_clean"
    metadata["layout_candidates"] = candidates
    metadata["panel_grid"] = {"rows": 2, "cols": 5, "column_widths": "content_weighted", "flow": "left_to_right_compact_labels"}
    metadata["export"] = {
        "target_width": "13.86 cm MDPI textwidth",
        "latex_width": "\\textwidth",
        "font_family": "TeX Gyre Pagella / Palatino fallback",
        "final_png_dpi": 600,
        "pdf_raster_dpi": 600,
        "label_policy": "column and row labels outside panels; subband/time labels are compact corner tags without reserved title bands",
    }
    metadata["layout_sources"] = [
        {"name": "IEEE graphics resolution and size", "url": "https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/create-graphics-for-your-article/resolution-and-size/"},
        {"name": "Nature figure panel arrangement and fonts", "url": "https://research-figure-guide.nature.com/figures/building-and-exporting-figure-panels/"},
        {"name": "Nature figure specifications", "url": "https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/"},
    ]
    metadata["panel_normalization"] = {
        "pixel_indicator": "fixed display scale vmin=0, vmax=1",
        "wavelet_response": "percentile display-normalized per panel",
        "branch_mask": "percentile display-normalized per panel",
        "fused_mask": "percentile display-normalized per panel",
    }


def to_jsonable(item: dict) -> dict:
    ref = item["ref"]
    maps = item["maps"]
    return {
        "kind": ref.kind,
        "sequence": ref.sequence,
        "frame_index": ref.frame_index,
        "frame_path": str(ref.frame_path),
        "resized_size_wh": item["resized_size"],
        "roi_xyxy": item["roi"],
        "cheap_score": ref.cheap_score,
        "model_score": item["score"],
        "mask_signal": item["mask_signal"],
        "pixel_signal": item["pixel_signal"],
        "max_fusion_error": maps["max_fusion_error"],
    }


def load_locked_selection(output_dir: Path) -> dict[str, dict] | None:
    metadata_path = output_dir / "metadata.json"
    if not metadata_path.exists():
        return None
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    selected = data.get("selected", {})
    synth = selected.get("synthetic_sparse_damage")
    real = selected.get("real_flicker")
    if not synth or not real:
        return None
    return {"synthetic": synth, "real": real}


def main() -> None:
    args = parse_args()
    ensure_inputs(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model, load_report = load_model(args.checkpoint, device)

    locked_selection = load_locked_selection(args.output_dir)
    synth, synth_candidates = choose_candidate(
        "synthetic", model, args, device, None if locked_selection is None else locked_selection["synthetic"]
    )
    real, real_candidates = choose_candidate(
        "real", model, args, device, None if locked_selection is None else locked_selection["real"]
    )
    max_fusion_error = max(float(synth["maps"]["max_fusion_error"]), float(real["maps"]["max_fusion_error"]))
    if max_fusion_error > 1e-6:
        raise RuntimeError(f"m_wavelet != max(m_local, m_flicker): {max_fusion_error}")

    synth_raw = save_raw(synth, args.output_dir, "synthetic_sparse_damage")
    real_raw = save_raw(real, args.output_dir, "real_flicker")
    metadata = {
        "figure": "Fig2_wavelet_frequency_motivation",
        "server": "L40_aliyun",
        "repo_root": str(REPO_ROOT),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "expected_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "datasets": {
            "synthetic_lq": str(args.synthetic_lq_root),
            "synthetic_gt": str(args.synthetic_gt_root),
            "real": str(args.real_root),
        },
        "model": "WaveletMaskABNet(num_feat=16,num_block=6,flow_type=raft)",
        "model_load": load_report,
        "processing": {
            "short_side": args.short_side,
            "align": args.align,
            "pixel_indicator": "mean of bidirectional RAFT-warped first-channel absolute differences",
            "max_fusion_error": max_fusion_error,
        },
        "selected": {
            "synthetic_sparse_damage": {**to_jsonable(synth), **synth_raw},
            "real_flicker": {**to_jsonable(real), **real_raw},
        },
        "candidate_summaries": {
            "synthetic": synth_candidates,
            "real": real_candidates,
        },
    }
    draw_figure(synth, real, args, metadata)
    metadata["outputs"] = {
        "pdf": str(args.figure_pdf),
        "svg": str(args.figure_svg),
        "png_600dpi": str(args.figure_png),
        "preview_png": str(args.output_dir / "Fig2_wavelet_frequency_motivation_preview.png"),
    }
    metadata["output_sha256"] = {
        "pdf": sha256(args.figure_pdf),
        "svg": sha256(args.figure_svg),
        "png_600dpi": sha256(args.figure_png),
    }
    metadata_path = args.output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({
        "metadata": str(metadata_path),
        "pdf": str(args.figure_pdf),
        "svg": str(args.figure_svg),
        "max_fusion_error": max_fusion_error,
        "pdf_sha256": metadata["output_sha256"]["pdf"],
    }, indent=2))


if __name__ == "__main__":
    main()
