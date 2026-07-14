#!/usr/bin/env python3
"""Compute paired clip-level statistics from BasicSR test logs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np


CLIP_RE = re.compile(
    r"->\s+(?P<clip>\d+)\s+metrics:\s+psnr:\s+(?P<psnr>[-+]?\d+(?:\.\d+)?)"
    r",\s+ssim:\s+(?P<ssim>[-+]?\d+(?:\.\d+)?)"
    r",\s+lpips:\s+(?P<lpips>[-+]?\d+(?:\.\d+)?)"
    r",\s+dists:\s+(?P<dists>[-+]?\d+(?:\.\d+)?)"
)

METRICS = {
    "psnr": {"higher_is_better": True, "label": "PSNR"},
    "ssim": {"higher_is_better": True, "label": "SSIM"},
    "lpips": {"higher_is_better": False, "label": "LPIPS"},
    "dists": {"higher_is_better": False, "label": "DISTS"},
}


def parse_log(path: Path) -> dict[str, dict[str, float]]:
    text = path.read_text()
    metrics: dict[str, dict[str, float]] = {}
    for match in CLIP_RE.finditer(text):
        clip = match.group("clip")
        metrics[clip] = {
            metric: float(match.group(metric))
            for metric in ("psnr", "ssim", "lpips", "dists")
        }
    if not metrics:
        raise ValueError(f"no clip metrics found in {path}")
    return metrics


def permutation_pvalue(diffs: np.ndarray, rng: np.random.Generator, draws: int) -> float:
    observed = abs(float(diffs.mean()))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(draws, diffs.size))
    null_means = np.abs((signs * diffs).mean(axis=1))
    return float((np.count_nonzero(null_means >= observed) + 1) / (draws + 1))


def bootstrap_ci(diffs: np.ndarray, rng: np.random.Generator, draws: int) -> tuple[float, float]:
    idx = rng.integers(0, diffs.size, size=(draws, diffs.size))
    means = diffs[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-log", required=True, type=Path)
    parser.add_argument("--waveletmask-log", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--draws", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260608)
    args = parser.parse_args()

    base = parse_log(args.baseline_log)
    ours = parse_log(args.waveletmask_log)
    common = sorted(set(base) & set(ours))
    if not common:
        raise ValueError("no overlapping clips between logs")

    rng = np.random.default_rng(args.seed)
    rows = []
    summary = []

    for metric, spec in METRICS.items():
        base_vals = np.array([base[clip][metric] for clip in common], dtype=float)
        ours_vals = np.array([ours[clip][metric] for clip in common], dtype=float)
        if spec["higher_is_better"]:
            improvement = ours_vals - base_vals
        else:
            improvement = base_vals - ours_vals

        ci_lo, ci_hi = bootstrap_ci(improvement, rng, args.draws)
        pvalue = permutation_pvalue(improvement, rng, args.draws)
        summary.append(
            {
                "metric": metric,
                "metric_label": spec["label"],
                "n_clips": len(common),
                "baseline_mean": float(base_vals.mean()),
                "waveletmask_mean": float(ours_vals.mean()),
                "mean_improvement": float(improvement.mean()),
                "median_improvement": float(np.median(improvement)),
                "ci95_low": ci_lo,
                "ci95_high": ci_hi,
                "permutation_pvalue": pvalue,
                "higher_is_better": spec["higher_is_better"],
            }
        )

        for clip, b, o, d in zip(common, base_vals, ours_vals, improvement):
            rows.append(
                {
                    "clip": clip,
                    "metric": metric,
                    "baseline": b,
                    "waveletmask": o,
                    "improvement": d,
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "paired_clip_stats_seed2021.csv"
    json_path = args.output_dir / "paired_clip_stats_seed2021.json"
    md_path = args.output_dir / "paired_clip_stats_seed2021.md"

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["clip", "metric", "baseline", "waveletmask", "improvement"],
        )
        writer.writeheader()
        writer.writerows(rows)

    json_path.write_text(
        json.dumps(
            {
                "baseline_log": str(args.baseline_log),
                "waveletmask_log": str(args.waveletmask_log),
                "n_clips": len(common),
                "draws": args.draws,
                "seed": args.seed,
                "summary": summary,
            },
            indent=2,
        )
    )

    md_lines = [
        "# Paired Clip Statistics (Seed 2021)",
        "",
        "Primary same-protocol paired comparison between the re-trained RTN-style baseline and WaveletMask.",
        "",
        f"- Baseline log: `{args.baseline_log}`",
        f"- WaveletMask log: `{args.waveletmask_log}`",
        f"- Number of paired clips: `{len(common)}`",
        f"- Bootstrap draws: `{args.draws}`",
        f"- Permutation draws: `{args.draws}`",
        "",
        "| Metric | Baseline mean | WaveletMask mean | Mean paired improvement | 95% bootstrap CI | Two-sided permutation p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in summary:
        md_lines.append(
            "| {label} | {base:.4f} | {ours:.4f} | {imp:.4f} | [{lo:.4f}, {hi:.4f}] | {p:.4g} |".format(
                label=item["metric_label"],
                base=item["baseline_mean"],
                ours=item["waveletmask_mean"],
                imp=item["mean_improvement"],
                lo=item["ci95_low"],
                hi=item["ci95_high"],
                p=item["permutation_pvalue"],
            )
        )
    md_lines.extend(
        [
            "",
            "Improvement is defined so that positive values always favor WaveletMask:",
            "",
            "- `PSNR`, `SSIM`: `WaveletMask - RTN`",
            "- `LPIPS`, `DISTS`: `RTN - WaveletMask`",
            "",
            "The permutation test uses random sign flips of the paired clip differences.",
        ]
    )
    md_path.write_text("\n".join(md_lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
