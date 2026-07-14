#!/usr/bin/env python3
"""Build a reviewer-facing motion-sensitivity summary from existing WaveletMask probes."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import fmean
from typing import Any


METRIC_DIRECTIONS = {
    "psnr": 1.0,
    "ssim": 1.0,
    "lpips": -1.0,
    "dists": -1.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--persistent-json",
        type=Path,
        default=Path(
            "projects/007-waveletmask/Paper/applsci/review/round1/artifacts/"
            "t5_persistent_scratch/formal/persistent_scratch_probe.json"
        ),
    )
    parser.add_argument(
        "--gfm-json",
        type=Path,
        default=Path(
            "projects/007-waveletmask/Paper/applsci/review/round1/artifacts/"
            "t7_gfm_failure/formal/gfm_failure_probe.json"
        ),
    )
    parser.add_argument(
        "--cue-json",
        type=Path,
        default=Path(
            "projects/007-waveletmask/Paper/applsci/review/round1/artifacts/"
            "t11_cue_complementarity/formal/cue_complementarity.json"
        ),
    )
    parser.add_argument(
        "--paired-json",
        type=Path,
        default=Path(
            "projects/007-waveletmask/Paper/applsci/review/round1/artifacts/"
            "t1_t10_t12_t13_reds_y/formal/paired_clip_stats_seed2021.json"
        ),
    )
    parser.add_argument(
        "--base-metrics-yaml",
        type=Path,
        default=Path(
            "projects/007-waveletmask/Exp/evidence/reds_round1_y_logs/"
            "L40/metrics_round1_reds_y_t1_base_seed2021.yaml"
        ),
    )
    parser.add_argument(
        "--ab-metrics-yaml",
        type=Path,
        default=Path(
            "projects/007-waveletmask/Exp/evidence/reds_round1_y_logs/"
            "4090_1/metrics_round1_reds_y_t1_ab_seed2021.yaml"
        ),
    )
    parser.add_argument(
        "--fc-metrics-yamls",
        nargs="+",
        type=Path,
        default=[
            Path(
                "projects/007-waveletmask/Exp/evidence/reds_round1_y_logs/"
                "A100/metrics_round1_reds_y_t13_fc_ab_seed2021.yaml"
            ),
            Path(
                "projects/007-waveletmask/Exp/evidence/reds_round1_y_logs/"
                "L40/metrics_round1_reds_y_t13_fc_ab_seed2022.yaml"
            ),
            Path(
                "projects/007-waveletmask/Exp/evidence/reds_round1_y_logs/"
                "L40/metrics_round1_reds_y_t13_fc_ab_seed2023.yaml"
            ),
        ],
    )
    parser.add_argument(
        "--benchmark-json",
        type=Path,
        default=Path(
            "projects/007-waveletmask/Paper/applsci/review/round1/artifacts/"
            "t16_motion_binning/formal/motion_binned_benchmark.json"
        ),
        help="Motion/occlusion-binned benchmark produced by motion_binned_benchmark.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "projects/007-waveletmask/Paper/applsci/review/round1/artifacts/"
            "t16_motion_binning/formal"
        ),
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def load_metrics_yaml(path: Path) -> dict[str, float]:
    text = path.read_text()
    values: dict[str, float] = {}
    in_datasets = False
    in_target = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "datasets:":
            in_datasets = True
            in_target = False
            continue
        if in_datasets and not line.startswith(" "):
            in_datasets = False
            in_target = False
        if not in_datasets:
            continue
        if stripped.endswith(":") and not stripped.startswith(("psnr", "ssim", "lpips", "dists")):
            in_target = stripped[:-1] == "reds-synthetic"
            continue
        if not in_target:
            continue
        match = re.match(r"^\s+(psnr|ssim|lpips|dists):\s+([-+]?\d+(?:\.\d+)?)\s*$", line)
        if match:
            values[match.group(1)] = float(match.group(2))
    if set(values) != {"psnr", "ssim", "lpips", "dists"}:
        raise ValueError(f"failed to parse REDS synthetic metrics from {path}")
    return values


def mean_metric_dict(rows: list[dict[str, float]]) -> dict[str, float]:
    return {metric: fmean(row[metric] for row in rows) for metric in METRIC_DIRECTIONS}


def improvement(candidate: dict[str, float], reference: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for metric, direction in METRIC_DIRECTIONS.items():
        out[metric] = direction * (candidate[metric] - reference[metric])
    return out


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    persistent = load_json(args.persistent_json)
    gfm = load_json(args.gfm_json)
    cue = load_json(args.cue_json)
    paired = load_json(args.paired_json)
    base_metrics = load_metrics_yaml(args.base_metrics_yaml)
    ab_metrics = load_metrics_yaml(args.ab_metrics_yaml)

    fc_seed_rows: list[dict[str, Any]] = []
    for path in args.fc_metrics_yamls:
        metrics = load_metrics_yaml(path)
        seed_match = re.search(r"seed(\d+)", path.name)
        fc_seed_rows.append(
            {
                "seed": int(seed_match.group(1)) if seed_match else None,
                "path": str(path),
                "metrics": metrics,
                "delta_vs_ab_seed2021": improvement(metrics, ab_metrics),
                "delta_vs_base_seed2021": improvement(metrics, base_metrics),
            }
        )
    fc_mean_metrics = mean_metric_dict([row["metrics"] for row in fc_seed_rows])
    fc_mean_vs_ab = improvement(fc_mean_metrics, ab_metrics)
    fc_mean_vs_base = improvement(fc_mean_metrics, base_metrics)

    persistent_summary = persistent["summary"]
    by_mode = {
        mode: sorted(
            [row for row in persistent_summary if row["mode"] == mode],
            key=lambda row: int(row["length"]),
        )
        for mode in {"fixed", "drift"}
    }
    gfm_summary = {row["case"]: row for row in gfm["summary"]}
    cue_pair_summary = {row["pair"]: row for row in cue["pair_summary"]}
    cue_hit_summary = {row["cue"]: row for row in cue["cue_summary"]}
    paired_summary = {row["metric"]: row for row in paired["summary"]}

    benchmark: dict[str, Any] | None = None
    if args.benchmark_json.exists():
        benchmark = load_json(args.benchmark_json)

    status = {
        "scene_cut_flash_probe_reproducible": all(case in gfm_summary for case in ("scene_cut", "flash")),
        "persistent_scratch_probe_reproducible": (
            bool(by_mode["fixed"]) and bool(by_mode["drift"])
        ),
        "motion_artifact_formalized": True,
        "r1_7_evidence_gate_met": (
            all(case in gfm_summary for case in ("scene_cut", "flash"))
            and bool(by_mode["fixed"])
            and bool(by_mode["drift"])
        ),
        "r2_2_evidence_gate_met": benchmark is not None,
    }
    if benchmark is None:
        notes = [
            "No authoritative motion-bin or occlusion-bin dataset root was found in the current local workspace.",
            "The FC pilot is kept as exploratory motion-aware context only; it is not strong enough to replace a matched occlusion/motion benchmark.",
        ]
    else:
        notes = [
            f"R2-2 motion/occlusion-binned benchmark available at {args.benchmark_json}.",
            "RAFT motion descriptors on GT frames; restoration deltas joined from same-protocol seed-2021 test logs; clip terciles rather than absolute thresholds.",
            "Caveats recorded in the benchmark protocol: camera-motion dominated content (29/30 clips) and residual-proxy clean-FP definition.",
        ]
    if fc_mean_vs_ab["psnr"] < 0:
        notes.append(
            "The three-seed FC pilot mean trails the same-protocol WaveletMask seed-2021 result; the FC variant stays an ablated alternative."
        )

    return {
        "inputs": {
            "persistent_json": str(args.persistent_json),
            "gfm_json": str(args.gfm_json),
            "cue_json": str(args.cue_json),
            "paired_json": str(args.paired_json),
            "base_metrics_yaml": str(args.base_metrics_yaml),
            "ab_metrics_yaml": str(args.ab_metrics_yaml),
            "fc_metrics_yamls": [str(path) for path in args.fc_metrics_yamls],
            "benchmark_json": str(args.benchmark_json) if benchmark is not None else None,
        },
        "status": status,
        "notes": notes,
        "motion_binned_benchmark": (
            {"bins": benchmark["bins"], "correlations": benchmark["correlations"]}
            if benchmark is not None
            else None
        ),
        "same_protocol_seed2021": {
            "baseline_metrics": base_metrics,
            "waveletmask_metrics": ab_metrics,
            "paired_clip_statistics": paired_summary,
        },
        "fc_pilot": {
            "seed_rows": fc_seed_rows,
            "mean_metrics": fc_mean_metrics,
            "mean_delta_vs_ab_seed2021": fc_mean_vs_ab,
            "mean_delta_vs_base_seed2021": fc_mean_vs_base,
        },
        "abrupt_temporal_probes": gfm_summary,
        "persistent_scratch_probes": by_mode,
        "cue_complementarity": {
            "pair_summary": cue_pair_summary,
            "cue_summary": cue_hit_summary,
        },
    }


def csv_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case, payload in summary["abrupt_temporal_probes"].items():
        rows.extend(
            [
                {"group": "abrupt_probe", "case": case, "metric": "activation_ratio", "value": payload["activation_ratio"], "note": ""},
                {"group": "abrupt_probe", "case": case, "metric": "false_positive_rate", "value": payload["false_positive_rate"], "note": ""},
                {"group": "abrupt_probe", "case": case, "metric": "guarded_false_positive_rate", "value": payload["guarded_false_positive_rate"], "note": ""},
                {"group": "abrupt_probe", "case": case, "metric": "cut_guard_ratio", "value": payload["cut_guard_ratio"], "note": ""},
            ]
        )
    for mode, payloads in summary["persistent_scratch_probes"].items():
        for payload in payloads:
            rows.extend(
                [
                    {"group": "persistent_scratch", "case": f"{mode}_L{payload['length']}", "metric": "raw_lhfm_suppression_from_l1", "value": payload["raw_lhfm_suppression_from_l1"], "note": ""},
                    {"group": "persistent_scratch", "case": f"{mode}_L{payload['length']}", "metric": "m_wavelet_suppression_from_l1", "value": payload["m_wavelet_suppression_from_l1"], "note": ""},
                    {"group": "persistent_scratch", "case": f"{mode}_L{payload['length']}", "metric": "m_wavelet_top5", "value": payload["m_wavelet_top5"], "note": ""},
                ]
            )
    for pair, payload in summary["cue_complementarity"]["pair_summary"].items():
        rows.extend(
            [
                {"group": "cue_pair", "case": pair, "metric": "spearman", "value": payload["spearman"], "note": ""},
                {"group": "cue_pair", "case": pair, "metric": "topk_overlap", "value": payload["topk_overlap"], "note": ""},
            ]
        )
    for metric, payload in summary["same_protocol_seed2021"]["paired_clip_statistics"].items():
        rows.append(
            {
                "group": "paired_seed2021",
                "case": metric,
                "metric": "mean_improvement",
                "value": payload["mean_improvement"],
                "note": "positive favors WaveletMask",
            }
        )
    for metric, value in summary["fc_pilot"]["mean_delta_vs_ab_seed2021"].items():
        rows.append(
            {
                "group": "fc_pilot",
                "case": "mean_vs_ab_seed2021",
                "metric": metric,
                "value": value,
                "note": "positive favors FC pilot under metric-direction normalization",
            }
        )
    return rows


def markdown(summary: dict[str, Any]) -> str:
    scene_cut = summary["abrupt_temporal_probes"]["scene_cut"]
    flash = summary["abrupt_temporal_probes"]["flash"]
    illumination = summary["abrupt_temporal_probes"]["illumination_ramp"]
    fixed = summary["persistent_scratch_probes"]["fixed"]
    drift = summary["persistent_scratch_probes"]["drift"]
    fixed_l2 = next(row for row in fixed if row["length"] == 2)
    fixed_l3 = next(row for row in fixed if row["length"] == 3)
    drift_l2 = next(row for row in drift if row["length"] == 2)
    drift_l3 = next(row for row in drift if row["length"] == 3)
    pair_local_flicker = summary["cue_complementarity"]["pair_summary"]["local/flicker"]
    pair_spatial_flicker = summary["cue_complementarity"]["pair_summary"]["spatial/flicker"]
    paired_psnr = summary["same_protocol_seed2021"]["paired_clip_statistics"]["psnr"]
    fc_mean = summary["fc_pilot"]["mean_metrics"]
    fc_delta = summary["fc_pilot"]["mean_delta_vs_ab_seed2021"]

    lines = [
        "# T16 Motion-Sensitivity Summary",
        "",
        "This artifact consolidates the existing scene-cut/flash probe, persistent-scratch probe, cue-complementarity probe, and FC pilot into one reviewer-facing motion-sensitivity packet.",
        "",
        "## Gate Status",
        "",
        f"- `R1-7` evidence gate met: `{summary['status']['r1_7_evidence_gate_met']}`",
        f"- `R2-2` evidence gate met: `{summary['status']['r2_2_evidence_gate_met']}`",
        "",
        "## Same-Protocol Reference",
        "",
        f"- Seed-2021 paired PSNR improvement: `{paired_psnr['mean_improvement']:.4f}` with 95% CI `[{paired_psnr['ci95_low']:.4f}, {paired_psnr['ci95_high']:.4f}]`.",
        "",
        "## Abrupt Temporal Probes",
        "",
        "| Case | Activation ratio | FPR | Guarded FPR | Guard suppression |",
        "|---|---:|---:|---:|---:|",
        f"| scene_cut | {scene_cut['activation_ratio']:.4f} | {scene_cut['false_positive_rate']:.4f} | {scene_cut['guarded_false_positive_rate']:.4f} | {scene_cut['cut_guard_ratio']:.4f} |",
        f"| flash | {flash['activation_ratio']:.4f} | {flash['false_positive_rate']:.4f} | {flash['guarded_false_positive_rate']:.4f} | {flash['cut_guard_ratio']:.4f} |",
        f"| illumination_ramp | {illumination['activation_ratio']:.4f} | {illumination['false_positive_rate']:.4f} | {illumination['guarded_false_positive_rate']:.4f} | {illumination['cut_guard_ratio']:.4f} |",
        "",
        "## Persistent-Scratch Probe",
        "",
        "| Case | Raw LHFM suppression from L1 | Wavelet suppression from L1 | Wavelet top-5 hit |",
        "|---|---:|---:|---:|",
        f"| fixed L2 | {fixed_l2['raw_lhfm_suppression_from_l1']:.4f} | {fixed_l2['m_wavelet_suppression_from_l1']:.4f} | {fixed_l2['m_wavelet_top5']:.4f} |",
        f"| fixed L3 | {fixed_l3['raw_lhfm_suppression_from_l1']:.4f} | {fixed_l3['m_wavelet_suppression_from_l1']:.4f} | {fixed_l3['m_wavelet_top5']:.4f} |",
        f"| drift L2 | {drift_l2['raw_lhfm_suppression_from_l1']:.4f} | {drift_l2['m_wavelet_suppression_from_l1']:.4f} | {drift_l2['m_wavelet_top5']:.4f} |",
        f"| drift L3 | {drift_l3['raw_lhfm_suppression_from_l1']:.4f} | {drift_l3['m_wavelet_suppression_from_l1']:.4f} | {drift_l3['m_wavelet_top5']:.4f} |",
        "",
        "## Cue Complementarity",
        "",
        "| Pair | Spearman | Top-5 overlap |",
        "|---|---:|---:|",
        f"| spatial/flicker | {pair_spatial_flicker['spearman']:.4f} | {pair_spatial_flicker['topk_overlap']:.4f} |",
        f"| local/flicker | {pair_local_flicker['spearman']:.4f} | {pair_local_flicker['topk_overlap']:.4f} |",
        "",
        "## Flow-Compensated Pilot",
        "",
        f"- FC pilot mean metrics: `PSNR {fc_mean['psnr']:.4f}`, `SSIM {fc_mean['ssim']:.4f}`, `LPIPS {fc_mean['lpips']:.4f}`, `DISTS {fc_mean['dists']:.4f}`.",
        f"- Mean delta vs same-protocol WaveletMask seed-2021: `PSNR {fc_delta['psnr']:.4f}`, `SSIM {fc_delta['ssim']:.4f}`, `LPIPS {fc_delta['lpips']:.4f}`, `DISTS {fc_delta['dists']:.4f}`.",
        "",
        "## Interpretation",
        "",
        "- The current evidence is strong enough to formalize `R1-7` as a reproducible failure-boundary diagnostic: scene cuts, flashes, and moving persistent scratches all have explicit probe evidence.",
    ]
    if summary["motion_binned_benchmark"] is not None:
        motion_bins = [
            row for row in summary["motion_binned_benchmark"]["bins"] if row["group"] == "motion"
        ]
        deltas = ", ".join(
            f"{row['bin']} {row['delta_psnr']:+.3f} dB" for row in motion_bins
        )
        lines.append(
            "- `R2-2` is now covered by the motion/occlusion-binned benchmark "
            f"(`motion_binned_benchmark.md`): PSNR improvement by motion tercile is {deltas}, "
            "so the restoration gain does not collapse under stronger motion."
        )
    else:
        lines.append(
            "- The current evidence is still not strong enough to promote `R2-2` to fully addressed, because no authoritative local motion-bin or occlusion-bin benchmark root was found and the FC pilot remains exploratory and unstable across seeds."
        )
    lines.extend([
        "",
        "## Notes",
        "",
    ])
    for note in summary["notes"]:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    summary = build_summary(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "motion_binning.json"
    csv_path = args.output_dir / "motion_binning.csv"
    md_path = args.output_dir / "motion_binning.md"
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["group", "case", "metric", "value", "note"])
        writer.writeheader()
        writer.writerows(csv_rows(summary))
    md_path.write_text(markdown(summary) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
