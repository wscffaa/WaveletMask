#!/usr/bin/env python3
"""Split existing follow-up ablation metrics into dedicated wavelet/fusion reviewer artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


METRICS = ("psnr", "ssim", "lpips", "dists")
METRIC_DIRECTIONS = {
    "psnr": 1.0,
    "ssim": 1.0,
    "lpips": -1.0,
    "dists": -1.0,
}
LOG_ROOT = Path("projects/007-waveletmask/Exp/evidence/reds_round1_y_logs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference-yaml",
        type=Path,
        default=Path(
            "projects/007-waveletmask/Exp/evidence/reds_round1_y_logs/"
            "4090_1/metrics_round1_reds_y_t1_ab_seed2021.yaml"
        ),
    )
    parser.add_argument(
        "--wavelet-yamls",
        nargs="+",
        type=Path,
        default=[
            Path(
                "projects/007-waveletmask/Exp/evidence/reds_round1_y_logs/"
                "A100/metrics_round1_reds_y_t10_2level_haar.yaml"
            ),
            Path(
                "projects/007-waveletmask/Exp/evidence/reds_round1_y_logs/"
                "A100/metrics_round1_reds_y_t10_db2.yaml"
            ),
            Path(
                "projects/007-waveletmask/Exp/evidence/reds_round1_y_logs/"
                "A100/metrics_round1_reds_y_t10_sym2.yaml"
            ),
            Path(
                "projects/007-waveletmask/Exp/evidence/reds_round1_y_logs/"
                "A100/metrics_round1_reds_y_t10_coif1.yaml"
            ),
            Path(
                "projects/007-waveletmask/Exp/evidence/reds_round1_y_logs/"
                "A100/metrics_round1_reds_y_t10_dtcwt.yaml"
            ),
        ],
    )
    parser.add_argument(
        "--fusion-yamls",
        nargs="+",
        type=Path,
        default=[
            Path(
                "projects/007-waveletmask/Exp/evidence/reds_round1_y_logs/"
                "4090_1/metrics_round1_reds_y_t12_attention_fusion.yaml"
            ),
            Path(
                "projects/007-waveletmask/Exp/evidence/reds_round1_y_logs/"
                "A100/metrics_round1_reds_y_t12_residual_fusion.yaml"
            ),
            Path(
                "projects/007-waveletmask/Exp/evidence/reds_round1_y_logs/"
                "A100/metrics_round1_reds_y_t12_bias0.yaml"
            ),
            Path(
                "projects/007-waveletmask/Exp/evidence/reds_round1_y_logs/"
                "A100/metrics_round1_reds_y_t12_bias2.yaml"
            ),
            Path(
                "projects/007-waveletmask/Exp/evidence/reds_round1_y_logs/"
                "A100/metrics_round1_reds_y_t12_scalar_gate.yaml"
            ),
            Path(
                "projects/007-waveletmask/Exp/evidence/reds_round1_y_logs/"
                "A100/metrics_round1_reds_y_t12_normalization_aware.yaml"
            ),
        ],
    )
    parser.add_argument(
        "--wavelet-out-dir",
        type=Path,
        default=Path(
            "projects/007-waveletmask/Paper/applsci/review/round1/artifacts/"
            "t18_wavelet_family/formal"
        ),
    )
    parser.add_argument(
        "--fusion-out-dir",
        type=Path,
        default=Path(
            "projects/007-waveletmask/Paper/applsci/review/round1/artifacts/"
            "t19_fusion_completion/formal"
        ),
    )
    return parser.parse_args()


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
        if stripped.endswith(":") and not stripped.startswith(METRICS):
            in_target = stripped[:-1] == "reds-synthetic"
            continue
        if not in_target:
            continue
        match = re.match(r"^\s+(psnr|ssim|lpips|dists):\s+([-+]?\d+(?:\.\d+)?)\s*$", line)
        if match:
            values[match.group(1)] = float(match.group(2))
    if set(values) != set(METRICS):
        raise ValueError(f"failed to parse REDS synthetic metrics from {path}")
    return values


def resolve_metric_paths(paths: list[Path]) -> list[Path]:
    resolved = []
    seen = set()
    for path in paths:
        candidates = [path]
        if not path.exists():
            candidates = sorted(LOG_ROOT.rglob(path.name))
        for candidate in candidates:
            if not candidate.exists():
                continue
            if candidate in seen:
                continue
            seen.add(candidate)
            resolved.append(candidate)
            break
    return resolved


def delta_vs_reference(candidate: dict[str, float], reference: dict[str, float]) -> dict[str, float]:
    return {
        metric: METRIC_DIRECTIONS[metric] * (candidate[metric] - reference[metric])
        for metric in METRICS
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["track", "variant", "metric", "value", "delta_vs_reference", "note"])
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def write_markdown(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n")


def summarize_wavelet(reference_metrics: dict[str, float], paths: list[Path], out_dir: Path) -> None:
    labels = {
        "metrics_round1_reds_y_t10_2level_haar.yaml": "haar_2level",
        "metrics_round1_reds_y_t10_db2.yaml": "db2",
        "metrics_round1_reds_y_t10_sym2.yaml": "sym2",
        "metrics_round1_reds_y_t10_coif1.yaml": "coif1",
        "metrics_round1_reds_y_t10_dtcwt.yaml": "dtcwt",
    }
    required = {"coif1": "coif1", "dtcwt": "DTCWT"}
    rows = []
    csv_rows = []
    tested = set()
    for path in resolve_metric_paths(paths):
        metrics = load_metrics_yaml(path)
        label = labels.get(path.name, path.stem)
        delta = delta_vs_reference(metrics, reference_metrics)
        tested.add(label)
        rows.append({"variant": label, "source": str(path), "metrics": metrics, "delta_vs_reference": delta})
        for metric in METRICS:
            csv_rows.append(
                {
                    "track": "wavelet_family",
                    "variant": label,
                    "metric": metric,
                    "value": metrics[metric],
                    "delta_vs_reference": delta[metric],
                    "note": "positive delta favors candidate under metric-direction normalization",
                }
            )
    missing = [display for key, display in required.items() if key not in tested]

    payload = {
        "reference_variant": "haar_1level_main_seed2021",
        "reference_metrics": reference_metrics,
        "tested_variants": rows,
        "missing_variants": missing,
        "status": {
            "coif1_tested": "coif1" in tested,
            "dtcwt_tested": "dtcwt" in tested,
            "ready_for_submission_promotion": not missing,
        },
        "notes": [
            "Current local evidence includes the wavelet-family metrics that can be resolved under reds_round1_y_logs.",
            "Missing reviewer-requested wavelet variants remain blocked until their metrics YAMLs exist in the current workspace.",
        ],
    }

    md_lines = [
        "# T18 Wavelet-Family Summary",
        "",
        "This artifact isolates the wavelet-family evidence already available in the current review workspace.",
        "",
        "## Reference",
        "",
        f"- Main WaveletMask seed-2021 metrics: `PSNR {reference_metrics['psnr']:.4f}`, `SSIM {reference_metrics['ssim']:.4f}`, `LPIPS {reference_metrics['lpips']:.4f}`, `DISTS {reference_metrics['dists']:.4f}`.",
        "",
        "## Tested Follow-Up Variants",
        "",
        "| Variant | PSNR | SSIM | LPIPS | DISTS | Delta PSNR | Delta SSIM | Delta LPIPS | Delta DISTS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        metrics = row["metrics"]
        delta = row["delta_vs_reference"]
        md_lines.append(
            f"| {row['variant']} | {metrics['psnr']:.4f} | {metrics['ssim']:.4f} | {metrics['lpips']:.4f} | {metrics['dists']:.4f} | "
            f"{delta['psnr']:.4f} | {delta['ssim']:.4f} | {delta['lpips']:.4f} | {delta['dists']:.4f} |"
        )
    md_lines.extend(
        ["", "## Missing Reviewer-Requested Variants", ""]
    )
    if missing:
        for item in missing:
            md_lines.append(f"- `{item}`: not tested in the current local workspace")
    else:
        md_lines.append("- None. All currently tracked reviewer-requested wavelet variants have local metrics.")
    md_lines.extend(
        [
            "",
            "## Decision",
            "",
            (
                "- Keep `R1-6` / `R2-3` as `Partially addressed` until `coif1` and DTCWT are either tested or explicitly ruled out with stronger scope evidence."
                if missing else
                "- All currently tracked wavelet-family reviewer variants now have local metrics; promote status only after manuscript/response text is updated from these results."
            ),
            "",
        ]
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "wavelet_family_summary.json", payload)
    write_csv(out_dir / "wavelet_family_summary.csv", csv_rows)
    write_markdown(out_dir / "wavelet_family_summary.md", md_lines)


def summarize_fusion(reference_metrics: dict[str, float], paths: list[Path], out_dir: Path) -> None:
    labels = {
        "metrics_round1_reds_y_t12_attention_fusion.yaml": "channel_attention",
        "metrics_round1_reds_y_t12_residual_fusion.yaml": "residual",
        "metrics_round1_reds_y_t12_bias0.yaml": "bias0",
        "metrics_round1_reds_y_t12_bias2.yaml": "bias2",
        "metrics_round1_reds_y_t12_scalar_gate.yaml": "scalar_gate",
        "metrics_round1_reds_y_t12_normalization_aware.yaml": "normalization_aware",
    }
    required = {
        "bias0": "bias0",
        "bias2": "bias2",
        "scalar_gate": "scalar_gate",
        "normalization_aware": "normalization_aware",
    }
    rows = []
    csv_rows = []
    tested = set()
    for path in resolve_metric_paths(paths):
        metrics = load_metrics_yaml(path)
        label = labels.get(path.name, path.stem)
        delta = delta_vs_reference(metrics, reference_metrics)
        tested.add(label)
        rows.append({"variant": label, "source": str(path), "metrics": metrics, "delta_vs_reference": delta})
        for metric in METRICS:
            csv_rows.append(
                {
                    "track": "fusion_completion",
                    "variant": label,
                    "metric": metric,
                    "value": metrics[metric],
                    "delta_vs_reference": delta[metric],
                    "note": "positive delta favors candidate under metric-direction normalization",
                }
            )
    missing = [display for key, display in required.items() if key not in tested]

    payload = {
        "reference_variant": "max_fusion_main_seed2021",
        "reference_metrics": reference_metrics,
        "tested_variants": rows,
        "missing_variants": missing,
        "status": {
            "attention_tested": "channel_attention" in tested,
            "residual_tested": "residual" in tested,
            "bias0_tested": "bias0" in tested,
            "bias2_tested": "bias2" in tested,
            "scalar_gate_tested": "scalar_gate" in tested,
            "normalization_aware_tested": "normalization_aware" in tested,
            "ready_for_submission_promotion": not missing,
        },
        "notes": [
            "Current local evidence includes the fusion metrics that can be resolved under reds_round1_y_logs.",
            "Missing stronger fusion variants remain blocked until their metrics YAMLs exist in the current workspace.",
        ],
    }

    md_lines = [
        "# T19 Fusion-Completion Summary",
        "",
        "This artifact isolates the fusion-ablation evidence already available in the current review workspace.",
        "",
        "## Reference",
        "",
        f"- Main max-fusion seed-2021 metrics: `PSNR {reference_metrics['psnr']:.4f}`, `SSIM {reference_metrics['ssim']:.4f}`, `LPIPS {reference_metrics['lpips']:.4f}`, `DISTS {reference_metrics['dists']:.4f}`.",
        "",
        "## Tested Follow-Up Variants",
        "",
        "| Variant | PSNR | SSIM | LPIPS | DISTS | Delta PSNR | Delta SSIM | Delta LPIPS | Delta DISTS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        metrics = row["metrics"]
        delta = row["delta_vs_reference"]
        md_lines.append(
            f"| {row['variant']} | {metrics['psnr']:.4f} | {metrics['ssim']:.4f} | {metrics['lpips']:.4f} | {metrics['dists']:.4f} | "
            f"{delta['psnr']:.4f} | {delta['ssim']:.4f} | {delta['lpips']:.4f} | {delta['dists']:.4f} |"
        )
    md_lines.extend(
        ["", "## Missing Stronger Variants", ""]
    )
    if missing:
        grouped = {
            "bias_pair": [item for item in ("bias0", "bias2") if item in missing],
            "scalar_gate": "scalar_gate" in missing,
            "normalization_aware": "normalization_aware" in missing,
        }
        if grouped["bias_pair"]:
            md_lines.append("- `" + "` / `".join(grouped["bias_pair"]) + "`: not tested in the current local workspace")
        if grouped["scalar_gate"]:
            md_lines.append("- `scalar_gate`: not tested in the current local workspace")
        if grouped["normalization_aware"]:
            md_lines.append("- `normalization_aware`: not tested in the current local workspace")
    else:
        md_lines.append("- None. All currently tracked stronger fusion variants have local metrics.")
    md_lines.extend(
        [
            "",
            "## Decision",
            "",
            (
                "- Keep `R1-8` / `R2-6` as `Partially addressed` until the stronger requested fusion variants are actually tested."
                if missing else
                "- All currently tracked stronger fusion variants now have local metrics; promote status only after manuscript/response text is updated from these results."
            ),
            "",
        ]
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "fusion_completion_summary.json", payload)
    write_csv(out_dir / "fusion_completion_summary.csv", csv_rows)
    write_markdown(out_dir / "fusion_completion_summary.md", md_lines)


def main() -> int:
    args = parse_args()
    reference_metrics = load_metrics_yaml(args.reference_yaml)
    summarize_wavelet(reference_metrics, args.wavelet_yamls, args.wavelet_out_dir)
    summarize_fusion(reference_metrics, args.fusion_yamls, args.fusion_out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
