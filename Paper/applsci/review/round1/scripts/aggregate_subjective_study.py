#!/usr/bin/env python3
"""Aggregate live WaveletMask human-study exports into reviewer-facing summaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

DIMENSIONS = ("degradation", "style", "temporal")
DEFAULT_NAME_PATTERNS = (r"^(SIM-|DRY-RUN-)",)
DEFAULT_EMAIL_PATTERNS = (r"@example\.invalid$",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses-csv", required=True, type=Path)
    parser.add_argument("--stats-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--chapter", default="waveletmask")
    parser.add_argument("--preferred-method", default="waveletmask")
    parser.add_argument("--baseline-method", default="rtn")
    parser.add_argument("--ordinary-min", type=int, default=15)
    parser.add_argument("--expert-min", type=int, default=2)
    parser.add_argument("--bootstrap-draws", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument("--expert-ids-file", type=Path)
    parser.add_argument("--expert-emails-file", type=Path)
    parser.add_argument("--exclude-name-regex", action="append")
    parser.add_argument("--exclude-email-regex", action="append")
    return parser.parse_args()


def compile_patterns(cli_values: list[str] | None, defaults: tuple[str, ...]) -> list[re.Pattern[str]]:
    patterns = list(defaults)
    if cli_values:
        patterns.extend(cli_values)
    return [re.compile(pattern, flags=re.IGNORECASE) for pattern in patterns]


def load_lines(path: Path | None) -> set[str]:
    if path is None:
        return set()
    return {
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


def parse_json_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return [str(item) for item in value] if isinstance(value, list) else []


def quantile(sorted_values: list[float], q: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = q * (len(sorted_values) - 1)
    lo = int(math.floor(position))
    hi = int(math.ceil(position))
    if lo == hi:
        return sorted_values[lo]
    weight = position - lo
    return sorted_values[lo] * (1.0 - weight) + sorted_values[hi] * weight


def bootstrap_ci(values: list[float], rng: random.Random, draws: int) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    size = len(values)
    means: list[float] = []
    for _ in range(draws):
        sample = [values[rng.randrange(size)] for _ in range(size)]
        means.append(sum(sample) / size)
    means.sort()
    return float(quantile(means, 0.025)), float(quantile(means, 0.975))


def safe_mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def bool_from_row(value: str | int | None) -> bool:
    return str(value).strip() in {"1", "true", "True"}


def pick_exclusion_reason(
    participant: dict[str, Any],
    name_patterns: list[re.Pattern[str]],
    email_patterns: list[re.Pattern[str]],
) -> str | None:
    name = str(participant.get("name") or "")
    email = str(participant.get("email") or "")
    for pattern in name_patterns:
        if pattern.search(name):
            return f"name:{pattern.pattern}"
    for pattern in email_patterns:
        if pattern.search(email):
            return f"email:{pattern.pattern}"
    return None


def participant_summary(
    participant_id: str,
    participant_meta: dict[str, Any],
    rows: list[dict[str, str]],
    preferred_method: str,
    baseline_method: str,
) -> dict[str, Any]:
    overall_counter = Counter(row.get("overall_best", "") for row in rows)
    confidence_counter = Counter((row.get("confidence") or "unknown").lower() for row in rows)
    method_order_variants = Counter(tuple(parse_json_list(row.get("method_order"))) for row in rows)

    dimension_rates: dict[str, float | None] = {}
    preferred_rank_means: dict[str, float | None] = {}
    baseline_rank_means: dict[str, float | None] = {}
    for dimension in DIMENSIONS:
        preferred_wins = 0
        comparable = 0
        preferred_ranks: list[float] = []
        baseline_ranks: list[float] = []
        field = f"rank_{dimension}"
        for row in rows:
            ranking = parse_json_list(row.get(field))
            if preferred_method in ranking:
                preferred_ranks.append(float(ranking.index(preferred_method) + 1))
            if baseline_method in ranking:
                baseline_ranks.append(float(ranking.index(baseline_method) + 1))
            if preferred_method in ranking and baseline_method in ranking:
                comparable += 1
                if ranking.index(preferred_method) < ranking.index(baseline_method):
                    preferred_wins += 1
        dimension_rates[dimension] = preferred_wins / comparable if comparable else None
        preferred_rank_means[dimension] = safe_mean(preferred_ranks)
        baseline_rank_means[dimension] = safe_mean(baseline_ranks)

    sample_count = len(rows)
    preferred_count = overall_counter.get(preferred_method, 0)
    baseline_count = overall_counter.get(baseline_method, 0)
    other_count = sample_count - preferred_count - baseline_count
    time_seconds = [float(row.get("time_spent_ms") or 0) / 1000.0 for row in rows]

    return {
        "participant_id": participant_id,
        "name": participant_meta.get("name"),
        "email": participant_meta.get("email"),
        "background": participant_meta.get("background"),
        "experience_years": participant_meta.get("experience_years"),
        "training_score": participant_meta.get("training_score"),
        "completed_at": participant_meta.get("completed_at"),
        "sample_count": sample_count,
        "preferred_count": preferred_count,
        "baseline_count": baseline_count,
        "other_count": other_count,
        "preferred_rate": preferred_count / sample_count if sample_count else None,
        "baseline_rate": baseline_count / sample_count if sample_count else None,
        "other_rate": other_count / sample_count if sample_count else None,
        "mean_time_seconds": safe_mean(time_seconds),
        "rushed_fraction": (
            sum(bool_from_row(row.get("is_rushed")) for row in rows) / sample_count
            if sample_count else None
        ),
        "distracted_fraction": (
            sum(bool_from_row(row.get("is_distracted")) for row in rows) / sample_count
            if sample_count else None
        ),
        "confidence_distribution": dict(confidence_counter),
        "method_order_variants": {"|".join(key): value for key, value in method_order_variants.items()},
        "dimension_preferred_rate": dimension_rates,
        "preferred_rank_mean": preferred_rank_means,
        "baseline_rank_mean": baseline_rank_means,
    }


def summarize_cohort(
    cohort_name: str,
    participants: list[dict[str, Any]],
    preferred_method: str,
    baseline_method: str,
    rng: random.Random,
    draws: int,
) -> dict[str, Any]:
    participant_rates = [row["preferred_rate"] for row in participants if row["preferred_rate"] is not None]
    ci_lo, ci_hi = bootstrap_ci(participant_rates, rng, draws)

    dimension_rates: dict[str, list[float]] = defaultdict(list)
    preferred_rank_means: dict[str, list[float]] = defaultdict(list)
    baseline_rank_means: dict[str, list[float]] = defaultdict(list)
    background_counter = Counter()
    confidence_counter = Counter()
    method_order_counter = Counter()
    total_rows = 0
    preferred_total = 0
    baseline_total = 0
    other_total = 0
    sample_counts: list[float] = []
    mean_times: list[float] = []
    rushed_values: list[float] = []
    distracted_values: list[float] = []

    for participant in participants:
        total_rows += int(participant["sample_count"])
        preferred_total += int(participant["preferred_count"])
        baseline_total += int(participant["baseline_count"])
        other_total += int(participant["other_count"])
        sample_counts.append(float(participant["sample_count"]))
        if participant["mean_time_seconds"] is not None:
            mean_times.append(float(participant["mean_time_seconds"]))
        if participant["rushed_fraction"] is not None:
            rushed_values.append(float(participant["rushed_fraction"]))
        if participant["distracted_fraction"] is not None:
            distracted_values.append(float(participant["distracted_fraction"]))
        if participant.get("background"):
            background_counter[str(participant["background"])] += 1
        for key, value in participant["confidence_distribution"].items():
            confidence_counter[key] += value
        for key, value in participant["method_order_variants"].items():
            method_order_counter[key] += value
        for dimension in DIMENSIONS:
            rate = participant["dimension_preferred_rate"].get(dimension)
            if rate is not None:
                dimension_rates[dimension].append(float(rate))
            pref_rank = participant["preferred_rank_mean"].get(dimension)
            if pref_rank is not None:
                preferred_rank_means[dimension].append(float(pref_rank))
            base_rank = participant["baseline_rank_mean"].get(dimension)
            if base_rank is not None:
                baseline_rank_means[dimension].append(float(base_rank))

    summary: dict[str, Any] = {
        "cohort": cohort_name,
        "n_participants": len(participants),
        "n_responses": total_rows,
        "sample_count_min": int(min(sample_counts)) if sample_counts else 0,
        "sample_count_max": int(max(sample_counts)) if sample_counts else 0,
        "mean_sample_count": safe_mean(sample_counts),
        "mean_time_seconds_per_response": safe_mean(mean_times),
        "rushed_fraction": safe_mean(rushed_values),
        "distracted_fraction": safe_mean(distracted_values),
        "background_distribution": dict(background_counter),
        "confidence_distribution": dict(confidence_counter),
        "method_order_distribution": dict(method_order_counter),
        "overall_preference": {
            preferred_method: preferred_total / total_rows if total_rows else None,
            baseline_method: baseline_total / total_rows if total_rows else None,
            "other": other_total / total_rows if total_rows else None,
            "preferred_participant_rate_mean": safe_mean(participant_rates),
            "preferred_participant_rate_ci95_low": ci_lo,
            "preferred_participant_rate_ci95_high": ci_hi,
        },
        "dimension_preference": {},
        "participants": participants,
    }
    for dimension in DIMENSIONS:
        rate_values = dimension_rates.get(dimension, [])
        dim_lo, dim_hi = bootstrap_ci(rate_values, rng, draws)
        summary["dimension_preference"][dimension] = {
            "preferred_rate_mean": safe_mean(rate_values),
            "preferred_rate_ci95_low": dim_lo,
            "preferred_rate_ci95_high": dim_hi,
            "preferred_rank_mean": safe_mean(preferred_rank_means.get(dimension, [])),
            "baseline_rank_mean": safe_mean(baseline_rank_means.get(dimension, [])),
        }
    return summary


def csv_rows_for_cohort(
    cohort_summary: dict[str, Any],
    preferred_method: str,
    baseline_method: str,
) -> list[dict[str, Any]]:
    cohort = cohort_summary["cohort"]
    rows = [
        {
            "cohort": cohort,
            "metric_group": "count",
            "metric": "participants",
            "method": "",
            "value": cohort_summary["n_participants"],
            "ci95_low": "",
            "ci95_high": "",
            "n_participants": cohort_summary["n_participants"],
            "n_responses": cohort_summary["n_responses"],
        },
        {
            "cohort": cohort,
            "metric_group": "overall_preference",
            "metric": "response_share",
            "method": preferred_method,
            "value": cohort_summary["overall_preference"][preferred_method],
            "ci95_low": cohort_summary["overall_preference"]["preferred_participant_rate_ci95_low"],
            "ci95_high": cohort_summary["overall_preference"]["preferred_participant_rate_ci95_high"],
            "n_participants": cohort_summary["n_participants"],
            "n_responses": cohort_summary["n_responses"],
        },
        {
            "cohort": cohort,
            "metric_group": "overall_preference",
            "metric": "response_share",
            "method": baseline_method,
            "value": cohort_summary["overall_preference"][baseline_method],
            "ci95_low": "",
            "ci95_high": "",
            "n_participants": cohort_summary["n_participants"],
            "n_responses": cohort_summary["n_responses"],
        },
        {
            "cohort": cohort,
            "metric_group": "overall_preference",
            "metric": "response_share",
            "method": "other",
            "value": cohort_summary["overall_preference"]["other"],
            "ci95_low": "",
            "ci95_high": "",
            "n_participants": cohort_summary["n_participants"],
            "n_responses": cohort_summary["n_responses"],
        },
    ]
    for dimension, payload in cohort_summary["dimension_preference"].items():
        rows.append(
            {
                "cohort": cohort,
                "metric_group": "dimension_preference",
                "metric": dimension,
                "method": preferred_method,
                "value": payload["preferred_rate_mean"],
                "ci95_low": payload["preferred_rate_ci95_low"],
                "ci95_high": payload["preferred_rate_ci95_high"],
                "n_participants": cohort_summary["n_participants"],
                "n_responses": cohort_summary["n_responses"],
            }
        )
    return rows


def markdown_for_result(result: dict[str, Any]) -> str:
    preferred_method = result["preferred_method"]
    baseline_method = result["baseline_method"]
    lines = [
        "# Subjective Study Summary",
        "",
        "This artifact summarizes the live WaveletMask human-evaluation export after excluding known simulated or dry-run records.",
        "",
        f"- Chapter: `{result['chapter']}`",
        f"- Preferred method under review: `{preferred_method}`",
        f"- Baseline method: `{baseline_method}`",
        f"- Raw response rows: `{result['counts']['raw_response_rows']}`",
        f"- Raw participants: `{result['counts']['raw_participants']}`",
        f"- Eligible ordinary participants: `{result['counts']['eligible_ordinary_participants']}`",
        f"- Eligible expert participants: `{result['counts']['eligible_expert_participants']}`",
        "",
        "## Gate Status",
        "",
        f"- Ordinary-participant gate (`>= {result['ordinary_min_required']}`): `{result['gates']['ordinary_count_met']}`",
        f"- Expert-participant gate (`>= {result['expert_min_required']}`): `{result['gates']['expert_count_met']}`",
        f"- Submission-promotion ready: `{result['gates']['ready_for_submission_promotion']}`",
        "",
    ]
    if result["excluded_participants"]:
        lines.extend(
            [
                "## Excluded Participants",
                "",
                f"- Excluded participant count: `{len(result['excluded_participants'])}`",
                f"- Exclusion reasons: `{json.dumps(result['excluded_reason_counts'], ensure_ascii=False)}`",
                "",
            ]
        )

    for key in ("ordinary", "expert"):
        cohort = result[key]
        lines.extend(
            [
                f"## {cohort['cohort'].title()} Cohort",
                "",
                f"- Participants: `{cohort['n_participants']}`",
                f"- Responses: `{cohort['n_responses']}`",
                f"- Mean sample count: `{cohort['mean_sample_count']}`",
                f"- Mean time per response (s): `{cohort['mean_time_seconds_per_response']}`",
                f"- Preferred response share (`{preferred_method}`): `{cohort['overall_preference'][preferred_method]}`",
                "",
                "| Dimension | Preferred-rate mean | 95% CI | Preferred mean rank | Baseline mean rank |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for dimension, payload in cohort["dimension_preference"].items():
            ci = (
                f"[{payload['preferred_rate_ci95_low']:.4f}, {payload['preferred_rate_ci95_high']:.4f}]"
                if payload["preferred_rate_ci95_low"] is not None and payload["preferred_rate_ci95_high"] is not None
                else "n/a"
            )
            pref_rank = f"{payload['preferred_rank_mean']:.4f}" if payload["preferred_rank_mean"] is not None else "n/a"
            base_rank = f"{payload['baseline_rank_mean']:.4f}" if payload["baseline_rank_mean"] is not None else "n/a"
            rate = f"{payload['preferred_rate_mean']:.4f}" if payload["preferred_rate_mean"] is not None else "n/a"
            lines.append(f"| {dimension} | {rate} | {ci} | {pref_rank} | {base_rank} |")
        lines.append("")

    if result["notes"]:
        lines.extend(["## Notes", ""])
        for note in result["notes"]:
            lines.append(f"- {note}")
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    name_patterns = compile_patterns(args.exclude_name_regex, DEFAULT_NAME_PATTERNS)
    email_patterns = compile_patterns(args.exclude_email_regex, DEFAULT_EMAIL_PATTERNS)
    expert_ids = load_lines(args.expert_ids_file)
    expert_emails = {value.lower() for value in load_lines(args.expert_emails_file)}

    with args.stats_json.open() as f:
        stats = json.load(f)
    participants_meta = {
        str(row["id"]): row for row in stats.get("participants", [])
    }

    rows_by_participant: dict[str, list[dict[str, str]]] = defaultdict(list)
    raw_row_count = 0
    with args.responses_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if args.chapter and row.get("chapter") != args.chapter:
                continue
            raw_row_count += 1
            rows_by_participant[str(row["participant_id"])].append(row)

    eligible_ordinary: list[dict[str, Any]] = []
    eligible_expert: list[dict[str, Any]] = []
    excluded_participants: list[dict[str, Any]] = []
    excluded_reason_counts = Counter()
    incomplete_participants: list[str] = []

    for participant_id, meta in participants_meta.items():
        participant_rows = rows_by_participant.get(participant_id, [])
        if not participant_rows:
            continue
        if not meta.get("completed_at"):
            incomplete_participants.append(participant_id)
            continue
        exclusion_reason = pick_exclusion_reason(meta, name_patterns, email_patterns)
        if exclusion_reason is not None:
            excluded_reason_counts[exclusion_reason] += 1
            excluded_participants.append(
                {
                    "participant_id": participant_id,
                    "name": meta.get("name"),
                    "email": meta.get("email"),
                    "reason": exclusion_reason,
                }
            )
            continue

        summary = participant_summary(
            participant_id=participant_id,
            participant_meta=meta,
            rows=participant_rows,
            preferred_method=args.preferred_method,
            baseline_method=args.baseline_method,
        )
        email = str(meta.get("email") or "").lower()
        if participant_id in expert_ids or email in expert_emails:
            eligible_expert.append(summary)
        else:
            eligible_ordinary.append(summary)

    rng = random.Random(args.seed)
    ordinary_summary = summarize_cohort(
        cohort_name="ordinary",
        participants=eligible_ordinary,
        preferred_method=args.preferred_method,
        baseline_method=args.baseline_method,
        rng=rng,
        draws=args.bootstrap_draws,
    )
    expert_summary = summarize_cohort(
        cohort_name="expert",
        participants=eligible_expert,
        preferred_method=args.preferred_method,
        baseline_method=args.baseline_method,
        rng=rng,
        draws=args.bootstrap_draws,
    )

    gates = {
        "ordinary_count_met": len(eligible_ordinary) >= args.ordinary_min,
        "expert_count_met": len(eligible_expert) >= args.expert_min,
    }
    gates["ready_for_submission_promotion"] = gates["ordinary_count_met"] and gates["expert_count_met"]

    notes = []
    if raw_row_count == 0:
        notes.append("No chapter rows were found in the provided responses CSV.")
    if not eligible_ordinary and not eligible_expert:
        notes.append("No eligible real participants remained after applying the default simulated/dry-run filters.")
    if not gates["ordinary_count_met"]:
        notes.append("Ordinary-participant gate is still open.")
    if not gates["expert_count_met"]:
        notes.append("Expert-participant gate is still open. Provide expert participant IDs or emails when available.")

    result = {
        "chapter": args.chapter,
        "preferred_method": args.preferred_method,
        "baseline_method": args.baseline_method,
        "ordinary_min_required": args.ordinary_min,
        "expert_min_required": args.expert_min,
        "input_files": {
            "responses_csv": str(args.responses_csv),
            "stats_json": str(args.stats_json),
        },
        "filters": {
            "exclude_name_regex": [pattern.pattern for pattern in name_patterns],
            "exclude_email_regex": [pattern.pattern for pattern in email_patterns],
            "expert_ids_file": str(args.expert_ids_file) if args.expert_ids_file else None,
            "expert_emails_file": str(args.expert_emails_file) if args.expert_emails_file else None,
        },
        "counts": {
            "raw_response_rows": raw_row_count,
            "raw_participants": len(rows_by_participant),
            "eligible_ordinary_participants": len(eligible_ordinary),
            "eligible_expert_participants": len(eligible_expert),
            "excluded_participants": len(excluded_participants),
            "incomplete_participants": len(incomplete_participants),
        },
        "ordinary": ordinary_summary,
        "expert": expert_summary,
        "excluded_reason_counts": dict(excluded_reason_counts),
        "excluded_participants": excluded_participants,
        "incomplete_participant_ids": incomplete_participants,
        "gates": gates,
        "status": (
            "ready_for_submission_promotion"
            if gates["ready_for_submission_promotion"]
            else "blocked_pending_real_exports"
        ),
        "notes": notes,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "subjective_study_summary.json"
    csv_path = args.output_dir / "subjective_study_summary.csv"
    md_path = args.output_dir / "subjective_study_summary.md"

    json_path.write_text(json.dumps(result, indent=2) + "\n")

    csv_rows = csv_rows_for_cohort(ordinary_summary, args.preferred_method, args.baseline_method)
    csv_rows.extend(csv_rows_for_cohort(expert_summary, args.preferred_method, args.baseline_method))
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "cohort",
                "metric_group",
                "metric",
                "method",
                "value",
                "ci95_low",
                "ci95_high",
                "n_participants",
                "n_responses",
            ],
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    md_path.write_text(markdown_for_result(result) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
