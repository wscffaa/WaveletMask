#!/usr/bin/env python3
"""Statistical tests for the 34-participant subjective study (T14).

Adds the significance analysis requested in the round-1 revision plan:
- chi-square goodness-of-fit of the overall preference counts against the
  uniform (chance) distribution over the 4 methods;
- one-sided binomial test of the WaveletMask preference share against the
  25% chance level, with a Wilson 95% confidence interval;
- the same pair of tests for the rank-1 counts of each of the three
  perceptual dimensions.

Caveat (also stated in the manuscript): tests treat the 476 votes as
independent; participant-level clustering is not modelled because the
released export only contains aggregated counts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[7]
T14_DIR = (
    REPO_ROOT
    / "projects/007-waveletmask/Paper/applsci/review/round1/artifacts/t14_realworld_human_eval"
)


def chi2_sf(x: float, df: int) -> float:
    """Survival function of the chi-square distribution (df even or odd)."""
    if x <= 0:
        return 1.0
    if df % 2 == 0:
        # closed form: exp(-x/2) * sum_{k=0}^{df/2-1} (x/2)^k / k!
        term = 1.0
        total = 1.0
        for k in range(1, df // 2):
            term *= (x / 2) / k
            total += term
        return math.exp(-x / 2) * total
    # odd df: use regularized upper incomplete gamma via series/continued fraction
    return float(_gammaincc(df / 2.0, x / 2.0))


def _gammaincc(a: float, x: float) -> float:
    """Regularized upper incomplete gamma Q(a, x) (Numerical Recipes style)."""
    if x < a + 1.0:
        # series for P(a,x), then Q = 1 - P
        ap = a
        summ = 1.0 / a
        delta = summ
        for _ in range(200):
            ap += 1.0
            delta *= x / ap
            summ += delta
            if abs(delta) < abs(summ) * 1e-15:
                break
        p = summ * math.exp(-x + a * math.log(x) - math.lgamma(a))
        return 1.0 - p
    # continued fraction for Q(a,x)
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 200):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return h * math.exp(-x + a * math.log(x) - math.lgamma(a))


def binom_sf_geq(k: int, n: int, p: float) -> float:
    """P[X >= k] for X ~ Binomial(n, p), exact summation in log space."""
    log_p = math.log(p)
    log_q = math.log1p(-p)
    total = 0.0
    for i in range(k, n + 1):
        log_pmf = (
            math.lgamma(n + 1)
            - math.lgamma(i + 1)
            - math.lgamma(n - i + 1)
            + i * log_p
            + (n - i) * log_q
        )
        total += math.exp(log_pmf)
    return min(1.0, total)


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    phat = k / n
    denom = 1.0 + z * z / n
    centre = phat + z * z / (2 * n)
    margin = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    return ((centre - margin) / denom, (centre + margin) / denom)


def chi2_gof(counts: list[int]) -> tuple[float, int, float]:
    n = sum(counts)
    k = len(counts)
    expected = n / k
    stat = sum((c - expected) ** 2 / expected for c in counts)
    df = k - 1
    return stat, df, chi2_sf(stat, df)


def fmt_p(p: float) -> str:
    if p <= 0 or p < 1e-300:
        return "<1e-300"
    return f"{p:.3g}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=T14_DIR / "formal/user_study_data.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=T14_DIR / "formal",
    )
    args = parser.parse_args()

    data = json.loads(args.data.read_text())
    tables = data["dataTables"]

    rows: list[dict] = []

    # Overall preference
    overall = tables["overallPreference"]
    counts = [row["count"] for row in overall]
    labels = [row["methodLabel"] for row in overall]
    total = overall[0]["totalVotes"]
    assert sum(counts) == total, "overall counts must sum to total votes"
    wm_idx = labels.index("WaveletMask")
    wm_count = counts[wm_idx]

    chi2, df, chi2_p = chi2_gof(counts)
    binom_p = binom_sf_geq(wm_count, total, 0.25)
    lo, hi = wilson_ci(wm_count, total)
    rows.append(
        {
            "scope": "overall_preference",
            "labels": "/".join(labels),
            "counts": "/".join(str(c) for c in counts),
            "total": total,
            "wm_count": wm_count,
            "wm_share": wm_count / total,
            "wilson95_lo": lo,
            "wilson95_hi": hi,
            "chi2_stat": chi2,
            "chi2_df": df,
            "chi2_p": chi2_p,
            "binom_p_greater_chance": binom_p,
        }
    )

    # Per-dimension rank-1 counts
    ranking = tables["rankingStatistics"]
    dims: dict[str, dict[str, int]] = {}
    dim_titles: dict[str, str] = {}
    for row in ranking:
        dims.setdefault(row["dimensionKey"], {})[row["methodLabel"]] = row["rank1Count"]
        dim_titles[row["dimensionKey"]] = row["dimensionTitleEn"]
    for key, method_counts in dims.items():
        labels_d = list(method_counts.keys())
        counts_d = [method_counts[m] for m in labels_d]
        total_d = sum(counts_d)
        wm_d = method_counts["WaveletMask"]
        chi2_d, df_d, chi2_p_d = chi2_gof(counts_d)
        binom_p_d = binom_sf_geq(wm_d, total_d, 0.25)
        lo_d, hi_d = wilson_ci(wm_d, total_d)
        rows.append(
            {
                "scope": f"rank1_{key}",
                "labels": "/".join(labels_d),
                "counts": "/".join(str(c) for c in counts_d),
                "total": total_d,
                "wm_count": wm_d,
                "wm_share": wm_d / total_d,
                "wilson95_lo": lo_d,
                "wilson95_hi": hi_d,
                "chi2_stat": chi2_d,
                "chi2_df": df_d,
                "chi2_p": chi2_p_d,
                "binom_p_greater_chance": binom_p_d,
            }
        )

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "meta": {
            "participants": data["meta"]["totalParticipants"],
            "total_responses": data["meta"]["totalResponses"],
            "chance_level": 0.25,
            "tests": [
                "chi-square goodness-of-fit vs uniform distribution over 4 methods",
                "one-sided exact binomial test of WaveletMask share vs 0.25",
                "Wilson 95% CI for WaveletMask share",
            ],
            "caveat": (
                "Votes are treated as independent; participant-level clustering "
                "(34 participants x 14 clips) is not modelled because only "
                "aggregated counts are released."
            ),
        },
        "results": rows,
    }
    (out_dir / "subjective_study_stats.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )

    with (out_dir / "subjective_study_stats.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md_lines = [
        "# T14 Subjective Study Significance Tests",
        "",
        f"Participants: {data['meta']['totalParticipants']}; total votes per scope: 476.",
        "Chance level for a 4-method forced ranking: 25%.",
        "",
        "| Scope | Counts (WM/Mamba/RRTN/RTN) | WM share | Wilson 95% CI | chi2 (df=3) | chi2 p | Binomial p (WM > chance) |",
        "|---|---|---:|---|---:|---|---|",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['scope']} | {r['counts']} | {r['wm_share']*100:.2f}% | "
            f"[{r['wilson95_lo']*100:.2f}%, {r['wilson95_hi']*100:.2f}%] | "
            f"{r['chi2_stat']:.2f} | {fmt_p(r['chi2_p'])} | "
            f"{fmt_p(r['binom_p_greater_chance'])} |"
        )
    md_lines += [
        "",
        "Caveat: tests treat the 476 votes as independent draws; "
        "participant-level clustering is not modelled (aggregated export only).",
        "",
    ]
    (out_dir / "subjective_study_stats.md").write_text("\n".join(md_lines))

    for r in rows:
        print(
            f"{r['scope']}: counts={r['counts']} share={r['wm_share']*100:.2f}% "
            f"CI=[{r['wilson95_lo']*100:.2f},{r['wilson95_hi']*100:.2f}]% "
            f"chi2={r['chi2_stat']:.2f} p={fmt_p(r['chi2_p'])} "
            f"binom_p={fmt_p(r['binom_p_greater_chance'])}"
        )


if __name__ == "__main__":
    main()
