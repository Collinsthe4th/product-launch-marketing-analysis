"""Full launch analysis: data quality, group tests, pre-launch models and figures.

Usage:
    python scripts/run_analysis.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from popup_analysis.analysis import compare_groups, score_prelaunch  # noqa: E402
from popup_analysis.data import load  # noqa: E402

RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"


def main() -> None:
    df, quality = load()

    groups = {g: compare_groups(df, g) for g in ["event_type", "location_type", "region", "category", "price_band"]}
    revenue_by_event = compare_groups(df, "event_type", "revenue_per_day")

    num = df[["lease_length_days", "price_usd", "avg_daily_footfall", "units_sold", "sell_through_pct"]]
    corr = num.corr(method="spearman").round(3)

    models = {t: score_prelaunch(df, t) for t in ["sell_through_pct", "units_sold", "revenue_usd"]}

    top = df.groupby("event_type")["sell_through_pct"].mean().sort_values(ascending=False)
    results = {
        "data_quality": quality,
        "date_range": [str(df["start_date"].min().date()), str(df["start_date"].max().date())],
        "brands": int(df["brand"].nunique()),
        "totals": {
            "units_sold": int(df["units_sold"].sum()),
            "revenue_usd": round(float(df["revenue_usd"].sum()), 2),
            "mean_sell_through_pct": round(float(df["sell_through_pct"].mean()), 2),
        },
        "sell_through_tests": {g: {k: v for k, v in r.items()} for g, r in groups.items()},
        "revenue_per_day_by_event_type": revenue_by_event,
        "spearman_correlations": corr.to_dict(),
        "prelaunch_models_cv": models,
        "best_vs_worst_event_type_gap_points": round(float(top.iloc[0] - top.iloc[-1]), 2),
    }
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    (RESULTS / "metrics.json").write_text(json.dumps(results, indent=2, default=str))

    # Figure 1: sell-through by event type with 95% CIs.
    g = df.groupby("event_type")["sell_through_pct"]
    stats_ = g.agg(["mean", "std", "size"]).sort_values("mean")
    ci = 1.96 * stats_["std"] / np.sqrt(stats_["size"])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(stats_.index, stats_["mean"], xerr=ci, color="#b7791f", capsize=4)
    ax.axvline(df["sell_through_pct"].mean(), color="black", ls="--", lw=1, label="All events")
    ax.set_xlim(65, 80)
    ax.set_xlabel("Mean sell-through (%) with 95% confidence interval")
    pv = groups["event_type"]["p_value"]
    ax.set_title("Sell-through by event format (Kruskal-Wallis " + ("p < 0.0001" if pv < 1e-4 else f"p = {pv:.4f}") + ")")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "sell_through_by_event_type.png", dpi=150)
    plt.close(fig)

    # Figure 2: footfall against units sold.
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.scatter(df["avg_daily_footfall"], df["units_sold"], s=6, alpha=0.35, color="#2b6cb0")
    ax.set_xlabel("Average daily footfall")
    ax.set_ylabel("Units sold")
    rho = corr.loc["avg_daily_footfall", "units_sold"]
    ax.set_title(f"Footfall against units sold (Spearman rho = {rho:.3f})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "footfall_vs_units.png", dpi=150)
    plt.close(fig)

    print(json.dumps({"quality": quality, "models": models, "event_type_p": groups["event_type"]["p_value"],
                      "event_type_eps2": groups["event_type"]["epsilon_squared"]}, indent=2))


if __name__ == "__main__":
    main()
