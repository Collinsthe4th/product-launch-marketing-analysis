"""Group comparisons and the pre-launch prediction model."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Only information that is known before a pop-up opens.
PRE_LAUNCH_CATEGORICAL = ["brand", "region", "location_type", "event_type", "category"]
PRE_LAUNCH_NUMERIC = ["lease_length_days", "price_usd", "avg_daily_footfall", "start_month"]


def holm(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (controls false positives across many pairs)."""
    order = np.argsort(pvalues)
    m = len(pvalues)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvalues[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()


def compare_groups(df: pd.DataFrame, group: str, value: str = "sell_through_pct") -> dict:
    """Kruskal-Wallis across all groups, epsilon-squared effect size, Holm-adjusted pairs.

    Kruskal-Wallis compares groups without assuming normal data. Epsilon-squared
    is the share of the variation in ``value`` that the grouping explains.
    """
    groups = {k: g[value].to_numpy() for k, g in df.groupby(group, observed=True)}
    h, p = stats.kruskal(*groups.values())
    n, k = len(df), len(groups)
    epsilon_sq = (h - k + 1) / (n - k)
    pairs = list(combinations(sorted(groups), 2))
    raw = [stats.mannwhitneyu(groups[a], groups[b]).pvalue for a, b in pairs]
    adj = holm(raw)
    summary = (
        df.groupby(group, observed=True)[value]
        .agg(events="size", mean="mean", median="median")
        .round(2)
        .sort_values("mean", ascending=False)
    )
    return {
        "group": group,
        "kruskal_h": float(h),
        "p_value": float(p),
        "epsilon_squared": float(max(epsilon_sq, 0.0)),
        "summary": summary.reset_index().to_dict(orient="records"),
        "significant_pairs": [
            {"a": a, "b": b, "p_holm": round(q, 4)} for (a, b), q in zip(pairs, adj) if q < 0.05
        ],
    }


def prelaunch_models(random_state: int = 42) -> dict[str, Pipeline]:
    prep = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), PRE_LAUNCH_CATEGORICAL),
            ("num", StandardScaler(), PRE_LAUNCH_NUMERIC),
        ]
    )
    return {
        "Mean (baseline)": Pipeline([("prep", prep), ("m", DummyRegressor())]),
        "Ridge regression": Pipeline([("prep", prep), ("m", RidgeCV(alphas=np.logspace(-2, 3, 12)))]),
        "Gradient boosting": Pipeline(
            [("prep", prep), ("m", HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=300, random_state=random_state))]
        ),
    }


def score_prelaunch(df: pd.DataFrame, target: str = "sell_through_pct", folds: int = 5) -> dict:
    """Cross-validated R² and MAE of each model when predicting ``target`` before launch."""
    X = df[PRE_LAUNCH_CATEGORICAL + PRE_LAUNCH_NUMERIC]
    y = df[target]
    cv = KFold(folds, shuffle=True, random_state=42)
    out = {}
    for name, model in prelaunch_models().items():
        r2 = cross_val_score(model, X, y, cv=cv, scoring="r2")
        mae = -cross_val_score(model, X, y, cv=cv, scoring="neg_mean_absolute_error")
        out[name] = {"r2_mean": round(float(r2.mean()), 4), "r2_std": round(float(r2.std()), 4), "mae": round(float(mae.mean()), 3)}
    return out
