import numpy as np
import pandas as pd
import pytest

from popup_analysis.analysis import compare_groups, holm, score_prelaunch
from popup_analysis.data import load, split_product


def make_csv(tmp_path, n=200, seed=0):
    rng = np.random.default_rng(seed)
    event = rng.choice(["Flash Event", "Mall Kiosk"], n)
    df = pd.DataFrame(
        {
            "event_id": [f"E{i}" for i in range(n)],
            "brand": rng.choice(["Dior", "Hermès Beauty"], n),
            "region": rng.choice(["Europe", "Asia-Pacific"], n),
            "city": rng.choice(["Paris", None], n, p=[0.9, 0.1]),
            "location_type": rng.choice(["Luxury Mall", "High-Street"], n),
            "event_type": event,
            "start_date": "2024-03-01",
            "end_date": "2024-03-11",
            "lease_length_days": 10,
            "sku": [f"S{i}" for i in range(n)],
            "price_usd": rng.uniform(30, 200, n).round(2),
            "avg_daily_footfall": rng.integers(500, 3000, n),
            "units_sold": rng.integers(100, 4000, n),
            # Flash events sell through 10 points better in this synthetic data.
            "sell_through_pct": np.clip(np.where(event == "Flash Event", 80, 70) + rng.normal(0, 5, n), 0, 100).round(2),
        }
    )
    df["product_name"] = df["brand"] + " Velvet Lipstick"
    df.loc[0, "end_date"] = None
    path = tmp_path / "popups.csv"
    df.to_csv(path, index=False)
    return path


def test_split_product():
    assert split_product("Dior Velvet Lipstick", "Dior") == ("Velvet", "Lipstick")
    assert split_product("Hermès Beauty Pearl Eyeshadow Palette", "Hermès Beauty") == ("Pearl", "Eyeshadow Palette")


def test_load_repairs_and_reports(tmp_path):
    df, report = load(make_csv(tmp_path))
    assert report["missing_end_date"] == 1
    assert df["end_date"].isna().sum() == 0
    assert df.loc[0, "end_date"] == pd.Timestamp("2024-03-11")
    assert (df["city"] == "Unknown").sum() == report["missing_city"]
    assert set(df["category"]) == {"Lipstick"}
    assert np.allclose(df["revenue_usd"], df["price_usd"] * df["units_sold"])


def test_holm_is_monotone_and_capped():
    adj = holm([0.01, 0.04, 0.03, 0.5])
    assert adj[0] == pytest.approx(0.04)
    assert max(adj) <= 1.0
    assert adj[1] >= adj[2] >= adj[0]


def test_compare_groups_detects_planted_effect(tmp_path):
    df, _ = load(make_csv(tmp_path))
    res = compare_groups(df, "event_type")
    assert res["p_value"] < 0.001
    assert res["epsilon_squared"] > 0.3
    assert res["summary"][0]["event_type"] == "Flash Event"
    assert res["significant_pairs"]


def test_prelaunch_models_find_planted_effect(tmp_path):
    df, _ = load(make_csv(tmp_path, n=400))
    scores = score_prelaunch(df)
    assert scores["Ridge regression"]["r2_mean"] > 0.3
    assert abs(scores["Mean (baseline)"]["r2_mean"]) < 0.05
