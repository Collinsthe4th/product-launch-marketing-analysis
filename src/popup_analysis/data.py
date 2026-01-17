"""Loading, validating and enriching the pop-up events table."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "luxury_cosmetics_popups.csv"

REQUIRED = [
    "event_id", "brand", "region", "city", "location_type", "event_type", "start_date", "end_date",
    "lease_length_days", "sku", "product_name", "price_usd", "avg_daily_footfall", "units_sold", "sell_through_pct",
]


def split_product(product_name: str, brand: str) -> tuple[str, str]:
    """Split "Dior Velvet Lipstick" into the line ("Velvet") and the category ("Lipstick")."""
    rest = product_name[len(brand):].strip() if product_name.startswith(brand) else product_name
    words = rest.split()
    if len(words) < 2:
        return "", rest
    return words[0], " ".join(words[1:])


def load(path: Path = RAW_PATH) -> tuple[pd.DataFrame, dict]:
    """Return the cleaned table and a data-quality report.

    Fixes:
        * ``end_date`` is missing for some events. It is rebuilt from
          ``start_date + lease_length_days``, which matches the given end date
          for every one of the 2,097 rows that have one.
        * ``city`` is missing for some events. It is labelled "Unknown" and kept,
          because the city is not used in the models.
    """
    df = pd.read_csv(path)
    missing_cols = set(REQUIRED) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns: {sorted(missing_cols)}")

    df["start_date"] = pd.to_datetime(df["start_date"])
    df["end_date"] = pd.to_datetime(df["end_date"])
    rebuilt = df["start_date"] + pd.to_timedelta(df["lease_length_days"], unit="D")
    has_end = df["end_date"].notna()
    report = {
        "rows": int(len(df)),
        "duplicate_event_ids": int(df["event_id"].duplicated().sum()),
        "missing_city": int(df["city"].isna().sum()),
        "missing_end_date": int((~has_end).sum()),
        "end_date_matches_lease_pct": round(100 * float((df.loc[has_end, "end_date"] == rebuilt[has_end]).mean()), 1),
        "sell_through_out_of_range": int((~df["sell_through_pct"].between(0, 100)).sum()),
    }
    df["end_date"] = df["end_date"].fillna(rebuilt)
    df["city"] = df["city"].fillna("Unknown")

    parts = [split_product(p, b) for p, b in zip(df["product_name"], df["brand"])]
    df["product_line"] = [p[0] for p in parts]
    df["category"] = [p[1] for p in parts]
    df["revenue_usd"] = df["price_usd"] * df["units_sold"]
    df["revenue_per_day"] = df["revenue_usd"] / df["lease_length_days"].clip(lower=1)
    df["start_month"] = df["start_date"].dt.month
    df["price_band"] = pd.qcut(df["price_usd"], 4, labels=["Entry", "Mid", "Upper", "Prestige"])
    return df, report
