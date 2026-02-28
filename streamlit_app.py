"""Streamlit dashboard for the luxury cosmetics pop-up launch analysis.

Run locally:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from popup_analysis.analysis import compare_groups  # noqa: E402
from popup_analysis.data import load  # noqa: E402

st.set_page_config(page_title="Luxury Cosmetics Pop-Up Launch Analysis", layout="wide")

HEADER = "https://images.unsplash.com/photo-1560472354-b33ff0c44a43?auto=format&fit=crop&w=1600&h=340&q=80"
st.image(HEADER, use_container_width=True)


@st.cache_data
def get_data() -> tuple[pd.DataFrame, dict, dict]:
    df, quality = load()
    metrics = json.loads((ROOT / "results" / "metrics.json").read_text())
    return df, quality, metrics


df, quality, metrics = get_data()

st.title("Luxury Cosmetics Pop-Up Launch Analysis")
st.caption("I analysed 2,133 pop-up launch events for 24 beauty brands between February 2024 and August 2025. Use the filters in the sidebar, and the tests below rerun on the events you select.")

st.sidebar.header("Filter events")
regions = st.sidebar.multiselect("Region", sorted(df["region"].unique()), default=sorted(df["region"].unique()))
brands = st.sidebar.multiselect("Brand", sorted(df["brand"].unique()), default=[])
f = df[df["region"].isin(regions)]
if brands:
    f = f[f["brand"].isin(brands)]

if f.empty:
    st.info("No events match these filters.")
    st.stop()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Events", f"{len(f):,}")
k2.metric("Units sold", f"{f['units_sold'].sum():,}")
k3.metric("Revenue", f"${f['revenue_usd'].sum() / 1e6:,.1f}M")
k4.metric("Mean sell-through", f"{f['sell_through_pct'].mean():.1f}%")

labels = {"event_type": "Event format", "location_type": "Location type", "region": "Region", "category": "Product category", "price_band": "Price band"}
group = st.selectbox("Compare sell-through by", list(labels), format_func=labels.get)

left, right = st.columns([1.6, 1])
with left:
    means = f.groupby(group, observed=True)["sell_through_pct"].mean().sort_values(ascending=False).rename("Mean sell-through (%)")
    st.bar_chart(means, horizontal=True)
with right:
    if f[group].nunique() >= 2 and len(f) >= 30:
        res = compare_groups(f, group)
        st.metric("Kruskal-Wallis p-value", f"{res['p_value']:.4f}")
        st.metric("Share of variation explained (epsilon²)", f"{res['epsilon_squared']:.4f}")
        if res["p_value"] < 0.05:
            st.success(f"{labels[group]} is linked to sell-through, but the effect is {'small' if res['epsilon_squared'] < 0.06 else 'meaningful'}.")
            if res["significant_pairs"]:
                st.write("Pairs that differ after the Holm correction:")
                st.dataframe(pd.DataFrame(res["significant_pairs"]), hide_index=True, use_container_width=True)
        else:
            st.info(f"No evidence that {labels[group].lower()} changes sell-through.")
    else:
        st.info("Not enough events or groups in the current filter to test.")

st.subheader("Footfall against units sold")
st.scatter_chart(f, x="avg_daily_footfall", y="units_sold", size=8)
rho = f[["avg_daily_footfall", "units_sold"]].corr(method="spearman").iloc[0, 1]
st.caption(f"Spearman correlation for the selected events: {rho:.3f}. Footfall and units sold are almost unrelated in this data.")

st.subheader("Predicting a launch before it opens")
rows = []
for target, models in metrics["prelaunch_models_cv"].items():
    for name, s in models.items():
        rows.append({"Target": target.replace("_", " "), "Model": name, "Cross-validated R²": s["r2_mean"]})
table = pd.DataFrame(rows).pivot(index="Target", columns="Model", values="Cross-validated R²")
st.dataframe(table, use_container_width=True)
st.caption("Sell-through and units cannot be predicted from pre-launch information (R² close to 0). Revenue only looks predictable because revenue = price × units and price is an input.")

st.markdown("[Source code and README](https://github.com/Collinsthe4th/product-launch-marketing-analysis) · Header photo from Unsplash.")
