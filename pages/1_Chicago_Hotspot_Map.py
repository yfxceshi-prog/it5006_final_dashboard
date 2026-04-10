import streamlit as st
import pydeck as pdk
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.data_loader import load_chicago, iso_week_to_label
from utils.map_utils import (
    build_chicago_layer, chicago_view_state, chicago_tooltip, MAP_STYLES
)
from utils.chart_utils import hotspot_trend_chart

st.set_page_config(page_title="Chicago Hotspot Map", layout="wide")

# ── Sidebar controls ──────────────────────────────────────────────────────────
st.sidebar.header("Controls")

crime = st.sidebar.selectbox(
    "Crime Type",
    ["THEFT", "BATTERY", "CRIMINAL_DAMAGE"],
    format_func=lambda x: x.replace("_", " "),
)

week = st.sidebar.slider(
    "ISO Week (2025)", min_value=1, max_value=52, value=15,
    help="Select the target prediction week",
)
st.sidebar.caption(iso_week_to_label(week))

model = st.sidebar.radio(
    "Model",
    ["MLP", "XGBoost", "True Labels"],
    index=0,
)

map_style_name = st.sidebar.selectbox("Map Style", list(MAP_STYLES.keys()), index=0)
pitch = st.sidebar.slider("3-D Pitch", 0, 60, 30, 5)

# ── Load data ─────────────────────────────────────────────────────────────────
df = load_chicago()
df_week = df[df["iso_week"] == week].copy()

# ── KPI row ───────────────────────────────────────────────────────────────────
st.title(f"Chicago Hotspot Map — {crime.replace('_', ' ')}  |  {iso_week_to_label(week)}")

total_grids = len(df_week)
if model == "MLP":
    pred_col = f"mlp_pred_{crime}"
elif model == "XGBoost":
    pred_col = f"xgb_pred_{crime}"
else:
    pred_col = f"true_{crime}"

hot_count = int(df_week[pred_col].sum()) if pred_col in df_week.columns else 0
true_count = int(df_week[f"true_{crime}"].sum())

if model != "True Labels" and pred_col in df_week.columns:
    correct = int((df_week[f"true_{crime}"] == df_week[pred_col]).sum())
    accuracy = correct / len(df_week) * 100 if len(df_week) > 0 else 0
    acc_str = f"{accuracy:.1f}%"
else:
    acc_str = "N/A"

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Grids (this week)", total_grids)
col2.metric(f"Predicted Hotspots ({model})", hot_count,
            f"{hot_count/total_grids*100:.1f}% of grids" if total_grids else "")
col3.metric("Actual Hotspots", true_count,
            f"{true_count/total_grids*100:.1f}% of grids" if total_grids else "")
col4.metric("Grid-level Accuracy", acc_str)

# ── Map ───────────────────────────────────────────────────────────────────────
layer = build_chicago_layer(df_week, crime, model)
view = chicago_view_state(pitch=pitch)
tooltip = chicago_tooltip(crime, model)

deck = pdk.Deck(
    layers=[layer],
    initial_view_state=view,
    map_style=MAP_STYLES[map_style_name],
    tooltip=tooltip,
)
st.pydeck_chart(deck, use_container_width=True, height=550)

# ── Trend chart + Top-10 table ────────────────────────────────────────────────
col_left, col_right = st.columns([6, 4])

with col_left:
    if model in ("MLP", "XGBoost"):
        fig = hotspot_trend_chart(df, crime, model, week)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Switch to MLP or XGBoost to see the weekly prediction trend.")

with col_right:
    st.subheader("Top-10 Highest-Risk Grids")
    if model == "MLP":
        sort_col = f"mlp_prob_{crime}"
        display_cols = ["grid_id", sort_col, f"true_{crime}", f"count_{crime}"]
        display_names = ["Grid ID", "MLP Prob", "Actual", "Count"]
    elif model == "XGBoost":
        sort_col = f"xgb_pred_{crime}"
        display_cols = ["grid_id", sort_col, f"true_{crime}", f"count_{crime}"]
        display_names = ["Grid ID", "XGB Pred", "Actual", "Count"]
    else:
        sort_col = f"true_{crime}"
        display_cols = ["grid_id", sort_col, f"count_{crime}"]
        display_names = ["Grid ID", "Actual", "Count"]

    top10 = df_week.nlargest(10, sort_col)[display_cols].copy()
    top10.columns = display_names
    top10 = top10.reset_index(drop=True)
    top10.index += 1

    st.dataframe(
        top10,
        use_container_width=True,
        column_config={
            "MLP Prob": st.column_config.ProgressColumn(
                "MLP Prob", min_value=0, max_value=1, format="%.3f"
            ) if model == "MLP" else None,
        },
    )
