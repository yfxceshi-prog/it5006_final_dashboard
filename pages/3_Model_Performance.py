import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.chart_utils import model_comparison_bar

st.set_page_config(page_title="Model Performance", layout="wide")

st.title("Model Performance — Chicago 2025 Test Set")
st.caption("34,424 grid-week rows (Jan–Dec 2025). Models trained on Chicago 2015–2024 only.")

FULL_METRICS = [
    {"Model": "LR (Baseline)", "Crime": "THEFT",           "Precision": 0.858, "Recall": 0.631, "F1": 0.727, "AUC": 0.790},
    {"Model": "LR (Baseline)", "Crime": "BATTERY",         "Precision": 0.847, "Recall": 0.611, "F1": 0.710, "AUC": 0.793},
    {"Model": "LR (Baseline)", "Crime": "CRIMINAL DAMAGE", "Precision": 0.699, "Recall": 0.561, "F1": 0.622, "AUC": 0.733},
    {"Model": "MLP",           "Crime": "THEFT",           "Precision": 0.844, "Recall": 0.704, "F1": 0.767, "AUC": 0.810},
    {"Model": "MLP",           "Crime": "BATTERY",         "Precision": 0.845, "Recall": 0.632, "F1": 0.723, "AUC": 0.808},
    {"Model": "MLP",           "Crime": "CRIMINAL DAMAGE", "Precision": 0.679, "Recall": 0.654, "F1": 0.667, "AUC": 0.752},
    {"Model": "Random Forest", "Crime": "THEFT",           "Precision": 0.803, "Recall": 0.778, "F1": 0.790, "AUC": 0.851},
    {"Model": "Random Forest", "Crime": "BATTERY",         "Precision": 0.776, "Recall": 0.782, "F1": 0.779, "AUC": 0.851},
    {"Model": "Random Forest", "Crime": "CRIMINAL DAMAGE", "Precision": 0.629, "Recall": 0.746, "F1": 0.683, "AUC": 0.793},
    {"Model": "XGBoost",       "Crime": "THEFT",           "Precision": 0.775, "Recall": 0.829, "F1": 0.801, "AUC": 0.852},
    {"Model": "XGBoost",       "Crime": "BATTERY",         "Precision": 0.765, "Recall": 0.811, "F1": 0.787, "AUC": 0.852},
    {"Model": "XGBoost",       "Crime": "CRIMINAL DAMAGE", "Precision": 0.667, "Recall": 0.667, "F1": 0.667, "AUC": 0.795},
]

df_metrics = pd.DataFrame(FULL_METRICS)

fig = model_comparison_bar()
st.plotly_chart(fig, use_container_width=True)

st.subheader("Radar: Precision / Recall / F1 / AUC — THEFT")
MODEL_COLORS_LIST = ["#FFD93D", "#4D96FF", "#6BCB77", "#FF6B6B"]
categories = ["Precision", "Recall", "F1", "AUC"]

fig_radar = go.Figure()
for model, color in zip(["LR (Baseline)", "MLP", "Random Forest", "XGBoost"], MODEL_COLORS_LIST):
    row = df_metrics[(df_metrics["Model"] == model) & (df_metrics["Crime"] == "THEFT")].iloc[0]
    vals = [row["Precision"], row["Recall"], row["F1"], row["AUC"]]
    vals_closed = vals + [vals[0]]
    fig_radar.add_trace(go.Scatterpolar(
        r=vals_closed,
        theta=categories + [categories[0]],
        fill="toself",
        name=model,
        line_color=color,
        fillcolor=color,
        opacity=0.25,
    ))
fig_radar.update_layout(
    polar=dict(
        radialaxis=dict(range=[0.55, 0.92], showticklabels=True, gridcolor="#2A2D3E"),
        bgcolor="#0E1117",
    ),
    paper_bgcolor="#0E1117",
    font_color="#FAFAFA",
    legend=dict(orientation="h", y=-0.1),
    height=380,
    margin=dict(l=40, r=40, t=30, b=60),
)
st.plotly_chart(fig_radar, use_container_width=True)

st.subheader("Full Metrics Table")
crime_filter = st.selectbox("Filter by Crime", ["All", "THEFT", "BATTERY", "CRIMINAL DAMAGE"])
if crime_filter != "All":
    show_df = df_metrics[df_metrics["Crime"] == crime_filter].copy()
else:
    show_df = df_metrics.copy()

show_df = show_df.sort_values(["Crime", "F1"], ascending=[True, False]).reset_index(drop=True)
st.dataframe(
    show_df.style
    .highlight_max(subset=["F1", "AUC"], color="#1e3a1e")
    .highlight_max(subset=["Precision"], color="#1a2a3a")
    .format({"Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}", "AUC": "{:.3f}"}),
    use_container_width=True,
    height=420,
)

st.subheader("Model Selection Guide")
guide = pd.DataFrame([
    {"Use Case": "Known city, long-term deployment",          "Recommended": "XGBoost",            "Reason": "Best in-domain F1 (0.801/0.787)"},
    {"Use Case": "New city, no local history",                "Recommended": "Random Forest",       "Reason": "Best zero-shot AUC (0.877/0.851/0.837)"},
    {"Use Case": "High precision required (minimize alerts)", "Recommended": "MLP",                 "Reason": "Highest precision (0.844/0.845)"},
    {"Use Case": "Audit / explainability required",           "Recommended": "Logistic Regression", "Reason": "Fully interpretable coefficients"},
])
st.dataframe(guide, use_container_width=True, hide_index=True)
