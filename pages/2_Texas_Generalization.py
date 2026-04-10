import streamlit as st
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.data_loader import load_texas
from utils.chart_utils import texas_auc_bar

st.set_page_config(page_title="Texas Generalization", layout="wide")

st.title("Texas Zero-Shot Generalization")
st.caption("Models trained exclusively on Chicago 2015–2024, evaluated on Texas NIBRS agencies with no re-training.")

METRICS = {
    "MLP": {
        "THEFT":           {"F1": 0.707, "AUC": 0.772, "AUC_drop": -4.7},
        "BATTERY":         {"F1": 0.591, "AUC": 0.783, "AUC_drop": -3.1},
        "CRIMINAL_DAMAGE": {"F1": 0.630, "AUC": 0.819, "AUC_drop": +8.9},
    },
    "Random Forest": {
        "THEFT":           {"F1": 0.732, "AUC": 0.877, "AUC_drop": -2.7},
        "BATTERY":         {"F1": 0.684, "AUC": 0.851, "AUC_drop": -0.02},
        "CRIMINAL_DAMAGE": {"F1": 0.563, "AUC": 0.837, "AUC_drop": +5.5},
    },
    "XGBoost": {
        "THEFT":           {"F1": 0.698, "AUC": 0.837, "AUC_drop": -1.7},
        "BATTERY":         {"F1": 0.592, "AUC": 0.805, "AUC_drop": -5.5},
        "CRIMINAL_DAMAGE": {"F1": 0.527, "AUC": 0.806, "AUC_drop": +1.4},
    },
}

st.sidebar.header("Filter")
crime = st.sidebar.selectbox(
    "Crime Type", ["THEFT", "BATTERY", "CRIMINAL_DAMAGE"],
    format_func=lambda x: x.replace("_", " "),
)

col1, col2, col3 = st.columns(3)
col1.metric("MLP AUC (Texas)",
            f"{METRICS['MLP'][crime]['AUC']:.3f}",
            f"{METRICS['MLP'][crime]['AUC_drop']:+.1f}% vs Chicago")
col2.metric("Random Forest AUC (Texas)",
            f"{METRICS['Random Forest'][crime]['AUC']:.3f}",
            f"{METRICS['Random Forest'][crime]['AUC_drop']:+.1f}% vs Chicago")
col3.metric("XGBoost AUC (Texas)",
            f"{METRICS['XGBoost'][crime]['AUC']:.3f}",
            f"{METRICS['XGBoost'][crime]['AUC_drop']:+.1f}% vs Chicago")

st.markdown("---")

fig = texas_auc_bar()
st.plotly_chart(fig, use_container_width=True)

st.subheader(f"Detailed Metrics — {crime.replace('_', ' ')}")
rows = []
for model in ["MLP", "Random Forest", "XGBoost"]:
    m = METRICS[model][crime]
    rows.append({
        "Model": model,
        "F1": m["F1"],
        "AUC-ROC": m["AUC"],
        "AUC Drop vs Chicago": f"{m['AUC_drop']:+.1f}%",
    })

table_df = pd.DataFrame(rows).set_index("Model")
st.dataframe(
    table_df.style.highlight_max(subset=["F1", "AUC-ROC"], color="#2d4a2d"),
    use_container_width=True,
)

st.success(
    "**Random Forest dominates zero-shot transfer** — AUC 0.877 / 0.851 / 0.837 across all three crimes. "
    "AUC drop is <5% for most cases, confirming that 42 temporal features are genuinely geography-agnostic."
)

st.info(
    "**Why does generalization work?** The 42 features capture only *temporal autocorrelation* "
    "(recent crime counts and rolling averages). Crime patterns exhibit similar temporal regularities "
    "across geographies — weekly rhythms, seasonal peaks, and inertia effects — independent of location."
)

with st.expander("Texas Data Preview (first 100 rows)"):
    df_tx = load_texas()
    st.write(f"Shape: {df_tx.shape[0]:,} rows × {df_tx.shape[1]} columns")
    st.dataframe(df_tx.head(100), use_container_width=True)
