import streamlit as st

st.set_page_config(
    page_title="Crime Hotspot Prediction",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Predictive Policing: Spatiotemporal Crime Hotspot System")
st.caption("IT5006 Group 19 — AY2025/26 Sem 2 | Chen Wudi, Yu Lingfeng, Zhou Zhenhuan, Ye Fuxian, Yang Luo")

st.markdown("---")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Monitoring Grids (Chicago)", "662", "1 km² each")
col2.metric("Best In-Domain F1", "0.801", "XGBoost — THEFT")
col3.metric("Best Zero-Shot AUC", "0.877", "RF — Texas THEFT")
col4.metric("Features (all temporal)", "42", "No lat/lon inputs")

st.markdown("---")

st.markdown("""
## Project Overview

This dashboard presents a **weekly crime hotspot prediction system** trained on Chicago 2015–2024
and evaluated on held-out 2025 data and a **zero-shot transfer** to Texas NIBRS agencies.

### Key Design Choices
- **42 purely temporal features** — lag and rolling statistics over 7 time scales (1–52 weeks),
  covering total crimes + 3 crime types (THEFT, BATTERY, CRIMINAL DAMAGE), plus cyclic week encoding.
  No latitude, longitude, demographic, or land-use inputs.
- **Geography-agnostic generalization** — the same models trained on Chicago achieve AUC > 0.83
  on Texas agencies with zero additional training data.
- **Four models compared** — Logistic Regression (baseline), MLP (neural network), Random Forest, XGBoost.

### Pages
| Page | Content |
|------|---------|
| **Chicago Hotspot Map** | Interactive weekly map — select crime type, week, model |
| **Texas Generalization** | Zero-shot transfer results — AUC / F1 across models and crimes |
| **Model Performance** | Full 4-model comparison on Chicago 2025 test set |
| **Feature Importance** | RF Gini importance — what the model relies on most |

---
*Use the sidebar to navigate between pages.*
""")
