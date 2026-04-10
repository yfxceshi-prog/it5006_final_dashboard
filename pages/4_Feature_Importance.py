import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.chart_utils import feature_importance_bar, feature_category_bar

st.set_page_config(page_title="Feature Importance", layout="wide")

st.title("Feature Importance")
st.caption("Random Forest Gini importance — top features across all 3 crime types.")

st.info(
    "**42 features, zero geographic inputs.** "
    "All features are temporal statistics computed per 1 km² grid cell: "
    "7 lag steps (1, 2, 4, 8, 13, 26, 52 weeks) × 4 crime streams (total + THEFT + BATTERY + CRIMINAL DAMAGE) = 28 lags, "
    "3 rolling means (4w, 12w, 26w) × 4 streams = 12 rolling features, "
    "plus sin/cos cyclic week encoding = 2 seasonal features."
)

col1, col2, col3 = st.columns(3)
col1.markdown("🔵 **Lag features** — recent crime count at fixed offsets")
col2.markdown("🟢 **Rolling means** — smoothed average over a window")
col3.markdown("🟡 **Cyclic encoding** — sin/cos of ISO week number")

st.markdown("---")

fig_imp = feature_importance_bar()
st.plotly_chart(fig_imp, use_container_width=True)

col_left, col_right = st.columns([5, 5])

with col_left:
    fig_cat = feature_category_bar()
    st.plotly_chart(fig_cat, use_container_width=True)

with col_right:
    st.subheader("Key Findings")
    st.markdown("""
**Short-term lags dominate** (lag_1w and lag_2w account for >20% total importance).
This reflects strong temporal autocorrelation: a grid that was a hotspot last week
is likely to remain one this week.

**Rolling means provide stable baselines** — the 4-week rolling mean smooths
noise while capturing recent activity levels.

**Long-term lags (52w) contribute little** — useful for seasonality context
but less predictive than recent history.

**Cyclic week encoding is relatively low-importance** overall, though it helps
for crime types with strong seasonal patterns (e.g., THEFT peaks in summer months).

**No location features** — the model learns *when* crime is likely, not *where*
based on geography. This is what enables zero-shot transfer to Texas.
    """)

st.markdown("---")
st.subheader("XGBoost Feature Importance (Gain)")
st.markdown("""
XGBoost feature importance (gain metric) tells a similar story:
- **lag_1w_THEFT** and **lag_1w_total** rank highest for THEFT prediction
- **lag_1w_BATTERY** tops BATTERY prediction
- Rolling features (roll_4w) remain consistently important across all crime types

The agreement between RF (Gini) and XGBoost (gain) importance provides **cross-validation**
that the short-term lag structure is genuinely informative, not an artifact of one model's inductive bias.
""")
