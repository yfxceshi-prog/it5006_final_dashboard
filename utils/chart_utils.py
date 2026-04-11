import plotly.graph_objects as go
import pandas as pd

CRIME_COLORS = {
    "THEFT": "#FF6B6B",
    "BATTERY": "#FFD93D",
    "CRIMINAL_DAMAGE": "#6BCB77",
}

MODEL_COLORS = {
    "MLP": "#4D96FF",
    "XGBoost": "#FF6B6B",
    "Random Forest": "#6BCB77",
    "LR (Baseline)": "#FFD93D",
}


def hotspot_trend_chart(df: pd.DataFrame, crime: str, model: str, selected_week: int) -> go.Figure:
    """Line chart: predicted vs actual hotspot grid count per week."""
    pred_col = {"MLP": f"mlp_pred_{crime}",
                "XGBoost": f"xgb_pred_{crime}",
                "Random Forest": f"rf_pred_{crime}"}[model]
    weekly = (
        df.groupby("iso_week")
        .agg(
            pred_hot=(pred_col, "sum"),
            true_hot=(f"true_{crime}", "sum"),
        )
        .reset_index()
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=weekly["iso_week"],
        y=weekly["pred_hot"],
        mode="lines",
        name=f"{model} Predicted",
        line=dict(color="#FF6B6B", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=weekly["iso_week"],
        y=weekly["true_hot"],
        mode="lines",
        name="Actual",
        line=dict(color="#6BCB77", width=2, dash="dot"),
    ))
    # Vertical line for selected week
    fig.add_vline(
        x=selected_week,
        line_width=1.5,
        line_dash="dash",
        line_color="#FFFFFF",
        annotation_text=f"Wk {selected_week}",
        annotation_position="top right",
        annotation_font_color="#FAFAFA",
    )
    fig.update_layout(
        title=dict(text=f"Weekly Hotspot Grid Count — {crime.replace('_', ' ')}", font_size=13),
        xaxis_title="ISO Week (2025)",
        yaxis_title="# Grids",
        legend=dict(orientation="h", y=1.02, x=0),
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font_color="#FAFAFA",
        margin=dict(l=40, r=20, t=50, b=40),
        height=320,
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#2A2D3E", zeroline=False)
    return fig


def model_comparison_bar(crimes=("THEFT", "BATTERY", "CRIMINAL_DAMAGE")) -> go.Figure:
    """Grouped bar chart: F1 by model × crime (Chicago 2025 hard-coded)."""
    models = ["LR (Baseline)", "MLP", "Random Forest", "XGBoost"]
    data = {
        "THEFT":           [0.727, 0.767, 0.790, 0.801],
        "BATTERY":         [0.710, 0.723, 0.779, 0.787],
        "CRIMINAL_DAMAGE": [0.622, 0.667, 0.683, 0.667],
    }
    fig = go.Figure()
    for model, color in MODEL_COLORS.items():
        fig.add_trace(go.Bar(
            name=model,
            x=[c.replace("_", " ") for c in crimes],
            y=[data[c][models.index(model)] for c in crimes],
            marker_color=color,
            text=[f"{data[c][models.index(model)]:.3f}" for c in crimes],
            textposition="outside",
        ))
    fig.update_layout(
        barmode="group",
        title=dict(text="F1 Score — Chicago 2025 Test Set", font_size=14),
        yaxis=dict(range=[0.55, 0.87], title="F1 Score"),
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font_color="#FAFAFA",
        legend=dict(orientation="h", y=1.05),
        margin=dict(l=40, r=20, t=60, b=40),
        height=380,
    )
    fig.update_yaxes(showgrid=True, gridcolor="#2A2D3E")
    return fig


def texas_auc_bar() -> go.Figure:
    """Grouped bar chart: Texas zero-shot AUC by model × crime."""
    models = ["MLP", "Random Forest", "XGBoost"]
    crimes = ["THEFT", "BATTERY", "CRIMINAL_DAMAGE"]
    auc_data = {
        "MLP":           [0.772, 0.783, 0.819],
        "Random Forest": [0.877, 0.851, 0.837],
        "XGBoost":       [0.837, 0.805, 0.806],
    }
    colors = ["#4D96FF", "#6BCB77", "#FF6B6B"]
    fig = go.Figure()
    for model, color in zip(models, colors):
        fig.add_trace(go.Bar(
            name=model,
            x=[c.replace("_", " ") for c in crimes],
            y=auc_data[model],
            marker_color=color,
            text=[f"{v:.3f}" for v in auc_data[model]],
            textposition="outside",
        ))
    # Reference lines
    for y_val, label, color in [(0.5, "Random", "#888"), (0.8, "Good", "#FFD93D")]:
        fig.add_hline(y=y_val, line_dash="dash", line_color=color,
                      annotation_text=label, annotation_position="right")
    fig.update_layout(
        barmode="group",
        title=dict(text="Texas Zero-Shot AUC (No Texas Training Data)", font_size=14),
        yaxis=dict(range=[0.45, 0.95], title="AUC-ROC"),
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font_color="#FAFAFA",
        legend=dict(orientation="h", y=1.05),
        margin=dict(l=40, r=20, t=60, b=40),
        height=380,
    )
    fig.update_yaxes(showgrid=True, gridcolor="#2A2D3E")
    return fig


def feature_importance_bar() -> go.Figure:
    """Horizontal bar chart: RF top-20 feature importance (hard-coded approximate values)."""
    features = [
        "lag_1w_THEFT", "lag_1w_total", "roll_4w_THEFT", "lag_2w_THEFT",
        "lag_1w_BATTERY", "roll_4w_total", "lag_1w_CRIMINAL_DAMAGE",
        "lag_2w_total", "roll_4w_BATTERY", "lag_2w_BATTERY",
        "roll_4w_CRIMINAL_DAMAGE", "lag_4w_THEFT", "lag_2w_CRIMINAL_DAMAGE",
        "lag_4w_total", "roll_12w_THEFT", "lag_4w_BATTERY",
        "lag_8w_THEFT", "lag_4w_CRIMINAL_DAMAGE", "cos_week", "sin_week",
    ]
    importance = [
        0.122, 0.098, 0.091, 0.085, 0.078, 0.071, 0.065,
        0.059, 0.054, 0.049, 0.044, 0.042, 0.039, 0.035,
        0.033, 0.031, 0.028, 0.026, 0.012, 0.011,
    ]
    colors = []
    for f in features:
        if "sin_" in f or "cos_" in f:
            colors.append("#FFD93D")
        elif "roll_" in f:
            colors.append("#6BCB77")
        else:
            colors.append("#4D96FF")

    fig = go.Figure(go.Bar(
        x=importance,
        y=features,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.3f}" for v in importance],
        textposition="outside",
    ))
    fig.update_layout(
        title=dict(text="RF Feature Importance — Top 20 (Gini)", font_size=14),
        xaxis_title="Mean Importance",
        yaxis=dict(autorange="reversed"),
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font_color="#FAFAFA",
        margin=dict(l=200, r=60, t=60, b=40),
        height=520,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#2A2D3E")
    return fig


def feature_category_bar() -> go.Figure:
    """Summary bar: average importance by feature category."""
    categories = ["Short-term Lags\n(1–4w)", "Medium-term Lags\n(8–26w)",
                  "Long-term Lags\n(52w)", "Rolling Means", "Cyclic (week)"]
    avg_imp = [0.078, 0.038, 0.018, 0.051, 0.012]
    colors = ["#FF6B6B", "#FFD93D", "#888888", "#6BCB77", "#4D96FF"]
    fig = go.Figure(go.Bar(
        x=categories,
        y=avg_imp,
        marker_color=colors,
        text=[f"{v:.3f}" for v in avg_imp],
        textposition="outside",
    ))
    fig.update_layout(
        title=dict(text="Avg Importance by Feature Category", font_size=13),
        yaxis_title="Mean Gini Importance",
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font_color="#FAFAFA",
        margin=dict(l=40, r=20, t=50, b=60),
        height=320,
    )
    fig.update_yaxes(showgrid=True, gridcolor="#2A2D3E")
    return fig
