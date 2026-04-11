import pydeck as pdk
import pandas as pd


MAP_STYLES = {
    "Dark": "mapbox://styles/mapbox/dark-v10",
    "Light": "mapbox://styles/mapbox/light-v10",
    "Satellite": "mapbox://styles/mapbox/satellite-streets-v11",
}

# Red-blue continuous color: high prob → red, low prob → blue
def _prob_to_color(prob: float):
    """Map [0,1] probability to [R,G,B] array (red=high, blue=low)."""
    r = int(255 * prob)
    g = int(30 * (1 - abs(prob - 0.5) * 2))
    b = int(255 * (1 - prob))
    a = int(80 + 160 * prob)
    return [r, g, b, a]


def build_chicago_layer(df_week: pd.DataFrame, crime: str, model: str) -> pdk.Layer:
    """Build a GridCellLayer for the Chicago hotspot map."""
    df = df_week.copy()

    if model == "MLP":
        prob_col = f"mlp_prob_{crime}"
        df["_color"] = df[prob_col].apply(_prob_to_color)
        df["_elev"] = (df[prob_col] * 800).clip(0, 800)
    elif model == "Random Forest":
        prob_col = f"rf_prob_{crime}"
        if prob_col not in df.columns:
            # fallback: use binary pred if prob column missing (old CSV)
            pred_col = f"rf_pred_{crime}"
            df["_color"] = df[pred_col].apply(
                lambda v: [220, 53, 69, 200] if v == 1 else [68, 68, 170, 80]
            )
            df["_elev"] = df[pred_col] * 400
        else:
            df["_color"] = df[prob_col].apply(_prob_to_color)
            df["_elev"] = (df[prob_col] * 800).clip(0, 800)
    elif model == "XGBoost":
        pred_col = f"xgb_pred_{crime}"
        df["_color"] = df[pred_col].apply(
            lambda v: [220, 53, 69, 200] if v == 1 else [68, 68, 170, 80]
        )
        df["_elev"] = df[pred_col] * 400
    else:  # True Labels
        true_col = f"true_{crime}"
        df["_color"] = df[true_col].apply(
            lambda v: [255, 140, 0, 200] if v == 1 else [40, 60, 100, 60]
        )
        df["_elev"] = df[true_col] * 400

    layer = pdk.Layer(
        "GridCellLayer",
        data=df,
        get_position=["lon", "lat"],
        cell_size_meters=900,
        get_color="_color",
        get_elevation="_elev",
        elevation_scale=1,
        pickable=True,
        auto_highlight=True,
        extruded=True,
    )
    return layer


def chicago_view_state(pitch: int = 30) -> pdk.ViewState:
    return pdk.ViewState(
        longitude=-87.72,
        latitude=41.83,
        zoom=10,
        pitch=pitch,
        bearing=0,
    )


def chicago_tooltip(crime: str, model: str) -> dict:
    if model == "MLP":
        prob_key = f"mlp_prob_{crime}"
        pred_key = f"mlp_pred_{crime}"
        html = (
            f"<b>{{grid_id}}</b><br/>"
            f"MLP Prob: {{{prob_key}}}<br/>"
            f"MLP Pred: {{{pred_key}}}<br/>"
            f"Actual: {{true_{crime}}}<br/>"
            f"Count: {{count_{crime}}}"
        )
    elif model == "XGBoost":
        pred_key = f"xgb_pred_{crime}"
        html = (
            f"<b>{{grid_id}}</b><br/>"
            f"XGB Pred: {{{pred_key}}}<br/>"
            f"Actual: {{true_{crime}}}<br/>"
            f"Count: {{count_{crime}}}"
        )
    else:
        html = (
            f"<b>{{grid_id}}</b><br/>"
            f"Actual: {{true_{crime}}}<br/>"
            f"Count: {{count_{crime}}}"
        )
    return {"html": html, "style": {"backgroundColor": "#1E2130", "color": "#FAFAFA"}}
