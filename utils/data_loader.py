import pandas as pd
import streamlit as st
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


@st.cache_data
def load_chicago():
    df = pd.read_csv(DATA_DIR / "predictions_chicago.csv")
    return df


@st.cache_data
def load_texas():
    df = pd.read_csv(DATA_DIR / "predictions_tx.csv")
    return df


def iso_week_to_label(week: int) -> str:
    from datetime import datetime, timedelta
    jan4 = datetime(2025, 1, 4)
    start = jan4 + timedelta(weeks=week - 1) - timedelta(days=jan4.weekday())
    end = start + timedelta(days=6)
    if start.month == end.month:
        return f"Week {week}  ({start.strftime('%b %d')}–{end.strftime('%d, %Y')})"
    return f"Week {week}  ({start.strftime('%b %d')} – {end.strftime('%b %d, %Y')})"
