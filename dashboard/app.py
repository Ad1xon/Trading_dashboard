"""Quant Dashboard — Streamlit entry point."""

import streamlit as st
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dashboard.views import (
    render_scanner_view, render_backtester_view,
    render_alert_view, render_sentiment_view,
)
from config import load_translations

st.set_page_config(page_title="Quant Dashboard", layout="wide")

if 'lang' not in st.session_state:
    st.session_state.lang = "EN"

st.sidebar.title("Language")
lang_choice = st.sidebar.radio(
    "Select Language", ["English", "Polski"],
    index=0 if st.session_state.lang == "EN" else 1,
    key="lang_radio",
)
st.session_state.lang = "EN" if lang_choice == "English" else "PL"

T = load_translations(st.session_state.lang)

st.title(T.get('title', 'Quant Dashboard'))
tab1, tab2, tab3, tab4 = st.tabs([
    T.get("tab_scanner", "Market Scanner (VWAP)"),
    T.get("tab_backtester", "Advanced Backtester"),
    T.get("tab_alerts", "Alert Center"),
    "Macro & Sentiment",
])

with tab1:
    render_scanner_view(st.session_state.lang)
with tab2:
    render_backtester_view(st.session_state.lang)
with tab3:
    render_alert_view(st.session_state.lang)
with tab4:
    render_sentiment_view(st.session_state.lang)