"""Quant Dashboard — Streamlit entry point."""

import warnings
import os
import logging

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", message=".*Accessing '__path__'.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)

import streamlit as st
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dashboard.views import (
    render_scanner_view, render_backtester_view,
    render_alert_view, render_sentiment_view,
    render_pairs_trading_view, render_strategy_comparison_view,
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
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    T.get("tab_scanner", "Market Scanner (VWAP)"),
    T.get("tab_backtester", "Advanced Backtester"),
    "Strategy Leaderboard",
    T.get("tab_alerts", "Alert Center"),
    "Macro & Sentiment",
    "Pairs Trading",
])

with tab1:
    render_scanner_view(st.session_state.lang)
with tab2:
    render_backtester_view(st.session_state.lang)
with tab3:
    render_strategy_comparison_view(st.session_state.lang)
with tab4:
    render_alert_view(st.session_state.lang)
with tab5:
    render_sentiment_view(st.session_state.lang)
with tab6:
    render_pairs_trading_view(st.session_state.lang)