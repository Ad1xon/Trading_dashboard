# dashboard/views.py
"""Dashboard views — scanner, backtester, alert center."""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import timedelta

from config import (
    load_translations, MT5_SYMBOLS, COMMISSION_USD_PER_LOT,
    CONTRACT_SIZES
)
from data_feed.mt5_connector import get_mt5_data
from quant_engine.data_processor import generate_synthetic_range_bars
from quant_engine.backtester import run_advanced_backtest
from quant_engine.strategies import STRATEGY_REGISTRY, detect_liquidity_sweep
from quant_engine.indicators import (
    calculate_vwap_with_bands, calculate_rsi, calculate_atr,
)
from alerts.alert_manager import AlertManager


def _get_alert_manager() -> AlertManager:
    if 'alert_manager' not in st.session_state:
        st.session_state.alert_manager = AlertManager()
    return st.session_state.alert_manager


def render_scanner_view(lang: str):
    T = load_translations(lang)
    st.sidebar.header(T["settings"])

    selected_symbols = st.sidebar.multiselect(
        T["symbol_input"],
        list(MT5_SYMBOLS.keys()),
        default=list(MT5_SYMBOLS.keys())[:3],
        key="scanner_symbols",
    )
    scan_days = st.sidebar.slider("Scan Depth (days)", 1, 30, 5, key="scanner_days")
    range_val = st.sidebar.number_input(
        T["range_size"], value=0.001, format="%.5f", key="scanner_range_size",
    )

    if st.button(T["run_scanner"], key="btn_scanner"):
        with st.spinner(T["scanner_running"]):
            scan_results = []

            from utils.event_bus import EventBus
            bus = EventBus()

            for sym_name in selected_symbols:
                mt5_sym = MT5_SYMBOLS[sym_name]
                df_raw = get_mt5_data(mt5_sym, scan_days)
                if df_raw.empty:
                    continue

                rb = generate_synthetic_range_bars(df_raw, range_val)
                if len(rb) < 30:
                    continue

                rb_vwap = calculate_vwap_with_bands(rb)
                rb_vwap['RSI'] = calculate_rsi(rb_vwap['Close'], 14)
                rb_vwap['ATR'] = calculate_atr(rb_vwap, 14)

                latest = rb_vwap.iloc[-1]
                signal = "—"
                confidence = 0.0

                sweep = detect_liquidity_sweep(rb)
                if sweep["signal"]:
                    signal = sweep["type"]
                    confidence = 0.75
                    bus.publish_sync("LIQUIDITY_SWEEP", {
                        "symbol": sym_name,
                        "message": sweep["message"],
                        "signal_type": sweep["type"],
                        "confidence": confidence
                    })

                scan_results.append({
                    T["col_symbol"]: sym_name,
                    T["col_price"]: f"{latest['Close']:.5f}",
                    T["col_vwap"]: f"{latest.get('VWAP', 0):.5f}",
                    T["col_rsi"]: f"{latest.get('RSI', 0):.1f}",
                    T["col_atr"]: f"{latest.get('ATR', 0):.5f}",
                    T["col_signal"]: signal,
                    T["col_confidence"]: f"{confidence * 100:.0f}%",
                })

            if scan_results:
                st.subheader(T["scanner_results"])
                res_df = pd.DataFrame(scan_results)

                def _highlight_signal(val):
                    if "BULLISH" in str(val):
                        return "background-color: #0d4f2b; color: #00ff88"
                    elif "BEARISH" in str(val):
                        return "background-color: #4f0d0d; color: #ff4444"
                    return ""

                styled = res_df.style.map(_highlight_signal, subset=[T["col_signal"]])
                st.dataframe(styled, use_container_width=True)
            else:
                st.info(T["scanner_no_data"])


def render_backtester_view(lang: str):
    T = load_translations(lang)
    st.sidebar.header(T["backtest_params"])

    selected_name = st.sidebar.selectbox(
        T["symbol_input"], list(MT5_SYMBOLS.keys()), key="bt_symbol",
    )
    symbol = MT5_SYMBOLS[selected_name]

    days_back = st.sidebar.slider(T["history_days"], 1, 365, 30, key="bt_days")

    strat_options = list(STRATEGY_REGISTRY.keys())
    strategy_choice = st.sidebar.selectbox(
        T["strat_select"], strat_options, key="bt_strategy",
    )

    capital = st.sidebar.number_input(T["capital"], value=10000.0, key="bt_capital")
    risk = st.sidebar.number_input(T["risk"], value=2.0, key="bt_risk") / 100.0

    def_range = 0.0010
    def_slip = 0.0001
    if "DAX" in selected_name or "NAS" in selected_name:
        def_range, def_slip = 15.0, 0.5
    elif "XAU" in selected_name:
        def_range, def_slip = 1.5, 0.1
    elif "JPY" in selected_name:
        def_range, def_slip = 0.1, 0.01

    range_val = st.sidebar.number_input(
        T["range_size"], value=def_range, format="%.5f", key=f"bt_range_{selected_name}",
    )
    slippage = st.sidebar.number_input(
        T["slippage"], value=def_slip, format="%.5f", key=f"bt_slip_{selected_name}",
    )

    if st.button(T["run_sim"], key="btn_backtest"):
        with st.spinner("Processing..."):
            df_raw = get_mt5_data(symbol, days_back)
            if df_raw.empty:
                st.error(T["error_data"])
                return

            range_bars = generate_synthetic_range_bars(df_raw, range_val)
            if len(range_bars) < 50:
                st.warning("Insufficient range bars generated.")
                return

            strat_cls = STRATEGY_REGISTRY[strategy_choice]
            strat = strat_cls()

            from config import COMMISSION_USD_PER_LOT, CONTRACT_SIZES
            contract = CONTRACT_SIZES.get(symbol, 100000)
            comm_pct = COMMISSION_USD_PER_LOT / contract
            results = run_advanced_backtest(
                range_bars, capital, risk, slippage, strat, comm_pct, symbol=symbol
            )

            c1, c2, c3, c4 = st.columns(4)
            c1.metric(T["stats_bars"], len(range_bars))
            c2.metric(T["stats_trades"], results['n_trades'])
            _colored_metric(c3, T["stats_return"], results['total_return'] * 100, fmt="{:.2f}%")
            _colored_metric(c4, T["stats_dd"], results['max_drawdown'] * 100, fmt="{:.2f}%", invert=True)

            c5, c6, c7, c8 = st.columns(4)
            _colored_metric(c5, T["stats_sharpe"], results['sharpe_ratio'], fmt="{:.2f}")
            _colored_metric(c6, T["stats_sortino"], results['sortino_ratio'], fmt="{:.2f}")
            c7.metric(T["stats_winrate"], f"{results['win_rate'] * 100:.1f}%")
            c8.metric(T["stats_pf"], f"{results['profit_factor']:.2f}")

            c9, c10, c11, c12 = st.columns(4)
            _colored_metric(c9, T["stats_calmar"], results['calmar_ratio'], fmt="{:.2f}")
            c10.metric(T["stats_avg_wl"], f"{results['avg_win_loss_ratio']:.2f}")
            c11.metric(T["stats_max_consec_loss"], results['max_consecutive_losses'])
            c12.metric("", "")

            _render_price_chart(range_bars, results, T)
            _render_equity_chart(results, T)
            _render_drawdown_chart(results, T)
            _render_pnl_distribution(results, T)
            _render_trade_table(results, T)

            if hasattr(strat, 'ml_model') and getattr(strat.ml_model, 'is_trained', False):
                st.subheader(f"{strategy_choice} Feature Importance")
                if getattr(strat.ml_model, 'is_cached', False):
                    st.success("⚡ Model & Predictions Loaded from joblib Cache (0.01s)")
                fig = strat.ml_model.plot_feature_importance()
                if fig is not None:
                    st.pyplot(fig)

                fi = strat.ml_model.get_feature_importance()
                if fi:
                    total = sum(fi.values())
                    weak = [f for f, v in fi.items() if v / total < 0.01]
                    if weak:
                        st.info(f"Low-importance features (<1%): {', '.join(weak)}. Consider dropping.")

                if getattr(strat.ml_model, 'cv_scores_', None):
                    avg_cv = np.mean(strat.ml_model.cv_scores_)
                    st.metric("Walk-Forward CV Accuracy", f"{avg_cv * 100:.1f}%")


def render_alert_view(lang: str):
    T = load_translations(lang)
    alert_mgr = _get_alert_manager()

    st.sidebar.header("⚡ Alerts Config")
    webhook = st.sidebar.text_input(
        T["alert_webhook"], type="password", key="discord_webhook",
    )
    if webhook:
        alert_mgr.configure_discord(webhook)

    alert_mgr.enabled = st.sidebar.checkbox(
        T["alert_enabled"], value=True, key="alerts_enabled_toggle",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Per-Symbol Toggles")
    for sym_name in MT5_SYMBOLS.keys():
        enabled = st.sidebar.checkbox(
            f"  {sym_name}", value=True, key=f"alert_toggle_{sym_name}",
        )
        alert_mgr.set_threshold(sym_name, enabled=enabled)

    if st.button(T["alert_test"], key="btn_test_alert"):
        from utils.event_bus import EventBus
        bus = EventBus()
        bus.publish_sync("TRADE_SIGNAL", {
            "symbol": "TEST",
            "message": "This is a test alert from the dashboard.",
            "signal_type": "INFO"
        })
        st.success("Test alert dispatched via Event Bus!")

    st.subheader(T["alert_history"])
    history = alert_mgr.get_history(50)
    if history:
        st.dataframe(pd.DataFrame(history), use_container_width=True, height=400)
    else:
        st.info("No alerts yet.")


def _render_price_chart(range_bars, results, T):
    st.subheader("Price & Trades")
    cutoff_date = range_bars.index.max() - timedelta(days=7)
    plot_bars = range_bars[range_bars.index >= cutoff_date].copy()
    plot_bars = plot_bars.reset_index()
    if 'Timestamp' not in plot_bars.columns:
        plot_bars['Timestamp'] = plot_bars.iloc[:, 0]
    plot_bars['_x'] = plot_bars['Timestamp'].astype(str)
    dupes = plot_bars['_x'].duplicated(keep=False)
    if dupes.any():
        counts: dict = {}
        labels = []
        for val in plot_bars['_x']:
            counts[val] = counts.get(val, 0) + 1
            labels.append(f"{val} [{counts[val]}]" if counts[val] > 1 or val in [v for v, c in counts.items() if c > 0 and dupes.any()] else val)
        plot_bars['_x'] = [f"{v}.{i}" for i, v in enumerate(plot_bars['_x'])]

    x_strings = plot_bars['_x'].tolist()
    ts_values = plot_bars['Timestamp'].values

    plot_trades = []
    for t in results['trades_history']:
        entry_ts = pd.Timestamp(t['entry_idx'])
        exit_ts = pd.Timestamp(t['exit_idx'])
        if entry_ts >= cutoff_date or exit_ts >= cutoff_date:
            plot_trades.append({**t, '_entry_ts': entry_ts, '_exit_ts': exit_ts})

    fig = go.Figure(data=[go.Candlestick(
        x=x_strings,
        open=plot_bars['Open'], high=plot_bars['High'],
        low=plot_bars['Low'], close=plot_bars['Close'],
        name="Price",
    )])

    if plot_trades:
        def _snap(ts):
            diffs = np.abs(ts_values - np.datetime64(ts))
            idx = int(np.argmin(diffs))
            return x_strings[idx]

        e_lx, e_ly, e_sx, e_sy, ex_x, ex_y = [], [], [], [], [], []
        for t in plot_trades:
            if t['type'] == 1:
                e_lx.append(_snap(t['_entry_ts']))
                e_ly.append(t['entry_price'])
            else:
                e_sx.append(_snap(t['_entry_ts']))
                e_sy.append(t['entry_price'])
            ex_x.append(_snap(t['_exit_ts']))
            ex_y.append(t['exit_price'])

        if e_lx:
            fig.add_trace(go.Scatter(
                x=e_lx, y=e_ly, mode='markers',
                marker=dict(symbol='triangle-up', size=12, color='lime'), name='Long',
            ))
        if e_sx:
            fig.add_trace(go.Scatter(
                x=e_sx, y=e_sy, mode='markers',
                marker=dict(symbol='triangle-down', size=12, color='red'), name='Short',
            ))
        if ex_x:
            fig.add_trace(go.Scatter(
                x=ex_x, y=ex_y, mode='markers',
                marker=dict(symbol='x', size=8, color='yellow'), name='Exit',
            ))

    fig.update_layout(
        template="plotly_dark", height=600,
        xaxis_type='category', xaxis_rangeslider_visible=False,
    )
    fig.update_xaxes(nticks=12)
    st.plotly_chart(fig, use_container_width=True)


def _render_equity_chart(results, T):
    st.subheader(T["equity_curve"])
    eq_fig = go.Figure()
    eq_data = results['equity_curve']
    eq_fig.add_trace(go.Scatter(
        y=eq_data['Strategy_Equity'], mode='lines',
        name='Strategy', line=dict(color='#00ff88', width=2),
    ))
    eq_fig.add_trace(go.Scatter(
        y=eq_data['BuyHold_Equity'], mode='lines',
        name='Buy & Hold', line=dict(color='#3399ff', width=1, dash='dash'),
    ))
    eq_fig.update_layout(template="plotly_dark", height=350)
    st.plotly_chart(eq_fig, use_container_width=True)


def _render_drawdown_chart(results, T):
    st.subheader(T["drawdown_chart"])
    dd_series = results.get('drawdown_series')
    if dd_series is None or len(dd_series) == 0:
        return
    dd_fig = go.Figure()
    dd_fig.add_trace(go.Scatter(
        y=dd_series.values * 100, mode='lines',
        fill='tozeroy', line=dict(color='#ff4444', width=1),
        fillcolor='rgba(255,68,68,0.2)', name='Drawdown %',
    ))
    dd_fig.update_layout(template="plotly_dark", height=250, yaxis_title="Drawdown %")
    st.plotly_chart(dd_fig, use_container_width=True)


def _render_pnl_distribution(results, T):
    if not results['trades_history']:
        return
    st.subheader(T["trade_dist"])
    pnls = [t['pnl'] for t in results['trades_history']]
    colors = ['#00ff88' if p > 0 else '#ff4444' for p in pnls]
    hist_fig = go.Figure()
    hist_fig.add_trace(go.Bar(y=pnls, marker_color=colors, name='Trade PnL'))
    hist_fig.update_layout(template="plotly_dark", height=250, yaxis_title="PnL ($)")
    st.plotly_chart(hist_fig, use_container_width=True)


def _render_trade_table(results, T):
    if not results['trades_history']:
        return
    st.subheader(T["trade_history"])
    trade_df = pd.DataFrame(results['trades_history'])
    trade_df[T["col_type"]] = trade_df['type'].map({1: 'LONG', -1: 'SHORT'})
    trade_df = trade_df.rename(columns={
        'entry_idx': T["col_entry"],
        'exit_idx': T["col_exit"],
        'entry_price': 'Entry Price',
        'exit_price': 'Exit Price',
        'pnl': T["col_pnl"],
        'bars_held': T["col_bars_held"],
        'exit_reason': T["col_exit_reason"],
    })
    display_cols = [
        T["col_entry"], T["col_exit"], 'Entry Price', 'Exit Price',
        T["col_type"], T["col_pnl"], T["col_bars_held"], T["col_exit_reason"],
    ]
    existing_cols = [c for c in display_cols if c in trade_df.columns]
    st.dataframe(
        trade_df[existing_cols].style.map(
            lambda v: "color: #00ff88" if isinstance(v, (int, float)) and v > 0
            else ("color: #ff4444" if isinstance(v, (int, float)) and v < 0 else ""),
            subset=[T["col_pnl"]],
        ),
        use_container_width=True,
        height=300,
    )


def _colored_metric(col, label: str, value: float, fmt: str = "{:.2f}", invert: bool = False):
    formatted = fmt.format(value)
    good = value < 0 if invert else value > 0
    color = "#00ff88" if good else "#ff4444"
    col.markdown(
        f"<div style='text-align:center'>"
        f"<small style='color:#aaa'>{label}</small><br>"
        f"<span style='font-size:1.4em;font-weight:bold;color:{color}'>{formatted}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )