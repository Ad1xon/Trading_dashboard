"""Dashboard views — scanner, backtester, alert center, macro & sentiment, pairs trading, risk analytics."""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from datetime import timedelta

from config import (
    load_translations, MT5_SYMBOLS, COMMISSION_USD_PER_LOT,
    CONTRACT_SIZES, PAIRS_TRADING_PAIRS,
)
from data_feed.mt5_connector import get_mt5_data
from data_feed.nlp_engine import SentimentEngine
from quant_engine.data_processor import generate_synthetic_range_bars
from quant_engine.backtester import run_advanced_backtest
from quant_engine.strategies import STRATEGY_REGISTRY, detect_liquidity_sweep
from quant_engine.indicators import (
    calculate_vwap_with_bands, calculate_rsi, calculate_atr,
)
from quant_engine.risk_metrics import (
    compute_kelly_fraction, compute_mae_mfe, compute_parametric_var,
)
from quant_engine.strategy_optimizer import monte_carlo_simulation
from quant_engine.stat_arb import test_cointegration
from alerts.alert_manager import AlertManager
from quant_engine.macro_filter import MacroFilter
from quant_engine.scanner import MarketScanner


SWING_STRATEGIES = {'LSTM Swing', 'Composite Alpha (MFT)', 'Regime Switch (HMM)', 'Ultimate Swing'}


def _get_alert_manager() -> AlertManager:
    """Retrieve or create the singleton AlertManager from session state."""
    if 'alert_manager' not in st.session_state:
        st.session_state.alert_manager = AlertManager()
    return st.session_state.alert_manager


def render_scanner_view(lang: str):
    """Render the multi-symbol market scanner tab."""
    t = load_translations(lang)
    st.sidebar.header(t.get("settings", "Settings"))

    selected_symbols = st.sidebar.multiselect(
        t.get("symbol_input", "Scanner Symbols"),
        list(MT5_SYMBOLS.keys()),
        default=list(MT5_SYMBOLS.keys())[:3],
        key="scanner_symbols",
    )

    tf_options = {
        "Range Bars (M1 Base)": mt5.TIMEFRAME_M1,
        "M15 (Time-based)": mt5.TIMEFRAME_M15,
        "H1 (Time-based)": mt5.TIMEFRAME_H1,
        "H4 (Time-based)": mt5.TIMEFRAME_H4,
        "D1 (Time-based)": mt5.TIMEFRAME_D1,
    }

    data_mode_scan = st.sidebar.selectbox(
        "Timeframe / Data Mode", list(tf_options.keys()), key="scanner_data_mode",
    )

    scan_days = st.sidebar.slider("Scan Depth (days)", 1, 30, 5, key="scanner_days")

    if "Range Bars" in data_mode_scan:
        range_val = st.sidebar.number_input(
            t.get("range_size", "Range Bar Size"),
            value=0.001, format="%.5f", key="scanner_range_size",
        )
    else:
        range_val = None

    if st.button(t.get("run_scanner", "Run Scanner"), key="btn_scanner"):
        with st.spinner(t.get("scanner_running", "Scanning markets...")):
            scanner_engine = MarketScanner()
            raw_results = scanner_engine.run_scan(
                selected_symbols=selected_symbols,
                data_mode_scan=data_mode_scan,
                tf_options=tf_options,
                scan_days=scan_days,
                range_val=range_val
            )
            
            scan_results = []
            for res in raw_results:
                scan_results.append({
                    t.get("col_symbol", "Symbol"): res["symbol"],
                    t.get("col_price", "Price"): f"{res['price']:.5f}",
                    t.get("col_vwap", "VWAP"): f"{res['vwap']:.5f}",
                    t.get("col_rsi", "RSI"): f"{res['rsi']:.1f}",
                    t.get("col_atr", "ATR"): f"{res['atr']:.5f}",
                    t.get("col_signal", "Signal"): res["signal"],
                    t.get("col_confidence", "Confidence"): f"{res['confidence'] * 100:.0f}%",
                })

            if scan_results:
                st.subheader(t.get("scanner_results", "Scanner Results"))
                res_df = pd.DataFrame(scan_results)

                def _highlight_signal(val):
                    if "BULLISH" in str(val):
                        return "background-color: #0d4f2b; color: #00ff88"
                    elif "BEARISH" in str(val):
                        return "background-color: #4f0d0d; color: #ff4444"
                    return ""

                styled = res_df.style.map(
                    _highlight_signal, subset=[t.get("col_signal", "Signal")],
                )
                st.dataframe(styled, use_container_width=True)
            else:
                st.info(t.get("scanner_no_data", "No data available for scan."))


def render_backtester_view(lang: str):
    """Render the advanced backtester tab with strategy-aware data defaults."""
    t = load_translations(lang)
    st.sidebar.header(t.get("backtest_params", "Backtest Parameters"))

    selected_name = st.sidebar.selectbox(
        t.get("symbol_input", "Scanner Symbols"),
        list(MT5_SYMBOLS.keys()), key="bt_symbol",
    )
    symbol = MT5_SYMBOLS[selected_name]

    strat_options = list(STRATEGY_REGISTRY.keys())
    strategy_choice = st.sidebar.selectbox(
        t.get("strat_select", "Select Strategy"), strat_options, key="bt_strategy",
    )

    is_swing = strategy_choice in SWING_STRATEGIES
    default_days = 730 if is_swing else 365

    tf_options = {
        "Range Bars (M1 Base)": mt5.TIMEFRAME_M1,
        "M15 (Time-based)": mt5.TIMEFRAME_M15,
        "H1 (Time-based)": mt5.TIMEFRAME_H1,
        "H4 (Time-based)": mt5.TIMEFRAME_H4,
        "D1 (Time-based)": mt5.TIMEFRAME_D1,
    }

    default_tf_idx = 2 if is_swing else 0
    data_mode = st.sidebar.selectbox(
        "Timeframe / Data Mode", list(tf_options.keys()),
        index=default_tf_idx, key="bt_data_mode",
    )

    max_days = {
        "Range Bars (M1 Base)": 90,
        "M15 (Time-based)": 180,
        "H1 (Time-based)": 730,
        "H4 (Time-based)": 2000,
        "D1 (Time-based)": 5000,
    }.get(data_mode, 365)

    days_back = st.sidebar.slider(
        t.get("history_days", "History Depth (Days)"),
        1, max_days, min(default_days, max_days), key="bt_days",
    )

    capital = st.sidebar.number_input(
        t.get("capital", "Initial Capital (USD)"), value=10000.0, key="bt_capital",
    )
    risk = st.sidebar.number_input(
        t.get("risk", "Risk per Trade (%)"), value=2.0, key="bt_risk",
    ) / 100.0

    def_range = 0.0010
    def_slip = 0.0001
    if "DAX" in selected_name or "NAS" in selected_name:
        def_range, def_slip = 15.0, 0.5
    elif "XAU" in selected_name:
        def_range, def_slip = 1.5, 0.1
    elif "JPY" in selected_name:
        def_range, def_slip = 0.1, 0.01

    if "Range Bars" in data_mode:
        range_val = st.sidebar.number_input(
            t.get("range_size", "Range Bar Size"),
            value=def_range, format="%.5f",
            key=f"bt_range_{selected_name}",
        )
    else:
        range_val = None

    slippage = st.sidebar.number_input(
        t.get("slippage", "Price Slippage"),
        value=def_slip, format="%.5f",
        key=f"bt_slip_{selected_name}",
    )

    if st.button(t.get("run_sim", "Run Simulation"), key="btn_backtest"):
        with st.spinner("Processing..."):
            mt5_tf = tf_options[data_mode]
            df_raw = get_mt5_data(symbol, days_back, timeframe=mt5_tf)

            if df_raw.empty:
                st.error(t.get("error_data", "Failed to retrieve MT5 data."))
                return

            if "Range Bars" in data_mode:
                trading_data = generate_synthetic_range_bars(df_raw, range_val)
                if len(trading_data) < 50:
                    st.warning("Insufficient range bars generated.")
                    return
            else:
                trading_data = df_raw.copy()
                if len(trading_data) < 50:
                    st.warning("Insufficient time bars fetched. Increase history depth.")
                    return

            strat_cls = STRATEGY_REGISTRY[strategy_choice]
            strat = strat_cls()

            contract = CONTRACT_SIZES.get(symbol, 100000)
            comm_pct = COMMISSION_USD_PER_LOT / contract
            results = run_advanced_backtest(
                trading_data, capital, risk, slippage, strat,
                comm_pct, symbol=symbol,
            )

            if results.get('bankrupt'):
                st.error("MARGIN CALL — Account went bankrupt during simulation!")

            c1, c2, c3, c4 = st.columns(4)
            data_days = (trading_data.index[-1] - trading_data.index[0]).days if isinstance(trading_data.index, pd.DatetimeIndex) and len(trading_data) > 0 else 0
            
            train_days = 0 
            if hasattr(strat, 'ml_model') and hasattr(strat.ml_model, 'horizon_param'):
                train_sample = min(len(trading_data) // 3, 5000) # proxy
                train_days = int(data_days * (train_sample / max(1, len(trading_data))))
            test_days = data_days - train_days

            c1.metric(t.get("stats_bars", "Analyzed Bars"), len(trading_data))
            c2.metric(t.get("stats_trades", "Total Trades"), results['n_trades'])
            c3.metric("Est. Phase Split", f"{train_days}d Train / {test_days}d Test" if train_days > 0 else f"{data_days}d Overall")
            _colored_metric(c4, t.get("stats_return", "Total Return"), results['total_return'] * 100, fmt="{:.2f}%")
            
            c5, c6, c7, c8 = st.columns(4)
            _colored_metric(c5, t.get("stats_dd", "Max Drawdown"), results['max_drawdown'] * 100, fmt="{:.2f}%", invert=True)
            c6.metric(t.get("stats_winrate", "Win Rate"), f"{results['win_rate'] * 100:.1f}%")
            c7.metric(t.get("stats_pf", "Profit Factor"), f"{results['profit_factor']:.2f}")
            c8.metric(t.get("stats_avg_wl", "Avg Win/Loss Ratio"), f"{results['avg_win_loss_ratio']:.2f}")

            c9, c10, c11, c12 = st.columns(4)
            _colored_metric(c9, t.get("stats_sharpe", "Sharpe Ratio"), results['sharpe_ratio'], fmt="{:.2f}",
                            help="Risk-adjusted return vs volatility limit.")
            _colored_metric(c10, t.get("stats_sortino", "Sortino Ratio"), results['sortino_ratio'], fmt="{:.2f}",
                            help="Risk-adjusted return strictly vs downside volatility.")
            _colored_metric(c11, t.get("stats_calmar", "Calmar Ratio"), results['calmar_ratio'], fmt="{:.2f}",
                            help="Annualized Return scaled by Max Drawdown.")
            c12.metric(t.get("stats_max_consec_loss", "Max Consec. Losses"), results['max_consecutive_losses'])

            c13, c14, c15, c16 = st.columns(4)
            _colored_metric(c13, "VaR (95%)", results.get('var_95', 0.0) * 100, fmt="{:.3f}%", invert=True,
                            help="Value at Risk: max loss at 95% confidence. Formula: percentile(returns, 5%).")
            _colored_metric(c14, "CVaR (95%)", results.get('cvar_95', 0.0) * 100, fmt="{:.3f}%", invert=True,
                            help="Expected Shortfall: E[loss | loss > VaR]. Mean of worst 5% scenarios.")

            kelly = compute_kelly_fraction(
                results['win_rate'], results['avg_win'], results['avg_loss'],
            )
            c15.metric("Kelly Fraction", f"{kelly * 100:.1f}%",
                       help="Optimal bet size: f* = (p·b − q)/b, capped at 25%.")

            mae_mfe = compute_mae_mfe(results['trades_history'])
            c16.metric("Trade Efficiency", f"{mae_mfe['efficiency'] * 100:.1f}%",
                       help="MAE/MFE ratio — how much of the move is captured.")

            _render_price_chart(trading_data, results)
            _render_equity_chart(results, t)
            _render_drawdown_chart(results, t)
            _render_pnl_distribution(results, t)
            _render_trade_table(results, t)
            _render_monthly_heatmap(results)
            _render_rolling_sharpe(results)
            _render_monte_carlo(results, capital)

            if hasattr(strat, 'ml_model') and getattr(strat.ml_model, 'is_trained', False):
                st.subheader(f"{strategy_choice} Feature Importance")
                if getattr(strat.ml_model, 'is_cached', False):
                    st.success("⚡ Model & Predictions Loaded from joblib Cache")

                if hasattr(strat.ml_model, 'plot_feature_importance'):
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
    """Render the alert center tab."""
    t = load_translations(lang)
    alert_mgr = _get_alert_manager()

    st.sidebar.header("Alerts Config")
    webhook = st.sidebar.text_input(
        t.get("alert_webhook", "Discord Webhook URL"),
        type="password", key="discord_webhook",
    )
    if webhook:
        alert_mgr.configure_discord(webhook)

    alert_mgr.enabled = st.sidebar.checkbox(
        t.get("alert_enabled", "Enable Alerts"),
        value=True, key="alerts_enabled_toggle",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Per-Symbol Toggles")
    for sym_name in MT5_SYMBOLS.keys():
        enabled = st.sidebar.checkbox(
            f"  {sym_name}", value=True, key=f"alert_toggle_{sym_name}",
        )
        alert_mgr.set_threshold(sym_name, enabled=enabled)

    if st.button(t.get("alert_test", "Send Test Alert"), key="btn_test_alert"):
        from utils.event_bus import EventBus
        bus = EventBus()
        bus.publish_sync("TRADE_SIGNAL", {
            "symbol": "TEST",
            "message": "This is a test alert from the dashboard.",
            "signal_type": "INFO",
        })
        st.success("Test alert dispatched via Event Bus!")

    st.subheader(t.get("alert_history", "Alert History"))
    history = alert_mgr.get_history(50)
    if history:
        st.dataframe(pd.DataFrame(history), use_container_width=True, height=400)
    else:
        st.info("No alerts yet.")


def render_sentiment_view(lang: str):
    """Render the Global Macro & Sentiment Heatmap tab."""
    st.subheader("Global Macro & Sentiment Heatmap")
    st.markdown("Powered by **FinBERT NLP** — Sources: Google News, Yahoo Finance, Reddit, Investing.com")

    if st.button("Refresh Global Data", key="btn_refresh_macro"):
        with st.spinner("Analyzing multi-source NLP sentiment & macro calendar..."):

            nlp = SentimentEngine()
            sent_data = []

            for sym_name in MT5_SYMBOLS.keys():
                search_term = sym_name.split(' ')[0] if ' ' in sym_name else sym_name
                score = nlp.fetch_rss_sentiment(search_term)

                if score > 0.15:
                    bias = "BULLISH 🟢"
                elif score < -0.15:
                    bias = "BEARISH 🔴"
                else:
                    bias = "NEUTRAL ⚪"

                sent_data.append({
                    "Instrument": sym_name,
                    "Aggregate Score": score,
                    "Regime Bias": bias,
                })

            df_sent = pd.DataFrame(sent_data)

            def _color_bias(val):
                if "BULLISH" in str(val):
                    return "color: #00ff88; font-weight: bold"
                if "BEARISH" in str(val):
                    return "color: #ff4444; font-weight: bold"
                return "color: #aaaaaa"

            st.markdown("### Multi-Source NLP Sentiment")
            st.dataframe(
                df_sent.style.map(
                    _color_bias, subset=["Regime Bias"],
                ).format({"Aggregate Score": "{:.3f}"}),
                use_container_width=True,
            )

            st.markdown("---")
            st.markdown("### High-Impact Macro Events (Red Folders)")
            mf = MacroFilter()
            events = mf.high_impact_events

            if not events.empty:
                ev_df = pd.DataFrame({"Event Time (Platform/MT5 Time)": events})
                ev_df['Event Time (Platform/MT5 Time)'] = ev_df[
                    'Event Time (Platform/MT5 Time)'
                ].dt.strftime('%Y-%m-%d %H:%M')
                st.dataframe(ev_df, use_container_width=True)
            else:
                st.info(
                    "No High-Impact ('Red Folder') events detected "
                    "for the remainder of this week."
                )


def _render_price_chart(trading_data, results):
    """Render candlestick price chart with trade markers."""
    st.subheader("Price & Trades")
    cutoff_date = trading_data.index.max() - timedelta(days=7)
    plot_bars = trading_data[trading_data.index >= cutoff_date].copy()
    plot_bars = plot_bars.reset_index()

    time_col = plot_bars.columns[0]
    if 'time' in plot_bars.columns:
        time_col = 'time'
    elif 'Timestamp' in plot_bars.columns:
        time_col = 'Timestamp'

    plot_bars['_x'] = plot_bars[time_col].astype(str)
    dupes = plot_bars['_x'].duplicated(keep=False)
    if dupes.any():
        plot_bars['_x'] = [f"{v}.{i}" for i, v in enumerate(plot_bars['_x'])]

    x_strings = plot_bars['_x'].tolist()
    ts_values = plot_bars[time_col].values

    plot_trades = []
    for tr in results['trades_history']:
        entry_ts = pd.Timestamp(tr['entry_idx'])
        exit_ts = pd.Timestamp(tr['exit_idx'])
        if entry_ts >= cutoff_date or exit_ts >= cutoff_date:
            plot_trades.append({**tr, '_entry_ts': entry_ts, '_exit_ts': exit_ts})

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
        for tr in plot_trades:
            if tr['type'] == 1:
                e_lx.append(_snap(tr['_entry_ts']))
                e_ly.append(tr['entry_price'])
            else:
                e_sx.append(_snap(tr['_entry_ts']))
                e_sy.append(tr['entry_price'])
            ex_x.append(_snap(tr['_exit_ts']))
            ex_y.append(tr['exit_price'])

        if e_lx:
            fig.add_trace(go.Scatter(
                x=e_lx, y=e_ly, mode='markers',
                marker=dict(symbol='triangle-up', size=12, color='lime'),
                name='Long',
            ))
        if e_sx:
            fig.add_trace(go.Scatter(
                x=e_sx, y=e_sy, mode='markers',
                marker=dict(symbol='triangle-down', size=12, color='red'),
                name='Short',
            ))
        if ex_x:
            fig.add_trace(go.Scatter(
                x=ex_x, y=ex_y, mode='markers',
                marker=dict(symbol='x', size=8, color='yellow'),
                name='Exit',
            ))

    fig.update_layout(
        template="plotly_dark", height=600,
        xaxis_type='category', xaxis_rangeslider_visible=False,
    )
    fig.update_xaxes(nticks=12)
    st.plotly_chart(fig, use_container_width=True)


def _render_equity_chart(results, t):
    """Render strategy vs. buy-and-hold equity curves."""
    st.subheader(t.get("equity_curve", "Equity Curve"))
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


def _render_drawdown_chart(results, t):
    """Render drawdown time-series area chart."""
    st.subheader(t.get("drawdown_chart", "Drawdown Chart"))
    dd_series = results.get('drawdown_series')
    if dd_series is None or len(dd_series) == 0:
        return
    dd_fig = go.Figure()
    dd_fig.add_trace(go.Scatter(
        y=dd_series.values * 100, mode='lines',
        fill='tozeroy', line=dict(color='#ff4444', width=1),
        fillcolor='rgba(255,68,68,0.2)', name='Drawdown %',
    ))
    dd_fig.update_layout(
        template="plotly_dark", height=250, yaxis_title="Drawdown %",
    )
    st.plotly_chart(dd_fig, use_container_width=True)


def _render_pnl_distribution(results, t):
    """Render per-trade PnL bar chart."""
    if not results['trades_history']:
        return
    st.subheader(t.get("trade_dist", "Trade PnL Distribution"))
    pnls = [tr['pnl'] for tr in results['trades_history']]
    colors = ['#00ff88' if p > 0 else '#ff4444' for p in pnls]
    hist_fig = go.Figure()
    hist_fig.add_trace(go.Bar(y=pnls, marker_color=colors, name='Trade PnL'))
    hist_fig.update_layout(template="plotly_dark", height=250, yaxis_title="PnL ($)")
    st.plotly_chart(hist_fig, use_container_width=True)


def _render_trade_table(results, t):
    """Render the trade log table with colour-coded PnL."""
    if not results['trades_history']:
        return
    st.subheader(t.get("trade_history", "Trade Log"))
    trade_df = pd.DataFrame(results['trades_history'])
    trade_df[t.get("col_type", "Type")] = trade_df['type'].map({1: 'LONG', -1: 'SHORT'})
    trade_df = trade_df.rename(columns={
        'entry_idx': t.get("col_entry", "Entry"),
        'exit_idx': t.get("col_exit", "Exit"),
        'entry_price': 'Entry Price',
        'exit_price': 'Exit Price',
        'pnl': t.get("col_pnl", "PnL"),
        'bars_held': t.get("col_bars_held", "Bars Held"),
        'exit_reason': t.get("col_exit_reason", "Exit Reason"),
    })
    display_cols = [
        t.get("col_entry", "Entry"), t.get("col_exit", "Exit"),
        'Entry Price', 'Exit Price',
        t.get("col_type", "Type"), t.get("col_pnl", "PnL"),
        t.get("col_bars_held", "Bars Held"),
        t.get("col_exit_reason", "Exit Reason"),
    ]
    existing_cols = [c for c in display_cols if c in trade_df.columns]

    st.dataframe(
        trade_df[existing_cols].style.map(
            lambda v: "color: #00ff88" if isinstance(v, (int, float)) and v > 0
            else ("color: #ff4444" if isinstance(v, (int, float)) and v < 0 else ""),
            subset=[t.get("col_pnl", "PnL")],
        ),
        use_container_width=True,
        height=300,
    )


def _render_monthly_heatmap(results):
    """Monthly returns heatmap — classic quant fund tearsheet element."""
    eq = results['equity_curve']
    if not isinstance(eq.index, pd.DatetimeIndex):
        return
    strat_eq = eq['Strategy_Equity']
    monthly = strat_eq.resample('ME').last().pct_change().dropna()
    if len(monthly) < 2:
        return
    st.subheader("Monthly Returns Heatmap")
    df_m = pd.DataFrame({
        'Year': monthly.index.year,
        'Month': monthly.index.month,
        'Return': monthly.values * 100,
    })
    pivot = df_m.pivot_table(index='Year', columns='Month', values='Return', aggfunc='sum')
    pivot.columns = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][:len(pivot.columns)]
    fig = px.imshow(
        pivot.values, x=pivot.columns.tolist(), y=[str(y) for y in pivot.index],
        color_continuous_scale='RdYlGn', aspect='auto',
        labels={'color': 'Return %'},
    )
    fig.update_layout(template='plotly_dark', height=250)
    st.plotly_chart(fig, use_container_width=True)


def _render_rolling_sharpe(results):
    """Rolling 60-bar Sharpe ratio chart."""
    eq = results['equity_curve']['Strategy_Equity']
    rets = eq.pct_change().dropna()
    if len(rets) < 60:
        return
    st.subheader("Rolling Sharpe Ratio (60-bar)")
    rolling_sharpe = (rets.rolling(60).mean() / (rets.rolling(60).std() + 1e-10)) * np.sqrt(252)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=rolling_sharpe.values, mode='lines',
        line=dict(color='#00ccff', width=1.5), name='Rolling Sharpe',
    ))
    fig.add_hline(y=0, line_dash='dash', line_color='#555')
    fig.add_hline(y=1, line_dash='dot', line_color='#00ff88', annotation_text='Target')
    fig.update_layout(template='plotly_dark', height=250, yaxis_title='Sharpe')
    st.plotly_chart(fig, use_container_width=True)


def _render_monte_carlo(results, initial_capital):
    """Bootstrap Monte Carlo simulation cone chart."""
    trades = results.get('trades_history', [])
    if len(trades) < 10:
        return
    st.subheader("Monte Carlo Simulation (1000 paths)")
    mc = monte_carlo_simulation(trades, initial_capital, n_simulations=1000)
    eq_pct = mc['terminal_equity_percentiles']
    dd_pct = mc['max_drawdown_percentiles']
    c1, c2, c3 = st.columns(3)
    c1.metric("P(Ruin)", f"{mc['ruin_probability']*100:.1f}%")
    c2.metric("Median Terminal", f"${eq_pct.get('50%', 0):,.0f}")
    c3.metric("5th Pctile DD", f"{dd_pct.get('5%', 0)*100:.1f}%")
    pctile_df = pd.DataFrame({
        'Percentile': list(eq_pct.keys()),
        'Terminal Equity': [f"${v:,.0f}" for v in eq_pct.values()],
        'Max Drawdown': [f"{v*100:.1f}%" for v in dd_pct.values()],
    })
    st.dataframe(pctile_df, use_container_width=True, height=200)


def render_pairs_trading_view(lang: str):
    """Render Pairs Trading / Statistical Arbitrage analysis tab."""
    st.subheader("Pairs Trading — Statistical Arbitrage")
    st.markdown("Multi-method cointegration testing (Engle-Granger + ADF) with log-normalized spreads.")

    pair_options = [f"{a} / {b}" for a, b in PAIRS_TRADING_PAIRS]
    selected_pair = st.selectbox("Select Pair", pair_options, key="pairs_select")
    days = st.slider("Lookback (days)", 30, 730, 365, key="pairs_days")

    if st.button("Analyze Pair", key="btn_pairs"):
        idx = pair_options.index(selected_pair)
        mt5_a, mt5_b = PAIRS_TRADING_PAIRS[idx]
        with st.spinner("Fetching data & running cointegration test..."):
            df_a = get_mt5_data(mt5_a, days, timeframe=mt5.TIMEFRAME_H1)
            df_b = get_mt5_data(mt5_b, days, timeframe=mt5.TIMEFRAME_H1)
            if df_a.empty or df_b.empty:
                st.error("Failed to retrieve data for one or both instruments.")
                return
            min_len = min(len(df_a), len(df_b))
            close_a = df_a['Close'].iloc[-min_len:].reset_index(drop=True)
            close_b = df_b['Close'].iloc[-min_len:].reset_index(drop=True)
            result = test_cointegration(close_a, close_b)

            c1, c2, c3, c4 = st.columns(4)
            coint_color = "#00ff88" if result['is_cointegrated'] else "#ff4444"
            c1.markdown(f"<div style='text-align:center'><small style='color:#aaa'>Cointegrated</small><br>"
                        f"<span style='font-size:1.4em;font-weight:bold;color:{coint_color}'>"
                        f"{'YES' if result['is_cointegrated'] else 'NO'}</span></div>",
                        unsafe_allow_html=True)
            c2.metric("Best P-Value", f"{result['p_value']:.4f}")
            c3.metric("Hedge Ratio (β)", f"{result['beta']:.4f}")
            c4.metric("Method", result.get('method', 'N/A'))

            if result.get('adf_p_value') is not None:
                st.caption(f"ADF Spread Stationarity p-value: {result['adf_p_value']:.4f}")

            norm_a = (close_a - close_a.mean()) / (close_a.std() + 1e-10)
            norm_b = (close_b - close_b.mean()) / (close_b.std() + 1e-10)
            fig_norm = go.Figure()
            fig_norm.add_trace(go.Scatter(y=norm_a.values, mode='lines', name=mt5_a,
                                          line=dict(color='#00ccff', width=1.5)))
            fig_norm.add_trace(go.Scatter(y=norm_b.values, mode='lines', name=mt5_b,
                                          line=dict(color='#ffaa00', width=1.5)))
            fig_norm.update_layout(template='plotly_dark', height=250,
                                   title='Normalized Price Overlay (z-score)')
            st.plotly_chart(fig_norm, use_container_width=True)

            fig = go.Figure()
            z = result['z_score'].dropna()
            fig.add_trace(go.Scatter(y=z.values, mode='lines', name='Z-Score',
                                     line=dict(color='#00ccff', width=1.5)))
            fig.add_hline(y=2, line_dash='dash', line_color='#ff4444')
            fig.add_hline(y=-2, line_dash='dash', line_color='#00ff88')
            fig.add_hline(y=0, line_dash='dot', line_color='#555')
            fig.update_layout(template='plotly_dark', height=350, title='Spread Z-Score')
            st.plotly_chart(fig, use_container_width=True)

            spread = result['spread'].dropna()
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(y=spread.values, mode='lines', name='Spread',
                                      line=dict(color='#ffaa00', width=1.5)))
            fig2.update_layout(template='plotly_dark', height=250,
                               title=f'Spread: log({mt5_a}) − β·log({mt5_b})')
            st.plotly_chart(fig2, use_container_width=True)


def render_strategy_comparison_view(lang: str):
    """Render multi-strategy comparison tab."""
    t = load_translations(lang)
    st.subheader("Strategy Leaderboard — Multi-Strategy Comparison")
    st.markdown("Compare all selected strategies on the same instrument and timeframe.")

    selected_name = st.selectbox("Instrument", list(MT5_SYMBOLS.keys()), key="cmp_symbol")
    symbol = MT5_SYMBOLS[selected_name]

    strat_choices = st.multiselect(
        "Select Strategies to Compare",
        list(STRATEGY_REGISTRY.keys()),
        default=list(STRATEGY_REGISTRY.keys())[:4],
        key="cmp_strategies",
    )

    tf_options = {"M15": mt5.TIMEFRAME_M15, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4, "D1": mt5.TIMEFRAME_D1}
    data_mode = st.selectbox("Timeframe", list(tf_options.keys()), index=1, key="cmp_tf")
    days_back = st.slider("History (days)", 30, 2000, 365, key="cmp_days")
    capital = st.number_input("Capital ($)", value=10000.0, key="cmp_capital")
    risk = st.number_input("Risk (%)", value=2.0, key="cmp_risk") / 100.0

    if st.button("Run Comparison", key="btn_compare") and strat_choices:
        mt5_tf = tf_options[data_mode]
        df_raw = get_mt5_data(symbol, days_back, timeframe=mt5_tf)
        if df_raw.empty or len(df_raw) < 50:
            st.error("Failed to retrieve sufficient data.")
            return

        contract = CONTRACT_SIZES.get(symbol, 100000)
        comm_pct = COMMISSION_USD_PER_LOT / contract
        all_results = {}
        progress = st.progress(0)

        for i, sn in enumerate(strat_choices):
            try:
                strat = STRATEGY_REGISTRY[sn]()
                res = run_advanced_backtest(df_raw.copy(), capital, risk, 0.0001, strat, comm_pct, symbol=symbol)
                all_results[sn] = res
            except Exception as exc:
                st.warning(f"{sn}: {exc}")
            progress.progress((i + 1) / len(strat_choices))

        if not all_results:
            return

        rows = []
        for name, res in all_results.items():
            rows.append({
                "Strategy": name, "Return %": f"{res['total_return']*100:.2f}",
                "Sharpe": f"{res['sharpe_ratio']:.2f}", "Sortino": f"{res['sortino_ratio']:.2f}",
                "Calmar": f"{res['calmar_ratio']:.2f}", "Max DD %": f"{res['max_drawdown']*100:.2f}",
                "Win %": f"{res['win_rate']*100:.1f}", "PF": f"{res['profit_factor']:.2f}",
                "Trades": res['n_trades'], "VaR95 %": f"{res.get('var_95',0)*100:.3f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, height=350)

        eq_fig = go.Figure()
        colors = ['#00ff88','#ff4444','#00ccff','#ffaa00','#cc66ff','#ff6699','#66ffcc','#ff9933','#9999ff','#33cc33']
        for i, (name, res) in enumerate(all_results.items()):
            eq = res['equity_curve']['Strategy_Equity']
            eq_fig.add_trace(go.Scatter(y=eq.values, mode='lines', name=name, line=dict(color=colors[i%10], width=2)))
        eq_fig.update_layout(template='plotly_dark', height=450, yaxis_title='Equity ($)')
        st.plotly_chart(eq_fig, use_container_width=True)

        dd_fig = go.Figure()
        for i, (name, res) in enumerate(all_results.items()):
            dd = res.get('drawdown_series')
            if dd is not None:
                dd_fig.add_trace(go.Scatter(y=dd.values*100, mode='lines', name=name, line=dict(color=colors[i%10], width=1.5)))
        dd_fig.update_layout(template='plotly_dark', height=300, yaxis_title='Drawdown %')
        st.plotly_chart(dd_fig, use_container_width=True)


def _colored_metric(
    col,
    label: str,
    value: float,
    fmt: str = "{:.2f}",
    invert: bool = False,
    help: str | None = None,
):
    """Render a colour-coded metric (green for positive, red for negative)."""
    formatted = fmt.format(value)
    good = value < 0 if invert else value > 0
    color = "#00ff88" if good else "#ff4444"
    help_html = f" title='{help}'" if help else ""
    col.markdown(
        f"<div{help_html} style='text-align:center; cursor:default;'>"
        f"<small style='color:#aaa'>{label}</small><br>"
        f"<span style='font-size:1.4em;font-weight:bold;color:{color}'>{formatted}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )
