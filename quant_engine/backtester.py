"""
Advanced backtester — bar-by-bar simulation with SL/TP, MFE trailing,
lot cap, margin-call circuit breaker, and comprehensive metrics.
"""

import pandas as pd
import numpy as np
from .strategies.base import BaseStrategy


def run_advanced_backtest(
    trading_data: pd.DataFrame,
    initial_capital: float,
    risk_percent: float,
    slippage: float,
    strategy: BaseStrategy,
    commission_pct: float,
    symbol: str = "EURUSD",
) -> dict:
    """Run an event-driven backtest on OHLCV or range-bar data.

    PnL is tracked exclusively via bar-by-bar equity updates (no
    double-counting).  Supports ATR-based SL/TP, MFE trailing stops,
    and a configurable max-holding period from strategy columns.

    Safety mechanisms:
        * **Lot cap** — position size is capped at ``MAX_ALLOWED_LOTS``
          to prevent absurd leverage when equity grows large.
        * **Margin-call circuit breaker** — if equity drops to zero or
          below, the simulation halts immediately (account bankruptcy).

    Args:
        trading_data:    DataFrame with OHLCV columns (range bars *or*
                         time-based bars).
        initial_capital: Starting account equity in USD.
        risk_percent:    Fraction of equity risked per trade (e.g. 0.02).
        slippage:        Per-trade slippage in price units.
        strategy:        ``BaseStrategy`` subclass instance.
        commission_pct:  Commission expressed as fraction of contract.
        symbol:          Instrument name (used for contract-size lookup).

    Returns:
        Dict with performance metrics, equity curve, and trade log.
    """
    from config import (
        MFE_ACTIVATION_MULTIPLIER, MFE_TRAIL_PCT,
        CONTRACT_SIZES, MAX_ALLOWED_LOTS,
    )

    df = trading_data.copy()
    df = strategy.generate_signals(df)

    n = len(df)
    equity_curve = np.zeros(n)
    equity_curve[0] = initial_capital

    contract_size = CONTRACT_SIZES.get(symbol, 100_000.0)

    closes = df['Close'].values
    highs = df['High'].values
    lows = df['Low'].values
    signals = df['Signal'].values
    exit_longs = df['Exit_Long'].values if 'Exit_Long' in df.columns else np.zeros(n, dtype=bool)
    exit_shorts = df['Exit_Short'].values if 'Exit_Short' in df.columns else np.zeros(n, dtype=bool)
    stds = df['Std'].values if 'Std' in df.columns else np.full(n, np.nan)

    has_sl = 'SL_Price' in df.columns
    has_tp = 'TP_Price' in df.columns
    sl_prices = df['SL_Price'].values if has_sl else np.full(n, np.nan)
    tp_prices = df['TP_Price'].values if has_tp else np.full(n, np.nan)
    max_hold = int(df['Max_Hold'].iloc[0]) if 'Max_Hold' in df.columns else 100

    strat_mfe_activation = strategy.params.get('mfe_activation', [MFE_ACTIVATION_MULTIPLIER])[0]
    strat_mfe_trail_pct = strategy.params.get('mfe_trail_pct', [MFE_TRAIL_PCT])[0]

    current_equity = initial_capital
    current_position = 0
    entry_price = 0.0
    position_size_usd = 0.0
    bars_held = 0
    entry_idx = 0

    high_since_entry = 0.0
    low_since_entry = float('inf')
    dynamic_sl = np.nan

    max_trades = n
    t_entry_idx = np.zeros(max_trades, dtype=int)
    t_exit_idx = np.zeros(max_trades, dtype=int)
    t_entry_price = np.zeros(max_trades)
    t_exit_price = np.zeros(max_trades)
    t_type = np.zeros(max_trades, dtype=int)
    t_pnl = np.zeros(max_trades)
    t_bars_held = np.zeros(max_trades, dtype=int)
    t_exit_reason = np.zeros(max_trades, dtype=object)
    trade_count = 0

    bankrupt = False

    for i in range(1, n):
        if current_equity <= 0:
            current_equity = 0.01
            equity_curve[i:] = 0.01
            bankrupt = True
            break

        sig = signals[i]
        pnl = 0.0

        if current_position != 0:
            price_change_pct = (closes[i] - closes[i - 1]) / (closes[i - 1] + 1e-8)
            pnl = price_change_pct * position_size_usd * current_position
            current_equity += pnl
            bars_held += 1

            high_since_entry = max(high_since_entry, highs[i])
            low_since_entry = min(low_since_entry, lows[i])

            if current_position == 1:
                mfe = high_since_entry - entry_price
                risk_dist = entry_price - sl_prices[entry_idx] if has_sl else entry_price * 0.01
                if risk_dist > 0 and mfe > risk_dist * strat_mfe_activation:
                    candidate = entry_price + mfe * strat_mfe_trail_pct
                    dynamic_sl = max(dynamic_sl, candidate) if not np.isnan(dynamic_sl) else candidate
            else:
                mfe = entry_price - low_since_entry
                risk_dist = sl_prices[entry_idx] - entry_price if has_sl else entry_price * 0.01
                if risk_dist > 0 and mfe > risk_dist * strat_mfe_activation:
                    candidate = entry_price - mfe * strat_mfe_trail_pct
                    dynamic_sl = min(dynamic_sl, candidate) if not np.isnan(dynamic_sl) else candidate

            if current_equity <= 0:
                current_equity = 0.01
                equity_curve[i:] = 0.01
                bankrupt = True
                break

        exit_signal = (
            (current_position == 1 and exit_longs[i])
            or (current_position == -1 and exit_shorts[i])
        )

        sl_hit = False
        tp_hit = False
        active_sl = np.nan
        if current_position != 0:
            active_sl = dynamic_sl if not np.isnan(dynamic_sl) else (sl_prices[entry_idx] if has_sl else np.nan)

            if not np.isnan(active_sl):
                if current_position == 1 and lows[i] <= active_sl:
                    sl_hit = True
                elif current_position == -1 and highs[i] >= active_sl:
                    sl_hit = True

            if has_tp and not np.isnan(tp_prices[entry_idx]):
                if current_position == 1 and highs[i] >= tp_prices[entry_idx]:
                    tp_hit = True
                elif current_position == -1 and lows[i] <= tp_prices[entry_idx]:
                    tp_hit = True

        hold_exit = max_hold > 0 and bars_held >= max_hold
        should_exit = current_position != 0 and (exit_signal or sl_hit or tp_hit or hold_exit)

        if should_exit:
            exit_slip = slippage if current_position == 1 else -slippage

            if sl_hit:
                actual_exit = active_sl - exit_slip
            elif tp_hit:
                actual_exit = tp_prices[entry_idx]
            else:
                actual_exit = closes[i] - exit_slip

            trade_pnl_raw = (
                (actual_exit - entry_price) / (entry_price + 1e-8)
            ) * position_size_usd * current_position

            if contract_size == 100_000:
                nominal_lot_value = contract_size
            else:
                nominal_lot_value = contract_size * entry_price

            lots_traded = position_size_usd / (nominal_lot_value + 1e-8)
            commission_cost = lots_traded * 6.0 * 2

            trade_pnl = trade_pnl_raw - commission_cost
            current_equity += (trade_pnl - pnl)

            if sl_hit and not np.isnan(dynamic_sl) and active_sl == dynamic_sl:
                exit_reason = 'MFE_TRAIL'
            elif sl_hit:
                exit_reason = 'SL'
            elif tp_hit:
                exit_reason = 'TP'
            elif hold_exit:
                exit_reason = 'MAX_HOLD'
            else:
                exit_reason = 'SIGNAL'

            t_entry_idx[trade_count] = entry_idx
            t_exit_idx[trade_count] = i
            t_entry_price[trade_count] = entry_price
            t_exit_price[trade_count] = actual_exit
            t_type[trade_count] = current_position
            t_pnl[trade_count] = trade_pnl
            t_bars_held[trade_count] = bars_held
            t_exit_reason[trade_count] = exit_reason
            trade_count += 1

            current_position = 0
            position_size_usd = 0.0
            bars_held = 0
            dynamic_sl = np.nan

        if current_position == 0 and sig != 0:
            entry_idx = i
            real_slippage = slippage if sig == 1 else -slippage
            entry_price = closes[i] + real_slippage
            vol = stds[i] if not np.isnan(stds[i]) and stds[i] > 0 else (closes[i] * 0.001)
            risk_amt = current_equity * risk_percent
            calculated_usd = risk_amt / ((vol * 2) / (entry_price + 1e-8) + 1e-6)

            if contract_size == 100_000:
                nominal_lot_value = contract_size
            else:
                nominal_lot_value = contract_size * entry_price

            calculated_lots = calculated_usd / (nominal_lot_value + 1e-8)
            capped_lots = min(calculated_lots, MAX_ALLOWED_LOTS)
            position_size_usd = capped_lots * nominal_lot_value

            current_position = sig
            bars_held = 0
            high_since_entry = highs[i]
            low_since_entry = lows[i]
            dynamic_sl = np.nan

        equity_curve[i] = current_equity

    trades_history = []
    index_vals = df.index.values
    for k in range(trade_count):
        trades_history.append({
            'entry_idx': index_vals[t_entry_idx[k]],
            'exit_idx': index_vals[t_exit_idx[k]],
            'entry_price': t_entry_price[k],
            'exit_price': t_exit_price[k],
            'type': t_type[k],
            'pnl': t_pnl[k],
            'bars_held': t_bars_held[k],
            'exit_reason': t_exit_reason[k],
        })

    df['Strategy_Equity'] = equity_curve
    df['Market_Return'] = df['Close'].pct_change()
    df['BuyHold_Equity'] = initial_capital * (1 + df['Market_Return'].fillna(0)).cumprod()

    metrics = _compute_metrics(equity_curve, initial_capital, trades_history)
    metrics['equity_curve'] = df[['Strategy_Equity', 'BuyHold_Equity']]
    metrics['trades_history'] = trades_history
    metrics['bankrupt'] = bankrupt
    return metrics


def _compute_metrics(
    equity_curve: np.ndarray,
    initial_capital: float,
    trades: list,
) -> dict:
    """Compute risk-adjusted performance metrics.

    Includes Sharpe, Sortino, Calmar, Win Rate, Profit Factor,
    max consecutive losses, and annualised return.
    """
    total_return = (equity_curve[-1] / initial_capital) - 1 if equity_curve[-1] > 0 else -1.0

    eq_series = pd.Series(equity_curve)
    running_max = eq_series.cummax()
    drawdown = (eq_series / running_max) - 1
    max_drawdown = drawdown.min()

    eq_nonzero = eq_series.replace(0, np.nan)
    bar_returns = eq_nonzero.pct_change().dropna()

    from config import BARS_PER_YEAR
    bars_per_year = BARS_PER_YEAR

    sharpe = (
        (bar_returns.mean() / (bar_returns.std() + 1e-10)) * np.sqrt(bars_per_year)
        if len(bar_returns) > 1 else 0.0
    )

    downside = bar_returns[bar_returns < 0]
    sortino = (
        (bar_returns.mean() / (downside.std() + 1e-10)) * np.sqrt(bars_per_year)
        if len(downside) > 1 else 0.0
    )

    n_bars = max(len(equity_curve), 1)
    ann_return = (1 + total_return) ** (bars_per_year / n_bars) - 1
    calmar = ann_return / (abs(max_drawdown) + 1e-10)

    pnls = [t['pnl'] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    n_trades = len(pnls)
    win_rate = len(wins) / n_trades if n_trades > 0 else 0.0
    avg_win = np.mean(wins) if wins else 0.0
    avg_loss = abs(np.mean(losses)) if losses else 0.0
    avg_win_loss_ratio = avg_win / (avg_loss + 1e-10) if avg_loss > 0 else 0.0
    profit_factor = sum(wins) / (abs(sum(losses)) + 1e-10) if losses else float('inf')

    return {
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'drawdown_series': drawdown,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'calmar_ratio': calmar,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'avg_win_loss_ratio': avg_win_loss_ratio,
        'profit_factor': profit_factor,
        'n_trades': n_trades,
        'max_consecutive_losses': _max_consecutive_losses(pnls),
    }


def _max_consecutive_losses(pnls: list) -> int:
    """Count the longest streak of consecutive losing trades."""
    max_streak = 0
    current_streak = 0
    for p in pnls:
        if p <= 0:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0
    return max_streak