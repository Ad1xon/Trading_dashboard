"""Strategy optimiser — grid search, walk-forward optimisation, and Monte Carlo simulation."""

import itertools
from typing import Type

import numpy as np
import pandas as pd

from .strategies.base import BaseStrategy
from .backtester import run_advanced_backtest


def grid_search(
    strategy_cls: Type[BaseStrategy],
    trading_data: pd.DataFrame,
    initial_capital: float,
    risk_percent: float,
    slippage: float,
    commission_pct: float,
    param_grid: dict | None = None,
    metric: str = 'sharpe_ratio',
    top_n: int = 10,
) -> list[dict]:
    """Exhaustive parameter grid search ranked by a target metric."""
    if param_grid is None:
        param_grid = _auto_grid(strategy_cls)

    keys = list(param_grid.keys())
    combos = list(itertools.product(*[param_grid[k] for k in keys]))
    results = []

    for combo in combos:
        kwargs = dict(zip(keys, combo))
        try:
            strat = strategy_cls(**kwargs)
            bt = run_advanced_backtest(
                trading_data, initial_capital, risk_percent,
                slippage, strat, commission_pct,
            )
            record = {'params': kwargs}
            record.update({
                k: v for k, v in bt.items()
                if k not in ('equity_curve', 'trades_history', 'drawdown_series')
            })
            results.append(record)
        except Exception:
            continue

    results.sort(key=lambda x: x.get(metric, -999), reverse=True)
    return results[:top_n]


def _auto_grid(strategy_cls: Type[BaseStrategy]) -> dict:
    """Generate a parameter grid from the strategy's ``params`` class attribute."""
    grid = {}
    for name, (default, lo, hi, step) in strategy_cls.params.items():
        if isinstance(default, int):
            grid[name] = list(range(int(lo), int(hi) + 1, int(step)))
        else:
            grid[name] = list(np.arange(lo, hi + step / 2, step))
    return grid


def walk_forward_optimization(
    strategy_cls: Type[BaseStrategy],
    trading_data: pd.DataFrame,
    initial_capital: float,
    risk_percent: float,
    slippage: float,
    commission_pct: float,
    n_splits: int = 5,
    train_ratio: float = 0.7,
    param_grid: dict | None = None,
    metric: str = 'sharpe_ratio',
    indicator_warmup: int = 120,
) -> dict:
    """Walk-forward optimisation with in-sample optimisation and     out-of-sample validation on each fold."""
    if param_grid is None:
        param_grid = _auto_grid(strategy_cls)

    n = len(trading_data)
    fold_size = n // n_splits
    oos_results = []
    best_params_per_fold = []

    for fold in range(n_splits):
        fold_start = fold * fold_size
        fold_end = min((fold + 1) * fold_size, n)
        train_end = fold_start + int((fold_end - fold_start) * train_ratio)

        if train_end - fold_start < 50 or fold_end - train_end < 20:
            continue

        train_data = trading_data.iloc[fold_start:train_end].copy()
        warmup_start = max(fold_start, train_end - indicator_warmup)
        test_data_with_warmup = trading_data.iloc[warmup_start:fold_end].copy()

        is_results = grid_search(
            strategy_cls, train_data, initial_capital,
            risk_percent, slippage, commission_pct,
            param_grid=param_grid, metric=metric, top_n=1,
        )

        if not is_results:
            continue

        best_params = is_results[0]['params']
        best_params_per_fold.append(best_params)

        strat = strategy_cls(**best_params)
        oos_bt = run_advanced_backtest(
            test_data_with_warmup, initial_capital,
            risk_percent, slippage, strat, commission_pct,
        )
        oos_record = {
            k: v for k, v in oos_bt.items()
            if k not in ('equity_curve', 'trades_history', 'drawdown_series')
        }
        oos_record['fold'] = fold
        oos_results.append(oos_record)

    return {
        'oos_results': oos_results,
        'best_params_per_fold': best_params_per_fold,
    }


def monte_carlo_simulation(
    trades_history: list,
    initial_capital: float,
    n_simulations: int = 1000,
    confidence_levels: tuple = (0.05, 0.25, 0.50, 0.75, 0.95),
) -> dict:
    """Bootstrap Monte Carlo simulation of trade-sequence risk."""
    if not trades_history:
        return {
            'terminal_equity_percentiles': {},
            'max_drawdown_percentiles': {},
            'ruin_probability': 0.0,
        }

    pnls = np.array([t['pnl'] for t in trades_history])
    n_trades = len(pnls)
    terminal_equities = np.zeros(n_simulations)
    max_drawdowns = np.zeros(n_simulations)
    ruin_count = 0
    rng = np.random.default_rng(42)

    for sim in range(n_simulations):
        shuffled = rng.choice(pnls, size=n_trades, replace=True)
        equity_path = initial_capital + np.cumsum(shuffled)
        terminal_equities[sim] = equity_path[-1]
        running_max = np.maximum.accumulate(
            np.concatenate([[initial_capital], equity_path]),
        )
        dd = (np.concatenate([[initial_capital], equity_path]) / running_max) - 1
        max_drawdowns[sim] = dd.min()
        if np.any(equity_path <= 0):
            ruin_count += 1

    eq_pct = {
        f'{int(c * 100)}%': np.percentile(terminal_equities, c * 100)
        for c in confidence_levels
    }
    dd_pct = {
        f'{int(c * 100)}%': np.percentile(max_drawdowns, c * 100)
        for c in confidence_levels
    }

    return {
        'terminal_equity_percentiles': eq_pct,
        'max_drawdown_percentiles': dd_pct,
        'ruin_probability': ruin_count / n_simulations,
    }
