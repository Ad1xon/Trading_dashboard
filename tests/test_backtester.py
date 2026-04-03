"""Tests for quant_engine.backtester — PnL correctness, metrics, SL/TP."""

import numpy as np
import pandas as pd
import pytest

from quant_engine.backtester import run_advanced_backtest, _max_consecutive_losses
from quant_engine.strategies import ZScoreMeanReversion, VolatilityBreakout



class TestPnLCorrectness:
    def test_no_trades_return_zero(self):
        """With no signals, total return should be ~0 (minus any tiny float drift)."""
        np.random.seed(99)
        n = 100
        df = pd.DataFrame({
            'Open':   np.full(n, 100.0),
            'High':   np.full(n, 101.0),
            'Low':    np.full(n, 99.0),
            'Close':  np.full(n, 100.0),
            'Volume': np.full(n, 1000.0),
        }, index=pd.date_range('2025-01-01', periods=n, freq='1min'))

        class NoSignalStrategy:
            params = {}
            def generate_signals(self, df):
                df['Signal'] = 0
                df['Exit_Long'] = False
                df['Exit_Short'] = False
                df['Std'] = 1.0
                return df

        results = run_advanced_backtest(df, 10000.0, 0.02, 0.0, NoSignalStrategy(), 0.0)
        assert abs(results['total_return']) < 1e-6, "No trades → zero return"
        assert results['n_trades'] == 0

    def test_equity_decreases_on_losing_long(self):
        """A long trade on a falling price should lose money."""
        n = 50
        prices = np.linspace(100, 90, n)  
        df = pd.DataFrame({
            'Open': prices,
            'High': prices + 0.5,
            'Low': prices - 0.5,
            'Close': prices,
            'Volume': np.full(n, 1000.0),
        }, index=pd.date_range('2025-01-01', periods=n, freq='1min'))

        class AlwaysLong:
            params = {}
            def generate_signals(self, df):
                df['Signal'] = 0
                df.iloc[1, df.columns.get_loc('Signal')] = 1  
                df['Exit_Long'] = False
                df.iloc[-1, df.columns.get_loc('Exit_Long')] = True  
                df['Exit_Short'] = False
                df['Std'] = 1.0
                return df

        results = run_advanced_backtest(df, 10000.0, 0.02, 0.0, AlwaysLong(), 0.0)
        assert results['total_return'] < 0, "Long on falling price should lose"
        assert results['n_trades'] == 1



class TestMetrics:
    def test_metrics_present(self, range_bars):
        if len(range_bars) < 50:
            pytest.skip("Not enough range bars")
        strat = ZScoreMeanReversion()
        results = run_advanced_backtest(range_bars, 10000.0, 0.02, 0.0001, strat, 0.00006)
        expected_keys = [
            'total_return', 'max_drawdown', 'sharpe_ratio', 'sortino_ratio',
            'calmar_ratio', 'win_rate', 'avg_win', 'avg_loss',
            'avg_win_loss_ratio', 'profit_factor', 'n_trades',
            'max_consecutive_losses', 'equity_curve', 'trades_history',
            'drawdown_series',
        ]
        for key in expected_keys:
            assert key in results, f"Missing metric: {key}"

    def test_max_drawdown_negative(self, range_bars):
        if len(range_bars) < 50:
            pytest.skip("Not enough range bars")
        strat = ZScoreMeanReversion()
        results = run_advanced_backtest(range_bars, 10000.0, 0.02, 0.0001, strat, 0.00006)
        assert results['max_drawdown'] <= 0, "Max drawdown should be <= 0"

    def test_win_rate_bounds(self, range_bars):
        if len(range_bars) < 50:
            pytest.skip("Not enough range bars")
        strat = VolatilityBreakout()
        results = run_advanced_backtest(range_bars, 10000.0, 0.02, 0.0001, strat, 0.00006)
        assert 0.0 <= results['win_rate'] <= 1.0



class TestMaxConsecutiveLosses:
    def test_empty(self):
        assert _max_consecutive_losses([]) == 0

    def test_all_wins(self):
        assert _max_consecutive_losses([10, 20, 30]) == 0

    def test_all_losses(self):
        assert _max_consecutive_losses([-5, -3, -10]) == 3

    def test_mixed(self):
        assert _max_consecutive_losses([10, -5, -3, 20, -1, -2, -4, 5]) == 3



class TestSLTP:
    def test_trades_have_exit_reason(self, range_bars):
        if len(range_bars) < 50:
            pytest.skip("Not enough range bars")
        strat = ZScoreMeanReversion()
        results = run_advanced_backtest(range_bars, 10000.0, 0.02, 0.0001, strat, 0.00006)
        for trade in results['trades_history']:
            assert 'exit_reason' in trade
            assert trade['exit_reason'] in ('SIGNAL', 'SL', 'TP', 'MAX_HOLD', 'MFE_TRAIL')
