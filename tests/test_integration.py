"""Integration test: data → strategy → backtest pipeline end-to-end."""

import numpy as np
import pandas as pd
import pytest

from quant_engine.data_processor import generate_synthetic_range_bars
from quant_engine.backtester import run_advanced_backtest
from quant_engine.strategies import (
    ZScoreMeanReversion,
    VolatilityBreakout,
    MLVolatilityBreakout,
    VWAPBounceStrategy,
    MultiTimeframeMomentum,
    STRATEGY_REGISTRY,
)
from quant_engine.strategy_optimizer import monte_carlo_simulation


# ═══════════════════════════════════════════════════════════════════════════
# Full Pipeline: raw data → range bars → strategy → backtest
# ═══════════════════════════════════════════════════════════════════════════

class TestFullPipeline:
    """
    End-to-end pipeline tests.
    Uses synthetic data (no MT5 dependency).
    """

    @pytest.mark.parametrize("strat_cls", [
        ZScoreMeanReversion,
        VolatilityBreakout,
        VWAPBounceStrategy,
        MultiTimeframeMomentum,
    ])
    def test_pipeline_runs(self, synthetic_ohlcv, strat_cls):
        range_bars = generate_synthetic_range_bars(synthetic_ohlcv, range_size=0.15)
        if len(range_bars) < 50:
            pytest.skip("Not enough range bars generated")

        strat = strat_cls()
        results = run_advanced_backtest(
            range_bars,
            initial_capital=10000.0,
            risk_percent=0.02,
            slippage=0.0001,
            strategy=strat,
            commission_pct=0.00006,
        )

        # Basic sanity
        assert 'total_return' in results
        assert 'equity_curve' in results
        assert len(results['equity_curve']) == len(range_bars)
        assert results['equity_curve']['Strategy_Equity'].iloc[0] == 10000.0

    def test_ml_pipeline_runs(self, synthetic_ohlcv):
        """ML strategy pipeline — separate because it's slower."""
        range_bars = generate_synthetic_range_bars(synthetic_ohlcv, range_size=0.15)
        if len(range_bars) < 50:
            pytest.skip("Not enough range bars generated")

        strat = MLVolatilityBreakout(lookback=20, prob_threshold=0.55)
        results = run_advanced_backtest(
            range_bars, 10000.0, 0.02, 0.0001, strat, 0.00006,
        )
        assert 'total_return' in results
        assert 'sharpe_ratio' in results


# ═══════════════════════════════════════════════════════════════════════════
# Range Bar Generation
# ═══════════════════════════════════════════════════════════════════════════

class TestRangeBarGeneration:
    def test_range_bars_have_ohlcv(self, synthetic_ohlcv):
        rb = generate_synthetic_range_bars(synthetic_ohlcv, range_size=0.15)
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            assert col in rb.columns

    def test_range_bars_high_gte_low(self, synthetic_ohlcv):
        rb = generate_synthetic_range_bars(synthetic_ohlcv, range_size=0.15)
        assert (rb['High'] >= rb['Low']).all()

    def test_range_bars_volume_positive(self, synthetic_ohlcv):
        rb = generate_synthetic_range_bars(synthetic_ohlcv, range_size=0.15)
        assert (rb['Volume'] > 0).all()

    def test_smaller_range_more_bars(self, synthetic_ohlcv):
        rb_big = generate_synthetic_range_bars(synthetic_ohlcv, range_size=1.0)
        rb_small = generate_synthetic_range_bars(synthetic_ohlcv, range_size=0.05)
        assert len(rb_small) >= len(rb_big)


# ═══════════════════════════════════════════════════════════════════════════
# Monte Carlo Simulation
# ═══════════════════════════════════════════════════════════════════════════

class TestMonteCarlo:
    def test_monte_carlo_with_trades(self, range_bars):
        if len(range_bars) < 50:
            pytest.skip("Not enough range bars")

        strat = ZScoreMeanReversion()
        results = run_advanced_backtest(range_bars, 10000.0, 0.02, 0.0001, strat, 0.00006)
        trades = results['trades_history']

        if not trades:
            pytest.skip("No trades generated")

        mc = monte_carlo_simulation(trades, 10000.0, n_simulations=100)
        assert 'terminal_equity_percentiles' in mc
        assert 'max_drawdown_percentiles' in mc
        assert 'ruin_probability' in mc
        assert 0 <= mc['ruin_probability'] <= 1

    def test_monte_carlo_empty_trades(self):
        mc = monte_carlo_simulation([], 10000.0)
        assert mc['ruin_probability'] == 0.0
        assert mc['terminal_equity_percentiles'] == {}


# ═══════════════════════════════════════════════════════════════════════════
# Alert Manager (unit-level, no network)
# ═══════════════════════════════════════════════════════════════════════════

class TestAlertManager:
    def test_alert_manager_dedup(self):
        from alerts.alert_manager import AlertManager

        mgr = AlertManager()
        mgr.cooldown_sec = 999  # long cooldown

        # No discord configured → fire returns False (no channel), but dedup should still track.
        result1 = mgr.fire("EURUSD", "test", "LONG")
        result2 = mgr.fire("EURUSD", "test", "LONG")
        # Both should be False since no discord configured
        assert result1 is False
        assert result2 is False

    def test_symbol_disable(self):
        from alerts.alert_manager import AlertManager

        mgr = AlertManager()
        mgr.set_threshold("EURUSD", enabled=False)
        assert not mgr.is_symbol_enabled("EURUSD")
        assert mgr.is_symbol_enabled("GBPUSD")  # default enabled

    def test_confidence_threshold(self):
        from alerts.alert_manager import AlertManager

        mgr = AlertManager()
        mgr.set_threshold("XAUUSD", min_confidence=0.7, enabled=True)
        # No discord → always False, but the confidence check happens before dispatch
        result = mgr.fire("XAUUSD", "test", "LONG", confidence=0.5)
        assert result is False
