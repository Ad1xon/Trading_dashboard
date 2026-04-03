"""Tests for quant_engine.strategies — signal generation, SL/TP columns."""

import numpy as np
import pandas as pd
import pytest

from quant_engine.strategies import (
    ZScoreMeanReversion,
    VolatilityBreakout,
    MLVolatilityBreakout,
    VWAPBounceStrategy,
    MultiTimeframeMomentum,
    STRATEGY_REGISTRY,
    detect_liquidity_sweep,
)


# ═══════════════════════════════════════════════════════════════════════════
# Common contract: every strategy must produce required columns
# ═══════════════════════════════════════════════════════════════════════════

REQUIRED_COLUMNS = ['Signal', 'Exit_Long', 'Exit_Short', 'Std', 'SL_Price', 'Max_Hold']


class TestStrategyContract:
    """All strategies must produce the columns the backtester expects."""

    @pytest.mark.parametrize("strat_cls", [
        ZScoreMeanReversion,
        VolatilityBreakout,
        VWAPBounceStrategy,
        MultiTimeframeMomentum,
    ])
    def test_required_columns(self, synthetic_ohlcv, strat_cls):
        strat = strat_cls()
        result = strat.generate_signals(synthetic_ohlcv.copy())
        for col in REQUIRED_COLUMNS:
            assert col in result.columns, f"{strat_cls.__name__} missing column: {col}"

    @pytest.mark.parametrize("strat_cls", [
        ZScoreMeanReversion,
        VolatilityBreakout,
        VWAPBounceStrategy,
        MultiTimeframeMomentum,
    ])
    def test_signal_values(self, synthetic_ohlcv, strat_cls):
        """Signal column should only contain -1, 0, or 1."""
        strat = strat_cls()
        result = strat.generate_signals(synthetic_ohlcv.copy())
        assert set(result['Signal'].unique()).issubset({-1, 0, 1})


# ═══════════════════════════════════════════════════════════════════════════
# ML strategy (slower, separate test)
# ═══════════════════════════════════════════════════════════════════════════

class TestMLVolatilityBreakout:
    def test_required_columns(self, synthetic_ohlcv):
        strat = MLVolatilityBreakout(lookback=20, prob_threshold=0.55)
        result = strat.generate_signals(synthetic_ohlcv.copy())
        for col in REQUIRED_COLUMNS:
            assert col in result.columns, f"MLVolatilityBreakout missing column: {col}"

    def test_bull_prob_column(self, synthetic_ohlcv):
        strat = MLVolatilityBreakout(lookback=20, prob_threshold=0.55)
        result = strat.generate_signals(synthetic_ohlcv.copy())
        assert 'Bull_Prob' in result.columns
        valid = result['Bull_Prob'].dropna()
        assert (valid >= 0).all() and (valid <= 1).all()


# ═══════════════════════════════════════════════════════════════════════════
# ZScoreMeanReversion specific
# ═══════════════════════════════════════════════════════════════════════════

class TestZScoreMeanReversion:
    def test_extreme_z_triggers_signal(self):
        """Manufacturing a Z-score beyond threshold should trigger a signal."""
        np.random.seed(0)
        n = 200
        # Flat period then a steep drop — enough to push Z below -2.
        close = np.concatenate([np.full(180, 100.0), np.linspace(100, 90, 20)])
        # Volume spikes at the end so the volume-above-avg filter passes
        volume = np.concatenate([np.full(180, 500.0), np.full(20, 2000.0)])
        df = pd.DataFrame({
            'Open': close,
            'High': close + 0.5,
            'Low': close - 0.5,
            'Close': close,
            'Volume': volume,
        }, index=pd.date_range('2025-01-01', periods=n, freq='1min'))

        strat = ZScoreMeanReversion(z_window=20, z_entry=2.0, adx_max=100)
        result = strat.generate_signals(df)
        # The price drop should produce some long signals (Z < -2)
        assert (result['Signal'] == 1).any() or (result['Signal'] == -1).any()

    def test_params_dict(self):
        strat = ZScoreMeanReversion()
        params = strat.get_params()
        assert 'z_window' in params
        assert 'atr_sl_mult' in params
        ranges = strat.get_param_ranges()
        assert len(ranges) > 0


# ═══════════════════════════════════════════════════════════════════════════
# VolatilityBreakout specific
# ═══════════════════════════════════════════════════════════════════════════

class TestVolatilityBreakout:
    def test_breakout_on_new_high(self):
        """A price breaking above the rolling high should trigger long."""
        np.random.seed(1)
        n = 100
        close = np.concatenate([np.full(80, 100.0), np.linspace(100, 110, 20)])
        df = pd.DataFrame({
            'Open': close,
            'High': close + 0.3,
            'Low': close - 0.3,
            'Close': close,
            'Volume': np.full(n, 5000.0),  # high volume to pass filter
        }, index=pd.date_range('2025-01-01', periods=n, freq='1min'))

        strat = VolatilityBreakout(lookback=20, vol_mult=0.5)
        result = strat.generate_signals(df)
        assert (result['Signal'] == 1).any(), "Should detect breakout on new high"


# ═══════════════════════════════════════════════════════════════════════════
# Strategy Registry
# ═══════════════════════════════════════════════════════════════════════════

class TestStrategyRegistry:
    def test_registry_complete(self):
        expected = {'ZScoreMeanReversion', 'VolatilityBreakout', 'MLVolatilityBreakout',
                    'VWAPBounceStrategy', 'MultiTimeframeMomentum'}
        assert set(STRATEGY_REGISTRY.keys()) == expected

    def test_registry_instantiation(self):
        for name, cls in STRATEGY_REGISTRY.items():
            instance = cls()
            assert hasattr(instance, 'generate_signals'), f"{name} missing generate_signals"


# ═══════════════════════════════════════════════════════════════════════════
# Liquidity Sweep
# ═══════════════════════════════════════════════════════════════════════════

class TestLiquiditySweep:
    def test_no_crash_on_small_data(self, small_ohlcv):
        result = detect_liquidity_sweep(small_ohlcv)
        assert 'signal' in result
        assert 'type' in result
