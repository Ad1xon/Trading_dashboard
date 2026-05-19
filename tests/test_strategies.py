"""Tests for quant_engine.strategies — signal generation, SL/TP columns, registry."""

import numpy as np
import pandas as pd
import pytest

from quant_engine.strategies import (
    ZScoreMeanReversion,
    VolatilityBreakout,
    MLVolatilityBreakout,
    VWAPBounceStrategy,
    MultiTimeframeMomentum,
    PairsTradingStrategy,
    RegimeSwitchStrategy,
    CompositeAlphaStrategy,
    STRATEGY_REGISTRY,
    detect_liquidity_sweep,
)


REQUIRED_COLUMNS = ['Signal', 'Exit_Long', 'Exit_Short', 'Std', 'SL_Price', 'Max_Hold']


class TestStrategyContract:
    """All strategies must produce the columns the backtester expects."""

    @pytest.mark.parametrize("strat_cls", [
        ZScoreMeanReversion,
        VolatilityBreakout,
        VWAPBounceStrategy,
        MultiTimeframeMomentum,
        CompositeAlphaStrategy,
        RegimeSwitchStrategy,
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
        CompositeAlphaStrategy,
        RegimeSwitchStrategy,
    ])
    def test_signal_values(self, synthetic_ohlcv, strat_cls):
        """Signal column should only contain -1, 0, or 1."""
        strat = strat_cls()
        result = strat.generate_signals(synthetic_ohlcv.copy())
        assert set(result['Signal'].unique()).issubset({-1, 0, 1})


class TestCompositeAlpha:
    """Composite Alpha must generate trades — previously zero due to threshold bug."""

    def test_generates_signals(self, synthetic_ohlcv):
        strat = CompositeAlphaStrategy()
        result = strat.generate_signals(synthetic_ohlcv.copy())
        n_signals = (result['Signal'] != 0).sum()
        assert n_signals > 0, "Composite Alpha should generate at least some signals"

    def test_composite_score_column(self, synthetic_ohlcv):
        strat = CompositeAlphaStrategy()
        result = strat.generate_signals(synthetic_ohlcv.copy())
        assert 'Composite_Score' in result.columns
        assert result['Composite_Score'].notna().sum() > 0

    def test_garch_vol_present(self, synthetic_ohlcv):
        strat = CompositeAlphaStrategy()
        result = strat.generate_signals(synthetic_ohlcv.copy())
        assert 'GARCH_Vol' in result.columns

    def test_sl_tp_set_on_signals(self, synthetic_ohlcv):
        strat = CompositeAlphaStrategy()
        result = strat.generate_signals(synthetic_ohlcv.copy())
        has_signal = result['Signal'] != 0
        if has_signal.any():
            sl_set = result.loc[has_signal, 'SL_Price'].notna().any()
            assert sl_set, "SL should be set on signal bars"


class TestRegimeSwitch:
    def test_regime_column_added(self, synthetic_ohlcv):
        strat = RegimeSwitchStrategy()
        result = strat.generate_signals(synthetic_ohlcv.copy())
        assert 'Regime' in result.columns
        assert set(result['Regime'].unique()).issubset({'bear', 'range', 'bull'})


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


class TestZScoreMeanReversion:
    def test_extreme_z_triggers_signal(self):
        """Manufacturing a Z-score beyond threshold should trigger a signal."""
        np.random.seed(0)
        n = 200
        close = np.concatenate([np.full(180, 100.0), np.linspace(100, 90, 20)])
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
        assert (result['Signal'] == 1).any() or (result['Signal'] == -1).any()

    def test_params_dict(self):
        strat = ZScoreMeanReversion()
        params = strat.get_params()
        assert 'z_window' in params
        assert 'atr_sl_mult' in params
        ranges = strat.get_param_ranges()
        assert len(ranges) > 0


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
            'Volume': np.full(n, 5000.0),
        }, index=pd.date_range('2025-01-01', periods=n, freq='1min'))

        strat = VolatilityBreakout(lookback=20, vol_mult=0.5)
        result = strat.generate_signals(df)
        assert (result['Signal'] == 1).any(), "Should detect breakout on new high"


class TestStrategyRegistry:
    def test_registry_complete(self):
        expected = {
            'Ultimate MFT', 'Ultimate Swing',
            'Composite Alpha (MFT)', 'LSTM Swing', 'Regime Switch (HMM)',
            'XGB Breakout (ML)', 'LGBM Arab Scalp (ML)', 'Pairs Trading (Stat Arb)',
            'MTF Momentum', 'VWAP Bounce', 'SMC Breakout', 'ZScore Rev',
        }
        assert set(STRATEGY_REGISTRY.keys()) == expected

    def test_registry_instantiation(self):
        for name, cls in STRATEGY_REGISTRY.items():
            if 'LSTM' in name:
                try:
                    instance = cls()
                except TypeError:
                    pytest.skip(f"{name} requires torch")
                    continue
            else:
                instance = cls()
            assert hasattr(instance, 'generate_signals'), f"{name} missing generate_signals"


class TestLiquiditySweep:
    def test_no_crash_on_small_data(self, small_ohlcv):
        result = detect_liquidity_sweep(small_ohlcv)
        assert 'signal' in result
        assert 'type' in result
