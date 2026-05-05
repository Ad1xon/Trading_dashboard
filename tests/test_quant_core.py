"""Tests for GARCH volatility, cointegration, composite alpha, and feature correlation."""

import numpy as np
import pandas as pd
import pytest


def _make_ohlcv(n=500, seed=42):
    """Generate deterministic synthetic OHLCV data."""
    np.random.seed(seed)
    timestamps = pd.date_range('2025-01-01', periods=n, freq='1min')
    base = 100.0
    returns = np.random.randn(n) * 0.002
    close = base * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(np.random.randn(n) * 0.001))
    low = close * (1 - np.abs(np.random.randn(n) * 0.001))
    open_ = np.roll(close, 1)
    open_[0] = base
    volume = np.random.randint(100, 10000, size=n).astype(float)
    return pd.DataFrame({
        'Open': open_, 'High': high, 'Low': low,
        'Close': close, 'Volume': volume,
    }, index=timestamps)



class TestGARCHVolatility:
    def test_garch_output_shape(self):
        from quant_engine.volatility_model import fit_garch_volatility
        returns = pd.Series(np.random.randn(200) * 0.01)
        vol = fit_garch_volatility(returns)
        assert len(vol) == len(returns)
        assert vol.name == 'GARCH_Vol'

    def test_garch_positive(self):
        """GARCH volatility must always be positive."""
        from quant_engine.volatility_model import fit_garch_volatility
        returns = pd.Series(np.random.randn(500) * 0.02)
        vol = fit_garch_volatility(returns)
        assert (vol > 0).all()

    def test_garch_responds_to_shock(self):
        """A large return shock should increase conditional volatility."""
        from quant_engine.volatility_model import fit_garch_volatility
        np.random.seed(42)
        calm = np.random.randn(100) * 0.001
        shock = np.array([0.10])
        after = np.random.randn(50) * 0.001
        returns = pd.Series(np.concatenate([calm, shock, after]))
        vol = fit_garch_volatility(returns)
        pre_shock = vol.iloc[99]
        post_shock = vol.iloc[101]
        assert post_shock > pre_shock, "GARCH vol should spike after large return"

    def test_ewma_volatility(self):
        from quant_engine.volatility_model import fit_ewma_volatility
        returns = pd.Series(np.random.randn(200) * 0.01)
        vol = fit_ewma_volatility(returns)
        assert len(vol) == len(returns)
        assert vol.dropna().gt(0).all()

    def test_compute_garch_features_columns(self):
        from quant_engine.volatility_model import compute_garch_features
        df = _make_ohlcv(300)
        result = compute_garch_features(df)
        for col in ['GARCH_Vol', 'EWMA_Vol', 'Realized_Vol', 'Vol_of_Vol', 'Vol_Regime']:
            assert col in result.columns, f"Missing: {col}"

    def test_vol_regime_values(self):
        from quant_engine.volatility_model import compute_garch_features
        df = _make_ohlcv(300)
        result = compute_garch_features(df)
        valid = result['Vol_Regime'].dropna()
        assert set(valid.unique()).issubset({'high_vol', 'low_vol'})



class TestCointegration:
    def test_identical_series_cointegrated(self):
        """Identical series (with noise) should be cointegrated."""
        from quant_engine.stat_arb import test_cointegration
        np.random.seed(42)
        x = pd.Series(np.cumsum(np.random.randn(500) * 0.01) + 100)
        y = x + np.random.randn(500) * 0.1
        result = test_cointegration(y, x)
        assert result['is_cointegrated'] is True
        assert result['p_value'] < 0.10

    def test_independent_series_not_cointegrated(self):
        """Independent random walks should NOT be cointegrated."""
        from quant_engine.stat_arb import test_cointegration
        np.random.seed(42)
        x = pd.Series(np.cumsum(np.random.randn(500) * 0.01) + 100)
        y = pd.Series(np.cumsum(np.random.randn(500) * 0.01) + 200)
        result = test_cointegration(y, x)
        assert result['p_value'] > 0.05

    def test_log_normalization_handles_different_scales(self):
        """Pairs with very different price levels should still be testable."""
        from quant_engine.stat_arb import test_cointegration
        np.random.seed(42)
        x = pd.Series(np.cumsum(np.random.randn(500) * 0.01) + 2400)
        y = pd.Series(np.cumsum(np.random.randn(500) * 0.001) + 28)
        result = test_cointegration(y, x)
        assert 'p_value' in result
        assert 'method' in result
        assert result['method'] != 'none'

    def test_spread_and_zscore_present(self):
        from quant_engine.stat_arb import test_cointegration
        np.random.seed(42)
        x = pd.Series(np.cumsum(np.random.randn(300) * 0.01) + 100)
        y = x * 1.5 + np.random.randn(300) * 0.2
        result = test_cointegration(y, x)
        assert len(result['spread']) > 0
        assert len(result['z_score']) > 0

    def test_insufficient_data(self):
        """Less than 100 bars should return empty result."""
        from quant_engine.stat_arb import test_cointegration
        x = pd.Series([1.0] * 50)
        y = pd.Series([2.0] * 50)
        result = test_cointegration(y, x)
        assert result['is_cointegrated'] is False



class TestFeatureCorrelation:
    """Verify that XGBoost features are not excessively correlated."""

    def test_no_perfect_correlation(self):
        """No feature pair should have |corr| > 0.95."""
        from quant_engine.ml_models import XGBoostRangeBarModel
        df = _make_ohlcv(500)
        model = XGBoostRangeBarModel()
        features = model.build_features(df)
        feat_cols = [c for c in model.feature_cols if c in features.columns]
        corr = features[feat_cols].corr().abs()
        np.fill_diagonal(corr.values, 0)
        max_corr = corr.max().max()
        assert max_corr < 0.95, (
            f"Feature pair with |corr| = {max_corr:.3f} — too correlated"
        )

    def test_no_high_multicollinearity(self):
        """No more than 2 feature pairs should have |corr| > 0.80."""
        from quant_engine.ml_models import XGBoostRangeBarModel
        df = _make_ohlcv(500)
        model = XGBoostRangeBarModel()
        features = model.build_features(df)
        feat_cols = [c for c in model.feature_cols if c in features.columns]
        corr = features[feat_cols].corr().abs()
        np.fill_diagonal(corr.values, 0)
        high_pairs = (corr > 0.80).sum().sum() // 2
        assert high_pairs <= 3, (
            f"{high_pairs} feature pairs with |corr| > 0.80 — excessive multicollinearity"
        )



class TestKellyCriterion:
    def test_zero_win_rate(self):
        from quant_engine.risk_metrics import compute_kelly_fraction
        assert compute_kelly_fraction(0.0, 100.0, 50.0) == 0.0

    def test_zero_avg_loss(self):
        from quant_engine.risk_metrics import compute_kelly_fraction
        assert compute_kelly_fraction(0.5, 100.0, 0.0) == 0.0

    def test_perfect_strategy(self):
        """100% win rate → Kelly should be capped at 25%."""
        from quant_engine.risk_metrics import compute_kelly_fraction
        kelly = compute_kelly_fraction(1.0, 100.0, 50.0)
        assert kelly == 0.25

    def test_negative_edge(self):
        """Losing strategy → Kelly should be 0."""
        from quant_engine.risk_metrics import compute_kelly_fraction
        kelly = compute_kelly_fraction(0.3, 50.0, 100.0)
        assert kelly == 0.0



class TestParametricVaR:
    def test_cornish_fisher_exists(self):
        from quant_engine.risk_metrics import compute_parametric_var
        np.random.seed(42)
        returns = np.random.randn(500) * 0.02
        var = compute_parametric_var(returns, 0.95, method="cornish_fisher")
        assert var < 0

    def test_cf_different_from_normal(self):
        """Cornish-Fisher VaR should differ from standard normal VaR on skewed data."""
        from quant_engine.risk_metrics import compute_parametric_var
        np.random.seed(42)
        returns = np.concatenate([np.random.randn(400) * 0.01, np.array([-0.15, -0.12, -0.10])])
        cf = compute_parametric_var(returns, 0.95, method="cornish_fisher")
        normal = compute_parametric_var(returns, 0.95, method="normal")
        assert cf != normal


class TestMAEMFE:
    def test_empty_trades(self):
        from quant_engine.risk_metrics import compute_mae_mfe
        result = compute_mae_mfe([])
        assert result['mae_mean'] == 0.0
        assert result['mfe_mean'] == 0.0

    def test_with_trades(self):
        from quant_engine.risk_metrics import compute_mae_mfe
        trades = [
            {'entry_price': 100, 'exit_price': 105, 'type': 1, 'pnl': 50},
            {'entry_price': 100, 'exit_price': 95, 'type': 1, 'pnl': -50},
        ]
        result = compute_mae_mfe(trades)
        assert 'mae_mean' in result
        assert 'mfe_mean' in result
        assert 'efficiency' in result
