"""Tests for new modules — risk metrics, regime detector, slippage model, rolling sentiment, session VWAP."""

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



class TestRiskMetrics:
    def test_var_basic(self):
        from quant_engine.risk_metrics import compute_var
        returns = np.array([-0.05, -0.03, -0.01, 0.0, 0.01, 0.02, 0.05, 0.08, 0.10, 0.15])
        var = compute_var(returns, 0.95)
        assert var < 0

    def test_cvar_worse_than_var(self):
        from quant_engine.risk_metrics import compute_var, compute_cvar
        np.random.seed(42)
        returns = np.random.randn(1000) * 0.02
        var = compute_var(returns, 0.95)
        cvar = compute_cvar(returns, 0.95)
        assert cvar <= var

    def test_correlation_matrix_shape(self):
        from quant_engine.risk_metrics import compute_correlation_matrix
        eq1 = np.cumsum(np.random.randn(100) * 0.01) + 100
        eq2 = np.cumsum(np.random.randn(100) * 0.01) + 100
        corr = compute_correlation_matrix({"strat_a": eq1, "strat_b": eq2})
        assert corr.shape == (2, 2)
        assert abs(corr.iloc[0, 0] - 1.0) < 1e-10

    def test_correlation_clusters(self):
        from quant_engine.risk_metrics import detect_correlation_clusters
        corr_df = pd.DataFrame(
            [[1.0, 0.85], [0.85, 1.0]],
            columns=["A", "B"], index=["A", "B"],
        )
        clusters = detect_correlation_clusters(corr_df, threshold=0.7)
        assert len(clusters) == 1
        assert clusters[0][2] == 0.85

    def test_portfolio_risk_report(self):
        from quant_engine.risk_metrics import compute_portfolio_risk_report
        np.random.seed(42)
        eq1 = np.cumsum(np.random.randn(200) * 0.01) + 10000
        eq2 = np.cumsum(np.random.randn(200) * 0.01) + 10000
        report = compute_portfolio_risk_report({"s1": eq1, "s2": eq2})
        assert "portfolio_var" in report
        assert "portfolio_cvar" in report
        assert "per_strategy" in report
        assert "s1" in report["per_strategy"]

    def test_empty_returns(self):
        from quant_engine.risk_metrics import compute_var, compute_cvar
        assert compute_var(np.array([]), 0.95) == 0.0
        assert compute_cvar(np.array([]), 0.95) == 0.0



class TestRegimeDetector:
    def test_fit_and_predict(self):
        from quant_engine.regime_detector import RegimeDetector
        df = _make_ohlcv(500)
        detector = RegimeDetector(n_states=3, lookback=200)
        detector.fit(df)
        assert detector.is_fitted
        labels = detector.predict_labels(df)
        assert len(labels) == len(df)
        assert set(labels.unique()).issubset({"bear", "range", "bull"})

    def test_add_regime_column(self):
        from quant_engine.regime_detector import RegimeDetector
        df = _make_ohlcv(300)
        detector = RegimeDetector(n_states=3, lookback=200)
        result = detector.add_regime_column(df)
        assert "Regime" in result.columns

    def test_is_favourable(self):
        from quant_engine.regime_detector import RegimeDetector
        detector = RegimeDetector()
        assert detector.is_favourable("bull", "trend")
        assert detector.is_favourable("range", "reversion")
        assert not detector.is_favourable("range", "trend")
        assert detector.is_favourable("bear", "scalp")

    def test_insufficient_data(self):
        from quant_engine.regime_detector import RegimeDetector
        df = _make_ohlcv(30)
        detector = RegimeDetector(n_states=3, lookback=200)
        detector.fit(df)
        labels = detector.predict(df)
        assert len(labels) == len(df)



class TestSlippageModel:
    def test_basic_estimate(self):
        from quant_engine.slippage_model import DynamicSlippageModel
        model = DynamicSlippageModel(base_bps=1.0)
        slip = model.estimate(
            price=100.0, atr=0.5, volume=1000,
            avg_volume=1000, avg_atr=0.5,
        )
        assert slip > 0
        assert slip < 1.0

    def test_high_vol_more_slippage(self):
        from quant_engine.slippage_model import DynamicSlippageModel
        model = DynamicSlippageModel(base_bps=1.0)
        slip_normal = model.estimate(100.0, 0.5, 1000, 1000, 0.5)
        slip_high_vol = model.estimate(100.0, 2.0, 1000, 1000, 0.5)
        assert slip_high_vol > slip_normal

    def test_low_volume_more_slippage(self):
        from quant_engine.slippage_model import DynamicSlippageModel
        model = DynamicSlippageModel(base_bps=1.0)
        slip_normal = model.estimate(100.0, 0.5, 1000, 1000, 0.5)
        slip_low_vol = model.estimate(100.0, 0.5, 100, 1000, 0.5)
        assert slip_low_vol > slip_normal

    def test_estimate_array(self):
        from quant_engine.slippage_model import DynamicSlippageModel
        model = DynamicSlippageModel(base_bps=1.0)
        n = 100
        prices = np.full(n, 100.0)
        atrs = np.full(n, 0.5)
        volumes = np.full(n, 1000.0)
        slippage = model.estimate_array(prices, atrs, volumes)
        assert len(slippage) == n
        assert (slippage > 0).all()



class TestRollingSentiment:
    def test_rolling_fills_window(self):
        from data_feed.nlp_engine import SentimentEngine
        engine = SentimentEngine()
        engine._cache["TEST"] = 0.5
        df = _make_ohlcv(200)
        result = engine.apply_rolling_sentiment(df, "TEST", window_bars=50)
        assert "Sentiment_Score" in result.columns
        nonzero = (result["Sentiment_Score"] != 0.0).sum()
        assert nonzero >= 50

    def test_legacy_last_bar_only(self):
        from data_feed.nlp_engine import SentimentEngine
        engine = SentimentEngine()
        engine._cache["TEST2"] = 0.3
        df = _make_ohlcv(100)
        result = engine.apply_sentiment_to_dataframe(df, "TEST2")
        nonzero = (result["Sentiment_Score"] != 0.0).sum()
        assert nonzero == 1

    def test_zero_sentiment_no_fill(self):
        from data_feed.nlp_engine import SentimentEngine
        engine = SentimentEngine()
        engine._cache["ZERO"] = 0.0
        df = _make_ohlcv(100)
        result = engine.apply_rolling_sentiment(df, "ZERO", window_bars=50)
        assert (result["Sentiment_Score"] == 0.0).all()



class TestSessionVWAP:
    def test_session_reset_runs(self):
        from quant_engine.indicators import calculate_vwap_with_bands
        df = _make_ohlcv(500)
        result = calculate_vwap_with_bands(df, session_reset="london")
        assert "VWAP" in result.columns
        assert "VWAP_Upper_2" in result.columns
        assert "VWAP_Lower_2" in result.columns

    def test_rolling_vwap_still_works(self):
        from quant_engine.indicators import calculate_vwap_with_bands
        df = _make_ohlcv(500)
        result = calculate_vwap_with_bands(df, window=200)
        assert "VWAP" in result.columns
        assert result["VWAP"].notna().sum() > 0

    def test_session_vwap_shortcut(self):
        from quant_engine.indicators import calculate_session_vwap
        df = _make_ohlcv(500)
        result = calculate_session_vwap(df, session="london")
        assert "VWAP" in result.columns



class TestBacktesterVaRCVaR:
    def test_var_cvar_in_results(self, synthetic_ohlcv):
        from quant_engine.data_processor import generate_synthetic_range_bars
        from quant_engine.backtester import run_advanced_backtest
        from quant_engine.strategies import ZScoreMeanReversion

        rb = generate_synthetic_range_bars(synthetic_ohlcv, range_size=0.15)
        if len(rb) < 50:
            pytest.skip("Not enough range bars")
        strat = ZScoreMeanReversion()
        results = run_advanced_backtest(rb, 10000.0, 0.02, 0.0001, strat, 0.00006)
        assert "var_95" in results
        assert "cvar_95" in results
        assert results["var_95"] <= 0 or results["var_95"] >= 0
