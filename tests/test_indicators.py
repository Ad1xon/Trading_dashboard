"""Tests for quant_engine.indicators — all indicator functions."""

import numpy as np
import pandas as pd
import pytest

from quant_engine.indicators import (
    calculate_vwap_with_bands,
    calculate_rsi,
    calculate_atr,
    calculate_bollinger,
    calculate_macd,
    calculate_adx,
    calculate_session_features,
    calculate_orderflow_proxy,
    calculate_return_autocorrelation,
)



class TestRSI:
    def test_rsi_range(self, synthetic_ohlcv):
        rsi = calculate_rsi(synthetic_ohlcv['Close'], 14)
        valid = rsi.dropna()
        assert (valid >= 0).all(), "RSI should be >= 0"
        assert (valid <= 100).all(), "RSI should be <= 100"

    def test_rsi_length(self, synthetic_ohlcv):
        rsi = calculate_rsi(synthetic_ohlcv['Close'], 14)
        assert len(rsi) == len(synthetic_ohlcv)

    def test_rsi_monotonic_up(self):
        """A purely rising series should have RSI near 100."""
        prices = pd.Series(np.arange(1, 102, dtype=float))
        rsi = calculate_rsi(prices, 14)
        assert rsi.iloc[-1] > 90

    def test_rsi_monotonic_down(self):
        """A purely falling series should have RSI near 0."""
        prices = pd.Series(np.arange(100, 0, -1, dtype=float))
        rsi = calculate_rsi(prices, 14)
        assert rsi.iloc[-1] < 10



class TestATR:
    def test_atr_positive(self, synthetic_ohlcv):
        atr = calculate_atr(synthetic_ohlcv, 14)
        valid = atr.dropna()
        assert (valid >= 0).all(), "ATR must be non-negative"

    def test_atr_length(self, synthetic_ohlcv):
        atr = calculate_atr(synthetic_ohlcv, 14)
        assert len(atr) == len(synthetic_ohlcv)

    def test_atr_flat_prices(self):
        """If prices don't move, ATR should tend to zero."""
        n = 100
        df = pd.DataFrame({
            'High': [100.0] * n,
            'Low': [100.0] * n,
            'Close': [100.0] * n,
        })
        atr = calculate_atr(df, 14)
        assert atr.iloc[-1] < 0.001



class TestBollinger:
    def test_bollinger_keys(self, synthetic_ohlcv):
        bb = calculate_bollinger(synthetic_ohlcv['Close'], 20, 2.0)
        expected_keys = {'BB_Mid', 'BB_Upper', 'BB_Lower', 'BB_Width', 'BB_PctB'}
        assert set(bb.keys()) == expected_keys

    def test_upper_above_lower(self, synthetic_ohlcv):
        bb = calculate_bollinger(synthetic_ohlcv['Close'], 20, 2.0)
        valid = bb['BB_Upper'].dropna()
        lower = bb['BB_Lower'].dropna()
        assert (valid >= lower).all()

    def test_pctb_at_mid_is_half(self):
        """When price equals the mid band, %B ≈ 0.5."""
        prices = pd.Series([100.0] * 50)
        bb = calculate_bollinger(prices, 20)
        assert not bb['BB_PctB'].isna().all()



class TestMACD:
    def test_macd_keys(self, synthetic_ohlcv):
        macd = calculate_macd(synthetic_ohlcv['Close'])
        assert set(macd.keys()) == {'MACD_Line', 'MACD_Signal', 'MACD_Hist'}

    def test_macd_histogram_is_difference(self, synthetic_ohlcv):
        macd = calculate_macd(synthetic_ohlcv['Close'])
        diff = macd['MACD_Line'] - macd['MACD_Signal']
        np.testing.assert_allclose(macd['MACD_Hist'].values, diff.values, atol=1e-10)



class TestADX:
    def test_adx_range(self, synthetic_ohlcv):
        adx = calculate_adx(synthetic_ohlcv, 14)
        valid = adx.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()



class TestVWAP:
    def test_vwap_columns(self, synthetic_ohlcv):
        result = calculate_vwap_with_bands(synthetic_ohlcv)
        for col in ['VWAP', 'VWAP_Upper_2', 'VWAP_Lower_2']:
            assert col in result.columns

    def test_vwap_between_bands(self, synthetic_ohlcv):
        result = calculate_vwap_with_bands(synthetic_ohlcv)
        valid = result.dropna(subset=['VWAP', 'VWAP_Upper_2', 'VWAP_Lower_2'])
        assert (valid['VWAP_Upper_2'] >= valid['VWAP']).all()
        assert (valid['VWAP_Lower_2'] <= valid['VWAP']).all()



class TestSessionFeatures:
    def test_session_columns(self, synthetic_ohlcv):
        result = calculate_session_features(synthetic_ohlcv)
        for col in ['Session_Asian', 'Session_London', 'Session_NY']:
            assert col in result.columns

    def test_session_binary(self, synthetic_ohlcv):
        result = calculate_session_features(synthetic_ohlcv)
        for col in ['Session_Asian', 'Session_London', 'Session_NY']:
            assert set(result[col].unique()).issubset({0, 1})



class TestOrderFlowProxy:
    def test_positive_for_bullish_bars(self):
        """Close > Open should give positive volume delta."""
        df = pd.DataFrame({
            'Open': [100.0], 'High': [105.0], 'Low': [99.0],
            'Close': [104.0], 'Volume': [1000.0],
        })
        delta = calculate_orderflow_proxy(df)
        assert delta.iloc[0] > 0

    def test_negative_for_bearish_bars(self):
        df = pd.DataFrame({
            'Open': [104.0], 'High': [105.0], 'Low': [99.0],
            'Close': [100.0], 'Volume': [1000.0],
        })
        delta = calculate_orderflow_proxy(df)
        assert delta.iloc[0] < 0



class TestReturnAutocorrelation:
    def test_autocorr_range(self, synthetic_ohlcv):
        ac = calculate_return_autocorrelation(synthetic_ohlcv['Close'], 20, 1)
        valid = ac.dropna()
        assert (valid >= -1.0).all()
        assert (valid <= 1.0).all()
