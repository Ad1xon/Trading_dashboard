"""Tests for live_engine.py — candle gate, anti-repainting, signal caching.

All MT5 interactions are mocked. No live broker connection required.
"""

import sys
import types
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import numpy as np
import pytest


def _build_mt5_mock():
    """Build a comprehensive MT5 module mock for engine import."""
    mt5_mock = MagicMock()
    mt5_mock.TIMEFRAME_M1 = 1
    mt5_mock.TIMEFRAME_M5 = 5
    mt5_mock.TIMEFRAME_M15 = 15
    mt5_mock.TIMEFRAME_M30 = 30
    mt5_mock.TIMEFRAME_H1 = 16385
    mt5_mock.TIMEFRAME_H4 = 16388
    mt5_mock.TIMEFRAME_D1 = 16408
    mt5_mock.ORDER_TYPE_BUY = 0
    mt5_mock.ORDER_TYPE_SELL = 1
    mt5_mock.TRADE_ACTION_DEAL = 1
    mt5_mock.TRADE_ACTION_SLTP = 6
    mt5_mock.ORDER_TIME_GTC = 0
    mt5_mock.ORDER_FILLING_FOK = 0
    mt5_mock.ORDER_FILLING_IOC = 1
    mt5_mock.ORDER_FILLING_RETURN = 2
    mt5_mock.TRADE_RETCODE_DONE = 10009
    return mt5_mock


def _make_rates(n=500, base_time=None, freq_seconds=300):
    """Generate synthetic structured array mimicking MT5 copy_rates_from_pos output."""
    if base_time is None:
        base_time = datetime(2025, 6, 1, 0, 0, 0)

    dtype = np.dtype([
        ("time", np.int64),
        ("open", np.float64),
        ("high", np.float64),
        ("low", np.float64),
        ("close", np.float64),
        ("tick_volume", np.int64),
        ("spread", np.int64),
        ("real_volume", np.int64),
    ])
    rates = np.zeros(n, dtype=dtype)
    np.random.seed(42)
    price = 100.0

    for i in range(n):
        ts = int((base_time + timedelta(seconds=i * freq_seconds)).timestamp())
        ret = np.random.randn() * 0.002
        o = price
        c = price * (1 + ret)
        h = max(o, c) * (1 + abs(np.random.randn() * 0.001))
        l = min(o, c) * (1 - abs(np.random.randn() * 0.001))
        rates[i] = (ts, o, h, l, c, np.random.randint(100, 5000), 1, 0)
        price = c

    return rates


@pytest.fixture
def mt5_mock():
    """Fixture that installs a mocked MetaTrader5 module into sys.modules."""
    mock = _build_mt5_mock()
    sys.modules["MetaTrader5"] = mock
    yield mock
    if "MetaTrader5" in sys.modules and sys.modules["MetaTrader5"] is mock:
        del sys.modules["MetaTrader5"]


@pytest.fixture
def engine_module(mt5_mock):
    """Import live_engine with mocked MT5 — returns (engine_class, mt5_mock)."""
    live_config_mock = types.ModuleType("live_config")
    live_config_mock.MT5_LOGIN = 12345678
    live_config_mock.MT5_PASSWORD = "test"
    live_config_mock.MT5_SERVER = "TestServer"
    live_config_mock.MT5_PATH = r"C:\fake\terminal64.exe"
    live_config_mock.MFT_SYMBOLS = ["EURUSD"]
    live_config_mock.SWING_SYMBOLS = ["EURUSD"]
    live_config_mock.MFT_TIMEFRAME_STR = "M5"
    live_config_mock.SWING_TIMEFRAME_STR = "H4"
    live_config_mock.POLL_INTERVAL = 5
    live_config_mock.MAX_DAILY_DRAWDOWN_PCT = 0.03
    live_config_mock.MAX_CONSECUTIVE_LOSSES = 5
    live_config_mock.MAX_TOTAL_EXPOSURE_LOTS = 10.0
    live_config_mock.RECONNECT_BASE_DELAY = 2.0
    live_config_mock.RECONNECT_MAX_DELAY = 300.0
    live_config_mock.HEARTBEAT_INTERVAL = 60
    live_config_mock.RISK_PER_TRADE_PCT = 0.01
    live_config_mock.DEFAULT_CAPITAL = 10000.0
    live_config_mock.LOG_FILE = "test_engine.log"
    live_config_mock.STATE_FILE = "test_state.json"
    live_config_mock.DISCORD_WEBHOOK = ""
    sys.modules["live_config"] = live_config_mock

    config_mock = types.ModuleType("config")
    config_mock.CONTRACT_SIZES = {}
    config_mock.MAX_ALLOWED_LOTS = 5.0
    config_mock.DEFAULT_MAX_HOLDING = 100
    config_mock.MFE_ACTIVATION_MULTIPLIER = 1.0
    config_mock.MFE_TRAIL_PCT = 0.5
    sys.modules["config"] = config_mock

    return mt5_mock


class TestCandleGate:
    """Verify that signals are only generated when a new closed bar is detected."""

    def test_signal_only_on_new_bar(self, engine_module):
        """When the closed bar timestamp is unchanged, no signal should fire."""
        mt5_mock = engine_module

        rates = _make_rates(n=10)
        mt5_mock.copy_rates_from_pos.return_value = rates

        from live_engine import LiveTradingEngine
        engine = LiveTradingEngine.__new__(LiveTradingEngine)
        engine._last_bar_time = {}

        """First call — new bar detected (no previous timestamp stored)."""
        assert engine._has_new_closed_bar("EURUSD", "M5") is True

        """Second call with same rates — no new bar."""
        assert engine._has_new_closed_bar("EURUSD", "M5") is False

    def test_new_bar_detected_on_timestamp_change(self, engine_module):
        """Changing the closed bar timestamp should trigger new bar detection."""
        mt5_mock = engine_module

        rates_old = _make_rates(n=10)
        rates_new = _make_rates(n=10, base_time=datetime(2025, 6, 1, 1, 0, 0))

        from live_engine import LiveTradingEngine
        engine = LiveTradingEngine.__new__(LiveTradingEngine)
        engine._last_bar_time = {}

        """First call: detect new bar for the first time."""
        mt5_mock.copy_rates_from_pos.return_value = rates_old
        first_result = engine._has_new_closed_bar("EURUSD", "M5")
        assert first_result is True

        """Second call with NEW timestamps: should detect change."""
        mt5_mock.copy_rates_from_pos.return_value = rates_new
        second_result = engine._has_new_closed_bar("EURUSD", "M5")
        assert second_result is True


class TestAntiRepainting:
    """Verify that signal reads use iloc[-2] (closed bar), not iloc[-1] (forming)."""

    def test_reads_closed_candle(self, engine_module, synthetic_ohlcv):
        """_get_latest_signal must read from iloc[-2], the last closed bar."""
        from live_engine import LiveTradingEngine
        engine = LiveTradingEngine.__new__(LiveTradingEngine)
        engine._signal_cache = {}

        mock_strategy = MagicMock()
        result_df = synthetic_ohlcv.copy()
        result_df["Signal"] = 0
        result_df.iloc[-1, result_df.columns.get_loc("Signal")] = 1
        result_df.iloc[-2, result_df.columns.get_loc("Signal")] = -1
        mock_strategy.generate_signals.return_value = result_df

        signal, cached = engine._get_latest_signal(result_df, mock_strategy)

        """Should read -1 from iloc[-2], NOT +1 from iloc[-1]."""
        assert signal == -1

    def test_sl_tp_reads_closed_bar(self, engine_module, synthetic_ohlcv):
        """_get_sl_tp must extract SL/TP from iloc[-2]."""
        from live_engine import LiveTradingEngine
        engine = LiveTradingEngine.__new__(LiveTradingEngine)

        mock_strategy = MagicMock()
        result_df = synthetic_ohlcv.copy()
        result_df["SL_Price"] = np.nan
        result_df["TP_Price"] = np.nan
        result_df.iloc[-2, result_df.columns.get_loc("SL_Price")] = 95.0
        result_df.iloc[-2, result_df.columns.get_loc("TP_Price")] = 110.0
        result_df.iloc[-1, result_df.columns.get_loc("SL_Price")] = 90.0
        result_df.iloc[-1, result_df.columns.get_loc("TP_Price")] = 120.0

        sl, tp = engine._get_sl_tp(result_df, 1, mock_strategy, cached_result=result_df)

        """Should get 95/110 from iloc[-2], NOT 90/120 from iloc[-1]."""
        assert sl == 95.0
        assert tp == 110.0


class TestSignalCaching:
    """Verify that signal results are cached to avoid double computation."""

    def test_cache_populated_on_signal_generation(self, engine_module, synthetic_ohlcv):
        """_get_latest_signal should populate _signal_cache when cache_key is given."""
        from live_engine import LiveTradingEngine
        engine = LiveTradingEngine.__new__(LiveTradingEngine)
        engine._signal_cache = {}

        mock_strategy = MagicMock()
        result_df = synthetic_ohlcv.copy()
        result_df["Signal"] = 0
        mock_strategy.generate_signals.return_value = result_df

        engine._get_latest_signal(result_df, mock_strategy, cache_key="EURUSD_MFT")

        assert "EURUSD_MFT" in engine._signal_cache


class TestCircuitBreakerIntegration:
    """Verify the circuit breaker blocks new entries when tripped."""

    def test_tripped_breaker_blocks_check(self):
        """A tripped circuit breaker should return False on check."""
        from live_engine import RiskCircuitBreaker
        breaker = RiskCircuitBreaker(10000.0)
        breaker.tripped = True
        breaker.trip_reason = "test"
        assert breaker.check(9000.0, 1.0) is False
