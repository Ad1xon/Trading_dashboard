import pytest
import pandas as pd
import numpy as np
from quant_engine.scanner import MarketScanner

class MockMT5Connector:
    def get_mt5_data(self, symbol, days_back, timeframe=None):
        dates = pd.date_range("2023-01-01", periods=100, freq="15min")
        df = pd.DataFrame({
            "Open": np.linspace(1.0, 1.1, 100),
            "High": np.linspace(1.05, 1.15, 100),
            "Low": np.linspace(0.95, 1.05, 100),
            "Close": np.linspace(1.02, 1.12, 100),
            "Volume": np.random.randint(100, 1000, size=100),
        }, index=dates)
        return df

@pytest.fixture
def scanner():
    return MarketScanner()

def test_market_scanner_initialization(scanner):
    assert scanner.strategies is not None
    assert "VWAP Bounce" in scanner.strategies
    assert "SMC Breakout" in scanner.strategies

def test_run_scan_with_mock_data(monkeypatch, scanner):
    mock = MockMT5Connector()
    monkeypatch.setattr("quant_engine.scanner.get_mt5_data", mock.get_mt5_data)
    
    tf_options = {"M15 (Time-based)": 15}
    results = scanner.run_scan(
        selected_symbols=["EURUSD", "GBPUSD"],
        data_mode_scan="M15 (Time-based)",
        tf_options=tf_options,
        scan_days=5
    )
    
    assert isinstance(results, list)
    assert len(results) == 2
    
    symbols = [r["symbol"] for r in results]
    assert "EURUSD" in symbols
    assert "GBPUSD" in symbols

    for res in results:
        assert isinstance(res["price"], float)
        assert isinstance(res["vwap"], float)
        assert isinstance(res["rsi"], float)
        assert isinstance(res["atr"], float)
        assert isinstance(res["signal"], str)
        assert isinstance(res["confidence"], float)
