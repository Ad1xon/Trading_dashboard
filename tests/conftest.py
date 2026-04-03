"""conftest.py — shared fixtures for all tests."""

import sys
import os
import pytest
import pandas as pd
import numpy as np

# Ensure project root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


@pytest.fixture
def synthetic_ohlcv():
    """
    Generate a deterministic synthetic OHLCV DataFrame (500 bars).
    Simulates a mean-reverting price process with known characteristics.
    """
    np.random.seed(42)
    n = 500
    timestamps = pd.date_range('2025-01-01', periods=n, freq='1min')
    base = 100.0
    returns = np.random.randn(n) * 0.002   # ~0.2% per bar

    close = base * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(np.random.randn(n) * 0.001))
    low = close * (1 - np.abs(np.random.randn(n) * 0.001))
    open_ = np.roll(close, 1)
    open_[0] = base
    volume = np.random.randint(100, 10000, size=n).astype(float)

    df = pd.DataFrame({
        'Open': open_,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume,
    }, index=timestamps)

    return df


@pytest.fixture
def small_ohlcv():
    """Minimal 10-bar OHLCV for edge-case tests."""
    data = {
        'Open':   [100, 101, 102, 101, 100, 99, 100, 101, 102, 103],
        'High':   [102, 103, 104, 103, 102, 101, 102, 103, 104, 105],
        'Low':    [99,  100, 101, 100, 99,  98,  99, 100, 101, 102],
        'Close':  [101, 102, 101, 100, 99,  100, 101, 102, 103, 104],
        'Volume': [500, 600, 700, 550, 800, 750, 650, 900, 850, 950],
    }
    index = pd.date_range('2025-06-01', periods=10, freq='1min')
    return pd.DataFrame(data, index=index, dtype=float)


@pytest.fixture
def range_bars(synthetic_ohlcv):
    """Range bars generated from synthetic data."""
    from quant_engine.data_processor import generate_synthetic_range_bars
    return generate_synthetic_range_bars(synthetic_ohlcv, range_size=0.15)
