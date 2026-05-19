"""Tests for ultimate_mft.py — cascading microstructure gate logic.

Validates the 4-stage cascade: GARCH regime → OFI impulse → CVD confirmation → execution gate.
Confirms that lagging indicators (RSI, MACD, Bollinger) are fully removed.
"""

import numpy as np
import pandas as pd
import pytest

from quant_engine.strategies.ultimate_mft import UltimateMFTStrategy


@pytest.fixture
def mft_strategy():
    """Fresh UltimateMFTStrategy with default params."""
    return UltimateMFTStrategy()


@pytest.fixture
def large_ohlcv():
    """Generate 500-bar deterministic OHLCV suitable for MFT microstructure features."""
    np.random.seed(42)
    n = 500
    timestamps = pd.date_range("2025-01-01", periods=n, freq="5min")
    base = 100.0
    returns = np.random.randn(n) * 0.002

    close = base * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(np.random.randn(n) * 0.001))
    low = close * (1 - np.abs(np.random.randn(n) * 0.001))
    open_ = np.roll(close, 1)
    open_[0] = base
    volume = np.random.randint(100, 10000, size=n).astype(float)

    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=timestamps,
    )


class TestCascadeStages:
    """Each cascade stage must independently gate or pass signals."""

    def test_high_vol_blocks_entry(self, mft_strategy, large_ohlcv):
        """When GARCH vol ratio exceeds cutoff, all signals must be blocked.

        We set an extremely low cutoff so that virtually all bars are blocked.
        """
        mft_strategy.garch_high_vol_cutoff = 0.01
        result = mft_strategy.generate_signals(large_ohlcv.copy())

        non_zero = (result["Signal"] != 0).sum()
        assert non_zero == 0, (
            f"High vol cutoff should block ALL entries, got {non_zero} signals"
        )

    def test_ofi_impulse_required(self, mft_strategy, large_ohlcv):
        """With an unreachable OFI threshold, no entries should pass Stage 2."""
        mft_strategy.ofi_impulse_threshold = 100.0
        result = mft_strategy.generate_signals(large_ohlcv.copy())

        non_zero = (result["Signal"] != 0).sum()
        assert non_zero == 0, (
            f"OFI threshold=100 should block ALL entries, got {non_zero} signals"
        )

    def test_cvd_confirmation_required(self, mft_strategy, large_ohlcv):
        """With an unreachable CVD threshold, no entries should pass Stage 3."""
        mft_strategy.cvd_confirm_threshold = 100.0
        result = mft_strategy.generate_signals(large_ohlcv.copy())

        non_zero = (result["Signal"] != 0).sum()
        assert non_zero == 0, (
            f"CVD threshold=100 should block ALL entries, got {non_zero} signals"
        )

    def test_full_cascade_pass_with_relaxed_thresholds(self, mft_strategy, large_ohlcv):
        """With very relaxed thresholds, at least some signals should pass all 4 stages."""
        mft_strategy.garch_high_vol_cutoff = 100.0
        mft_strategy.ofi_impulse_threshold = 0.01
        mft_strategy.cvd_confirm_threshold = 0.01
        mft_strategy.lob_confirm_threshold = 0.01
        mft_strategy.cost_hurdle_mult = 0.01
        mft_strategy.vol_floor_mult = 0.01

        result = mft_strategy.generate_signals(large_ohlcv.copy())
        non_zero = (result["Signal"] != 0).sum()
        assert non_zero > 0, "Relaxed thresholds should allow some signals through"


class TestLaggingIndicatorsRemoved:
    """Confirm that RSI, MACD, and Bollinger columns are NOT computed."""

    def test_no_rsi_column(self, mft_strategy, large_ohlcv):
        """RSI should not be present in the output DataFrame."""
        result = mft_strategy.generate_signals(large_ohlcv.copy())
        assert "RSI" not in result.columns, "RSI should be removed from MFT v3"

    def test_no_macd_columns(self, mft_strategy, large_ohlcv):
        """MACD and related columns should not be present."""
        result = mft_strategy.generate_signals(large_ohlcv.copy())
        macd_cols = [c for c in result.columns if "MACD" in c.upper()]
        assert len(macd_cols) == 0, f"MACD columns should be removed: {macd_cols}"

    def test_no_bollinger_columns(self, mft_strategy, large_ohlcv):
        """Bollinger Band columns should not be present."""
        result = mft_strategy.generate_signals(large_ohlcv.copy())
        bb_cols = [c for c in result.columns if "BB_" in c]
        assert len(bb_cols) == 0, f"Bollinger columns should be removed: {bb_cols}"

    def test_no_composite_score(self, mft_strategy, large_ohlcv):
        """Old Composite_Score column replaced with Cascade_Stage."""
        result = mft_strategy.generate_signals(large_ohlcv.copy())
        assert "Composite_Score" not in result.columns
        assert "Cascade_Stage" in result.columns


class TestOutputContract:
    """MFT strategy must produce all columns expected by backtester and engine."""

    REQUIRED_COLUMNS = ["Signal", "Exit_Long", "Exit_Short", "Std", "SL_Price", "Max_Hold"]

    def test_required_columns_present(self, mft_strategy, large_ohlcv):
        """All backtester-required columns must be present."""
        result = mft_strategy.generate_signals(large_ohlcv.copy())
        for col in self.REQUIRED_COLUMNS:
            assert col in result.columns, f"Missing required column: {col}"

    def test_signal_values_valid(self, mft_strategy, large_ohlcv):
        """Signal column should only contain -1, 0, or 1."""
        result = mft_strategy.generate_signals(large_ohlcv.copy())
        assert set(result["Signal"].unique()).issubset({-1, 0, 1})

    def test_cascade_stage_values(self, mft_strategy, large_ohlcv):
        """Cascade_Stage should contain integers 0-4."""
        result = mft_strategy.generate_signals(large_ohlcv.copy())
        assert set(result["Cascade_Stage"].unique()).issubset({0, 1, 2, 3, 4})

    def test_configurable_params(self):
        """All cascade thresholds must be in the params dict."""
        strat = UltimateMFTStrategy()
        ranges = strat.get_param_ranges()
        assert "garch_high_vol_cutoff" in ranges
        assert "ofi_impulse_threshold" in ranges
        assert "cvd_confirm_threshold" in ranges
        assert "lob_confirm_threshold" in ranges
