"""Tests for ultimate_swing.py — ML decoupling, background retraining.

Validates that generate_signals() NEVER trains the model, and that
update_model() properly sets the _model_ready flag.
"""

import time
from datetime import datetime
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pandas as pd
import pytest

from quant_engine.strategies.ultimate_swing import UltimateSwingStrategy


@pytest.fixture
def swing_strategy():
    """Fresh UltimateSwingStrategy with ML model mocked out."""
    strat = UltimateSwingStrategy()
    strat.ml_model = MagicMock()
    strat.ml_model.is_trained = False
    strat.ml_model.build_features.return_value = pd.DataFrame()
    strat.ml_model.predict_proba.return_value = np.full(500, 0.5)
    strat.ml_model.train.return_value = None
    return strat


@pytest.fixture
def swing_ohlcv():
    """Generate 500-bar deterministic OHLCV for swing testing."""
    np.random.seed(42)
    n = 500
    timestamps = pd.date_range("2025-01-01", periods=n, freq="4h")
    base = 100.0
    returns = np.random.randn(n) * 0.003

    close = base * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(np.random.randn(n) * 0.002))
    low = close * (1 - np.abs(np.random.randn(n) * 0.002))
    open_ = np.roll(close, 1)
    open_[0] = base
    volume = np.random.randint(500, 20000, size=n).astype(float)

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


class TestMLDecoupling:
    """generate_signals() must NEVER call ml_model.train()."""

    def test_generate_signals_does_not_train(self, swing_strategy, swing_ohlcv):
        """Calling generate_signals must not trigger model training."""
        swing_strategy._model_ready = False
        swing_strategy.generate_signals(swing_ohlcv.copy())

        swing_strategy.ml_model.train.assert_not_called()

    def test_generate_signals_does_not_train_even_when_ready(self, swing_strategy, swing_ohlcv):
        """Even with _model_ready=True, generate_signals must not call train."""
        swing_strategy._model_ready = True
        swing_strategy.ml_model.is_trained = True
        swing_strategy.generate_signals(swing_ohlcv.copy())

        swing_strategy.ml_model.train.assert_not_called()


class TestPredictProba:
    """When model is ready, generate_signals must use predict_proba for inference."""

    def test_predict_proba_called_when_ready(self, swing_strategy, swing_ohlcv):
        """With _model_ready=True, predict_proba should be invoked."""
        swing_strategy._model_ready = True
        swing_strategy.ml_model.is_trained = True
        swing_strategy.generate_signals(swing_ohlcv.copy())

        swing_strategy.ml_model.predict_proba.assert_called_once()

    def test_predict_proba_not_called_when_not_ready(self, swing_strategy, swing_ohlcv):
        """With _model_ready=False, predict_proba should NOT be invoked."""
        swing_strategy._model_ready = False
        swing_strategy.generate_signals(swing_ohlcv.copy())

        swing_strategy.ml_model.predict_proba.assert_not_called()


class TestNeutralScore:
    """When model is not trained, LSTM score must default to neutral (0.5)."""

    def test_neutral_lstm_prob_when_untrained(self, swing_strategy, swing_ohlcv):
        """LSTM_Prob should be 0.5 across all bars when model is not ready."""
        swing_strategy._model_ready = False
        result = swing_strategy.generate_signals(swing_ohlcv.copy())

        assert (result["LSTM_Prob"] == 0.5).all(), "LSTM_Prob must be 0.5 when untrained"


class TestUpdateModel:
    """update_model() must train the model and set the ready flag."""

    def test_update_model_calls_train(self, swing_strategy, swing_ohlcv):
        """update_model should call ml_model.train()."""
        swing_strategy.ml_model.is_trained = True
        swing_strategy.update_model(swing_ohlcv.copy())

        swing_strategy.ml_model.train.assert_called_once()

    def test_update_model_sets_ready_flag(self, swing_strategy, swing_ohlcv):
        """After update_model, _model_ready should be True."""
        swing_strategy.ml_model.is_trained = True
        swing_strategy.update_model(swing_ohlcv.copy())

        assert swing_strategy._model_ready is True

    def test_update_model_sets_timestamp(self, swing_strategy, swing_ohlcv):
        """After update_model, _last_model_update should be set."""
        swing_strategy.ml_model.is_trained = True
        swing_strategy.update_model(swing_ohlcv.copy())

        assert swing_strategy._last_model_update is not None
        assert isinstance(swing_strategy._last_model_update, datetime)


class TestBackgroundRetraining:
    """Background retraining thread lifecycle tests."""

    def test_start_stop_retraining(self, swing_strategy):
        """Background thread should start and stop cleanly."""
        fetcher = MagicMock(return_value=None)
        swing_strategy.retrain_interval_hours = 1

        swing_strategy.start_background_retraining(fetcher)
        assert swing_strategy._retrain_thread is not None
        assert swing_strategy._retrain_thread.is_alive()

        swing_strategy.stop_background_retraining()
        assert not swing_strategy._retrain_thread or not swing_strategy._retrain_thread.is_alive()

    def test_retrain_interval_configurable(self):
        """retrain_interval_hours should be in the params dict."""
        strat = UltimateSwingStrategy()
        ranges = strat.get_param_ranges()
        assert "retrain_interval_hours" in ranges


class TestOutputContract:
    """Swing strategy must produce all columns expected by backtester and engine."""

    REQUIRED_COLUMNS = ["Signal", "Exit_Long", "Exit_Short", "Std", "SL_Price", "Max_Hold"]

    def test_required_columns_present(self, swing_strategy, swing_ohlcv):
        """All backtester-required columns must be present."""
        result = swing_strategy.generate_signals(swing_ohlcv.copy())
        for col in self.REQUIRED_COLUMNS:
            assert col in result.columns, f"Missing required column: {col}"

    def test_signal_values_valid(self, swing_strategy, swing_ohlcv):
        """Signal column should only contain -1, 0, or 1."""
        result = swing_strategy.generate_signals(swing_ohlcv.copy())
        assert set(result["Signal"].unique()).issubset({-1, 0, 1})
