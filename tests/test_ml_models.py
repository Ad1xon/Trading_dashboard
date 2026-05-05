"""Tests for quant_engine.ml_models — walk-forward training, feature building."""

import numpy as np
import pandas as pd
import pytest

from quant_engine.ml_models import XGBoostRangeBarModel, StatArbMLFilter


class TestXGBoostRangeBarModel:
    def test_build_features_columns(self, synthetic_ohlcv):
        model = XGBoostRangeBarModel()
        features = model.build_features(synthetic_ohlcv)
        for col in model.feature_cols:
            assert col in features.columns, f"Missing feature column: {col}"

    def test_garch_vol_in_features(self, synthetic_ohlcv):
        """GARCH_Vol should be computed as a feature."""
        model = XGBoostRangeBarModel()
        features = model.build_features(synthetic_ohlcv)
        assert 'GARCH_Vol' in features.columns
        assert features['GARCH_Vol'].notna().sum() > 0

    def test_variance_ratio_in_features(self, synthetic_ohlcv):
        """Variance_Ratio should be computed as a feature."""
        model = XGBoostRangeBarModel()
        features = model.build_features(synthetic_ohlcv)
        assert 'Variance_Ratio' in features.columns

    def test_walk_forward_no_lookahead(self, synthetic_ohlcv):
        """Walk-forward predictions should ONLY exist for the OOS portion."""
        model = XGBoostRangeBarModel()
        result = model.train(synthetic_ohlcv, initial_train_frac=0.70)

        if result is None:
            pytest.skip("Not enough data for walk-forward")

        n = len(result)
        initial_end = int(n * 0.70)

        first_chunk = result['WF_Prediction'].iloc[:initial_end]
        assert first_chunk.isna().all(), "In-sample region should have NaN predictions"

        oos_chunk = result['WF_Prediction'].iloc[initial_end:]
        assert oos_chunk.notna().any(), "OOS region should have predictions"

    def test_predictions_in_range(self, synthetic_ohlcv):
        model = XGBoostRangeBarModel()
        result = model.train(synthetic_ohlcv)

        if result is None:
            pytest.skip("Not enough data")

        valid = result['WF_Prediction'].dropna()
        assert (valid >= 0).all() and (valid <= 1).all(), "Probabilities must be in [0, 1]"

    def test_feature_importance_after_training(self, synthetic_ohlcv):
        model = XGBoostRangeBarModel()
        model.train(synthetic_ohlcv)

        if not model.is_trained:
            pytest.skip("Model not trained")

        fi = model.get_feature_importance()
        assert len(fi) > 0
        assert all(v >= 0 for v in fi.values())

    def test_cv_scores(self, synthetic_ohlcv):
        model = XGBoostRangeBarModel()
        model.train(synthetic_ohlcv)

        if not model.is_trained:
            pytest.skip("Model not trained")

        assert len(model.cv_scores_) > 0
        assert all(0 <= s <= 1 for s in model.cv_scores_)

    def test_predict_proba_untrained(self, synthetic_ohlcv):
        """Untrained model should return 0.5 for all rows."""
        model = XGBoostRangeBarModel()
        probs = model.predict_proba(synthetic_ohlcv)
        assert (probs == 0.5).all()


class TestStatArbMLFilter:
    def test_prepare_features(self):
        np.random.seed(42)
        spread = pd.Series(np.cumsum(np.random.randn(200) * 0.1))
        z = (spread - spread.rolling(50).mean()) / (spread.rolling(50).std() + 1e-8)
        z = z.dropna()
        spread = spread.loc[z.index]

        filt = StatArbMLFilter()
        features = filt.prepare_features(spread, z)
        assert 'Target' in features.columns
        assert 'Z_Score' in features.columns
        assert len(features) > 0

    def test_predict_untrained(self):
        filt = StatArbMLFilter()
        dummy = pd.DataFrame({'Z_Score': [1.0], 'Spread_Momentum_3': [0.1], 'Spread_Volatility_10': [0.5]})
        prob = filt.predict_probability(dummy)
        assert prob == 0.5
