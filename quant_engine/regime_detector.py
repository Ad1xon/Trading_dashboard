"""HMM-based market regime detector — classifies bars into bear, range, or bull regimes."""

import logging

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from config import HMM_N_STATES, HMM_COVARIANCE_TYPE

logger = logging.getLogger(__name__)

REGIME_LABELS = {0: "bear", 1: "range", 2: "bull"}


class RegimeDetector:
    """Gaussian HMM regime classifier with 3 hidden states."""

    def __init__(
        self,
        n_states: int = HMM_N_STATES,
        covariance_type: str = HMM_COVARIANCE_TYPE,
        lookback: int = 252,
    ):
        self.n_states = n_states
        self.covariance_type = covariance_type
        self.lookback = lookback
        self.model = GaussianHMM(
            n_components=n_states,
            covariance_type=covariance_type,
            n_iter=200,
            random_state=42,
            tol=1e-4,
        )
        self.is_fitted = False
        self._state_map: dict[int, str] = {}

    def _build_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract returns and rolling volatility as HMM observation features."""
        returns = df["Close"].pct_change().fillna(0).values
        vol = pd.Series(returns).rolling(20).std().fillna(0).values
        features = np.column_stack([returns, vol])
        return features

    def fit(self, df: pd.DataFrame) -> "RegimeDetector":
        """Train the HMM on historical OHLCV data."""
        features = self._build_features(df)
        valid_start = max(20, len(features) - self.lookback)
        train_features = features[valid_start:]

        if len(train_features) < 50:
            logger.warning("Insufficient data for regime detection (%d bars)", len(train_features))
            return self

        rng = np.random.default_rng(42)
        train_features = train_features + rng.normal(0, 1e-6, train_features.shape)

        try:
            self.model.fit(train_features)
            self.is_fitted = True
            self._label_states(train_features)
        except Exception:
            try:
                self.model = GaussianHMM(
                    n_components=self.n_states, covariance_type="diag",
                    n_iter=200, random_state=42, tol=1e-4,
                )
                self.model.fit(train_features)
                self.is_fitted = True
                self._label_states(train_features)
            except Exception as exc:
                logger.error("HMM fitting failed: %s", exc)

        return self

    def _label_states(self, features: np.ndarray):
        """Assign semantic labels by sorting states on mean return."""
        states = self.model.predict(features)
        returns = features[:, 0]
        state_means = {}
        for s in range(self.n_states):
            mask = states == s
            state_means[s] = returns[mask].mean() if mask.any() else 0.0
        sorted_states = sorted(state_means, key=state_means.get)
        labels = ["bear", "range", "bull"]
        self._state_map = {sorted_states[i]: labels[i] for i in range(min(len(sorted_states), 3))}

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return integer regime labels for each bar."""
        if not self.is_fitted:
            return np.zeros(len(df), dtype=int)
        features = self._build_features(df)
        try:
            return self.model.predict(features)
        except Exception:
            return np.zeros(len(df), dtype=int)

    def predict_labels(self, df: pd.DataFrame) -> pd.Series:
        """Return human-readable regime labels for each bar."""
        raw_states = self.predict(df)
        if not self._state_map:
            return pd.Series(["range"] * len(df), index=df.index)
        return pd.Series(
            [self._state_map.get(s, "range") for s in raw_states],
            index=df.index,
        )

    def add_regime_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add a Regime column to the DataFrame."""
        df = df.copy()
        if not self.is_fitted:
            self.fit(df)
        df["Regime"] = self.predict_labels(df).values
        return df

    def is_favourable(self, regime: str, strategy_type: str) -> bool:
        """Check if a regime is favourable for the given strategy archetype."""
        rules = {
            "trend": regime in ("bull", "bear"),
            "reversion": regime == "range",
            "momentum": regime in ("bull", "bear"),
            "scalp": True,
        }
        return rules.get(strategy_type, True)
