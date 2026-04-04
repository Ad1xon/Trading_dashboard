"""Statistical Arbitrage ML Filter."""

import pandas as pd
import numpy as np
import xgboost as xgb

from config import (
    XGB_N_ESTIMATORS, XGB_MAX_DEPTH, XGB_LEARNING_RATE
)

class StatArbMLFilter:
    """ML-enhanced pair-trading filter for spread reversion probability."""

    def __init__(self):
        self.model = xgb.XGBClassifier(
            n_estimators=XGB_N_ESTIMATORS, max_depth=XGB_MAX_DEPTH,
            learning_rate=XGB_LEARNING_RATE,
            objective='binary:logistic', random_state=42,
        )
        self.is_trained = False

    def prepare_features(self, spread: pd.Series, z_score: pd.Series, window: int = 10) -> pd.DataFrame:
        """Build feature matrix for spread reversion classification."""
        df = pd.DataFrame()
        df['Z_Score'] = z_score
        df['Spread_Momentum_3'] = spread.diff(3)
        df['Spread_Volatility_10'] = spread.rolling(window=10).std()
        future_spread_change = spread.shift(-window) - spread
        is_reverting = np.where(
            (z_score > 1.5) & (future_spread_change < 0), 1,
            np.where((z_score < -1.5) & (future_spread_change > 0), 1, 0),
        )
        df['Target'] = is_reverting
        return df.dropna()

    def train(self, df_features: pd.DataFrame):
        """Train on 80% chronological split."""
        X = df_features.drop('Target', axis=1)
        y = df_features['Target']
        split_idx = int(len(X) * 0.8)
        if split_idx < 10:
            return
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        self.model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        self.is_trained = True

    def predict_probability(self, latest_features: pd.DataFrame) -> float:
        """Return P(reversion) for the latest observation."""
        if not self.is_trained:
            return 0.5
        return self.model.predict_proba(latest_features)[0][1]