"""XGBoost ML models — walk-forward training, MFE target labeling, feature importance."""

import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
import logging
from numpy.lib.stride_tricks import sliding_window_view
from joblib import Memory

from .indicators import (
    calculate_rsi, calculate_atr, calculate_bollinger,
    calculate_adx, calculate_orderflow_proxy, calculate_return_autocorrelation,
)
from config import (
    XGB_N_ESTIMATORS, XGB_MAX_DEPTH, XGB_LEARNING_RATE,
    XGB_REFIT_EVERY, XGB_HORIZON, XGB_TP_MULT, XGB_SL_MULT
)

logger = logging.getLogger(__name__)

cache_dir = os.path.join(os.path.dirname(__file__), '..', '.cache')
os.makedirs(cache_dir, exist_ok=True)
memory = Memory(cache_dir, verbose=0)


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


@memory.cache
def _cached_walk_forward_train(
    df_index, df_values, df_columns, initial_train_frac,
    n_estimators, max_depth, learning_rate,
    horizon, refit_every, tp_mult, sl_mult, feature_cols
):
    """Core walk-forward logic, extracted for joblib caching.
    Reconstructs DataFrame from primitive arrays to avoid hashing overhead/issues.
    """
    df = pd.DataFrame(df_values, index=df_index, columns=df_columns)
    
    model = xgb.XGBClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=learning_rate,
        objective='binary:logistic', random_state=42,
        subsample=0.8, colsample_bytree=0.8
    )
    
    dummy_instance = XGBoostRangeBarModel(tp_mult=tp_mult, sl_mult=sl_mult)
    data = dummy_instance.build_features(df)
    data['Target'] = dummy_instance._build_mfe_target(data)
    valid_mask = data[feature_cols + ['Target']].notna().all(axis=1)
    data = data.loc[valid_mask].copy()

    n = len(data)
    initial_train_end = int(n * initial_train_frac)
    if initial_train_end < 30:
        return None, None, {}, []

    X_all = data[feature_cols].values
    y_all = data['Target'].values
    predictions = np.full(n, np.nan)
    cursor = initial_train_end

    while cursor < n:
        segment_end = min(cursor + refit_every, n)
        train_end_purged = max(0, cursor - horizon)

        if train_end_purged > 0:
            X_train_sub = X_all[:train_end_purged]
            y_train_sub = y_all[:train_end_purged]
            model.fit(X_train_sub, y_train_sub, verbose=False)
            predictions[cursor:segment_end] = model.predict_proba(X_all[cursor:segment_end])[:, 1]

        cursor = segment_end

    data['WF_Prediction'] = predictions
    feature_importances = dict(zip(feature_cols, model.feature_importances_))
    
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = []
    X_train_init = X_all[:initial_train_end]
    y_train_init = y_all[:initial_train_end]
    for train_idx, val_idx in tscv.split(X_train_init):
        if len(train_idx) < 20:
            continue
        model.fit(X_train_init[train_idx], y_train_init[train_idx], verbose=False)
        cv_scores.append(accuracy_score(y_train_init[val_idx], model.predict(X_train_init[val_idx])))

    return data, model, feature_importances, cv_scores

class XGBoostRangeBarModel:
    """Walk-forward XGBoost classifier for range-bar price direction.

    Uses MFE-based target labeling: a bar is labeled 1 (bullish setup)
    if, within the next ``horizon`` bars, the price reaches +``tp_mult``
    range-bar-sizes BEFORE hitting -``sl_mult`` range-bar-sizes.

    Features: RSI, ATR, Bollinger %B/Width, ADX, volume delta,
    return autocorrelation, directional momentum.
    """

    FEATURE_COLS = [
        'Dir_Sum_5', 'Vol_Ratio', 'Close_Diff', 'Variance_Ratio',
        'RSI_14', 'ATR_14', 'BB_PctB', 'BB_Width',
        'ADX_14', 'Volume_Delta', 'Return_Autocorr', 'Return_5',
    ]

    def __init__(self, tp_mult: float = XGB_TP_MULT, sl_mult: float = XGB_SL_MULT):
        self.horizon_param = XGB_HORIZON
        self.refit_param = XGB_REFIT_EVERY
        self.tp_mult = tp_mult
        self.sl_mult = sl_mult
        self.model = xgb.XGBClassifier(
            n_estimators=XGB_N_ESTIMATORS, max_depth=XGB_MAX_DEPTH,
            learning_rate=XGB_LEARNING_RATE,
            objective='binary:logistic', random_state=42,
            subsample=0.8, colsample_bytree=0.8,
            early_stopping_rounds=10,
        )
        self.is_trained = False
        self.is_cached = False
        self.feature_importances_: dict = {}
        self.cv_scores_: list = []

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Construct full feature matrix from OHLCV data."""
        data = df.copy()
        data['Dir'] = np.where(data['Close'] > data['Open'], 1, -1)
        data['Dir_Sum_5'] = data['Dir'].rolling(5).sum()
        data['Vol_MA_5'] = data['Volume'].rolling(5).mean()
        data['Vol_Ratio'] = data['Volume'] / (data['Vol_MA_5'] + 1e-6)
        data['Close_Diff'] = data['Close'].diff(5)
        data['Var_Short'] = data['Close'].rolling(5).std()
        data['Var_Long'] = data['Close'].rolling(20).std()
        data['Variance_Ratio'] = data['Var_Short'] / (data['Var_Long'] + 1e-6)
        data['RSI_14'] = calculate_rsi(data['Close'], 14)
        data['ATR_14'] = calculate_atr(data, 14)
        bb = calculate_bollinger(data['Close'], 20, 2.0)
        data['BB_PctB'] = bb['BB_PctB']
        data['BB_Width'] = bb['BB_Width']
        data['ADX_14'] = calculate_adx(data, 14)
        data['Volume_Delta'] = calculate_orderflow_proxy(data)
        data['Return_Autocorr'] = calculate_return_autocorrelation(data['Close'], 20, 1)
        data['Return_5'] = np.log(data['Close'] / data['Close'].shift(5).clip(lower=1e-10))
        return data

    def _build_mfe_target(self, data: pd.DataFrame) -> np.ndarray:
        """MFE-based target: 1 if TP hit before SL within horizon, else 0.

        Optimized vectorized implementation O(1) via sliding_window_view.
        """
        closes = data['Close'].values
        highs = data['High'].values
        lows = data['Low'].values
        atr = data['ATR_14'].values
        horizon = self.horizon_param
        n = len(data)

        tp_levels = closes + self.tp_mult * atr
        sl_levels = closes - self.sl_mult * atr

        highs_padded = np.pad(highs, (0, horizon), mode='edge')
        lows_padded = np.pad(lows, (0, horizon), mode='edge')

        high_windows = sliding_window_view(highs_padded, window_shape=horizon)[1:n+1]
        low_windows = sliding_window_view(lows_padded, window_shape=horizon)[1:n+1]

        high_hits = high_windows >= tp_levels[:, None]
        low_hits = low_windows <= sl_levels[:, None]

        high_any = high_hits.any(axis=1)
        low_any = low_hits.any(axis=1)

        high_idx = np.where(high_any, np.argmax(high_hits, axis=1), horizon + 1)
        low_idx = np.where(low_any, np.argmax(low_hits, axis=1), horizon + 1)

        target = np.where((high_idx <= low_idx) & high_any, 1, 0).astype(float)

        invalid_mask = np.isnan(atr) | (atr < 1e-10)
        target[invalid_mask] = np.nan
        target[n - horizon:] = np.nan

        return target



    def train(self, df: pd.DataFrame, initial_train_frac: float = 0.70):
        """Expanding-window walk-forward training. Caches via joblib."""
        data, model, fi, cv_scores = _cached_walk_forward_train(
            df.index, df.values, df.columns, initial_train_frac,
            self.model.n_estimators, self.model.max_depth, self.model.learning_rate,
            self.horizon_param, self.refit_param, self.tp_mult, self.sl_mult, self.FEATURE_COLS
        )
        
        if data is None:
            return None
            
        self.is_cached = True 
        self.model = model
        self.feature_importances_ = fi
        self.cv_scores_ = cv_scores
        self.is_trained = True
        return data

    def _run_cv(self, X: np.ndarray, y: np.ndarray, n_splits: int = 5):
        """TimeSeriesSplit cross-validation on the training portion."""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = []
        for train_idx, val_idx in tscv.split(X):
            if len(train_idx) < 20:
                continue
            self.model.fit(X[train_idx], y[train_idx], verbose=False)
            scores.append(accuracy_score(y[val_idx], self.model.predict(X[val_idx])))
        self.cv_scores_ = scores



    def predict_proba(self, df_features: pd.DataFrame) -> np.ndarray:
        """Return bullish probability for each row."""
        if 'WF_Prediction' in df_features.columns:
            return df_features['WF_Prediction'].fillna(0.5).values
        if not self.is_trained:
            return np.full(len(df_features), 0.5)
        available = [c for c in self.FEATURE_COLS if c in df_features.columns]
        if len(available) < len(self.FEATURE_COLS):
            return np.full(len(df_features), 0.5)
        X = df_features[self.FEATURE_COLS]
        X = X.ffill().bfill().fillna(0)
        return self.model.predict_proba(X.values)[:, 1]

    def get_feature_importance(self) -> dict:
        """Feature importances sorted descending."""
        return dict(sorted(self.feature_importances_.items(), key=lambda x: x[1], reverse=True))

    def plot_feature_importance(self, save_path: str | None = None):
        """Render horizontal bar chart of feature importances via matplotlib.

        Returns the matplotlib Figure object. If ``save_path`` is given,
        the chart is also saved to disk (PNG).
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fi = self.get_feature_importance()
        if not fi:
            logger.warning("No feature importances available — model not trained.")
            return None

        features = list(fi.keys())
        values = list(fi.values())
        total = sum(values)
        pct = [v / total * 100 if total > 0 else 0 for v in values]

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(features)))
        bars = ax.barh(features[::-1], pct[::-1], color=colors[::-1], edgecolor='white', linewidth=0.5)

        for bar, p in zip(bars, pct[::-1]):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    f'{p:.1f}%', va='center', fontsize=9, color='#333')

        ax.set_xlabel('Relative Importance (%)', fontsize=11)
        ax.set_title('XGBoost Feature Importance (Walk-Forward)', fontsize=13, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='x', alpha=0.3)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info("Feature importance chart saved to %s", save_path)

        return fig