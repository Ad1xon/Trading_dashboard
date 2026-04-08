"""XGBoost walk-forward classifier and shared caching engine."""

import os
import warnings
import logging

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from numpy.lib.stride_tricks import sliding_window_view
from joblib import Memory

from ..indicators import (
    calculate_rsi, calculate_atr, calculate_bollinger,
    calculate_adx, calculate_orderflow_proxy, calculate_return_autocorrelation,
)
from config import (
    XGB_N_ESTIMATORS, XGB_MAX_DEPTH, XGB_LEARNING_RATE,
    XGB_REFIT_EVERY, XGB_HORIZON, XGB_TP_MULT, XGB_SL_MULT,
)

logger = logging.getLogger(__name__)

cache_dir = os.path.join(os.path.dirname(__file__), '..', '..', '.cache')
os.makedirs(cache_dir, exist_ok=True)
memory = Memory(cache_dir, verbose=0)


@memory.cache
def _cached_walk_forward_train(
    df_index,
    df_values,
    df_columns,
    initial_train_frac,
    n_estimators,
    max_depth,
    learning_rate,
    horizon,
    refit_every,
    tp_mult,
    sl_mult,
    feature_cols,
    model_type='xgboost',
):
    """Core expanding-window walk-forward logic, extracted for joblib caching."""
    df = pd.DataFrame(df_values, index=df_index, columns=df_columns)

    df = df.infer_objects()
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            pass

    if model_type == 'lightgbm':
        model = lgb.LGBMClassifier(
            num_leaves=31, max_depth=-1, learning_rate=learning_rate,
            n_estimators=n_estimators, objective='binary', random_state=42,
            subsample=0.8, colsample_bytree=0.8, n_jobs=-1, verbose=-1,
        )
    else:
        model = xgb.XGBClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, objective='binary:logistic',
            random_state=42, subsample=0.8, colsample_bytree=0.8,
        )

    dummy_instance = XGBoostRangeBarModel(tp_mult=tp_mult, sl_mult=sl_mult)
    data = dummy_instance.build_features(df)
    data['Target'] = dummy_instance._build_mfe_target(data)

    actual_features = [c for c in feature_cols if c in data.columns]
    valid_mask = data[actual_features + ['Target']].notna().all(axis=1)
    data = data.loc[valid_mask].copy()

    n = len(data)
    initial_train_end = int(n * initial_train_frac)
    if initial_train_end < 30:
        return None, None, {}, []

    X_all = data[actual_features]
    y_all = data['Target']
    predictions = np.full(n, np.nan)
    cursor = initial_train_end

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        while cursor < n:
            segment_end = min(cursor + refit_every, n)
            train_end_purged = max(0, cursor - horizon)

            if train_end_purged > 0:
                X_train_sub = X_all.iloc[:train_end_purged]
                y_train_sub = y_all.iloc[:train_end_purged]
                model.fit(X_train_sub, y_train_sub)
                predictions[cursor:segment_end] = model.predict_proba(
                    X_all.iloc[cursor:segment_end]
                )[:, 1]

            cursor = segment_end

        data['WF_Prediction'] = predictions
        feature_importances = dict(zip(actual_features, model.feature_importances_))

        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []
        X_train_init = X_all.iloc[:initial_train_end]
        y_train_init = y_all.iloc[:initial_train_end]

        for train_idx, val_idx in tscv.split(X_train_init):
            if len(train_idx) < 20:
                continue
            model.fit(X_train_init.iloc[train_idx], y_train_init.iloc[train_idx])
            cv_scores.append(
                accuracy_score(
                    y_train_init.iloc[val_idx],
                    model.predict(X_train_init.iloc[val_idx]),
                )
            )

    return data, model, feature_importances, cv_scores


class XGBoostRangeBarModel:
    """Walk-forward XGBoost classifier for price-direction prediction."""

    FEATURE_COLS = [
        'Dir_Sum_5', 'Vol_Ratio', 'Close_Diff', 'Variance_Ratio',
        'RSI_14', 'ATR_14', 'BB_PctB', 'BB_Width',
        'ADX_14', 'Volume_Delta', 'Return_Autocorr', 'Return_5',
        'Sentiment_Score',
    ]

    def __init__(
        self,
        tp_mult: float = XGB_TP_MULT,
        sl_mult: float = XGB_SL_MULT,
    ):
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
        """Construct the full feature matrix from OHLCV data."""
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

        if 'Sentiment_Score' not in data.columns:
            data['Sentiment_Score'] = 0.0

        return data

    def _build_mfe_target(self, data: pd.DataFrame) -> np.ndarray:
        """MFE-based binary target: 1 if TP hit before SL within horizon, else 0."""
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

        high_windows = sliding_window_view(highs_padded, window_shape=horizon)[1:n + 1]
        low_windows = sliding_window_view(lows_padded, window_shape=horizon)[1:n + 1]

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
        """Expanding-window walk-forward training with joblib cache."""
        data, model, fi, cv_scores = _cached_walk_forward_train(
            df.index, df.values, df.columns, initial_train_frac,
            self.model.n_estimators, self.model.max_depth,
            self.model.learning_rate,
            self.horizon_param, self.refit_param,
            self.tp_mult, self.sl_mult, self.FEATURE_COLS,
            model_type='xgboost',
        )

        if data is None:
            return None

        self.is_cached = True
        self.model = model
        self.feature_importances_ = fi
        self.cv_scores_ = cv_scores
        self.is_trained = True
        return data

    def predict_proba(self, df_features: pd.DataFrame) -> np.ndarray:
        """Return bullish probability for each row."""
        if 'WF_Prediction' in df_features.columns:
            return df_features['WF_Prediction'].fillna(0.5).values
        if not self.is_trained:
            return np.full(len(df_features), 0.5)

        actual_features = [c for c in self.FEATURE_COLS if c in df_features.columns]
        if len(actual_features) < len(self.FEATURE_COLS):
            for col in self.FEATURE_COLS:
                if col not in df_features.columns:
                    df_features[col] = 0.0

        X = df_features[self.FEATURE_COLS]
        X = X.ffill().fillna(0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self.model.predict_proba(X)[:, 1]

    def get_feature_importance(self) -> dict:
        """Feature importances sorted descending by value."""
        return dict(
            sorted(self.feature_importances_.items(), key=lambda x: x[1], reverse=True)
        )

    def plot_feature_importance(self, save_path: str | None = None):
        """Render horizontal bar chart of feature importances via matplotlib."""
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
        bars = ax.barh(features[::-1], pct[::-1], color=colors[::-1],
                       edgecolor='white', linewidth=0.5)

        for bar, p in zip(bars, pct[::-1]):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    f'{p:.1f}%', va='center', fontsize=9, color='#333')

        ax.set_xlabel('Relative Importance (%)', fontsize=11)
        ax.set_title('Feature Importance (Walk-Forward)', fontsize=13, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='x', alpha=0.3)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info("Feature importance chart saved to %s", save_path)

        return fig
