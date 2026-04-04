"""LightGBM model implementation."""

import pandas as pd
import lightgbm as lgb

from .xgb_model import XGBoostRangeBarModel, _cached_walk_forward_train
from config import (
    XGB_N_ESTIMATORS, XGB_LEARNING_RATE, XGB_TP_MULT, XGB_SL_MULT
)

class LGBMRangeBarModel(XGBoostRangeBarModel):
    """
    LightGBM model for Scalping.
    Uses Leaf-wise growth for extreme speed and precision on noisy Range Bars.
    Inherits feature engineering and Walk-Forward logic from XGBoostRangeBarModel.
    """
    def __init__(self, tp_mult: float = XGB_TP_MULT, sl_mult: float = XGB_SL_MULT):
        super().__init__(tp_mult=tp_mult, sl_mult=sl_mult)
        self.model = lgb.LGBMClassifier(
            num_leaves=31,
            max_depth=-1,
            learning_rate=XGB_LEARNING_RATE,
            n_estimators=XGB_N_ESTIMATORS,
            objective='binary',
            random_state=42,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            verbose=-1
        )

    def train(self, df: pd.DataFrame, initial_train_frac: float = 0.70):
        """Expanding-window walk-forward training for LightGBM. Caches via joblib."""
        data, model, fi, cv_scores = _cached_walk_forward_train(
            df.index, df.values, df.columns, initial_train_frac,
            self.model.n_estimators, -1, self.model.learning_rate,
            self.horizon_param, self.refit_param, self.tp_mult, self.sl_mult, self.FEATURE_COLS,
            model_type='lightgbm'
        )
        if data is None:
            return None
        self.is_cached = True
        self.model = model
        self.feature_importances_ = fi
        self.cv_scores_ = cv_scores
        self.is_trained = True
        return data