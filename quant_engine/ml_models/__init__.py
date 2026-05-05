"""Machine learning models module."""

from .xgb_model import XGBoostRangeBarModel
from .lgbm_model import LGBMRangeBarModel
from .stat_arb_filter import StatArbMLFilter

try:
    from .lstm_model import LSTMSwingModel
except ImportError:
    LSTMSwingModel = None

__all__ = ['XGBoostRangeBarModel', 'LGBMRangeBarModel', 'StatArbMLFilter', 'LSTMSwingModel']
