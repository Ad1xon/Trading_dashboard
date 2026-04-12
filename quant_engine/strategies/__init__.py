"""Trading strategies registry."""

from .reversion import ZScoreMeanReversion, VWAPBounceStrategy, MultiTimeframeMomentum
from .breakout import VolatilityBreakout, MLVolatilityBreakout
from .arabian_scalper import ArabianScalper
from .detectors import detect_liquidity_sweep
from .lstm_swing import LSTMSwingStrategy

STRATEGY_REGISTRY = {
    'ZScore Rev': ZScoreMeanReversion,
    'SMC Breakout': VolatilityBreakout,
    'XGB Breakout': MLVolatilityBreakout,
    'VWAP Bounce': VWAPBounceStrategy,
    'MTF Momentum': MultiTimeframeMomentum,
    'LGBM Arab Scalp': ArabianScalper,
    'LSTM Swing': LSTMSwingStrategy,
}
