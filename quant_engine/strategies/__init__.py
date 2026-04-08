"""Trading strategies registry."""

from .reversion import ZScoreMeanReversion, MLBounceReversion, VWAPBounceStrategy, MultiTimeframeMomentum
from .breakout import VolatilityBreakout, MLVolatilityBreakout
from .arabian_scalper import ArabianScalper
from .detectors import detect_liquidity_sweep, analyze_pair_opportunity
from .lstm_swing import LSTMSwingStrategy

STRATEGY_REGISTRY = {
    'ZScore Rev': ZScoreMeanReversion,
    'SMC Breakout': VolatilityBreakout,
    'XGB Breakout': MLVolatilityBreakout,
    'XGB Bounce': MLBounceReversion,
    'VWAP Bounce': VWAPBounceStrategy,
    'MTF Momentum': MultiTimeframeMomentum,
    'LGBM Arab Scalp': ArabianScalper,
    'LSTM Swing': LSTMSwingStrategy,
}
