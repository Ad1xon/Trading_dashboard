"""Trading strategies registry — ordered by complexity (most sophisticated first)."""

from .reversion import ZScoreMeanReversion, VWAPBounceStrategy, MultiTimeframeMomentum
from .breakout import VolatilityBreakout, MLVolatilityBreakout
from .arabian_scalper import ArabianScalper
from .detectors import detect_liquidity_sweep
from .lstm_swing import LSTMSwingStrategy
from .pairs_trading import PairsTradingStrategy
from .regime_switch import RegimeSwitchStrategy
from .composite_alpha import CompositeAlphaStrategy
from .ultimate_mft import UltimateMFTStrategy
from .ultimate_swing import UltimateSwingStrategy

STRATEGY_REGISTRY = {
    'Ultimate MFT': UltimateMFTStrategy,
    'Ultimate Swing': UltimateSwingStrategy,
    'Composite Alpha (MFT)': CompositeAlphaStrategy,
    'LSTM Swing': LSTMSwingStrategy,
    'Regime Switch (HMM)': RegimeSwitchStrategy,
    'XGB Breakout (ML)': MLVolatilityBreakout,
    'LGBM Arab Scalp (ML)': ArabianScalper,
    'Pairs Trading (Stat Arb)': PairsTradingStrategy,
    'MTF Momentum': MultiTimeframeMomentum,
    'VWAP Bounce': VWAPBounceStrategy,
    'SMC Breakout': VolatilityBreakout,
    'ZScore Rev': ZScoreMeanReversion,
}
