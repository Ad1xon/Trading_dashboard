"""Project configuration – symbols, defaults, trading constants, ML hyperparameters."""

import json
import os


def load_translations(lang_code: str) -> dict:
    """Load JSON i18n file based on language code ('EN' or 'PL')."""
    base_dir = os.path.dirname(__file__)
    file_path = os.path.join(base_dir, 'locales', f'{lang_code.lower()}.json')
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        with open(os.path.join(base_dir, 'locales', 'en.json'), 'r', encoding='utf-8') as f:
            return json.load(f)


MT5_SYMBOLS = {
    "EURUSD": "EURUSD.ecn",
    "GBPUSD": "GBPUSD.ecn",
    "XAUUSD (Gold)": "XAUUSDs",
    "NAS100 (Nasdaq)": "NASUSD.ecn",
    "DEUR40 (DAX)": "D40EUR.ecn",
    "GBPJPY": "GBPJPY.ecn",
    "USDJPY": "USDJPY.ecn",
}

COMMISSION_USD_PER_LOT = 6.0
CONTRACT_SIZES = {
    "EURUSD": 100000,
    "GBPUSD": 100000,
    "XAUUSD (Gold)": 100,
    "NAS100 (Nasdaq)": 1,
    "DEUR40 (DAX)": 1,
    "GBPJPY": 100000,
    "USDJPY": 100000,
}

MFE_ACTIVATION_MULTIPLIER = 1.0
MFE_TRAIL_PCT = 0.5
DEFAULT_MAX_HOLDING = 100
BARS_PER_YEAR = 252

MAX_ALLOWED_LOTS = 50.0


TRANSACTION_COST_BPS = 2.5
SLIPPAGE_PCT = 0.05
AVERAGE_SPREAD_PIPS = 1.0
EXECUTION_DELAY_BARS = 1
ORDER_FILL_PROB = 0.98

XGB_N_ESTIMATORS = 150
XGB_MAX_DEPTH = 3
XGB_LEARNING_RATE = 0.05
XGB_REFIT_EVERY = 1000
XGB_HORIZON = 10
XGB_TP_MULT = 1.5
XGB_SL_MULT = 1.5

MACRO_BLACKOUT_MINUTES = 15
USE_NLP_SENTIMENT = True

DEFAULT_STRATEGIES = {
    "ZScore Rev": "ZScoreMeanReversion",
    "SMC Breakout": "VolatilityBreakout",
    "XGB Breakout": "MLVolatilityBreakout",
    "XGB Bounce": "MLBounceReversion",
    "VWAP Bounce": "VWAPBounceStrategy",
    "MTF Momentum": "MultiTimeframeMomentum",
    "LGBM Arab Scalp": "ArabianScalper",
}
