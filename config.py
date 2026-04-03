"""Project configuration — symbols, defaults, magic numbers."""

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
        # Fallback to English if file missing
        with open(os.path.join(base_dir, 'locales', 'en.json'), 'r', encoding='utf-8') as f:
            return json.load(f)


# Market Configuration
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
DEFAULT_CONTRACT_SIZE = 100000

# Trading Logic Configurations
MFE_ACTIVATION_MULTIPLIER = 1.0
MFE_TRAIL_PCT = 0.5
DEFAULT_MAX_HOLDING = 100
BARS_PER_YEAR = 252

# ML Hyperparameters
XGB_N_ESTIMATORS = 150
XGB_MAX_DEPTH = 4
XGB_LEARNING_RATE = 0.05
XGB_REFIT_EVERY = 50
XGB_HORIZON = 15

# Strategy mapping for UI
DEFAULT_STRATEGIES = {
    "strategy_ml_breakout": "MLVolatilityBreakout",
    "strategy_breakout": "VolatilityBreakout",
    "strategy_reversion": "ZScoreMeanReversion",
    "strategy_vwap_bounce": "VWAPBounceStrategy",
    "strategy_mtf_momentum": "MultiTimeframeMomentum",
}