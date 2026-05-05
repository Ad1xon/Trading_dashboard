"""Project configuration — symbols, defaults, trading constants, ML hyperparameters."""

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
    "AUDUSD": "AUDUSD.ecn",
    "USDCAD": "USDCAD.ecn",
    "EURGBP": "EURGBP.ecn",
    "NZDUSD": "NZDUSD.ecn",
    "USDCHF": "USDCHF.ecn",
    "US500 (S&P)": "S&P500.ecn",
    "UK100 (FTSE)": "UK100.ecn",
    "XAGUSD (Silver)": "XAGUSDs",
    "USOIL (WTI)": "USOIL.ecn",
    "NGAS (Nat Gas)": "NGAS.ecn",
    "AAPL": "AAPL",
    "TSLA": "TSLA",
    "NVDA": "NVDA",
    "MSFT": "MSFT",
    "AMZN": "AMZN",
    "GOOG": "GOOG",
    "META": "META",
    "BTCUSD": "BTCUSDT.ecn",
    "ETHUSD": "ETHUSDT.ecn",
    "SOLUSD": "SOLUSDT.ecn",
    "BNBUSD": "BNBUSDT.ecn",
    "XRPUSD": "XRPUSDT.ecn",
    "ADAUSD": "ADAUSDT.ecn",
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
    "AUDUSD": 100000,
    "USDCAD": 100000,
    "EURGBP": 100000,
    "NZDUSD": 100000,
    "USDCHF": 100000,
    "US500 (S&P)": 1,
    "UK100 (FTSE)": 1,
    "XAGUSD (Silver)": 5000,
    "USOIL (WTI)": 1000,
    "NGAS (Nat Gas)": 10000,
    "AAPL": 1,
    "TSLA": 1,
    "NVDA": 1,
    "MSFT": 1,
    "AMZN": 1,
    "GOOG": 1,
    "META": 1,
    "BTCUSD": 1,
    "ETHUSD": 1,
    "SOLUSD": 1,
    "BNBUSD": 1,
    "XRPUSD": 1,
    "ADAUSD": 1,
}

PAIRS_TRADING_PAIRS = [
    ("EURUSD.ecn", "GBPUSD.ecn"),
    ("XAUUSDs", "XAGUSD.ecn"),
    ("AAPL", "MSFT"),
    ("BTCUSDT.ecn", "ETHUSDT.ecn"),
    ("USDJPY.ecn", "GBPJPY.ecn"),
    ("NASUSD.ecn", "SP500.ecn"),
]

MFE_ACTIVATION_MULTIPLIER = 1.0
MFE_TRAIL_PCT = 0.5
DEFAULT_MAX_HOLDING = 100
BARS_PER_YEAR = 252

BARS_PER_YEAR_MAP = {
    "M1": 252 * 1440,
    "M5": 252 * 288,
    "M15": 252 * 96,
    "M30": 252 * 48,
    "H1": 252 * 24,
    "H4": 252 * 6,
    "D1": 252,
    "RANGE": 252 * 500,
}

MAX_ALLOWED_LOTS = 50.0

TRANSACTION_COST_BPS = 2.5
AVERAGE_SPREAD_PIPS = 1.0
EXECUTION_DELAY_BARS = 1
ORDER_FILL_PROB = 0.98

SLIPPAGE_BASE_BPS = 1.0
SLIPPAGE_VOL_EXPONENT = 0.5

XGB_N_ESTIMATORS = 150
XGB_MAX_DEPTH = 3
XGB_LEARNING_RATE = 0.05
XGB_REFIT_EVERY = 1000
XGB_HORIZON = 10
XGB_TP_MULT = 1.5
XGB_SL_MULT = 1.5
XGB_EARLY_STOPPING_ROUNDS = 10

MACRO_BLACKOUT_MINUTES = 15
USE_NLP_SENTIMENT = True
SENTIMENT_ROLLING_WINDOW = 50

HMM_N_STATES = 3
HMM_COVARIANCE_TYPE = "full"

KELLY_FRACTION_CAP = 0.25

DEFAULT_STRATEGIES = {
    "Composite Alpha (MFT)": "CompositeAlphaStrategy",
    "LSTM Swing": "LSTMSwingStrategy",
    "Regime Switch (HMM)": "RegimeSwitchStrategy",
    "XGB Breakout (ML)": "MLVolatilityBreakout",
    "LGBM Arab Scalp (ML)": "ArabianScalper",
    "Pairs Trading (Stat Arb)": "PairsTradingStrategy",
    "MTF Momentum": "MultiTimeframeMomentum",
    "VWAP Bounce": "VWAPBounceStrategy",
    "SMC Breakout": "VolatilityBreakout",
    "ZScore Rev": "ZScoreMeanReversion",
}
