"""Live trading bot configuration — credentials, symbols, risk limits.

All sensitive values are loaded from environment variables.
Create a .env file in the project root with the required keys.
"""

import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


MT5_LOGIN = int(os.getenv("MT5_LOGIN", "12345678"))
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "dummy_password_CHANGE_ME")
MT5_SERVER = os.getenv("MT5_SERVER", "DemoServer-CHANGE_ME")
MT5_PATH = os.getenv("MT5_PATH", r"C:\Program Files\MetaTrader 5\terminal64.exe")

DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK", "")

MFT_SYMBOLS = os.getenv("LIVE_MFT_SYMBOLS", "EURUSD.ecn,GBPUSD.ecn,XAUUSDs").split(",")
SWING_SYMBOLS = os.getenv("LIVE_SWING_SYMBOLS", "EURUSD.ecn,XAUUSDs,NASUSD.ecn").split(",")

MFT_TIMEFRAME_STR = os.getenv("LIVE_MFT_TIMEFRAME", "M5")
SWING_TIMEFRAME_STR = os.getenv("LIVE_SWING_TIMEFRAME", "H4")

POLL_INTERVAL = int(os.getenv("LIVE_POLL_INTERVAL", "5"))
MAX_DAILY_DRAWDOWN_PCT = float(os.getenv("LIVE_MAX_DAILY_DD", "0.03"))
MAX_CONSECUTIVE_LOSSES = int(os.getenv("LIVE_MAX_CONSEC_LOSSES", "5"))
MAX_TOTAL_EXPOSURE_LOTS = float(os.getenv("LIVE_MAX_EXPOSURE_LOTS", "10.0"))
RECONNECT_BASE_DELAY = float(os.getenv("LIVE_RECONNECT_BASE", "2.0"))
RECONNECT_MAX_DELAY = float(os.getenv("LIVE_RECONNECT_MAX", "300.0"))
HEARTBEAT_INTERVAL = int(os.getenv("LIVE_HEARTBEAT_INTERVAL", "60"))

RISK_PER_TRADE_PCT = float(os.getenv("LIVE_RISK_PER_TRADE", "0.01"))
DEFAULT_CAPITAL = float(os.getenv("LIVE_CAPITAL", "10000.0"))

LOG_FILE = os.getenv("LIVE_LOG_FILE", "live_engine.log")
STATE_FILE = os.getenv("LIVE_STATE_FILE", "live_state.json")
