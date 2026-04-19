"""Market scanner logic encompassing data fetching, indicator application, and signal detection."""

from typing import List, Dict, Any, Optional

from config import MT5_SYMBOLS
from data_feed.mt5_connector import get_mt5_data
from quant_engine.data_processor import generate_synthetic_range_bars
from quant_engine.indicators import calculate_vwap_with_bands, calculate_rsi, calculate_atr
from quant_engine.strategies import detect_liquidity_sweep
from utils.event_bus import EventBus


class MarketScanner:
    """Core market scanner engine"""

    def __init__(self):
        """Initialise scanner with event bus."""
        self.bus = EventBus()

    def run_scan(
        self,
        selected_symbols: List[str],
        data_mode_scan: str,
        tf_options: Dict[str, Any],
        scan_days: int,
        range_val: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Execute market scan across selected symbols returning unformatted results."""
        results = []

        for sym_name in selected_symbols:
            mt5_sym = MT5_SYMBOLS.get(sym_name)
            if not mt5_sym:
                continue
                
            mt5_tf = tf_options.get(data_mode_scan)
            df_raw = get_mt5_data(mt5_sym, scan_days, timeframe=mt5_tf)
            
            if df_raw.empty:
                continue

            if "Range Bars" in data_mode_scan and range_val is not None:
                rb = generate_synthetic_range_bars(df_raw, range_val)
            else:
                rb = df_raw.copy()

            if len(rb) < 30:
                continue

            rb_vwap = calculate_vwap_with_bands(rb)
            rb_vwap['RSI'] = calculate_rsi(rb_vwap['Close'], 14)
            rb_vwap['ATR'] = calculate_atr(rb_vwap, 14)

            latest = rb_vwap.iloc[-1]
            signal = "—"
            confidence = 0.0

            sweep = detect_liquidity_sweep(rb)
            if sweep["signal"]:
                signal = sweep["type"]
                confidence = 0.75
                self.bus.publish_sync("LIQUIDITY_SWEEP", {
                    "symbol": sym_name,
                    "message": sweep["message"],
                    "signal_type": sweep["type"],
                    "confidence": confidence,
                })

            results.append({
                "symbol": sym_name,
                "price": latest['Close'],
                "vwap": latest.get('VWAP', 0.0),
                "rsi": latest.get('RSI', 0.0),
                "atr": latest.get('ATR', 0.0),
                "signal": signal,
                "confidence": confidence,
            })

        return results
