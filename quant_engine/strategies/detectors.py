"""Market-structure detectors — liquidity sweeps, statistical arbitrage setups."""

import pandas as pd
from ..indicators import calculate_vwap_with_bands


def detect_liquidity_sweep(df: pd.DataFrame) -> dict:
    """Detect VWAP-band liquidity sweeps on the latest two bars."""
    df = calculate_vwap_with_bands(df)
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    if prev['Low'] < prev['VWAP_Lower_2'] and latest['Close'] > latest['VWAP_Lower_2']:
        return {
            "signal": True,
            "type": "BULLISH_SWEEP",
            "message": f"Byczy Liquidity Sweep: {latest['VWAP_Lower_2']:.2f}.",
        }
    elif prev['High'] > prev['VWAP_Upper_2'] and latest['Close'] < latest['VWAP_Upper_2']:
        return {
            "signal": True,
            "type": "BEARISH_SWEEP",
            "message": f"Niedźwiedzi Liquidity Sweep: {latest['VWAP_Upper_2']:.2f}.",
        }
    return {"signal": False, "type": None, "message": ""}
