"""
MetaTrader5 data connector with multi-timeframe and chunked fetching support.
"""

import MetaTrader5 as mt5
import numpy as np
import pandas as pd


def get_mt5_data(
    symbol: str,
    days: int = 730,
    timeframe=mt5.TIMEFRAME_M1,
) -> pd.DataFrame:
    """Fetch OHLCV data from the MT5 terminal.

    Retrieves historical bars in chunks to avoid MT5 API limits.
    Default history depth is 730 days (≈2 years) so that swing
    strategies have enough data for indicator warm-up and statistical
    significance.

    Args:
        symbol:    MT5 broker symbol (e.g. ``EURUSD.ecn``).
        days:      Number of calendar days of history to request.
        timeframe: MT5 timeframe constant (``TIMEFRAME_M1``, ``TIMEFRAME_H1``, etc.).

    Returns:
        DataFrame indexed by datetime with columns
        ``['Open', 'High', 'Low', 'Close', 'Volume']``.
        Empty DataFrame on connection failure.
    """
    if not mt5.initialize():
        return pd.DataFrame()

    bars_per_day = {
        mt5.TIMEFRAME_M1: 1440,
        mt5.TIMEFRAME_M5: 288,
        mt5.TIMEFRAME_M15: 96,
        mt5.TIMEFRAME_M30: 48,
        mt5.TIMEFRAME_H1: 24,
        mt5.TIMEFRAME_H4: 6,
        mt5.TIMEFRAME_D1: 1,
    }

    multiplier = bars_per_day.get(timeframe, 1440)
    total_bars = days * multiplier
    chunk_size = 50_000
    all_rates = []

    pos = 0
    while pos < total_bars:
        fetch_size = min(chunk_size, total_bars - pos)
        rates = mt5.copy_rates_from_pos(symbol, timeframe, pos, fetch_size)
        if rates is None or len(rates) == 0:
            break
        all_rates.insert(0, rates)
        pos += fetch_size

    mt5.shutdown()

    if not all_rates:
        return pd.DataFrame()

    rates = np.concatenate(all_rates)

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    df.rename(columns={
        'tick_volume': 'Volume',
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
    }, inplace=True)
    return df[['Open', 'High', 'Low', 'Close', 'Volume']]