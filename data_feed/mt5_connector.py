"""MetaTrader5 data connector."""

import MetaTrader5 as mt5
import pandas as pd


def get_mt5_data(symbol: str, days: int, timeframe=mt5.TIMEFRAME_M1) -> pd.DataFrame:
    """Fetch M1 OHLCV data from MT5 terminal."""
    if not mt5.initialize():
        return pd.DataFrame()

    import numpy as np

    total_bars = days * 1440
    chunk_size = 50000
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
        'tick_volume': 'Volume', 'open': 'Open',
        'high': 'High', 'low': 'Low', 'close': 'Close',
    }, inplace=True)
    return df[['Open', 'High', 'Low', 'Close', 'Volume']]