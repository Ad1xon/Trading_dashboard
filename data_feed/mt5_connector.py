"""MetaTrader5 data connector."""

import MetaTrader5 as mt5
import pandas as pd


def get_mt5_data(symbol: str, days: int, timeframe=mt5.TIMEFRAME_M1) -> pd.DataFrame:
    """Fetch M1 OHLCV data from MT5 terminal."""
    if not mt5.initialize():
        return pd.DataFrame()

    total_bars = days * 1440
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, total_bars)
    mt5.shutdown()

    if rates is None or len(rates) == 0:
        return pd.DataFrame()

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    df.rename(columns={
        'tick_volume': 'Volume', 'open': 'Open',
        'high': 'High', 'low': 'Low', 'close': 'Close',
    }, inplace=True)
    return df[['Open', 'High', 'Low', 'Close', 'Volume']]