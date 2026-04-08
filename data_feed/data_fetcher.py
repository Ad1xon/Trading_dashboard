"""
Yahoo Finance historical data fetcher (fallback / alternative data source).
"""

import yfinance as yf
import pandas as pd


def fetch_historical_data(
    ticker: str,
    start,
    end,
    interval: str = "1m",
) -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance.

    Args:
        ticker:   Yahoo Finance ticker symbol.
        start:    Start date (str or datetime-like).
        end:      End date (str or datetime-like).
        interval: Bar interval string (``'1m'``, ``'1h'``, ``'1d'``, etc.).

    Returns:
        DataFrame with float-typed OHLCV columns.
        Empty DataFrame when no data is available.
    """
    df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.astype(float)