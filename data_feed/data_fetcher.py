"""Yahoo Finance historical data fetcher (fallback / alternative data source)."""

import yfinance as yf
import pandas as pd


def fetch_historical_data(
    ticker: str,
    start,
    end,
    interval: str = "1m",
) -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance."""
    df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.astype(float)
