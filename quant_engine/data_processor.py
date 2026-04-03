"""Range bar generator from M1 OHLCV data."""

import pandas as pd


def generate_synthetic_range_bars(df_1m: pd.DataFrame, range_size: float) -> pd.DataFrame:
    """Convert 1-minute OHLCV into range bars of specified size."""
    bars = []
    current_bar = None

    for index, row in df_1m.iterrows():
        if current_bar is None:
            current_bar = {
                'Timestamp': index,
                'Open': row['Open'],
                'High': row['High'],
                'Low': row['Low'],
                'Volume': row['Volume'],
            }
        else:
            current_bar['High'] = max(current_bar['High'], row['High'])
            current_bar['Low'] = min(current_bar['Low'], row['Low'])
            current_bar['Volume'] += row['Volume']

        if (current_bar['High'] - current_bar['Low']) >= range_size:
            current_bar['Close'] = row['Close']
            bars.append(current_bar)
            current_bar = None

    if current_bar is not None and 'Close' not in current_bar:
        current_bar['Close'] = row['Close']
        bars.append(current_bar)

    range_df = pd.DataFrame(bars)
    if not range_df.empty:
        range_df.set_index('Timestamp', inplace=True)
    return range_df