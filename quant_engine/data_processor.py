"""Range bar generator from M1 OHLCV data with large-candle splitting."""

import pandas as pd
import numpy as np


def generate_synthetic_range_bars(df_1m: pd.DataFrame, range_size: float) -> pd.DataFrame:
    """Convert 1-minute OHLCV into fixed-size range bars.

    Large M1 candles exceeding range_size are split into multiple
    synthetic bars to preserve the constant-range assumption required
    by downstream ML and indicator logic.
    """
    bars = []
    current_bar = None

    for index, row in df_1m.iterrows():
        candle_range = row['High'] - row['Low']

        if candle_range >= range_size * 2:
            if current_bar is not None:
                current_bar['Close'] = row['Open']
                bars.append(current_bar)
                current_bar = None

            n_splits = max(1, int(np.floor(candle_range / range_size)))
            is_bullish = row['Close'] >= row['Open']

            if is_bullish:
                start = row['Low']
                end = row['High']
            else:
                start = row['High']
                end = row['Low']

            step = (end - start) / n_splits
            vol_step = row['Volume'] / n_splits

            for s in range(n_splits):
                seg_open = start + s * step
                seg_close = start + (s + 1) * step
                bars.append({
                    'Timestamp': index + pd.Timedelta(microseconds=s*10),
                    'Open': seg_open,
                    'High': max(seg_open, seg_close),
                    'Low': min(seg_open, seg_close),
                    'Close': seg_close,
                    'Volume': vol_step,
                })
            continue

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