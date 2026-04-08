"""
Technical indicators — RSI, ATR, VWAP, Bollinger, ADX, MACD,
SuperTrend, Weis Wave Volume, session features, order-flow proxy.
"""

import pandas as pd
import numpy as np


def calculate_vwap_with_bands(df: pd.DataFrame, window: int = 200) -> pd.DataFrame:
    """Rolling anchored VWAP with ±2σ bands.

    Uses a rolling window instead of cumulative sum to prevent the
    indicator from flattening on multi-week datasets.

    Args:
        df:     DataFrame with ``High``, ``Low``, ``Close``, ``Volume``.
        window: Rolling lookback for VWAP calculation.

    Returns:
        Copy of *df* with added ``VWAP``, ``VWAP_Upper_2``, ``VWAP_Lower_2``.
    """
    df = df.copy()
    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['VP'] = df['Typical_Price'] * df['Volume']
    roll_vp = df['VP'].rolling(window=window, min_periods=1).sum()
    roll_vol = df['Volume'].rolling(window=window, min_periods=1).sum()
    df['VWAP'] = roll_vp / (roll_vol + 1e-10)
    price_diff_sq = ((df['Typical_Price'] - df['VWAP']) ** 2) * df['Volume']
    roll_diff_sq = price_diff_sq.rolling(window=window, min_periods=1).sum()
    df['VWAP_Std'] = np.sqrt(roll_diff_sq / (roll_vol + 1e-10))
    df['VWAP_Upper_2'] = df['VWAP'] + (2 * df['VWAP_Std'])
    df['VWAP_Lower_2'] = df['VWAP'] - (2 * df['VWAP_Std'])
    return df


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index with Wilder smoothing.

    Args:
        series: Price series (typically Close).
        period: RSI lookback period.

    Returns:
        Series of RSI values in the 0–100 range.
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range — absolute volatility measure.

    Computed via Wilder EMA of True Range.

    Args:
        df:     DataFrame with ``High``, ``Low``, ``Close``.
        period: Smoothing period.

    Returns:
        Series of ATR values.
    """
    close_prev = df['Close'].shift(1)
    tr1 = df['High'] - df['Low']
    tr2 = (df['High'] - close_prev).abs()
    tr3 = (df['Low'] - close_prev).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def calculate_bollinger(
    series: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> dict:
    """Bollinger Bands.

    Args:
        series:  Price series (typically Close).
        period:  Moving-average lookback.
        num_std: Band width in standard deviations.

    Returns:
        Dict with keys ``BB_Mid``, ``BB_Upper``, ``BB_Lower``,
        ``BB_Width``, ``BB_PctB``.
    """
    mid = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    width = (upper - lower) / (mid + 1e-10)
    pct_b = (series - lower) / (upper - lower + 1e-10)
    return {
        'BB_Mid': mid, 'BB_Upper': upper, 'BB_Lower': lower,
        'BB_Width': width, 'BB_PctB': pct_b,
    }


def calculate_macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> dict:
    """Moving Average Convergence Divergence.

    Returns:
        Dict with keys ``MACD_Line``, ``MACD_Signal``, ``MACD_Hist``.
    """
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return {
        'MACD_Line': macd_line, 'MACD_Signal': signal_line,
        'MACD_Hist': macd_line - signal_line,
    }


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index — trend strength gauge.

    Values above 25 indicate a trending market; below 20 a ranging one.

    Args:
        df:     DataFrame with ``High``, ``Low``, ``Close``.
        period: Smoothing period for DI and ADX.

    Returns:
        Series of ADX values.
    """
    plus_dm = df['High'].diff()
    minus_dm = -df['Low'].diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    atr = calculate_atr(df, period)
    alpha = 1 / period
    plus_di = 100 * (plus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean() / (atr + 1e-10))
    minus_di = 100 * (minus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean() / (atr + 1e-10))
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10))
    return dx.ewm(alpha=alpha, min_periods=period, adjust=False).mean()


def calculate_session_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary session columns (Asian / London / New York) based on UTC hour.

    Returns:
        Copy of *df* with ``Session_Asian``, ``Session_London``,
        ``Session_NY`` columns (0 or 1).
    """
    df = df.copy()
    hour = df.index.hour if hasattr(df.index, 'hour') else pd.Series(0, index=df.index)
    df['Session_Asian'] = ((hour >= 0) & (hour < 8)).astype(int)
    df['Session_London'] = ((hour >= 8) & (hour < 16)).astype(int)
    df['Session_NY'] = ((hour >= 13) & (hour < 22)).astype(int)
    return df


def calculate_orderflow_proxy(df: pd.DataFrame) -> pd.Series:
    """Estimate directional volume delta from OHLCV bars.

    Uses a Close-minus-Open heuristic normalised by bar range.
    """
    bar_range = df['High'] - df['Low']
    close_open = df['Close'] - df['Open']
    return (close_open / (bar_range + 1e-10)) * df['Volume']


def calculate_return_autocorrelation(
    series: pd.Series,
    window: int = 20,
    lag: int = 1,
) -> pd.Series:
    """Rolling autocorrelation of returns at the specified lag.

    Useful for detecting mean-reversion or momentum regimes.
    """
    returns = series.pct_change()
    return returns.rolling(window=window).apply(
        lambda x: pd.Series(x).autocorr(lag=lag) if len(x) > lag else 0,
        raw=False,
    )


def calculate_supertrend(
    df: pd.DataFrame,
    period: int = 10,
    multiplier: float = 3.0,
) -> pd.DataFrame:
    """Classic SuperTrend indicator.

    Convention:
        ``SuperTrend_Trend =  1`` → uptrend  (price above lower band).
        ``SuperTrend_Trend = -1`` → downtrend (price below upper band).

    The bands ratchet: the lower band can only rise, the upper band
    can only fall, until the trend flips.

    Args:
        df:         DataFrame with ``High``, ``Low``, ``Close``.
        period:     ATR lookback.
        multiplier: ATR multiplier for band width.

    Returns:
        DataFrame with ``SuperTrend_Upper``, ``SuperTrend_Lower``,
        ``SuperTrend_Trend``.
    """
    atr = calculate_atr(df, period)
    hl2 = (df['High'] + df['Low']) / 2.0

    basic_upper = hl2 + (multiplier * atr)
    basic_lower = hl2 - (multiplier * atr)

    close = df['Close'].values

    final_upper = np.zeros(len(df))
    final_lower = np.zeros(len(df))
    trend = np.ones(len(df))

    for i in range(len(df)):
        if i == 0:
            final_upper[i] = basic_upper.iloc[i]
            final_lower[i] = basic_lower.iloc[i]
            trend[i] = 1
            continue

        prev_upper = final_upper[i - 1]
        prev_lower = final_lower[i - 1]
        prev_close = close[i - 1]

        if basic_upper.iloc[i] < prev_upper or prev_close > prev_upper:
            final_upper[i] = basic_upper.iloc[i]
        else:
            final_upper[i] = prev_upper

        if basic_lower.iloc[i] > prev_lower or prev_close < prev_lower:
            final_lower[i] = basic_lower.iloc[i]
        else:
            final_lower[i] = prev_lower

        if trend[i - 1] == 1:
            if close[i] < final_lower[i]:
                trend[i] = -1
            else:
                trend[i] = 1
        else:
            if close[i] > final_upper[i]:
                trend[i] = 1
            else:
                trend[i] = -1

    res = pd.DataFrame(index=df.index)
    res['SuperTrend_Upper'] = final_upper
    res['SuperTrend_Lower'] = final_lower
    res['SuperTrend_Trend'] = trend
    return res


def calculate_weis_wave_volume(df: pd.DataFrame) -> pd.Series:
    """Weis Wave Volume — cumulative directional volume within each wave.

    A new wave starts whenever the bar direction (bullish / bearish)
    changes.  Positive values represent buying waves, negative values
    selling waves.
    """
    dir_series = np.where(df['Close'] >= df['Open'], 1, -1)
    dir_series = pd.Series(dir_series, index=df.index)
    trend_shift = dir_series != dir_series.shift(1)
    wave_id = trend_shift.cumsum()
    weis_vol = df.groupby(wave_id)['Volume'].cumsum() * dir_series
    return weis_vol