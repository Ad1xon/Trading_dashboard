"""Technical indicators — RSI, ATR, VWAP, Bollinger, ADX, MACD, SuperTrend, Weis Wave Volume, session features, order-flow proxy."""

import pandas as pd
import numpy as np


def calculate_vwap_with_bands(
    df: pd.DataFrame,
    window: int = 200,
    session_reset: str | None = None,
) -> pd.DataFrame:
    """Rolling anchored VWAP with +/-2σ bands, optionally resetting at session boundaries."""
    df = df.copy()
    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['VP'] = df['Typical_Price'] * df['Volume']

    if session_reset is not None and isinstance(df.index, pd.DatetimeIndex):
        session_groups = _detect_session_groups(df.index, session_reset)
        roll_vp = df.groupby(session_groups)['VP'].cumsum()
        roll_vol = df.groupby(session_groups)['Volume'].cumsum()
        df['VWAP'] = roll_vp / (roll_vol + 1e-10)
        price_diff_sq = ((df['Typical_Price'] - df['VWAP']) ** 2) * df['Volume']
        roll_diff_sq = df.groupby(session_groups).apply(
            lambda g: g[['Typical_Price', 'Volume']].assign(
                pds=((g['Typical_Price'] - (g['VP'].cumsum() / (g['Volume'].cumsum() + 1e-10))) ** 2) * g['Volume']
            )['pds'].cumsum()
        )
        if hasattr(roll_diff_sq, 'droplevel'):
            roll_diff_sq = roll_diff_sq.droplevel(0)
        roll_diff_sq = roll_diff_sq.reindex(df.index, fill_value=0)
        df['VWAP_Std'] = np.sqrt(roll_diff_sq / (roll_vol + 1e-10))
    else:
        roll_vp = df['VP'].rolling(window=window, min_periods=1).sum()
        roll_vol = df['Volume'].rolling(window=window, min_periods=1).sum()
        df['VWAP'] = roll_vp / (roll_vol + 1e-10)
        price_diff_sq = ((df['Typical_Price'] - df['VWAP']) ** 2) * df['Volume']
        roll_diff_sq = price_diff_sq.rolling(window=window, min_periods=1).sum()
        df['VWAP_Std'] = np.sqrt(roll_diff_sq / (roll_vol + 1e-10))

    df['VWAP_Upper_2'] = df['VWAP'] + (2 * df['VWAP_Std'])
    df['VWAP_Lower_2'] = df['VWAP'] - (2 * df['VWAP_Std'])
    return df


def _detect_session_groups(index: pd.DatetimeIndex, session: str) -> pd.Series:
    """Assign a session group ID to each bar based on session open boundaries."""
    session_hours = {
        "asian": 0,
        "london": 8,
        "new_york": 13,
    }
    reset_hour = session_hours.get(session.lower(), 8)
    hours = index.hour
    dates = index.date
    groups = pd.Series(0, index=index)
    group_id = 0
    prev_in_session = False

    for i in range(len(index)):
        at_reset = hours[i] == reset_hour
        crossed_midnight = i > 0 and dates[i] != dates[i - 1]
        if at_reset and not prev_in_session:
            group_id += 1
            prev_in_session = True
        elif crossed_midnight and not at_reset:
            prev_in_session = False
        elif hours[i] != reset_hour:
            prev_in_session = False
        groups.iloc[i] = group_id

    return groups


def calculate_session_vwap(df: pd.DataFrame, session: str = "london") -> pd.DataFrame:
    """VWAP that resets at the specified session boundary."""
    return calculate_vwap_with_bands(df, session_reset=session)


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index with Wilder smoothing."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range — absolute volatility measure."""
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
    """Bollinger Bands."""
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
    """Moving Average Convergence Divergence."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return {
        'MACD_Line': macd_line, 'MACD_Signal': signal_line,
        'MACD_Hist': macd_line - signal_line,
    }


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index — trend strength gauge."""
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
    """Add binary session columns (Asian / London / New York) based on UTC hour."""
    df = df.copy()
    hour = df.index.hour if hasattr(df.index, 'hour') else pd.Series(0, index=df.index)
    df['Session_Asian'] = ((hour >= 0) & (hour < 8)).astype(int)
    df['Session_London'] = ((hour >= 8) & (hour < 16)).astype(int)
    df['Session_NY'] = ((hour >= 13) & (hour < 22)).astype(int)
    return df


def calculate_orderflow_proxy(df: pd.DataFrame) -> pd.Series:
    """Estimate directional volume delta from OHLCV bars."""
    bar_range = df['High'] - df['Low']
    close_open = df['Close'] - df['Open']
    return (close_open / (bar_range + 1e-10)) * df['Volume']


def calculate_return_autocorrelation(
    series: pd.Series,
    window: int = 20,
    lag: int = 1,
) -> pd.Series:
    """Rolling autocorrelation of returns at the specified lag."""
    returns = series.pct_change()
    return returns.rolling(window=window).apply(
        lambda x: pd.Series(x).autocorr(lag=lag) if len(x) > lag else 0,
        raw=False,
    )


