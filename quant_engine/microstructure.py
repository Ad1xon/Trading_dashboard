"""Market microstructure analytics — Volume Profile, LOB imbalance, OFI, Cumulative Delta, SuperTrend.

Production-grade implementations designed for Mid-Frequency Trading on OHLCV + tick_volume data.
All metrics use tick_volume as a proxy since retail MT5 does not expose real depth-of-book.
"""

import numpy as np
import pandas as pd


def compute_volume_profile(
    df: pd.DataFrame,
    num_bins: int = 50,
    value_area_pct: float = 0.70,
    window: int = 100,
) -> pd.DataFrame:
    """Rolling Volume Profile with Point of Control, Value Area High, and Value Area Low.

    Distributes volume across price bins within a rolling lookback window.
    POC is the price level with the highest traded volume.
    VAH/VAL bound the smallest range containing value_area_pct of total volume.

    Returns the original DataFrame with VP_POC, VP_VAH, VP_VAL columns appended.
    """
    df = df.copy()
    n = len(df)
    poc_arr = np.full(n, np.nan)
    vah_arr = np.full(n, np.nan)
    val_arr = np.full(n, np.nan)

    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values
    volumes = df["Volume"].values

    for i in range(window, n):
        slice_start = i - window
        slice_highs = highs[slice_start:i]
        slice_lows = lows[slice_start:i]
        slice_closes = closes[slice_start:i]
        slice_volumes = volumes[slice_start:i]

        price_low = slice_lows.min()
        price_high = slice_highs.max()
        if price_high - price_low < 1e-10:
            poc_arr[i] = slice_closes[-1]
            vah_arr[i] = price_high
            val_arr[i] = price_low
            continue

        bin_edges = np.linspace(price_low, price_high, num_bins + 1)
        bin_volumes = np.zeros(num_bins)

        for j in range(len(slice_closes)):
            typical = (slice_highs[j] + slice_lows[j] + slice_closes[j]) / 3.0
            bin_idx = int((typical - price_low) / (price_high - price_low) * (num_bins - 1))
            bin_idx = max(0, min(bin_idx, num_bins - 1))
            bin_volumes[bin_idx] += slice_volumes[j]

        poc_bin = int(np.argmax(bin_volumes))
        poc_arr[i] = (bin_edges[poc_bin] + bin_edges[poc_bin + 1]) / 2.0

        total_vol = bin_volumes.sum()
        if total_vol < 1e-10:
            vah_arr[i] = price_high
            val_arr[i] = price_low
            continue

        target_vol = total_vol * value_area_pct
        sorted_indices = np.argsort(bin_volumes)[::-1]
        cum_vol = 0.0
        selected_bins = []
        for idx in sorted_indices:
            cum_vol += bin_volumes[idx]
            selected_bins.append(idx)
            if cum_vol >= target_vol:
                break

        selected_bins_sorted = sorted(selected_bins)
        val_arr[i] = bin_edges[selected_bins_sorted[0]]
        vah_arr[i] = bin_edges[selected_bins_sorted[-1] + 1]

    df["VP_POC"] = poc_arr
    df["VP_VAH"] = vah_arr
    df["VP_VAL"] = val_arr
    return df


def compute_lob_imbalance(
    df: pd.DataFrame,
    window: int = 20,
) -> pd.Series:
    """Limit Order Book imbalance proxy via Bulk Volume Classification (BVC).

    Classifies each bar's volume as buy- or sell-initiated based on
    the close position within the high-low range (Lee-Ready simplified).
    Returns a rolling z-scored imbalance series.

    Positive values indicate buy pressure dominance; negative = sell pressure.
    """
    bar_range = df["High"] - df["Low"]
    close_position = (df["Close"] - df["Low"]) / (bar_range + 1e-10)

    buy_volume = close_position * df["Volume"]
    sell_volume = (1.0 - close_position) * df["Volume"]

    imbalance = (buy_volume - sell_volume) / (buy_volume + sell_volume + 1e-10)

    imbalance_mean = imbalance.rolling(window=window, min_periods=5).mean()
    imbalance_std = imbalance.rolling(window=window, min_periods=5).std()
    imbalance_z = (imbalance - imbalance_mean) / (imbalance_std + 1e-10)

    return imbalance_z.rename("LOB_Imbalance_Z")


def compute_order_flow_imbalance(
    df: pd.DataFrame,
    window: int = 20,
) -> pd.DataFrame:
    """Order Flow Imbalance (OFI) approximated from OHLCV bars.

    Estimates aggressive buyer/seller flow by decomposing volume using
    bar direction and close-position ratio. Returns the original DataFrame
    with OFI_Raw, OFI_Cumulative, and OFI_Z columns.

    OFI_Z > +1.5 suggests strong buying aggression.
    OFI_Z < -1.5 suggests strong selling aggression.
    """
    df = df.copy()

    bar_range = df["High"] - df["Low"]
    close_position = (df["Close"] - df["Low"]) / (bar_range + 1e-10)

    direction = np.sign(df["Close"] - df["Open"])
    aggressive_buy = close_position * df["Volume"] * np.where(direction >= 0, 1.0, 0.5)
    aggressive_sell = (1.0 - close_position) * df["Volume"] * np.where(direction <= 0, 1.0, 0.5)

    ofi_raw = aggressive_buy - aggressive_sell
    df["OFI_Raw"] = ofi_raw
    df["OFI_Cumulative"] = ofi_raw.cumsum()

    ofi_mean = ofi_raw.rolling(window=window, min_periods=5).mean()
    ofi_std = ofi_raw.rolling(window=window, min_periods=5).std()
    df["OFI_Z"] = (ofi_raw - ofi_mean) / (ofi_std + 1e-10)

    return df


def compute_cumulative_delta(
    df: pd.DataFrame,
    divergence_window: int = 20,
) -> pd.DataFrame:
    """Cumulative Volume Delta with price-delta divergence detection.

    Delta is computed as buy_volume - sell_volume using close-position
    within the bar range. Divergence flags are set when price makes a
    new high/low but delta does not confirm.

    Returns DataFrame with CVD, CVD_MA, Delta_Divergence_Bull, Delta_Divergence_Bear.
    """
    df = df.copy()

    bar_range = df["High"] - df["Low"]
    close_position = (df["Close"] - df["Low"]) / (bar_range + 1e-10)
    buy_vol = close_position * df["Volume"]
    sell_vol = (1.0 - close_position) * df["Volume"]
    delta = buy_vol - sell_vol

    df["CVD"] = delta.cumsum()
    df["CVD_MA"] = df["CVD"].rolling(window=divergence_window, min_periods=5).mean()

    price_high_roll = df["Close"].rolling(window=divergence_window).max()
    price_low_roll = df["Close"].rolling(window=divergence_window).min()
    cvd_high_roll = df["CVD"].rolling(window=divergence_window).max()
    cvd_low_roll = df["CVD"].rolling(window=divergence_window).min()

    price_at_high = df["Close"] >= price_high_roll
    cvd_below_high = df["CVD"] < cvd_high_roll * 0.95

    price_at_low = df["Close"] <= price_low_roll
    cvd_above_low = df["CVD"] > cvd_low_roll * 0.95

    df["Delta_Divergence_Bear"] = (price_at_high & cvd_below_high).astype(int)
    df["Delta_Divergence_Bull"] = (price_at_low & cvd_above_low).astype(int)

    return df


def compute_supertrend(
    df: pd.DataFrame,
    period: int = 10,
    multiplier: float = 3.0,
) -> pd.DataFrame:
    """ATR-based SuperTrend indicator without look-ahead bias.

    Computes upper/lower bands from ATR and flips trend direction based on
    close crossing the active band. Returns DataFrame with SuperTrend_Value
    and SuperTrend_Direction (+1 = bull, -1 = bear).

    Causal implementation: each bar's SuperTrend depends only on past data.
    """
    df = df.copy()
    n = len(df)

    close_prev = df["Close"].shift(1)
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - close_prev).abs()
    tr3 = (df["Low"] - close_prev).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    hl2 = (df["High"] + df["Low"]) / 2.0
    upper_basic = hl2 + multiplier * atr
    lower_basic = hl2 - multiplier * atr

    closes = df["Close"].values
    upper_basic_vals = upper_basic.values
    lower_basic_vals = lower_basic.values

    upper_band = np.zeros(n)
    lower_band = np.zeros(n)
    supertrend = np.zeros(n)
    direction = np.ones(n, dtype=int)

    upper_band[0] = upper_basic_vals[0]
    lower_band[0] = lower_basic_vals[0]
    supertrend[0] = upper_basic_vals[0]
    direction[0] = -1

    for i in range(1, n):
        if np.isnan(upper_basic_vals[i]):
            upper_band[i] = upper_band[i - 1]
            lower_band[i] = lower_band[i - 1]
            direction[i] = direction[i - 1]
            supertrend[i] = supertrend[i - 1]
            continue

        if upper_basic_vals[i] < upper_band[i - 1] or closes[i - 1] > upper_band[i - 1]:
            upper_band[i] = upper_basic_vals[i]
        else:
            upper_band[i] = upper_band[i - 1]

        if lower_basic_vals[i] > lower_band[i - 1] or closes[i - 1] < lower_band[i - 1]:
            lower_band[i] = lower_basic_vals[i]
        else:
            lower_band[i] = lower_band[i - 1]

        if direction[i - 1] == 1:
            if closes[i] < lower_band[i]:
                direction[i] = -1
                supertrend[i] = upper_band[i]
            else:
                direction[i] = 1
                supertrend[i] = lower_band[i]
        else:
            if closes[i] > upper_band[i]:
                direction[i] = 1
                supertrend[i] = lower_band[i]
            else:
                direction[i] = -1
                supertrend[i] = upper_band[i]

    df["SuperTrend_Value"] = supertrend
    df["SuperTrend_Direction"] = direction
    return df
