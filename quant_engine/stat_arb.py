"""Cointegration test for statistical arbitrage pair trading."""

import logging

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller

logger = logging.getLogger(__name__)


def test_cointegration(
    asset_y: pd.Series,
    asset_x: pd.Series,
    p_threshold: float = 0.10,
    lookback: int = 60,
) -> dict:
    """Multi-method cointegration test with log-price normalization and ADF verification."""
    y = asset_y.dropna().astype(float)
    x = asset_x.dropna().astype(float)
    min_len = min(len(y), len(x))
    y = y.iloc[-min_len:].reset_index(drop=True)
    x = x.iloc[-min_len:].reset_index(drop=True)

    if min_len < 100:
        return _empty_result(y, "Insufficient data (need >= 100 bars)")

    y_log = np.log(y.clip(lower=1e-10))
    x_log = np.log(x.clip(lower=1e-10))

    y_norm = (y_log - y_log.mean()) / (y_log.std() + 1e-10)
    x_norm = (x_log - x_log.mean()) / (x_log.std() + 1e-10)

    best_p = 1.0
    best_method = "none"

    try:
        _, p_c, _ = coint(y_norm, x_norm, trend='c')
        if p_c < best_p:
            best_p = p_c
            best_method = "EG_constant"
    except Exception:
        pass

    try:
        _, p_ct, _ = coint(y_norm, x_norm, trend='ct')
        if p_ct < best_p:
            best_p = p_ct
            best_method = "EG_constant_trend"
    except Exception:
        pass

    try:
        _, p_raw, _ = coint(y, x, trend='c')
        if p_raw < best_p:
            best_p = p_raw
            best_method = "EG_raw_levels"
    except Exception:
        pass

    X_ols = sm.add_constant(x_norm)
    model = sm.OLS(y_norm, X_ols).fit()
    beta = float(model.params.iloc[1])

    spread = y_norm - beta * x_norm
    spread_mean = spread.rolling(window=lookback).mean()
    spread_std = spread.rolling(window=lookback).std()
    z_score = (spread - spread_mean) / (spread_std + 1e-10)

    adf_stat = None
    try:
        adf_result = adfuller(spread.dropna(), maxlag=10)
        adf_stat = adf_result[1]
        if adf_stat < best_p:
            best_p = adf_stat
            best_method = "ADF_spread"
    except Exception:
        pass

    is_cointegrated = best_p < p_threshold

    X_raw = sm.add_constant(x)
    raw_model = sm.OLS(y, X_raw).fit()
    raw_beta = float(raw_model.params.iloc[1])

    return {
        "is_cointegrated": is_cointegrated,
        "p_value": float(best_p),
        "beta": raw_beta,
        "beta_normalized": beta,
        "spread": spread,
        "z_score": z_score,
        "method": best_method,
        "adf_p_value": float(adf_stat) if adf_stat is not None else None,
    }


def _empty_result(series, reason: str) -> dict:
    """Return empty result dict when test cannot be performed."""
    logger.warning("Cointegration test failed: %s", reason)
    return {
        "is_cointegrated": False,
        "p_value": 1.0,
        "beta": 0.0,
        "beta_normalized": 0.0,
        "spread": pd.Series(0.0, index=range(len(series))),
        "z_score": pd.Series(0.0, index=range(len(series))),
        "method": "none",
        "adf_p_value": None,
    }
