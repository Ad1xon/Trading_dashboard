"""Cointegration test for statistical arbitrage."""

import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint


def test_cointegration(asset_y: pd.Series, asset_x: pd.Series) -> dict:
    """Test pair cointegration and return spread, Z-score, and beta."""
    score, p_value, _ = coint(asset_y, asset_x)
    X = sm.add_constant(asset_x)
    model = sm.OLS(asset_y, X).fit()
    beta = model.params.iloc[1]
    spread = asset_y - beta * asset_x
    spread_mean = spread.rolling(window=50).mean()
    spread_std = spread.rolling(window=50).std()
    z_score = (spread - spread_mean) / spread_std
    return {
        "is_cointegrated": p_value < 0.05,
        "p_value": p_value,
        "beta": beta,
        "spread": spread,
        "z_score": z_score,
    }