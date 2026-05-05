"""GARCH/ARCH volatility modeling for conditional variance forecasting."""

import numpy as np
import pandas as pd


def fit_garch_volatility(
    returns: pd.Series,
    omega: float = 0.00001,
    alpha: float = 0.05,
    beta: float = 0.90,
) -> pd.Series:
    """GARCH(1,1) conditional variance: σ²_t = ω + α·r²_{t-1} + β·σ²_{t-1}."""
    n = len(returns)
    variance = np.zeros(n)
    variance[0] = returns.var() if len(returns) > 1 else omega / (1 - alpha - beta + 1e-10)

    r = returns.values
    for t in range(1, n):
        variance[t] = omega + alpha * r[t - 1] ** 2 + beta * variance[t - 1]

    return pd.Series(np.sqrt(np.maximum(variance, 1e-12)), index=returns.index, name='GARCH_Vol')


def fit_ewma_volatility(returns: pd.Series, decay: float = 0.94) -> pd.Series:
    """RiskMetrics EWMA volatility: σ²_t = λ·σ²_{t-1} + (1-λ)·r²_{t-1}."""
    return returns.ewm(alpha=1 - decay, min_periods=5).std()


def compute_garch_features(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Add GARCH conditional volatility and vol-of-vol to a DataFrame."""
    df = df.copy()
    returns = df['Close'].pct_change().fillna(0)
    df['GARCH_Vol'] = fit_garch_volatility(returns)
    df['EWMA_Vol'] = fit_ewma_volatility(returns)
    df['Realized_Vol'] = returns.rolling(window).std()
    df['Vol_of_Vol'] = df['GARCH_Vol'].rolling(window).std()
    df['Vol_Regime'] = np.where(
        df['GARCH_Vol'] > df['GARCH_Vol'].rolling(60).mean(),
        'high_vol', 'low_vol',
    )
    return df
