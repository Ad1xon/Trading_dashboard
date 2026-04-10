"""Portfolio-level risk metrics — VaR, CVaR, correlation matrix, position correlation tracking."""

import numpy as np
import pandas as pd


def compute_var(returns: np.ndarray, confidence: float = 0.95) -> float:
    """Historical Value-at-Risk at the given confidence level."""
    if len(returns) < 2:
        return 0.0
    return float(np.percentile(returns, (1 - confidence) * 100))


def compute_cvar(returns: np.ndarray, confidence: float = 0.95) -> float:
    """Conditional VaR (Expected Shortfall) — mean of losses beyond VaR."""
    if len(returns) < 2:
        return 0.0
    var = compute_var(returns, confidence)
    tail = returns[returns <= var]
    return float(tail.mean()) if len(tail) > 0 else var


def compute_correlation_matrix(equity_dict: dict[str, np.ndarray]) -> pd.DataFrame:
    """Pairwise Pearson correlation from a dict of equity curves."""
    if len(equity_dict) < 2:
        return pd.DataFrame()
    returns_dict = {}
    for name, equity in equity_dict.items():
        eq_series = pd.Series(equity).replace(0, np.nan)
        returns_dict[name] = eq_series.pct_change().dropna()
    returns_df = pd.DataFrame(returns_dict)
    return returns_df.corr()


def compute_position_correlation(
    strategy_returns: dict[str, pd.Series],
    window: int = 50,
) -> dict[str, pd.DataFrame]:
    """Rolling pairwise correlation between strategy return streams."""
    if len(strategy_returns) < 2:
        return {"current": pd.DataFrame(), "rolling": pd.DataFrame()}
    returns_df = pd.DataFrame(strategy_returns)
    current_corr = returns_df.corr()
    rolling_corr = returns_df.rolling(window=window).corr()
    return {"current": current_corr, "rolling": rolling_corr}


def detect_correlation_clusters(
    corr_matrix: pd.DataFrame,
    threshold: float = 0.7,
) -> list[tuple[str, str, float]]:
    """Flag strategy pairs whose absolute correlation exceeds the threshold."""
    clusters = []
    if corr_matrix.empty:
        return clusters
    names = corr_matrix.columns.tolist()
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) >= threshold:
                clusters.append((names[i], names[j], float(corr_val)))
    return clusters


def compute_portfolio_risk_report(
    equity_dict: dict[str, np.ndarray],
    confidence: float = 0.95,
    bars_per_year: int = 252,
) -> dict:
    """Aggregated portfolio risk report with VaR, CVaR, correlation, per-strategy Sharpe."""
    all_returns = {}
    per_strategy = {}

    for name, equity in equity_dict.items():
        eq_series = pd.Series(equity).replace(0, np.nan)
        rets = eq_series.pct_change().dropna().values
        all_returns[name] = rets
        mean_ret = rets.mean() if len(rets) > 0 else 0.0
        std_ret = rets.std() if len(rets) > 1 else 1e-10
        per_strategy[name] = {
            "var": compute_var(rets, confidence),
            "cvar": compute_cvar(rets, confidence),
            "sharpe": (mean_ret / (std_ret + 1e-10)) * np.sqrt(bars_per_year),
        }

    corr_matrix = compute_correlation_matrix(equity_dict)
    clusters = detect_correlation_clusters(corr_matrix)

    combined_equity = np.mean(
        [eq for eq in equity_dict.values()], axis=0,
    )
    combined_returns = pd.Series(combined_equity).replace(0, np.nan).pct_change().dropna().values

    return {
        "per_strategy": per_strategy,
        "portfolio_var": compute_var(combined_returns, confidence),
        "portfolio_cvar": compute_cvar(combined_returns, confidence),
        "correlation_matrix": corr_matrix,
        "high_correlation_pairs": clusters,
    }
