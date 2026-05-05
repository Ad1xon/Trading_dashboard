"""Portfolio-level risk metrics — VaR, CVaR, Kelly, MAE/MFE, correlation matrix."""

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from config import KELLY_FRACTION_CAP


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


def compute_parametric_var(
    returns: np.ndarray,
    confidence: float = 0.95,
    method: str = "cornish_fisher",
) -> float:
    """Parametric VaR with optional Cornish-Fisher skewness/kurtosis correction."""
    if len(returns) < 30:
        return compute_var(returns, confidence)

    mu = np.mean(returns)
    sigma = np.std(returns, ddof=1)
    z = sp_stats.norm.ppf(1 - confidence)

    if method == "cornish_fisher":
        s = sp_stats.skew(returns)
        k = sp_stats.kurtosis(returns, fisher=True)
        z_cf = z + (z**2 - 1) * s / 6 + (z**3 - 3 * z) * k / 24 - (2 * z**3 - 5 * z) * s**2 / 36
        return float(mu + z_cf * sigma)

    return float(mu + z * sigma)


def compute_kelly_fraction(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """Kelly Criterion optimal fraction: f* = (p·b - q) / b."""
    if avg_loss <= 0 or win_rate <= 0:
        return 0.0
    b = avg_win / avg_loss
    q = 1.0 - win_rate
    kelly = (win_rate * b - q) / b
    return float(min(max(kelly, 0.0), KELLY_FRACTION_CAP))


def compute_mae_mfe(trades: list) -> dict:
    """Maximum Adverse Excursion and Maximum Favorable Excursion analysis."""
    if not trades:
        return {"mae_mean": 0.0, "mfe_mean": 0.0, "efficiency": 0.0}
    pnls = np.array([t['pnl'] for t in trades])
    entries = np.array([t['entry_price'] for t in trades])
    exits = np.array([t['exit_price'] for t in trades])
    types = np.array([t['type'] for t in trades])

    mae_pcts = []
    mfe_pcts = []
    efficiencies = []

    for i, t in enumerate(trades):
        move = (exits[i] - entries[i]) / (entries[i] + 1e-10) * types[i]
        if move > 0:
            mfe_pcts.append(move)
            mae_pcts.append(0.0)
        else:
            mfe_pcts.append(0.0)
            mae_pcts.append(abs(move))

        if abs(move) > 1e-10:
            efficiencies.append(move / (abs(move) + 1e-10))

    return {
        "mae_mean": float(np.mean(mae_pcts)) if mae_pcts else 0.0,
        "mfe_mean": float(np.mean(mfe_pcts)) if mfe_pcts else 0.0,
        "mae_values": mae_pcts,
        "mfe_values": mfe_pcts,
        "efficiency": float(np.mean(efficiencies)) if efficiencies else 0.0,
    }


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
            "parametric_var": compute_parametric_var(rets, confidence),
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
