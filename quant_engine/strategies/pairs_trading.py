"""Pairs Trading strategy — statistical arbitrage on cointegrated instrument pairs."""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

from .base import BaseStrategy
from ..indicators import calculate_atr


class PairsTradingStrategy(BaseStrategy):
    """Z-score spread reversion on cointegrated pairs with dynamic hedge ratio."""

    strategy_type = "reversion"

    params = {
        'z_entry': (2.0, 1.5, 3.0, 0.25),
        'z_exit': (0.5, 0.0, 1.0, 0.25),
        'lookback': (60, 30, 120, 10),
        'half_life_max': (50, 20, 100, 10),
        'atr_sl_mult': (2.5, 1.0, 4.0, 0.5),
        'max_holding': (100, 50, 200, 25),
    }

    def __init__(
        self,
        z_entry=2.0,
        z_exit=0.5,
        lookback=60,
        half_life_max=50,
        atr_sl_mult=2.5,
        max_holding=100,
        pair_data: pd.DataFrame | None = None,
    ):
        super().__init__()
        self.z_entry = z_entry
        self.z_exit = z_exit
        self.lookback = lookback
        self.half_life_max = half_life_max
        self.atr_sl_mult = atr_sl_mult
        self.max_holding = max_holding
        self.pair_data = pair_data
        self.cointegration_result: dict = {}

    def compute_spread(self, asset_y: pd.Series, asset_x: pd.Series) -> dict:
        """Compute spread analytics: OLS hedge ratio, z-score, half-life, cointegration p-value."""
        score, p_value, _ = coint(asset_y, asset_x)
        X = sm.add_constant(asset_x)
        model = sm.OLS(asset_y, X).fit()
        beta = model.params.iloc[1]

        spread = asset_y - beta * asset_x
        spread_mean = spread.rolling(window=self.lookback).mean()
        spread_std = spread.rolling(window=self.lookback).std()
        z_score = (spread - spread_mean) / (spread_std + 1e-10)

        half_life = self._estimate_half_life(spread)

        self.cointegration_result = {
            "is_cointegrated": p_value < 0.05,
            "p_value": float(p_value),
            "beta": float(beta),
            "half_life": half_life,
            "spread": spread,
            "z_score": z_score,
            "spread_mean": spread_mean,
            "spread_std": spread_std,
        }
        return self.cointegration_result

    @staticmethod
    def _estimate_half_life(spread: pd.Series) -> float:
        """Ornstein-Uhlenbeck half-life: τ = -ln(2) / ln(φ) from AR(1) on spread."""
        spread_lag = spread.shift(1)
        delta_spread = spread - spread_lag
        valid = ~(spread_lag.isna() | delta_spread.isna())
        if valid.sum() < 30:
            return 999.0
        X = sm.add_constant(spread_lag[valid])
        model = sm.OLS(delta_spread[valid], X).fit()
        phi = model.params.iloc[1]
        if phi >= 0:
            return 999.0
        return float(-np.log(2) / np.log(1 + phi))

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate spread z-score signals. Uses Close as primary, pair_data for hedge leg."""
        df['ATR'] = calculate_atr(df, 14)
        df['Std'] = df['Close'].rolling(20).std()

        if self.pair_data is not None and len(self.pair_data) >= len(df):
            pair_close = self.pair_data['Close'].iloc[-len(df):].values
            pair_series = pd.Series(pair_close, index=df.index)
            result = self.compute_spread(df['Close'], pair_series)
        else:
            result = self._generate_synthetic_spread(df)

        z = result['z_score']
        hl = result['half_life']

        df['Z_Score'] = z
        df['Spread'] = result['spread']

        valid_hl = hl < self.half_life_max
        df['Signal'] = 0

        if valid_hl:
            long_cond = z < -self.z_entry
            short_cond = z > self.z_entry
            df.loc[long_cond, 'Signal'] = 1
            df.loc[short_cond, 'Signal'] = -1

        df = self.apply_regime_filter(df)

        df['Exit_Long'] = z > -self.z_exit
        df['Exit_Short'] = z < self.z_exit

        df['SL_Price'] = np.nan
        long_entries = df['Signal'] == 1
        short_entries = df['Signal'] == -1
        df.loc[long_entries, 'SL_Price'] = (
            df.loc[long_entries, 'Close'] - self.atr_sl_mult * df.loc[long_entries, 'ATR']
        )
        df.loc[short_entries, 'SL_Price'] = (
            df.loc[short_entries, 'Close'] + self.atr_sl_mult * df.loc[short_entries, 'ATR']
        )
        df['TP_Price'] = np.nan
        df['Max_Hold'] = self.max_holding
        return df

    def _generate_synthetic_spread(self, df: pd.DataFrame) -> dict:
        """Fallback: use price vs. its own rolling mean as a synthetic spread."""
        spread = df['Close'] - df['Close'].rolling(self.lookback).mean()
        spread_mean = spread.rolling(self.lookback).mean()
        spread_std = spread.rolling(self.lookback).std()
        z_score = (spread - spread_mean) / (spread_std + 1e-10)
        half_life = self._estimate_half_life(spread)
        self.cointegration_result = {
            "is_cointegrated": False,
            "p_value": 1.0,
            "beta": 0.0,
            "half_life": half_life,
            "spread": spread,
            "z_score": z_score,
            "spread_mean": spread_mean,
            "spread_std": spread_std,
        }
        return self.cointegration_result
