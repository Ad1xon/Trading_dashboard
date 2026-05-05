"""Composite Alpha strategy — institutional-grade mid-frequency signal fusion."""

import logging

import pandas as pd
import numpy as np

from .base import BaseStrategy
from ..indicators import (
    calculate_rsi, calculate_atr, calculate_adx,
    calculate_bollinger, calculate_macd, calculate_orderflow_proxy,
)
from ..volatility_model import compute_garch_features

logger = logging.getLogger(__name__)


class CompositeAlphaStrategy(BaseStrategy):
    """Multi-factor alpha engine with GARCH-adaptive sizing and regime awareness.

    Fuses momentum, mean-reversion, volume microstructure, and volatility signals
    into a single composite score. No ML gate — the composite score IS the edge.
    Position risk scales with GARCH conditional volatility.
    """

    strategy_type = "scalp"

    params = {
        'composite_long': (0.15, 0.05, 0.40, 0.05),
        'composite_short': (0.15, 0.05, 0.40, 0.05),
        'atr_sl_mult': (2.0, 1.0, 3.5, 0.5),
        'atr_tp_mult': (3.5, 2.0, 6.0, 0.5),
        'max_holding': (120, 50, 250, 25),
    }

    def __init__(
        self,
        composite_long=0.15,
        composite_short=0.15,
        atr_sl_mult=2.0,
        atr_tp_mult=3.5,
        max_holding=120,
    ):
        super().__init__()
        self.composite_long = composite_long
        self.composite_short = composite_short
        self.atr_sl_mult = atr_sl_mult
        self.atr_tp_mult = atr_tp_mult
        self.max_holding = max_holding

    def _compute_momentum_score(self, df: pd.DataFrame) -> pd.Series:
        """Multi-horizon momentum z-score: 5/20/60 bars, normalized to [-1, 1]."""
        mom_5 = df['Close'].pct_change(5)
        mom_20 = df['Close'].pct_change(20)
        mom_60 = df['Close'].pct_change(60)
        composite = 0.5 * mom_5 + 0.3 * mom_20 + 0.2 * mom_60
        z = (composite - composite.rolling(60).mean()) / (composite.rolling(60).std() + 1e-10)
        return z.clip(-3, 3) / 3.0

    def _compute_reversion_score(self, df: pd.DataFrame) -> pd.Series:
        """Bollinger %B + RSI — anti-trend mean-reversion score in [-1, 1]."""
        bb = calculate_bollinger(df['Close'], 20, 2.0)
        pct_b = bb['BB_PctB']
        rsi = calculate_rsi(df['Close'], 14) / 100.0
        reversion = -((pct_b - 0.5) * 2) * 0.6 + -((rsi - 0.5) * 2) * 0.4
        return reversion.clip(-1, 1)

    def _compute_volume_score(self, df: pd.DataFrame) -> pd.Series:
        """Directional volume delta + volume surge, normalized to [-1, 1]."""
        delta = calculate_orderflow_proxy(df)
        delta_z = (delta - delta.rolling(20).mean()) / (delta.rolling(20).std() + 1e-10)
        vol_ratio = df['Volume'] / (df['Volume'].rolling(20).mean() + 1e-10)
        vol_surge = (vol_ratio - 1.0).clip(-2, 2) / 2.0
        return (0.7 * delta_z.clip(-3, 3) / 3.0 + 0.3 * vol_surge).clip(-1, 1)

    def _compute_volatility_score(self, df: pd.DataFrame) -> pd.Series:
        """GARCH regime signal — favour entries in low-vol environment."""
        if 'GARCH_Vol' not in df.columns:
            return pd.Series(0.0, index=df.index)
        garch_ma = df['GARCH_Vol'].rolling(60).mean()
        ratio = df['GARCH_Vol'] / (garch_ma + 1e-10)
        return -(ratio - 1.0).clip(-1, 1)

    def _compute_macd_score(self, df: pd.DataFrame) -> pd.Series:
        """MACD histogram direction normalized to [-1, 1]."""
        macd = calculate_macd(df['Close'])
        hist = macd['MACD_Hist']
        hist_z = (hist - hist.rolling(30).mean()) / (hist.rolling(30).std() + 1e-10)
        return hist_z.clip(-3, 3) / 3.0

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate composite alpha signals with ADX-adaptive factor weighting."""
        df['ATR'] = calculate_atr(df, 14)
        df['ADX'] = calculate_adx(df, 14)
        df['RSI'] = calculate_rsi(df['Close'], 14)
        df['Std'] = df['Close'].rolling(20).std()

        df = compute_garch_features(df)

        mom = self._compute_momentum_score(df)
        rev = self._compute_reversion_score(df)
        vol = self._compute_volume_score(df)
        garch = self._compute_volatility_score(df)
        macd = self._compute_macd_score(df)

        adx = df['ADX']
        trend_w = (adx / 40.0).clip(0, 1)
        rev_w = 1.0 - trend_w

        df['Composite_Score'] = (
            trend_w * mom * 0.35
            + rev_w * rev * 0.25
            + vol * 0.20
            + macd * 0.15
            + garch * 0.05
        )

        vol_above = df['Volume'] > df['Volume'].rolling(20).mean() * 0.8

        df['Signal'] = 0
        long_cond = (df['Composite_Score'] > self.composite_long) & vol_above
        short_cond = (df['Composite_Score'] < -self.composite_short) & vol_above
        df.loc[long_cond, 'Signal'] = 1
        df.loc[short_cond, 'Signal'] = -1

        df = self.apply_macro_filter(df)

        highest = df['High'].rolling(20).max()
        lowest = df['Low'].rolling(20).min()
        df['Exit_Long'] = (df['Close'] < (highest - 2.5 * df['ATR'])) | (df['Composite_Score'] < 0)
        df['Exit_Short'] = (df['Close'] > (lowest + 2.5 * df['ATR'])) | (df['Composite_Score'] > 0)

        df['SL_Price'] = np.nan
        df['TP_Price'] = np.nan
        long_e = df['Signal'] == 1
        short_e = df['Signal'] == -1

        garch_scale = (df['GARCH_Vol'] / (df['GARCH_Vol'].rolling(60).mean() + 1e-10)).clip(0.5, 2.0)
        adaptive_sl = self.atr_sl_mult * garch_scale
        adaptive_tp = self.atr_tp_mult * garch_scale

        df.loc[long_e, 'SL_Price'] = df.loc[long_e, 'Close'] - adaptive_sl.loc[long_e] * df.loc[long_e, 'ATR']
        df.loc[short_e, 'SL_Price'] = df.loc[short_e, 'Close'] + adaptive_sl.loc[short_e] * df.loc[short_e, 'ATR']
        df.loc[long_e, 'TP_Price'] = df.loc[long_e, 'Close'] + adaptive_tp.loc[long_e] * df.loc[long_e, 'ATR']
        df.loc[short_e, 'TP_Price'] = df.loc[short_e, 'Close'] - adaptive_tp.loc[short_e] * df.loc[short_e, 'ATR']

        df['Max_Hold'] = self.max_holding
        return df
