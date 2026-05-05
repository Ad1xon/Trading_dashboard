"""Regime-adaptive meta-strategy with GARCH volatility scaling."""

import pandas as pd
import numpy as np

from .base import BaseStrategy
from ..indicators import calculate_rsi, calculate_atr, calculate_adx, calculate_bollinger
from ..volatility_model import compute_garch_features


class RegimeSwitchStrategy(BaseStrategy):
    """HMM-driven strategy switcher with GARCH-scaled position risk."""

    strategy_type = "trend"

    params = {
        'trend_rsi_entry': (30, 20, 40, 5),
        'rev_z_entry': (2.0, 1.5, 3.0, 0.25),
        'rev_z_exit': (0.5, 0.0, 1.0, 0.25),
        'atr_sl_mult': (2.0, 1.0, 3.0, 0.5),
        'atr_tp_mult': (3.0, 1.5, 5.0, 0.5),
        'max_holding': (120, 50, 300, 25),
    }

    def __init__(
        self,
        trend_rsi_entry=30,
        rev_z_entry=2.0,
        rev_z_exit=0.5,
        atr_sl_mult=2.0,
        atr_tp_mult=3.0,
        max_holding=120,
    ):
        super().__init__()
        self.trend_rsi_entry = trend_rsi_entry
        self.rev_z_entry = rev_z_entry
        self.rev_z_exit = rev_z_exit
        self.atr_sl_mult = atr_sl_mult
        self.atr_tp_mult = atr_tp_mult
        self.max_holding = max_holding

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate adaptive signals — momentum in bull/bear, mean-reversion in range."""
        df['ATR'] = calculate_atr(df, 14)
        df['RSI'] = calculate_rsi(df['Close'], 14)
        df['ADX'] = calculate_adx(df, 14)
        df['Std'] = df['Close'].rolling(20).std()
        df['SMA_100'] = df['Close'].rolling(100).mean()
        df['Fast_RSI'] = calculate_rsi(df['Close'], 7)

        df = compute_garch_features(df)

        bb = calculate_bollinger(df['Close'], 20, 2.0)
        df['BB_Upper'] = bb['BB_Upper']
        df['BB_Lower'] = bb['BB_Lower']
        df['BB_Mid'] = bb['BB_Mid']

        z_window = 20
        df['Mean'] = df['Close'].rolling(z_window).mean()
        z_std = df['Close'].rolling(z_window).std()
        df['Z_Score'] = (df['Close'] - df['Mean']) / (z_std + 1e-10)

        df = self.regime_detector.add_regime_column(df)

        df['Signal'] = 0
        df['Exit_Long'] = False
        df['Exit_Short'] = False

        bull_mask = df['Regime'] == 'bull'
        bear_mask = df['Regime'] == 'bear'
        range_mask = df['Regime'] == 'range'

        low_vol = df['Vol_Regime'] == 'low_vol'

        trend_up = df['Close'] > df['SMA_100']
        trend_down = df['Close'] < df['SMA_100']
        bull_long = bull_mask & trend_up & (df['Fast_RSI'] < self.trend_rsi_entry) & low_vol
        bear_short = bear_mask & trend_down & (df['Fast_RSI'] > (100 - self.trend_rsi_entry)) & low_vol

        df.loc[bull_long, 'Signal'] = 1
        df.loc[bear_short, 'Signal'] = -1

        vol_above_avg = df['Volume'] > df['Volume'].rolling(20).mean()
        range_long = range_mask & (df['Z_Score'] < -self.rev_z_entry) & (df['RSI'] < 35) & vol_above_avg
        range_short = range_mask & (df['Z_Score'] > self.rev_z_entry) & (df['RSI'] > 65) & vol_above_avg
        df.loc[range_long, 'Signal'] = 1
        df.loc[range_short, 'Signal'] = -1

        df = self.apply_macro_filter(df)

        highest = df['High'].rolling(20).max()
        lowest = df['Low'].rolling(20).min()
        df.loc[bull_mask | bear_mask, 'Exit_Long'] = (
            df['Close'] < (highest - 2.5 * df['ATR'])
        )
        df.loc[bull_mask | bear_mask, 'Exit_Short'] = (
            df['Close'] > (lowest + 2.5 * df['ATR'])
        )
        df.loc[range_mask, 'Exit_Long'] = df['Z_Score'] > -self.rev_z_exit
        df.loc[range_mask, 'Exit_Short'] = df['Z_Score'] < self.rev_z_exit

        garch_scale = df['GARCH_Vol'] / (df['GARCH_Vol'].rolling(60).mean() + 1e-10)
        adaptive_sl = self.atr_sl_mult * garch_scale.clip(0.5, 2.0)
        adaptive_tp = self.atr_tp_mult * garch_scale.clip(0.5, 2.0)

        df['SL_Price'] = np.nan
        df['TP_Price'] = np.nan
        long_entries = df['Signal'] == 1
        short_entries = df['Signal'] == -1

        df.loc[long_entries, 'SL_Price'] = (
            df.loc[long_entries, 'Close'] - adaptive_sl.loc[long_entries] * df.loc[long_entries, 'ATR']
        )
        df.loc[short_entries, 'SL_Price'] = (
            df.loc[short_entries, 'Close'] + adaptive_sl.loc[short_entries] * df.loc[short_entries, 'ATR']
        )
        df.loc[long_entries, 'TP_Price'] = (
            df.loc[long_entries, 'Close'] + adaptive_tp.loc[long_entries] * df.loc[long_entries, 'ATR']
        )
        df.loc[short_entries, 'TP_Price'] = (
            df.loc[short_entries, 'Close'] - adaptive_tp.loc[short_entries] * df.loc[short_entries, 'ATR']
        )

        df['Max_Hold'] = self.max_holding
        return df
