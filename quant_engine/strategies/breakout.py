# quant_engine/strategies/breakout.py
"""Breakout strategies."""

import pandas as pd
import numpy as np
from .base import BaseStrategy
from ..indicators import calculate_atr
from ..ml_models import XGBoostRangeBarModel


class VolatilityBreakout(BaseStrategy):
    """Breakout above rolling high/low with volume + trend confirmation and Chandelier exit."""

    params = {
        'lookback': (20, 10, 40, 5), 'vol_mult': (1.5, 1.0, 3.0, 0.25),
        'atr_sl_mult': (1.5, 1.0, 3.0, 0.5), 'atr_trail_mult': (3.0, 2.0, 5.0, 0.5),
        'ma_trend_len': (50, 20, 100, 10), 'max_holding': (100, 50, 300, 50),
    }

    def __init__(self, lookback=20, vol_mult=1.5, atr_sl_mult=1.5,
                 atr_trail_mult=3.0, ma_trend_len=50, max_holding=100):
        self.lookback = lookback
        self.vol_mult = vol_mult
        self.atr_sl_mult = atr_sl_mult
        self.atr_trail_mult = atr_trail_mult
        self.ma_trend_len = ma_trend_len
        self.max_holding = max_holding

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df['Local_High'] = df['High'].rolling(window=self.lookback).max().shift(1)
        df['Local_Low'] = df['Low'].rolling(window=self.lookback).min().shift(1)
        df['Mean'] = df['Close'].rolling(window=self.lookback).mean()
        df['Std'] = df['Close'].rolling(window=self.lookback).std()

        atr = calculate_atr(df, 14)
        df['ATR'] = atr

        vol_avg = df['Volume'].rolling(20).mean()
        vol_conf = df['Volume'] > self.vol_mult * vol_avg
        atr_expanding = atr > atr.rolling(20).mean()

        ma_trend = df['Close'].rolling(self.ma_trend_len).mean()
        trend_up = ma_trend > ma_trend.shift(1)
        trend_down = ma_trend < ma_trend.shift(1)

        df['Signal'] = 0
        long_cond = (df['Close'] > df['Local_High']) & vol_conf & atr_expanding & trend_up
        short_cond = (df['Close'] < df['Local_Low']) & vol_conf & atr_expanding & trend_down
        df.loc[long_cond, 'Signal'] = 1
        df.loc[short_cond, 'Signal'] = -1

        highest = df['High'].rolling(window=self.lookback).max()
        lowest = df['Low'].rolling(window=self.lookback).min()
        df['Exit_Long'] = df['Close'] < (highest - self.atr_trail_mult * atr)
        df['Exit_Short'] = df['Close'] > (lowest + self.atr_trail_mult * atr)

        df['SL_Price'] = np.nan
        df.loc[long_cond, 'SL_Price'] = df.loc[long_cond, 'Close'] - self.atr_sl_mult * df.loc[long_cond, 'ATR']
        df.loc[short_cond, 'SL_Price'] = df.loc[short_cond, 'Close'] + self.atr_sl_mult * df.loc[short_cond, 'ATR']
        df['TP_Price'] = np.nan
        df['Max_Hold'] = self.max_holding
        return df


class MLVolatilityBreakout(BaseStrategy):
    """ML-enhanced breakout — walk-forward XGBoost probability filter with ATR trailing exit."""

    params = {
        'lookback': (20, 10, 40, 5), 'prob_threshold': (0.55, 0.50, 0.70, 0.05),
        'atr_trail_mult': (3.0, 2.0, 5.0, 0.5), 'atr_sl_mult': (1.5, 1.0, 3.0, 0.5),
        'atr_tp_mult': (2.0, 1.0, 4.0, 0.5), 'max_holding': (100, 50, 300, 50),
    }

    def __init__(self, lookback=20, prob_threshold=0.55,
                 atr_trail_mult=3.0, atr_sl_mult=1.5, atr_tp_mult=2.0, max_holding=100):
        self.lookback = lookback
        self.prob_threshold = prob_threshold
        self.atr_trail_mult = atr_trail_mult
        self.atr_sl_mult = atr_sl_mult
        self.atr_tp_mult = atr_tp_mult
        self.max_holding = max_holding
        self.ml_model = XGBoostRangeBarModel(tp_mult=atr_tp_mult, sl_mult=atr_sl_mult)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df['Macro_Trend'] = df['Close'].rolling(window=100).mean()
        atr = calculate_atr(df, 14)
        df['ATR'] = atr
        df['Vol_MA_20'] = df['Volume'].rolling(20).mean()
        df['Vol_Ratio'] = df['Volume'] / (df['Vol_MA_20'] + 1e-6)

        df['_uid'] = np.arange(len(df))
        df_copy = df.copy()
        wf_data = self.ml_model.train(df_copy)
        df['Bull_Prob'] = 0.5

        df_features = self.ml_model.build_features(df)
        df_features.fillna(0, inplace=True)

        if wf_data is not None and 'WF_Prediction' in wf_data.columns:
            mapped = wf_data.set_index('_uid')['WF_Prediction']
            df['Bull_Prob'] = df['_uid'].map(mapped).fillna(0.5)
        else:
            trained_mask = df_features.index.isin(df_features.dropna().index)
            if trained_mask.any() and self.ml_model.is_trained:
                df.loc[trained_mask, 'Bull_Prob'] = self.ml_model.predict_proba(df_features[trained_mask])

        df.drop(columns=['_uid'], inplace=True)

        df['Local_High'] = df['High'].rolling(window=self.lookback).max().shift(1)
        df['Local_Low'] = df['Low'].rolling(window=self.lookback).min().shift(1)

        bull_breakout = (df['Close'] > df['Local_High']) & (df['Close'] > df['Macro_Trend']) & (df['Vol_Ratio'] > 1.0)
        bear_breakout = (df['Close'] < df['Local_Low']) & (df['Close'] < df['Macro_Trend']) & (df['Vol_Ratio'] > 1.0)

        df['Signal'] = 0
        df.loc[bull_breakout & (df['Bull_Prob'] > self.prob_threshold), 'Signal'] = 1
        df.loc[bear_breakout & (df['Bull_Prob'] < (1 - self.prob_threshold)), 'Signal'] = -1

        highest = df['High'].rolling(window=self.lookback).max()
        lowest = df['Low'].rolling(window=self.lookback).min()
        df['Exit_Long'] = df['Close'] < (highest - self.atr_trail_mult * atr)
        df['Exit_Short'] = df['Close'] > (lowest + self.atr_trail_mult * atr)
        df['Std'] = df['Close'].rolling(window=self.lookback).std()

        df['SL_Price'] = np.nan
        df['TP_Price'] = np.nan
        long_entries = df['Signal'] == 1
        short_entries = df['Signal'] == -1
        df.loc[long_entries, 'SL_Price'] = df.loc[long_entries, 'Close'] - self.atr_sl_mult * df.loc[
            long_entries, 'ATR']
        df.loc[short_entries, 'SL_Price'] = df.loc[short_entries, 'Close'] + self.atr_sl_mult * df.loc[
            short_entries, 'ATR']

        df.loc[long_entries, 'TP_Price'] = df.loc[long_entries, 'Close'] + self.atr_tp_mult * df.loc[
            long_entries, 'ATR']
        df.loc[short_entries, 'TP_Price'] = df.loc[short_entries, 'Close'] - self.atr_tp_mult * df.loc[
            short_entries, 'ATR']

        df['Max_Hold'] = self.max_holding
        return df