"""Trading strategies — all with params for optimizer, ATR SL/TP, max holding period."""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

from .ml_models import XGBoostRangeBarModel, StatArbMLFilter
from .indicators import (
    calculate_vwap_with_bands, calculate_rsi, calculate_atr,
    calculate_adx, calculate_bollinger,
)
from .stat_arb import test_cointegration
from config import DEFAULT_MAX_HOLDING, MFE_ACTIVATION_MULTIPLIER, MFE_TRAIL_PCT

class BaseStrategy(ABC):
    """All strategies expose *params* dict for optimiser introspection."""

    params: dict = {}

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    def get_params(self) -> dict:
        base = {
            'mfe_activation': MFE_ACTIVATION_MULTIPLIER,
            'mfe_trail_pct': MFE_TRAIL_PCT,
            'max_holding': DEFAULT_MAX_HOLDING
        }
        base.update({k: v[0] for k, v in self.params.items()})
        return base

    def get_param_ranges(self) -> dict:
        base = {
            'mfe_activation': (0.5, 3.0, 0.5),
            'mfe_trail_pct': (0.1, 0.9, 0.1),
        }
        base.update({k: v[1:] for k, v in self.params.items()})
        return base


class ZScoreMeanReversion(BaseStrategy):
    """Mean-reversion on Z-score extremes with RSI confirmation and ADX regime filter."""

    params = {
        'z_window': (20, 10, 60, 5), 'z_entry': (2.0, 1.5, 3.0, 0.25),
        'z_exit': (0.5, 0.0, 1.0, 0.25), 'rsi_long': (30, 20, 40, 5),
        'rsi_short': (70, 60, 80, 5), 'adx_max': (25, 15, 35, 5),
        'atr_sl_mult': (2.0, 1.0, 4.0, 0.5), 'max_holding': (100, 50, 300, 50),
    }

    def __init__(self, z_window=20, z_entry=2.0, z_exit=0.5,
                 rsi_long=30, rsi_short=70, adx_max=25,
                 atr_sl_mult=2.0, max_holding=100):
        self.z_window = z_window
        self.z_entry = z_entry
        self.z_exit = z_exit
        self.rsi_long = rsi_long
        self.rsi_short = rsi_short
        self.adx_max = adx_max
        self.atr_sl_mult = atr_sl_mult
        self.max_holding = max_holding

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df['Mean'] = df['Close'].rolling(window=self.z_window).mean()
        df['Std'] = df['Close'].rolling(window=self.z_window).std()
        df['Z_Score'] = (df['Close'] - df['Mean']) / (df['Std'] + 1e-8)
        df['RSI'] = calculate_rsi(df['Close'], 14)
        df['ADX'] = calculate_adx(df, 14)
        atr = calculate_atr(df, 14)
        df['ATR'] = atr

        vol_above_avg = df['Volume'] > df['Volume'].rolling(20).mean()
        regime_ok = df['ADX'] < self.adx_max

        df['Signal'] = 0
        long_cond = (df['Z_Score'] < -self.z_entry) & (df['RSI'] < self.rsi_long) & vol_above_avg & regime_ok
        short_cond = (df['Z_Score'] > self.z_entry) & (df['RSI'] > self.rsi_short) & vol_above_avg & regime_ok
        df.loc[long_cond, 'Signal'] = 1
        df.loc[short_cond, 'Signal'] = -1

        df['Exit_Long'] = df['Z_Score'] > 0
        df['Exit_Short'] = df['Z_Score'] < 0

        df['SL_Price'] = np.nan
        df.loc[long_cond, 'SL_Price'] = df.loc[long_cond, 'Close'] - self.atr_sl_mult * df.loc[long_cond, 'ATR']
        df.loc[short_cond, 'SL_Price'] = df.loc[short_cond, 'Close'] + self.atr_sl_mult * df.loc[short_cond, 'ATR']
        df['TP_Price'] = np.nan
        df['Max_Hold'] = self.max_holding
        return df


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
        df.loc[long_entries, 'SL_Price'] = df.loc[long_entries, 'Close'] - self.atr_sl_mult * df.loc[long_entries, 'ATR']
        df.loc[short_entries, 'SL_Price'] = df.loc[short_entries, 'Close'] + self.atr_sl_mult * df.loc[short_entries, 'ATR']
        
        df.loc[long_entries, 'TP_Price'] = df.loc[long_entries, 'Close'] + self.atr_tp_mult * df.loc[long_entries, 'ATR']
        df.loc[short_entries, 'TP_Price'] = df.loc[short_entries, 'Close'] - self.atr_tp_mult * df.loc[short_entries, 'ATR']
        
        df['Max_Hold'] = self.max_holding
        return df


class MLBounceReversion(BaseStrategy):
    """Mean reversion strategy integrating XGBoost directional bias and Bollinger Bands.
    
    Trades against extreme momentum when ML confidence supports the reversal.
    Handles highly noisy and ranging M1 micro-structure periods much better than breakout.
    """

    params = {
        'bb_period': (20, 10, 40, 5), 'bb_std': (2.0, 1.5, 3.0, 0.5),
        'prob_threshold': (0.55, 0.50, 0.70, 0.05),
        'atr_sl_mult': (1.5, 1.0, 3.0, 0.5), 'atr_tp_mult': (1.5, 1.0, 3.0, 0.5),
        'max_holding': (50, 20, 150, 10),
    }

    def __init__(self, bb_period=20, bb_std=2.0, prob_threshold=0.55,
                 atr_sl_mult=1.5, atr_tp_mult=1.5, max_holding=50):
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.prob_threshold = prob_threshold
        self.atr_sl_mult = atr_sl_mult
        self.atr_tp_mult = atr_tp_mult
        self.max_holding = max_holding
        self.ml_model = XGBoostRangeBarModel(tp_mult=atr_tp_mult, sl_mult=atr_sl_mult)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        atr = calculate_atr(df, 14)
        df['ATR'] = atr
        bb = calculate_bollinger(df['Close'], self.bb_period, self.bb_std)
        df['BB_Lower'], df['BB_Upper'] = bb['BB_Lower'], bb['BB_Upper']

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

        bull_bounce = (df['Close'] < df['BB_Lower']) | (df['Low'] < df['BB_Lower'])
        bear_bounce = (df['Close'] > df['BB_Upper']) | (df['High'] > df['BB_Upper'])

        df['Signal'] = 0
        df.loc[bull_bounce & (df['Bull_Prob'] > self.prob_threshold), 'Signal'] = 1
        df.loc[bear_bounce & (df['Bull_Prob'] < (1 - self.prob_threshold)), 'Signal'] = -1

        df['Exit_Long'] = df['Close'] > bb['BB_Mid']
        df['Exit_Short'] = df['Close'] < bb['BB_Mid']

        df['SL_Price'] = np.nan
        df['TP_Price'] = np.nan
        long_entries = df['Signal'] == 1
        short_entries = df['Signal'] == -1
        
        df.loc[long_entries, 'SL_Price'] = df.loc[long_entries, 'Close'] - self.atr_sl_mult * df.loc[long_entries, 'ATR']
        df.loc[short_entries, 'SL_Price'] = df.loc[short_entries, 'Close'] + self.atr_sl_mult * df.loc[short_entries, 'ATR']
        
        df.loc[long_entries, 'TP_Price'] = df.loc[long_entries, 'Close'] + self.atr_tp_mult * df.loc[long_entries, 'ATR']
        df.loc[short_entries, 'TP_Price'] = df.loc[short_entries, 'Close'] - self.atr_tp_mult * df.loc[short_entries, 'ATR']
        
        df['Max_Hold'] = self.max_holding
        return df


class VWAPBounceStrategy(BaseStrategy):
    """VWAP band bounce with RSI + volume confirmation and ATR SL/TP."""

    params = {
        'vol_mult': (1.5, 1.0, 3.0, 0.25), 'rsi_oversold': (35, 20, 40, 5),
        'rsi_overbought': (65, 60, 80, 5), 'atr_sl_mult': (1.5, 1.0, 3.0, 0.5),
        'atr_tp_mult': (2.5, 1.5, 4.0, 0.5), 'max_holding': (80, 40, 200, 20),
    }

    def __init__(self, vol_mult=1.5, rsi_oversold=35, rsi_overbought=65,
                 atr_sl_mult=1.5, atr_tp_mult=2.5, max_holding=80):
        self.vol_mult = vol_mult
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.atr_sl_mult = atr_sl_mult
        self.atr_tp_mult = atr_tp_mult
        self.max_holding = max_holding

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = calculate_vwap_with_bands(df)
        df['RSI'] = calculate_rsi(df['Close'], 14)
        atr = calculate_atr(df, 14)
        df['ATR'] = atr
        df['Std'] = df['Close'].rolling(20).std()

        vol_avg = df['Volume'].rolling(20).mean()
        vol_conf = df['Volume'] > self.vol_mult * vol_avg

        near_lower = df['Close'] <= df['VWAP_Lower_2'] * 1.002
        near_upper = df['Close'] >= df['VWAP_Upper_2'] * 0.998
        long_cond = near_lower & (df['RSI'] < self.rsi_oversold) & vol_conf
        short_cond = near_upper & (df['RSI'] > self.rsi_overbought) & vol_conf

        df['Signal'] = 0
        df.loc[long_cond, 'Signal'] = 1
        df.loc[short_cond, 'Signal'] = -1

        df['Exit_Long'] = df['Close'] >= df['VWAP']
        df['Exit_Short'] = df['Close'] <= df['VWAP']

        df['SL_Price'] = np.nan
        df.loc[long_cond, 'SL_Price'] = df.loc[long_cond, 'Close'] - self.atr_sl_mult * df.loc[long_cond, 'ATR']
        df.loc[short_cond, 'SL_Price'] = df.loc[short_cond, 'Close'] + self.atr_sl_mult * df.loc[short_cond, 'ATR']
        df['TP_Price'] = np.nan
        df.loc[long_cond, 'TP_Price'] = df.loc[long_cond, 'Close'] + self.atr_tp_mult * df.loc[long_cond, 'ATR']
        df.loc[short_cond, 'TP_Price'] = df.loc[short_cond, 'Close'] - self.atr_tp_mult * df.loc[short_cond, 'ATR']
        df['Max_Hold'] = self.max_holding
        return df


class MultiTimeframeMomentum(BaseStrategy):
    """Trend on slow MA, entry on fast RSI pullback. ATR trailing exit."""

    params = {
        'slow_ma': (100, 50, 200, 25), 'fast_rsi_len': (7, 5, 14, 1),
        'rsi_entry': (30, 20, 40, 5), 'atr_sl_mult': (2.0, 1.0, 3.0, 0.5),
        'atr_trail_mult': (2.5, 1.5, 4.0, 0.5), 'max_holding': (150, 50, 300, 50),
    }

    def __init__(self, slow_ma=100, fast_rsi_len=7, rsi_entry=30,
                 atr_sl_mult=2.0, atr_trail_mult=2.5, max_holding=150):
        self.slow_ma = slow_ma
        self.fast_rsi_len = fast_rsi_len
        self.rsi_entry = rsi_entry
        self.atr_sl_mult = atr_sl_mult
        self.atr_trail_mult = atr_trail_mult
        self.max_holding = max_holding

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df['Slow_MA'] = df['Close'].rolling(self.slow_ma).mean()
        df['Fast_RSI'] = calculate_rsi(df['Close'], self.fast_rsi_len)
        atr = calculate_atr(df, 14)
        df['ATR'] = atr
        df['Std'] = df['Close'].rolling(20).std()

        trend_up = df['Close'] > df['Slow_MA']
        trend_down = df['Close'] < df['Slow_MA']
        long_cond = trend_up & (df['Fast_RSI'] < self.rsi_entry)
        short_cond = trend_down & (df['Fast_RSI'] > (100 - self.rsi_entry))

        df['Signal'] = 0
        df.loc[long_cond, 'Signal'] = 1
        df.loc[short_cond, 'Signal'] = -1

        highest = df['High'].rolling(window=20).max()
        lowest = df['Low'].rolling(window=20).min()
        df['Exit_Long'] = df['Close'] < (highest - self.atr_trail_mult * atr)
        df['Exit_Short'] = df['Close'] > (lowest + self.atr_trail_mult * atr)

        df['SL_Price'] = np.nan
        df.loc[long_cond, 'SL_Price'] = df.loc[long_cond, 'Close'] - self.atr_sl_mult * df.loc[long_cond, 'ATR']
        df.loc[short_cond, 'SL_Price'] = df.loc[short_cond, 'Close'] + self.atr_sl_mult * df.loc[short_cond, 'ATR']
        df['TP_Price'] = np.nan
        df['Max_Hold'] = self.max_holding
        return df


def detect_liquidity_sweep(df: pd.DataFrame) -> dict:
    """Detect VWAP-band liquidity sweeps on the last two bars."""
    df = calculate_vwap_with_bands(df)
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    if prev['Low'] < prev['VWAP_Lower_2'] and latest['Close'] > latest['VWAP_Lower_2']:
        return {"signal": True, "type": "BULLISH_SWEEP",
                "message": f"Byczy Liquidity Sweep: {latest['VWAP_Lower_2']:.2f}."}
    elif prev['High'] > prev['VWAP_Upper_2'] and latest['Close'] < latest['VWAP_Upper_2']:
        return {"signal": True, "type": "BEARISH_SWEEP",
                "message": f"Niedźwiedzi Liquidity Sweep: {latest['VWAP_Upper_2']:.2f}."}
    return {"signal": False, "type": None, "message": ""}


def analyze_pair_opportunity(df_y: pd.DataFrame, df_x: pd.DataFrame, ml_filter: StatArbMLFilter) -> dict:
    """Evaluate stat-arb pair trade opportunity with ML confirmation."""
    arb_data = test_cointegration(df_y['Close'], df_x['Close'])
    if not arb_data["is_cointegrated"]:
        return {"signal": False, "message": "Brak kointegracji statystycznej."}
    latest_z = arb_data["z_score"].iloc[-1]
    if abs(latest_z) >= 2.0:
        features = ml_filter.prepare_features(arb_data["spread"], arb_data["z_score"])
        if features.empty:
            return {"signal": False, "message": "Brak danych ML."}
        latest_row = features.drop('Target', axis=1).iloc[-1:]
        prob_success = ml_filter.predict_probability(latest_row)
        if prob_success > 0.65:
            action = "SHORT Y, LONG X" if latest_z > 0 else "LONG Y, SHORT X"
            return {"signal": True,
                    "message": f"Setup StatArb. Z-Score: {latest_z:.2f}. ML szanse: {prob_success * 100:.1f}%. {action}"}
        return {"signal": False, "message": "Odrzucono przez model ML."}
    return {"signal": False, "message": "Z-score w normie."}


STRATEGY_REGISTRY = {
    'ZScoreMeanReversion': ZScoreMeanReversion,
    'VolatilityBreakout': VolatilityBreakout,
    'MLVolatilityBreakout': MLVolatilityBreakout,
    'MLBounceReversion': MLBounceReversion,
    'VWAPBounceStrategy': VWAPBounceStrategy,
    'MultiTimeframeMomentum': MultiTimeframeMomentum,
}