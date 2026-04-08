"""Arabian Scalper with Risk Management (1:3 R:R)."""

import pandas as pd
import numpy as np
from .base import BaseStrategy
from ..indicators import calculate_atr, calculate_supertrend, calculate_weis_wave_volume
from ..ml_models import LGBMRangeBarModel
from data_feed.nlp_engine import SentimentEngine


class ArabianScalper(BaseStrategy):
    """
    Arabian Scalper with Risk Management (1:3 R:R).
    Uses LightGBM for signal confirmation + Macro/NLP integration.
    """
    params = {
        'lookback': (10, 5, 20, 1),
        'prob_threshold': (0.60, 0.50, 0.70, 0.05),
        'risk_atr_cap': (1.5, 1.0, 2.5, 0.5),
        'reward_multiplier': (3.0, 2.0, 5.0, 0.5)
    }

    def __init__(self, lookback=10, prob_threshold=0.60, risk_atr_cap=1.5, reward_multiplier=3.0):
        super().__init__()
        self.lookback = lookback
        self.prob_threshold = prob_threshold
        self.risk_atr_cap = risk_atr_cap
        self.reward_multiplier = reward_multiplier
        self.ml_model = LGBMRangeBarModel(tp_mult=reward_multiplier, sl_mult=risk_atr_cap)
        self.nlp = SentimentEngine()

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.nlp.apply_sentiment_to_dataframe(df, "Market")

        df['ATR'] = calculate_atr(df, 14)
        df['Momentum'] = df['Close'].diff(self.lookback)
        df['Vol_Surge'] = df['Volume'] > df['Volume'].rolling(20).mean() * 1.5

        df['_uid'] = np.arange(len(df))
        wf_data = self.ml_model.train(df.copy())

        df_features = self.ml_model.build_features(df)
        df_features.fillna(0, inplace=True)

        df['LGBM_Prob'] = 0.5
        if wf_data is not None and 'WF_Prediction' in wf_data.columns:
            mapped = wf_data.set_index('_uid')['WF_Prediction']
            df['LGBM_Prob'] = df['_uid'].map(mapped).fillna(0.5)
        else:
            trained_mask = df_features.index.isin(df_features.dropna().index)
            if trained_mask.any() and self.ml_model.is_trained:
                df.loc[trained_mask, 'LGBM_Prob'] = self.ml_model.predict_proba(df_features[trained_mask])

        df.drop(columns=['_uid'], inplace=True)

        bull_signal = (df['Momentum'] > 0) & df['Vol_Surge']
        bear_signal = (df['Momentum'] < 0) & df['Vol_Surge']

        df['Signal'] = 0
        df.loc[bull_signal & (df['LGBM_Prob'] > self.prob_threshold), 'Signal'] = 1
        df.loc[bear_signal & (df['LGBM_Prob'] < (1 - self.prob_threshold)), 'Signal'] = -1

        df = self.apply_macro_filter(df)

        df['SL_Price'] = np.nan
        df['TP_Price'] = np.nan

        long_entries = df['Signal'] == 1
        short_entries = df['Signal'] == -1

        df.loc[long_entries, 'SL_Price'] = df.loc[long_entries, 'Close'] - (
                    self.risk_atr_cap * df.loc[long_entries, 'ATR'])
        df.loc[short_entries, 'SL_Price'] = df.loc[short_entries, 'Close'] + (
                    self.risk_atr_cap * df.loc[short_entries, 'ATR'])

        df.loc[long_entries, 'TP_Price'] = df.loc[long_entries, 'Close'] + (
                    self.risk_atr_cap * self.reward_multiplier * df.loc[long_entries, 'ATR'])
        df.loc[short_entries, 'TP_Price'] = df.loc[short_entries, 'Close'] - (
                    self.risk_atr_cap * self.reward_multiplier * df.loc[short_entries, 'ATR'])

        df['Max_Hold'] = 50
        return df