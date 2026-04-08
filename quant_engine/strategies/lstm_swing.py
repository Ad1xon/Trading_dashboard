"""
Deep-Learning swing strategy using LSTM neural network.

Designed for longer time-horizons (H1 / H4 / D1 OHLCV data) rather
than M1 range bars.
"""

import pandas as pd
import numpy as np
from .base import BaseStrategy
from ..ml_models.lstm_model import LSTMSwingModel
from ..indicators import calculate_atr


class LSTMSwingStrategy(BaseStrategy):
    """Swing trading strategy driven by an LSTM neural network.

    Requires raw OHLCV data at H1/H4/D1 resolution and at least two
    years of history for statistically meaningful predictions.  The
    LSTM outputs a bullish probability that is gated by a 200-bar
    trend filter before producing a signal.
    """

    params = {
        'prob_threshold': (0.65, 0.50, 0.80, 0.05),
        'atr_sl_mult': (2.0, 1.0, 4.0, 0.5),
        'atr_tp_mult': (4.0, 2.0, 8.0, 0.5),
        'max_holding': (150, 50, 300, 20),
    }

    def __init__(
        self,
        prob_threshold=0.53,
        atr_sl_mult=2.0,
        atr_tp_mult=4.0,
        max_holding=150,
    ):
        super().__init__()
        self.prob_threshold = prob_threshold
        self.atr_sl_mult = atr_sl_mult
        self.atr_tp_mult = atr_tp_mult
        self.max_holding = max_holding
        self.ml_model = LSTMSwingModel(sequence_length=30)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate LSTM-driven swing signals with SMA-200 trend gate."""
        df['ATR'] = calculate_atr(df, 14)
        df['Std'] = df['Close'].rolling(20).std()

        df_copy = df.copy()
        wf_data = self.ml_model.train(df_copy)

        df_features = self.ml_model.build_features(df)
        df['LSTM_Prob'] = 0.5

        if wf_data is not None and 'WF_Prediction' in wf_data.columns:
            df['LSTM_Prob'] = wf_data['WF_Prediction']
        else:
            if self.ml_model.is_trained:
                df['LSTM_Prob'] = self.ml_model.predict_proba(df_features)

        trend_up = df['Close'] > df['Close'].rolling(200).mean()
        trend_down = df['Close'] < df['Close'].rolling(200).mean()

        df['Signal'] = 0
        df.loc[trend_up & (df['LSTM_Prob'] > self.prob_threshold), 'Signal'] = 1
        df.loc[trend_down & (df['LSTM_Prob'] < (1 - self.prob_threshold)), 'Signal'] = -1

        df = self.apply_macro_filter(df)

        df['SL_Price'] = np.nan
        df['TP_Price'] = np.nan

        long_entries = df['Signal'] == 1
        short_entries = df['Signal'] == -1

        df.loc[long_entries, 'SL_Price'] = df.loc[long_entries, 'Close'] - (
            self.atr_sl_mult * df.loc[long_entries, 'ATR']
        )
        df.loc[short_entries, 'SL_Price'] = df.loc[short_entries, 'Close'] + (
            self.atr_sl_mult * df.loc[short_entries, 'ATR']
        )

        df.loc[long_entries, 'TP_Price'] = df.loc[long_entries, 'Close'] + (
            self.atr_tp_mult * df.loc[long_entries, 'ATR']
        )
        df.loc[short_entries, 'TP_Price'] = df.loc[short_entries, 'Close'] - (
            self.atr_tp_mult * df.loc[short_entries, 'ATR']
        )

        df['Max_Hold'] = self.max_holding
        return df