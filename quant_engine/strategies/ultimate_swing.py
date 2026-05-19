"""Ultimate Swing Strategy v3 — Decoupled ML with Background Retraining.
Designed for H4/D1 swing trading.
"""

import logging
import threading
from datetime import datetime

import pandas as pd
import numpy as np

from .base import BaseStrategy
from ..indicators import calculate_atr, calculate_rsi, calculate_adx
from ..volatility_model import compute_garch_features

logger = logging.getLogger(__name__)

try:
    from ..ml_models.lstm_model import LSTMSwingModel
except ImportError:
    LSTMSwingModel = None

try:
    from data_feed.nlp_engine import SentimentEngine
except ImportError:
    SentimentEngine = None


class UltimateSwingStrategy(BaseStrategy):
    """Dynamic-scoring swing strategy with decoupled ML training.

    Model Lifecycle:
        1. update_model(df) trains the LSTM via walk-forward expansion.
           Called once at engine startup, then every retrain_interval_hours
           by a background daemon thread.
        2. generate_signals(df) performs FAST inference only via predict_proba().
           No training occurs in the hot path — ever.
        3. _model_ready flag gates inference: if False, LSTM score = 0.5 (neutral).

    Score Components (each produces a value in [-1, +1]):
        1. LSTM prediction      — weight 0.35 (primary directional signal)
        2. Trend alignment      — weight 0.25 (SMA-50/200 + price position)
        3. Regime (HMM)         — weight 0.20 (bull/bear/range encoding)
        4. NLP Sentiment        — weight 0.10 (market mood overlay)
        5. RSI Momentum         — weight 0.10 (overbought/oversold confirmation)

    Entry: composite_score > entry_threshold (long) or < -entry_threshold (short)
    Exit: score drops below exit_threshold, regime flips, or momentum collapses
    """

    strategy_type = "momentum"

    params = {
        "entry_threshold": (0.30, 0.15, 0.60, 0.05),
        "exit_threshold": (0.05, -0.10, 0.20, 0.05),
        "atr_sl_mult": (2.5, 1.5, 4.0, 0.5),
        "atr_tp_mult": (5.0, 3.0, 8.0, 0.5),
        "max_holding": (40, 15, 80, 5),
        "lstm_weight": (0.35, 0.20, 0.50, 0.05),
        "trend_weight": (0.25, 0.10, 0.40, 0.05),
        "regime_weight": (0.20, 0.05, 0.35, 0.05),
        "sentiment_weight": (0.10, 0.0, 0.30, 0.05),
        "rsi_weight": (0.10, 0.0, 0.20, 0.05),
        "retrain_interval_hours": (6, 1, 24, 1),
    }

    def __init__(
        self,
        entry_threshold=0.30,
        exit_threshold=0.05,
        atr_sl_mult=2.5,
        atr_tp_mult=5.0,
        max_holding=40,
        lstm_weight=0.35,
        trend_weight=0.25,
        regime_weight=0.20,
        sentiment_weight=0.10,
        rsi_weight=0.10,
        retrain_interval_hours=6,
    ):
        super().__init__()
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.atr_sl_mult = atr_sl_mult
        self.atr_tp_mult = atr_tp_mult
        self.max_holding = max_holding
        self.lstm_weight = lstm_weight
        self.trend_weight = trend_weight
        self.regime_weight = regime_weight
        self.sentiment_weight = sentiment_weight
        self.rsi_weight = rsi_weight
        self.retrain_interval_hours = retrain_interval_hours

        self._model_ready = False
        self._last_model_update = None
        self._model_lock = threading.Lock()
        self._retrain_thread = None
        self._retrain_stop_event = threading.Event()

        if LSTMSwingModel is not None:
            self.ml_model = LSTMSwingModel(sequence_length=30, epochs=20, n_wf_folds=3)
        else:
            self.ml_model = None
            logger.warning("LSTMSwingModel not available — LSTM score will be neutral")

        if SentimentEngine is not None:
            self.nlp = SentimentEngine()
        else:
            self.nlp = None

    def update_model(self, df: pd.DataFrame):
        """Train LSTM offline — call on startup or from background thread.

        This method runs walk-forward LSTM training on the provided DataFrame.
        After training, the model weights and scaler are cached so that
        generate_signals() can perform fast inference via predict_proba().

        Thread-safe: acquires _model_lock before mutating model state.
        """
        if self.ml_model is None:
            logger.warning("No LSTM model available — skipping update_model")
            return

        try:
            logger.info("LSTM model training started at %s", datetime.utcnow().isoformat())
            df_copy = df.copy()
            wf_data = self.ml_model.train(df_copy)

            with self._model_lock:
                self._model_ready = self.ml_model.is_trained
                self._last_model_update = datetime.utcnow()

            if self._model_ready:
                logger.info(
                    "LSTM model training completed — model ready for inference"
                )
            else:
                logger.warning("LSTM model training completed but model reports not trained")
        except Exception as exc:
            logger.error("LSTM model training failed: %s", exc)

    def start_background_retraining(self, data_fetcher_fn):
        """Launch a daemon thread that retrains the model every N hours.

        data_fetcher_fn: callable that returns a pd.DataFrame with OHLCV data.
        The thread will call data_fetcher_fn() → update_model(df) on schedule.
        """
        if self._retrain_thread is not None and self._retrain_thread.is_alive():
            logger.warning("Background retraining thread already running")
            return

        self._retrain_stop_event.clear()

        def _retrain_loop():
            """Daemon loop — retrain on interval until stop event is set."""
            interval_seconds = self.retrain_interval_hours * 3600
            while not self._retrain_stop_event.is_set():
                self._retrain_stop_event.wait(timeout=interval_seconds)
                if self._retrain_stop_event.is_set():
                    break
                try:
                    df = data_fetcher_fn()
                    if df is not None and len(df) >= 200:
                        self.update_model(df)
                except Exception as exc:
                    logger.error("Background retrain failed: %s", exc)

        self._retrain_thread = threading.Thread(
            target=_retrain_loop, daemon=True, name="lstm-retrain"
        )
        self._retrain_thread.start()
        logger.info(
            "Background LSTM retraining started — interval=%dh",
            self.retrain_interval_hours,
        )

    def stop_background_retraining(self):
        """Signal the background retraining thread to stop."""
        self._retrain_stop_event.set()
        if self._retrain_thread is not None:
            self._retrain_thread.join(timeout=5)
            self._retrain_thread = None
            logger.info("Background LSTM retraining stopped")

    def _apply_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enrich DataFrame with NLP sentiment if available."""
        df = df.copy()
        if self.nlp is not None:
            try:
                df = self.nlp.apply_rolling_sentiment(df, "Market")
            except Exception as exc:
                logger.warning("Sentiment fetch failed: %s", exc)
                if "Sentiment_Score" not in df.columns:
                    df["Sentiment_Score"] = 0.0
        else:
            if "Sentiment_Score" not in df.columns:
                df["Sentiment_Score"] = 0.0
        return df

    def _compute_lstm_score(self, df: pd.DataFrame) -> pd.Series:
        """LSTM probability mapped to [-1, +1]: 0.5 → 0, 1.0 → +1, 0.0 → -1."""
        return (df["LSTM_Prob"] - 0.5) * 2.0

    def _compute_trend_score(self, df: pd.DataFrame) -> pd.Series:
        """Soft trend score from SMA alignment — no hard gate.

        +1 when Close > SMA200 > SMA50 alignment (strong uptrend)
        -1 when Close < SMA200 < SMA50 alignment (strong downtrend)
        Intermediate values for partial alignment.
        """
        sma_200 = df["Close"].rolling(200, min_periods=50).mean()
        sma_50 = df["Close"].rolling(50, min_periods=20).mean()

        price_vs_200 = np.sign(df["Close"] - sma_200)
        sma_cross = np.sign(sma_50 - sma_200)
        sma_slope = sma_200.pct_change(10).clip(-0.01, 0.01) * 100

        return ((price_vs_200 * 0.4 + sma_cross * 0.4 + sma_slope * 0.2)).clip(-1, 1)

    def _compute_regime_score(self, df: pd.DataFrame) -> pd.Series:
        """Regime encoded as directional score: bull=+1, bear=-1, range=0."""
        regime_map = {"bull": 1.0, "bear": -1.0, "range": 0.0}
        return df["Regime"].map(regime_map).fillna(0.0)

    def _compute_sentiment_score(self, df: pd.DataFrame) -> pd.Series:
        """NLP sentiment clipped to [-1, +1]."""
        return df["Sentiment_Score"].clip(-1, 1)

    def _compute_rsi_score(self, df: pd.DataFrame) -> pd.Series:
        """RSI mapped to directional score: oversold=+1, overbought=-1, neutral=0."""
        rsi = df["RSI"]
        return -((rsi - 50) / 50.0).clip(-1, 1)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate swing signals — FAST inference only, no ML training.

        The LSTM component uses cached model weights via predict_proba().
        If the model has not been trained yet (via update_model), LSTM
        score defaults to 0.5 (neutral) and other components compensate.
        """
        df["ATR"] = calculate_atr(df, 14)
        df["RSI"] = calculate_rsi(df["Close"], 14)
        df["ADX"] = calculate_adx(df, 14)
        df["Std"] = df["Close"].rolling(20).std()

        df = compute_garch_features(df)
        df = self._apply_sentiment_features(df)
        df = self.regime_detector.add_regime_column(df)

        df["LSTM_Prob"] = 0.5
        with self._model_lock:
            if self._model_ready and self.ml_model is not None:
                try:
                    df_features = self.ml_model.build_features(df)
                    df["LSTM_Prob"] = self.ml_model.predict_proba(df_features)
                except Exception as exc:
                    logger.warning("LSTM inference failed: %s", exc)

        lstm_s = self._compute_lstm_score(df)
        trend_s = self._compute_trend_score(df)
        regime_s = self._compute_regime_score(df)
        sent_s = self._compute_sentiment_score(df)
        rsi_s = self._compute_rsi_score(df)

        df["Swing_Score"] = (
            self.lstm_weight * lstm_s
            + self.trend_weight * trend_s
            + self.regime_weight * regime_s
            + self.sentiment_weight * sent_s
            + self.rsi_weight * rsi_s
        ).clip(-1, 1)

        df["Signal"] = 0
        long_cond = df["Swing_Score"] > self.entry_threshold
        short_cond = df["Swing_Score"] < -self.entry_threshold
        df.loc[long_cond, "Signal"] = 1
        df.loc[short_cond, "Signal"] = -1

        df = self.apply_macro_filter(df)

        df["SL_Price"] = np.nan
        df["TP_Price"] = np.nan
        long_entries = df["Signal"] == 1
        short_entries = df["Signal"] == -1

        garch_scale = (
            df["GARCH_Vol"] / (df["GARCH_Vol"].rolling(60).mean() + 1e-10)
        ).clip(0.5, 2.0)
        adaptive_sl = self.atr_sl_mult * garch_scale
        adaptive_tp = self.atr_tp_mult * garch_scale

        df.loc[long_entries, "SL_Price"] = (
            df.loc[long_entries, "Close"]
            - adaptive_sl.loc[long_entries] * df.loc[long_entries, "ATR"]
        )
        df.loc[short_entries, "SL_Price"] = (
            df.loc[short_entries, "Close"]
            + adaptive_sl.loc[short_entries] * df.loc[short_entries, "ATR"]
        )
        df.loc[long_entries, "TP_Price"] = (
            df.loc[long_entries, "Close"]
            + adaptive_tp.loc[long_entries] * df.loc[long_entries, "ATR"]
        )
        df.loc[short_entries, "TP_Price"] = (
            df.loc[short_entries, "Close"]
            - adaptive_tp.loc[short_entries] * df.loc[short_entries, "ATR"]
        )

        """Dynamic exit: score drops below exit_threshold or regime flips."""
        df["Exit_Long"] = (
            (df["Swing_Score"] < self.exit_threshold)
            | (df["Regime"] == "bear")
        )
        df["Exit_Short"] = (
            (df["Swing_Score"] > -self.exit_threshold)
            | (df["Regime"] == "bull")
        )

        df["Max_Hold"] = self.max_holding
        return df
