"""Ultimate MFT Strategy v3 — Cascading Microstructure Gate Engine.
Designed for M1/M5/M15 or Range Bar data.
"""

import logging

import pandas as pd
import numpy as np

from .base import BaseStrategy
from ..indicators import calculate_atr, calculate_adx
from ..volatility_model import compute_garch_features
from ..microstructure import (
    compute_volume_profile,
    compute_lob_imbalance,
    compute_order_flow_imbalance,
    compute_cumulative_delta,
    compute_supertrend,
)

logger = logging.getLogger(__name__)


class UltimateMFTStrategy(BaseStrategy):
    """Cascading microstructure gate engine — pure order-flow alpha.

    Cascade Logic (each stage must pass before the next is evaluated):

        Stage 0 (Baseline):
            Compute all microstructure features. Every bar starts here.

        Stage 1 — GARCH Volatility Regime:
            Compute GARCH vol ratio = current_vol / MA(60).
            If ratio > garch_high_vol_cutoff → HOLD (vol too high for MFT).
            Pass → proceed to Stage 2.

        Stage 2 — OFI Impulse Detection:
            OFI_Z must exceed ±ofi_impulse_threshold.
            Bullish impulse: OFI_Z > +threshold.
            Bearish impulse: OFI_Z < -threshold.
            No impulse → HOLD.

        Stage 3 — CVD Directional Confirmation:
            CVD 5-bar slope (z-scored) must agree with OFI direction.
            LOB imbalance used as secondary confirmation (same sign).
            If CVD opposes OFI → flow conflict → HOLD.

        Stage 4 — Execution Gate:
            ATR must exceed cost_hurdle_mult × estimated round-trip cost.
            SuperTrend direction must align with trade direction.
            Volume must exceed vol_floor_mult × 20-bar MA.
            All pass → ENTER.

    Active Exits:
        OFI z-score reversal (flow collapse).
        Chandelier exit (2.5 ATR from rolling extreme).
    """

    strategy_type = "scalp"

    params = {
        "garch_high_vol_cutoff": (1.5, 1.0, 3.0, 0.1),
        "ofi_impulse_threshold": (1.0, 0.5, 2.5, 0.1),
        "cvd_confirm_threshold": (0.3, 0.1, 1.0, 0.1),
        "lob_confirm_threshold": (0.3, 0.1, 1.0, 0.1),
        "cost_hurdle_mult": (2.5, 1.5, 5.0, 0.5),
        "vol_floor_mult": (1.2, 0.8, 2.0, 0.1),
        "atr_sl_mult": (2.0, 1.0, 3.5, 0.5),
        "atr_tp_mult": (3.5, 2.0, 6.0, 0.5),
        "max_holding": (80, 20, 200, 20),
        "supertrend_period": (10, 7, 20, 1),
        "supertrend_mult": (3.0, 2.0, 4.0, 0.5),
        "chandelier_atr_mult": (2.5, 1.5, 4.0, 0.5),
        "ofi_exit_threshold": (0.3, 0.1, 1.0, 0.1),
    }

    def __init__(
        self,
        garch_high_vol_cutoff=1.5,
        ofi_impulse_threshold=1.0,
        cvd_confirm_threshold=0.3,
        lob_confirm_threshold=0.3,
        cost_hurdle_mult=2.5,
        vol_floor_mult=1.2,
        atr_sl_mult=2.0,
        atr_tp_mult=3.5,
        max_holding=80,
        supertrend_period=10,
        supertrend_mult=3.0,
        chandelier_atr_mult=2.5,
        ofi_exit_threshold=0.3,
    ):
        super().__init__()
        self.garch_high_vol_cutoff = garch_high_vol_cutoff
        self.ofi_impulse_threshold = ofi_impulse_threshold
        self.cvd_confirm_threshold = cvd_confirm_threshold
        self.lob_confirm_threshold = lob_confirm_threshold
        self.cost_hurdle_mult = cost_hurdle_mult
        self.vol_floor_mult = vol_floor_mult
        self.atr_sl_mult = atr_sl_mult
        self.atr_tp_mult = atr_tp_mult
        self.max_holding = max_holding
        self.supertrend_period = supertrend_period
        self.supertrend_mult = supertrend_mult
        self.chandelier_atr_mult = chandelier_atr_mult
        self.ofi_exit_threshold = ofi_exit_threshold

    def _compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all microstructure and volatility features in a single pass."""
        df["ATR"] = calculate_atr(df, 14)
        df["ADX"] = calculate_adx(df, 14)
        df["Std"] = df["Close"].rolling(20).std()

        df = compute_garch_features(df)
        df = compute_order_flow_imbalance(df, window=20)
        df = compute_cumulative_delta(df, divergence_window=20)
        df = compute_volume_profile(df, num_bins=50, value_area_pct=0.70, window=100)
        df["LOB_Imbalance_Z"] = compute_lob_imbalance(df, window=20)
        df = compute_supertrend(
            df, period=self.supertrend_period, multiplier=self.supertrend_mult
        )
        return df

    def _stage1_garch_regime(self, df: pd.DataFrame) -> pd.Series:
        """Stage 1: GARCH volatility regime filter.

        Returns a boolean Series — True where vol regime is acceptable (low/normal).
        Bars with GARCH ratio above the cutoff are blocked.
        """
        garch_ma = df["GARCH_Vol"].rolling(60, min_periods=10).mean()
        garch_ratio = df["GARCH_Vol"] / (garch_ma + 1e-10)
        return garch_ratio <= self.garch_high_vol_cutoff

    def _stage2_ofi_impulse(self, df: pd.DataFrame) -> tuple:
        """Stage 2: OFI impulse detection.

        Returns (bull_impulse, bear_impulse) boolean Series.
        An impulse requires OFI_Z to exceed the configurable threshold.
        """
        bull_impulse = df["OFI_Z"] > self.ofi_impulse_threshold
        bear_impulse = df["OFI_Z"] < -self.ofi_impulse_threshold
        return bull_impulse, bear_impulse

    def _stage3_cvd_confirmation(self, df: pd.DataFrame) -> tuple:
        """Stage 3: CVD directional confirmation with LOB tiebreaker.

        CVD 5-bar slope is z-scored against a 20-bar window.
        Bullish confirmation: CVD slope z-score > cvd_confirm_threshold.
        Bearish confirmation: CVD slope z-score < -cvd_confirm_threshold.
        LOB imbalance must also agree (same sign) for full confirmation.
        """
        cvd = df["CVD"]
        cvd_slope = cvd.diff(5)
        cvd_slope_std = cvd_slope.rolling(20, min_periods=5).std()
        cvd_slope_z = cvd_slope / (cvd_slope_std + 1e-10)

        lob_z = df["LOB_Imbalance_Z"]

        cvd_bull = cvd_slope_z > self.cvd_confirm_threshold
        cvd_bear = cvd_slope_z < -self.cvd_confirm_threshold
        lob_bull = lob_z > self.lob_confirm_threshold
        lob_bear = lob_z < -self.lob_confirm_threshold

        confirmed_bull = cvd_bull & lob_bull
        confirmed_bear = cvd_bear & lob_bear

        return confirmed_bull, confirmed_bear

    def _stage4_execution_gate(self, df: pd.DataFrame) -> tuple:
        """Stage 4: Cost hurdle, SuperTrend alignment, and volume floor.

        All three conditions must be satisfied for the execution gate to open.
        """
        spread_estimate = df["ATR"] * 0.05
        round_trip_cost = 2.0 * spread_estimate
        cost_ok = df["ATR"] > self.cost_hurdle_mult * round_trip_cost

        st_bull = df["SuperTrend_Direction"] == 1
        st_bear = df["SuperTrend_Direction"] == -1

        vol_ok = df["Volume"] > df["Volume"].rolling(20).mean() * self.vol_floor_mult

        gate_long = cost_ok & st_bull & vol_ok
        gate_short = cost_ok & st_bear & vol_ok

        return gate_long, gate_short

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate signals via 4-stage cascading microstructure gate.

        Each bar progresses through stages 1-4. A bar can only
        generate a signal if ALL stages pass. The Cascade_Stage column
        records the highest stage reached (0-4) for diagnostic purposes.
        """
        df = self._compute_features(df)

        vol_ok = self._stage1_garch_regime(df)
        ofi_bull, ofi_bear = self._stage2_ofi_impulse(df)
        cvd_bull, cvd_bear = self._stage3_cvd_confirmation(df)
        gate_long, gate_short = self._stage4_execution_gate(df)

        """Build cascade stage tracker for diagnostics."""
        cascade = pd.Series(0, index=df.index, dtype=int)
        cascade = cascade.where(~vol_ok, 1)
        cascade = cascade.where(~(vol_ok & (ofi_bull | ofi_bear)), 2)
        cascade = cascade.where(
            ~(vol_ok & ((ofi_bull & cvd_bull) | (ofi_bear & cvd_bear))), 3
        )

        long_entry = vol_ok & ofi_bull & cvd_bull & gate_long
        short_entry = vol_ok & ofi_bear & cvd_bear & gate_short

        cascade = cascade.where(~(long_entry | short_entry), 4)
        df["Cascade_Stage"] = cascade

        df["Signal"] = 0
        df.loc[long_entry, "Signal"] = 1
        df.loc[short_entry, "Signal"] = -1

        df = self.apply_macro_filter(df)

        highest = df["High"].rolling(20).max()
        lowest = df["Low"].rolling(20).min()
        ofi_reversal_long = df["OFI_Z"] < -self.ofi_exit_threshold
        ofi_reversal_short = df["OFI_Z"] > self.ofi_exit_threshold
        chandelier_exit_long = df["Close"] < (
            highest - self.chandelier_atr_mult * df["ATR"]
        )
        chandelier_exit_short = df["Close"] > (
            lowest + self.chandelier_atr_mult * df["ATR"]
        )

        df["Exit_Long"] = chandelier_exit_long | ofi_reversal_long
        df["Exit_Short"] = chandelier_exit_short | ofi_reversal_short

        df["SL_Price"] = np.nan
        df["TP_Price"] = np.nan
        long_e = df["Signal"] == 1
        short_e = df["Signal"] == -1

        garch_scale = (
            df["GARCH_Vol"] / (df["GARCH_Vol"].rolling(60).mean() + 1e-10)
        ).clip(0.5, 2.0)
        adaptive_sl = self.atr_sl_mult * garch_scale
        adaptive_tp = self.atr_tp_mult * garch_scale

        df.loc[long_e, "SL_Price"] = (
            df.loc[long_e, "Close"] - adaptive_sl.loc[long_e] * df.loc[long_e, "ATR"]
        )
        df.loc[short_e, "SL_Price"] = (
            df.loc[short_e, "Close"] + adaptive_sl.loc[short_e] * df.loc[short_e, "ATR"]
        )
        df.loc[long_e, "TP_Price"] = (
            df.loc[long_e, "Close"] + adaptive_tp.loc[long_e] * df.loc[long_e, "ATR"]
        )
        df.loc[short_e, "TP_Price"] = (
            df.loc[short_e, "Close"] - adaptive_tp.loc[short_e] * df.loc[short_e, "ATR"]
        )

        df["Max_Hold"] = self.max_holding
        return df
