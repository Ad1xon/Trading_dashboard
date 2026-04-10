"""Dynamic slippage model — scales with volume and volatility conditions."""

import numpy as np

from config import SLIPPAGE_BASE_BPS, SLIPPAGE_VOL_EXPONENT, SLIPPAGE_VOLUME_EXPONENT


class DynamicSlippageModel:
    """Slippage estimator that increases with volatility and decreases with liquidity."""

    def __init__(
        self,
        base_bps: float = SLIPPAGE_BASE_BPS,
        vol_exponent: float = SLIPPAGE_VOL_EXPONENT,
        volume_exponent: float = SLIPPAGE_VOLUME_EXPONENT,
    ):
        self.base_bps = base_bps
        self.vol_exponent = vol_exponent
        self.volume_exponent = volume_exponent

    def estimate(
        self,
        price: float,
        atr: float,
        volume: float,
        avg_volume: float,
        avg_atr: float,
    ) -> float:
        """Compute per-bar slippage in price units."""
        if avg_atr < 1e-10 or avg_volume < 1e-10 or price < 1e-10:
            return self.base_bps * 1e-4 * price

        vol_ratio = (atr / avg_atr) ** self.vol_exponent
        liquidity_ratio = (avg_volume / max(volume, 1.0)) ** self.volume_exponent
        adjusted_bps = self.base_bps * vol_ratio * liquidity_ratio
        return adjusted_bps * 1e-4 * price

    def estimate_array(
        self,
        prices: np.ndarray,
        atrs: np.ndarray,
        volumes: np.ndarray,
        atr_window: int = 20,
        vol_window: int = 20,
    ) -> np.ndarray:
        """Vectorised slippage estimation for an entire bar series."""
        n = len(prices)
        slippage = np.full(n, self.base_bps * 1e-4)

        avg_atr = np.full(n, np.nan)
        avg_vol = np.full(n, np.nan)
        for i in range(atr_window, n):
            avg_atr[i] = np.mean(atrs[max(0, i - atr_window):i])
            avg_vol[i] = np.mean(volumes[max(0, i - vol_window):i])

        valid = (avg_atr > 1e-10) & (avg_vol > 1e-10) & (atrs > 1e-10)
        vol_ratio = np.where(valid, (atrs / np.where(avg_atr > 1e-10, avg_atr, 1.0)) ** self.vol_exponent, 1.0)
        liq_ratio = np.where(
            valid,
            (np.where(avg_vol > 1e-10, avg_vol, 1.0) / np.maximum(volumes, 1.0)) ** self.volume_exponent,
            1.0,
        )
        slippage = self.base_bps * 1e-4 * vol_ratio * liq_ratio * prices
        return slippage
