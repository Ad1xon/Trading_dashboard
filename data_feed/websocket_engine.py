"""
WebSocket engine — volume-bar aggregation with callback system
and multi-symbol support via MT5 tick streaming.
"""

import asyncio
import time
from datetime import datetime
from typing import Callable

import numpy as np
import MetaTrader5 as mt5


class VolumeBarAggregator:
    """Aggregate tick data into volume bars and fire callbacks on completion.

    Maintains a rolling NumPy ring-buffer of completed bars for
    low-latency historical access.

    Args:
        volume_threshold: Cumulative volume required to close a bar.
        symbol:           Instrument identifier attached to bar dicts.
    """

    def __init__(self, volume_threshold: float, symbol: str = "UNKNOWN"):
        self.volume_threshold = volume_threshold
        self.symbol = symbol
        self.current_volume = 0.0
        self.bar_open = 0.0
        self.bar_high = 0.0
        self.bar_low = float('inf')
        self.bar_close = 0.0
        self.is_new_bar = True

        self.buffer_size = 500
        self.hist_arr = np.zeros((self.buffer_size, 5), dtype=np.float64)
        self.hist_timestamps = np.zeros(self.buffer_size, dtype=np.float64)
        self.hist_idx = 0
        self.hist_count = 0

        self._callbacks: list[Callable] = []

    def on_bar_complete(self, callback: Callable):
        """Register a callback invoked when a volume bar completes.

        Signature: ``callback(bar_data: dict, symbol: str)``.
        """
        self._callbacks.append(callback)

    def process_tick(
        self,
        price: float,
        volume: float,
        timestamp: int,
    ) -> dict | None:
        """Process a single tick.

        Accumulates volume and updates the running bar.  When the
        threshold is reached, the bar is finalised, stored in the ring
        buffer, and all registered callbacks are fired.

        Returns:
            Completed bar dict or ``None`` if the bar is still open.
        """
        if self.is_new_bar:
            self.bar_open = price
            self.bar_high = price
            self.bar_low = price
            self.is_new_bar = False

        self.bar_high = max(self.bar_high, price)
        self.bar_low = min(self.bar_low, price)
        self.current_volume += volume

        if self.current_volume >= self.volume_threshold:
            self.bar_close = price
            ts_sec = timestamp / 1000.0

            idx = self.hist_idx
            self.hist_arr[idx, 0] = self.bar_open
            self.hist_arr[idx, 1] = self.bar_high
            self.hist_arr[idx, 2] = self.bar_low
            self.hist_arr[idx, 3] = self.bar_close
            self.hist_arr[idx, 4] = self.current_volume
            self.hist_timestamps[idx] = ts_sec

            self.hist_idx = (self.hist_idx + 1) % self.buffer_size
            if self.hist_count < self.buffer_size:
                self.hist_count += 1

            completed_bar = {
                "timestamp": datetime.fromtimestamp(ts_sec),
                "Open": self.bar_open, "High": self.bar_high,
                "Low": self.bar_low, "Close": self.bar_close,
                "Volume": self.current_volume, "Symbol": self.symbol,
            }
            self.current_volume = 0.0
            self.is_new_bar = True

            for cb in self._callbacks:
                try:
                    cb(completed_bar, self.symbol)
                except Exception:
                    pass
            return completed_bar
        return None


async def mt5_trade_stream(
    symbol: str,
    volume_threshold: float,
    on_bar: Callable | None = None,
):
    """Stream ticks from MT5 and aggregate into volume bars.

    Polls ``mt5.copy_ticks_from`` every second, feeding each new tick
    into a ``VolumeBarAggregator``.
    """
    if not mt5.initialize():
        return

    aggregator = VolumeBarAggregator(volume_threshold, symbol=symbol.upper())
    if on_bar:
        aggregator.on_bar_complete(on_bar)

    last_time_msc = int(time.time() * 1000)

    while True:
        try:
            ticks = mt5.copy_ticks_from(
                symbol, last_time_msc, 100_000, mt5.COPY_TICKS_ALL,
            )
            if ticks is not None and len(ticks) > 0:
                for tick in ticks:
                    if tick['time_msc'] > last_time_msc:
                        price = tick['last'] if tick['last'] > 0 else tick['bid']
                        volume = tick['volume'] if tick['volume'] > 0 else 1.0
                        aggregator.process_tick(
                            float(price), float(volume), int(tick['time_msc']),
                        )
                        last_time_msc = max(last_time_msc, int(tick['time_msc']))
            await asyncio.sleep(1.0)
        except Exception:
            await asyncio.sleep(5.0)


async def multi_symbol_stream(
    symbols: list[str],
    volume_threshold: float,
    on_bar: Callable | None = None,
):
    """Run parallel tick streams for multiple symbols."""
    await asyncio.gather(*[
        mt5_trade_stream(sym, volume_threshold, on_bar) for sym in symbols
    ])