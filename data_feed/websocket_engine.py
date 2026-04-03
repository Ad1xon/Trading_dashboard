"""WebSocket engine — volume-bar aggregation with callback system and multi-symbol support."""

import asyncio
import websockets
import json
from datetime import datetime
from typing import Callable


class VolumeBarAggregator:
    """Aggregate tick data into volume bars and fire callbacks on completion."""

    def __init__(self, volume_threshold: float, symbol: str = "UNKNOWN"):
        self.volume_threshold = volume_threshold
        self.symbol = symbol
        self.current_volume = 0.0
        self.bar_open = 0.0
        self.bar_high = 0.0
        self.bar_low = float('inf')
        self.bar_close = 0.0
        self.is_new_bar = True
        self.historical_bars = []
        self._callbacks: list[Callable] = []

    def on_bar_complete(self, callback: Callable):
        """Register callback: callback(bar_data: dict, symbol: str)."""
        self._callbacks.append(callback)

    def process_tick(self, price: float, volume: float, timestamp: int) -> dict | None:
        """Process a single tick. Returns completed bar dict or None."""
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
            completed_bar = {
                "timestamp": datetime.fromtimestamp(timestamp / 1000),
                "Open": self.bar_open, "High": self.bar_high,
                "Low": self.bar_low, "Close": self.bar_close,
                "Volume": self.current_volume, "Symbol": self.symbol,
            }
            self.current_volume = 0.0
            self.is_new_bar = True
            self.historical_bars.append(completed_bar)
            if len(self.historical_bars) > 500:
                self.historical_bars.pop(0)

            for cb in self._callbacks:
                try:
                    cb(completed_bar, self.symbol)
                except Exception:
                    pass
            return completed_bar
        return None


async def binance_trade_stream(
    symbol: str, volume_threshold: float, on_bar: Callable | None = None,
):
    """Connect to a single Binance trade stream and aggregate volume bars."""
    stream_url = f"wss://stream.binance.com:9443/ws/{symbol.lower()}@trade"
    aggregator = VolumeBarAggregator(volume_threshold, symbol=symbol.upper())
    if on_bar:
        aggregator.on_bar_complete(on_bar)

    async for websocket in websockets.connect(stream_url):
        try:
            async for message in websocket:
                data = json.loads(message)
                aggregator.process_tick(float(data['p']), float(data['q']), int(data['T']))
        except websockets.ConnectionClosed:
            await asyncio.sleep(2)
        except Exception:
            await asyncio.sleep(5)


async def multi_symbol_stream(
    symbols: list[str], volume_threshold: float, on_bar: Callable | None = None,
):
    """Run parallel streams for multiple symbols."""
    await asyncio.gather(*[
        binance_trade_stream(sym, volume_threshold, on_bar) for sym in symbols
    ])