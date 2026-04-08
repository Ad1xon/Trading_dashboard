"""Discord notifier with rate limiting, rich embeds, and alert history."""

import time
import logging
import threading
from datetime import datetime, timezone

import requests

logger = logging.getLogger(__name__)


class DiscordNotifier:
    """Send formatted embed alerts to a Discord webhook with rate limiting."""

    def __init__(self, webhook_url: str, rate_limit_per_minute: int = 10):
        self.webhook_url = webhook_url
        self.rate_limit = rate_limit_per_minute
        self._send_times: list[float] = []
        self.history: list[dict] = []

    def send_alert(
        self,
        asset: str,
        message: str,
        signal_type: str = "INFO",
        confidence: float | None = None,
    ) -> bool:
        """Send a formatted embed alert via Discord webhook."""
        if not self.webhook_url:
            return False

        now = time.time()
        self._send_times = [t for t in self._send_times if now - t < 60]
        if len(self._send_times) >= self.rate_limit:
            logger.warning("Rate limit reached — alert suppressed for %s", asset)
            return False

        emoji = _signal_emoji(signal_type)
        conf_str = (
            f" | Confidence: **{confidence * 100:.1f}%**"
            if confidence is not None else ""
        )
        payload = {
            "embeds": [{
                "title": f"{emoji} {signal_type}: {asset}",
                "description": f"{message}{conf_str}",
                "color": _signal_color(signal_type),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "footer": {"text": "Institutional Quant Dashboard"},
            }],
        }

        def _async_post(payload_data):
            try:
                requests.post(self.webhook_url, json=payload_data, timeout=5)
            except Exception as exc:
                logger.error("Discord send failed: %s", exc)

        threading.Thread(target=_async_post, args=(payload,), daemon=True).start()

        self._send_times.append(now)
        self.history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "asset": asset,
            "signal_type": signal_type,
            "message": message,
            "sent": True,
        })
        if len(self.history) > 500:
            self.history = self.history[-500:]

        return True

    def get_history(self, last_n: int = 50) -> list[dict]:
        """Return the last *last_n* alert records."""
        return self.history[-last_n:]


def _signal_emoji(signal_type: str) -> str:
    """Map signal types to Discord-friendly emoji."""
    return {
        "BULLISH_SWEEP": "🟢🔄", "BEARISH_SWEEP": "🔴🔄",
        "LONG": "🟢📈", "SHORT": "🔴📉",
        "EXIT": "⚪⏹", "INFO": "ℹ️",
    }.get(signal_type, "🚨")


def _signal_color(signal_type: str) -> int:
    """Map signal types to Discord embed colour codes."""
    return {
        "BULLISH_SWEEP": 0x00FF88, "BEARISH_SWEEP": 0xFF4444,
        "LONG": 0x00CC66, "SHORT": 0xCC3333,
        "EXIT": 0x888888, "INFO": 0x3399FF,
    }.get(signal_type, 0xFFAA00)
