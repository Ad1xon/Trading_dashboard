"""Alert manager — central coordinator with dedup, per-symbol thresholds, multi-channel dispatch."""

import time
import logging
from typing import Optional
from .notifier import DiscordNotifier
from utils.event_bus import EventBus

logger = logging.getLogger(__name__)


class AlertManager:
    """Central alert manager linking Scanner/Engine signals to notification channels.

    Subscribes to EventBus topics (`TRADE_SIGNAL`, `LIQUIDITY_SWEEP`).
    Features: duplicate suppression (cooldown window), per-symbol thresholds.
    """

    DEFAULT_COOLDOWN_SEC = 300

    def __init__(self):
        self._discord: Optional[DiscordNotifier] = None
        self._thresholds: dict[str, dict] = {}
        self._recent_alerts: dict[str, float] = {}
        self.cooldown_sec = self.DEFAULT_COOLDOWN_SEC
        self.enabled = True
        
        self.bus = EventBus()
        self.bus.subscribe("TRADE_SIGNAL", self._on_event)
        self.bus.subscribe("LIQUIDITY_SWEEP", self._on_event)

    async def _on_event(self, topic: str, data: dict):
        """Handle incoming events from the bus."""
        symbol = data.get("symbol", "UNKNOWN")
        msg = data.get("message", "")
        sig_type = data.get("signal_type", "INFO")
        conf = data.get("confidence", 0.0)
        self.fire(symbol, msg, signal_type=sig_type, confidence=conf)

    def configure_discord(self, webhook_url: str, rate_limit: int = 10):
        """Set up Discord notification channel."""
        self._discord = DiscordNotifier(webhook_url, rate_limit_per_minute=rate_limit)

    def set_threshold(self, symbol: str, min_confidence: float = 0.5, enabled: bool = True):
        """Configure per-symbol alert thresholds."""
        self._thresholds[symbol] = {"min_confidence": min_confidence, "enabled": enabled}

    def is_symbol_enabled(self, symbol: str) -> bool:
        """Check if alerts are enabled for a symbol (default: True)."""
        cfg = self._thresholds.get(symbol)
        if cfg is None:
            return True
        return cfg.get("enabled", True)

    def fire(self, symbol: str, message: str, signal_type: str = "INFO",
             confidence: float | None = None) -> bool:
        """Attempt to send an alert. Returns True if dispatched.

        Suppresses if globally disabled, symbol disabled, confidence below
        threshold, or duplicate within cooldown window.
        """
        if not self.enabled or not self.is_symbol_enabled(symbol):
            return False

        cfg = self._thresholds.get(symbol, {})
        min_conf = cfg.get("min_confidence", 0.0)
        if confidence is not None and confidence < min_conf:
            logger.debug("Alert suppressed for %s — confidence %.2f < threshold %.2f", symbol, confidence, min_conf)
            return False

        key = f"{symbol}|{signal_type}"
        now = time.time()
        if now - self._recent_alerts.get(key, 0) < self.cooldown_sec:
            logger.debug("Alert suppressed for %s — duplicate within cooldown", symbol)
            return False

        sent = False
        if self._discord:
            sent = self._discord.send_alert(symbol, message, signal_type, confidence)

        if sent:
            self._recent_alerts[key] = now
        return sent

    def get_history(self, last_n: int = 50) -> list[dict]:
        """Return recent alert history from Discord notifier."""
        if self._discord:
            return self._discord.get_history(last_n)
        return []

    def clear_cooldowns(self):
        """Reset all dedup cooldown timers."""
        self._recent_alerts.clear()
