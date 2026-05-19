"""Production-grade MT5 Live Trading Engine — continuous bot with dual-strategy fusion.

Integrates Ultimate MFT and Ultimate Swing strategies with MetaTrader 5 execution,
risk circuit breakers, GARCH-adaptive trailing stops, and VPS-hardened reconnection.
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import MetaTrader5 as mt5

from live_config import (
    MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, MT5_PATH,
    MFT_SYMBOLS, SWING_SYMBOLS,
    MFT_TIMEFRAME_STR, SWING_TIMEFRAME_STR,
    POLL_INTERVAL, MAX_DAILY_DRAWDOWN_PCT, MAX_CONSECUTIVE_LOSSES,
    MAX_TOTAL_EXPOSURE_LOTS, RECONNECT_BASE_DELAY, RECONNECT_MAX_DELAY,
    HEARTBEAT_INTERVAL, RISK_PER_TRADE_PCT, DEFAULT_CAPITAL,
    LOG_FILE, STATE_FILE, DISCORD_WEBHOOK,
)
from config import CONTRACT_SIZES, MAX_ALLOWED_LOTS
from quant_engine.indicators import calculate_atr
from quant_engine.volatility_model import fit_garch_volatility


TIMEFRAME_MAP = {
    "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15, "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
}


def _setup_logging():
    """Configure dual-output logging: file + console."""
    logger = logging.getLogger("LiveEngine")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


logger = _setup_logging()


class PositionSide(Enum):
    """Trade direction enumeration."""
    FLAT = 0
    LONG = 1
    SHORT = -1


class RiskCircuitBreaker:
    """Risk management circuit breaker — kills trading on safety violations."""

    def __init__(self, initial_equity: float):
        self.initial_equity = initial_equity
        self.day_start_equity = initial_equity
        self.consecutive_losses = 0
        self.daily_pnl = 0.0
        self.tripped = False
        self.trip_reason = ""
        self.last_reset_date = datetime.utcnow().date()

    def reset_daily(self, current_equity: float):
        """Reset daily counters at the start of a new trading day."""
        today = datetime.utcnow().date()
        if today != self.last_reset_date:
            self.day_start_equity = current_equity
            self.daily_pnl = 0.0
            self.last_reset_date = today
            if self.tripped and "daily" in self.trip_reason.lower():
                self.tripped = False
                self.trip_reason = ""
                logger.info("Daily circuit breaker reset")

    def record_trade(self, pnl: float):
        """Record a closed trade and check for breaker conditions."""
        self.daily_pnl += pnl
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

    def check(self, current_equity: float, total_exposure_lots: float) -> bool:
        """Return True if trading is allowed, False if circuit breaker is tripped."""
        if self.tripped:
            return False

        daily_dd = (self.day_start_equity - current_equity) / (self.day_start_equity + 1e-10)
        if daily_dd > MAX_DAILY_DRAWDOWN_PCT:
            self.tripped = True
            self.trip_reason = f"Daily drawdown {daily_dd:.2%} exceeds limit {MAX_DAILY_DRAWDOWN_PCT:.2%}"
            logger.critical("CIRCUIT BREAKER: %s", self.trip_reason)
            return False

        if self.consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
            self.tripped = True
            self.trip_reason = f"Consecutive losses ({self.consecutive_losses}) hit limit"
            logger.critical("CIRCUIT BREAKER: %s", self.trip_reason)
            return False

        if total_exposure_lots > MAX_TOTAL_EXPOSURE_LOTS:
            self.tripped = True
            self.trip_reason = f"Total exposure {total_exposure_lots:.2f} lots exceeds limit"
            logger.critical("CIRCUIT BREAKER: %s", self.trip_reason)
            return False

        return True


class MT5ConnectionManager:
    """Manages MT5 terminal connection with exponential backoff reconnection."""

    def __init__(self):
        self.connected = False
        self.reconnect_delay = RECONNECT_BASE_DELAY
        self.last_connect_attempt = 0.0

    def connect(self) -> bool:
        """Attempt to connect to MT5 terminal."""
        try:
            if not mt5.initialize(path=MT5_PATH, login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
                error = mt5.last_error()
                logger.error("MT5 init failed: %s", error)
                return False

            account = mt5.account_info()
            if account is None:
                logger.error("MT5 account_info returned None")
                return False

            self.connected = True
            self.reconnect_delay = RECONNECT_BASE_DELAY
            logger.info(
                "MT5 connected: account=%d server=%s balance=%.2f",
                account.login, account.server, account.balance,
            )
            return True
        except Exception as exc:
            logger.error("MT5 connection exception: %s", exc)
            return False

    def ensure_connected(self) -> bool:
        """Ensure connection is alive, reconnect with backoff if needed."""
        if self.connected:
            try:
                info = mt5.terminal_info()
                if info is not None:
                    return True
            except Exception:
                pass
            self.connected = False

        now = time.time()
        if now - self.last_connect_attempt < self.reconnect_delay:
            return False

        self.last_connect_attempt = now
        logger.info("Reconnecting to MT5 (delay=%.1fs)...", self.reconnect_delay)

        if self.connect():
            return True

        self.reconnect_delay = min(self.reconnect_delay * 2, RECONNECT_MAX_DELAY)
        return False

    def shutdown(self):
        """Gracefully shut down MT5 connection."""
        try:
            mt5.shutdown()
        except Exception:
            pass
        self.connected = False


class TrackedPosition:
    """State tracker for a single open position."""

    def __init__(
        self,
        ticket: int,
        symbol: str,
        side: PositionSide,
        entry_price: float,
        lots: float,
        sl: float,
        tp: float,
        strategy_name: str,
    ):
        self.ticket = ticket
        self.symbol = symbol
        self.side = side
        self.entry_price = entry_price
        self.lots = lots
        self.initial_sl = sl
        self.current_sl = sl
        self.tp = tp
        self.strategy_name = strategy_name
        self.entry_time = datetime.utcnow()
        self.bars_held = 0
        self.high_since_entry = entry_price
        self.low_since_entry = entry_price
        self.mfe_trail_active = False

    def update_extremes(self, high: float, low: float):
        """Update tracked high/low since entry."""
        self.high_since_entry = max(self.high_since_entry, high)
        self.low_since_entry = min(self.low_since_entry, low)
        self.bars_held += 1

    def to_dict(self) -> dict:
        """Serialize to dictionary for state persistence."""
        return {
            "ticket": self.ticket,
            "symbol": self.symbol,
            "side": self.side.value,
            "entry_price": self.entry_price,
            "lots": self.lots,
            "initial_sl": self.initial_sl,
            "current_sl": self.current_sl,
            "tp": self.tp,
            "strategy_name": self.strategy_name,
            "entry_time": self.entry_time.isoformat(),
            "bars_held": self.bars_held,
            "high_since_entry": self.high_since_entry,
            "low_since_entry": self.low_since_entry,
            "mfe_trail_active": self.mfe_trail_active,
        }


class OrderExecutor:
    """Sends market orders to MT5 with retry logic and validation."""

    @staticmethod
    def send_market_order(
        symbol: str,
        side: PositionSide,
        lots: float,
        sl: float,
        tp: float,
        comment: str = "",
    ) -> int | None:
        """Send a market order and return the ticket number, or None on failure."""
        sym_info = mt5.symbol_info(symbol)
        if sym_info is None:
            logger.error("Symbol %s not found in MT5", symbol)
            return None

        if not sym_info.visible:
            mt5.symbol_select(symbol, True)

        lot_step = sym_info.volume_step
        lots = max(sym_info.volume_min, round(lots / lot_step) * lot_step)
        lots = min(lots, sym_info.volume_max, MAX_ALLOWED_LOTS)

        price = mt5.symbol_info_tick(symbol)
        if price is None:
            logger.error("Cannot get tick for %s", symbol)
            return None

        if side == PositionSide.LONG:
            order_type = mt5.ORDER_TYPE_BUY
            entry_price = price.ask
        else:
            order_type = mt5.ORDER_TYPE_SELL
            entry_price = price.bid

        filling = OrderExecutor._detect_filling_mode(sym_info)

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lots,
            "type": order_type,
            "price": entry_price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": 202605,
            "comment": comment[:31],
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling,
        }

        for attempt in range(3):
            result = mt5.order_send(request)
            if result is None:
                logger.error("order_send returned None (attempt %d)", attempt + 1)
                time.sleep(0.5)
                continue

            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(
                    "ORDER FILLED: %s %s %.2f lots @ %.5f SL=%.5f TP=%.5f ticket=%d",
                    side.name, symbol, lots, result.price, sl, tp, result.order,
                )
                return result.order

            logger.warning(
                "Order rejected (attempt %d): retcode=%d comment=%s",
                attempt + 1, result.retcode, result.comment,
            )
            time.sleep(0.5)

        return None

    @staticmethod
    def _detect_filling_mode(sym_info):
        """Detect supported filling mode from symbol info bitmask.

        Brokers advertise filling_mode as a bitmask:
            bit 0 (1) = ORDER_FILLING_FOK
            bit 1 (2) = ORDER_FILLING_IOC
        If neither is set, fall back to ORDER_FILLING_RETURN (book brokers).
        """
        fm = sym_info.filling_mode if sym_info else 0
        if fm & 1:
            return mt5.ORDER_FILLING_FOK
        if fm & 2:
            return mt5.ORDER_FILLING_IOC
        return mt5.ORDER_FILLING_RETURN

    @staticmethod
    def modify_sl(ticket: int, symbol: str, new_sl: float) -> bool:
        """Modify the stop-loss of an open position."""
        positions = mt5.positions_get(ticket=ticket)
        if positions is None or len(positions) == 0:
            return False

        pos = positions[0]
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "symbol": symbol,
            "sl": new_sl,
            "tp": pos.tp,
            "magic": 202605,
        }

        result = mt5.order_send(request)
        if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info("SL modified: ticket=%d new_sl=%.5f", ticket, new_sl)
            return True
        return False

    @staticmethod
    def close_position(ticket: int, symbol: str, lots: float, side: PositionSide) -> bool:
        """Close an open position by sending an opposite market order."""
        price = mt5.symbol_info_tick(symbol)
        if price is None:
            return False

        if side == PositionSide.LONG:
            order_type = mt5.ORDER_TYPE_SELL
            close_price = price.bid
        else:
            order_type = mt5.ORDER_TYPE_BUY
            close_price = price.ask

        sym_info = mt5.symbol_info(symbol)
        filling = OrderExecutor._detect_filling_mode(sym_info)

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lots,
            "type": order_type,
            "position": ticket,
            "price": close_price,
            "deviation": 20,
            "magic": 202605,
            "comment": "close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling,
        }

        result = mt5.order_send(request)
        if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info("POSITION CLOSED: ticket=%d %s %.2f lots", ticket, symbol, lots)
            return True
        return False


class TrailingSLManager:
    """GARCH-adaptive trailing stop with MFE activation."""

    MFE_ACTIVATION_MULT = 1.0
    MFE_TRAIL_PCT = 0.5

    @classmethod
    def update_trailing(cls, pos: TrackedPosition, current_atr: float, garch_vol_ratio: float):
        """Update trailing stop for a tracked position based on MFE progress."""
        risk_dist = abs(pos.entry_price - pos.initial_sl)
        if risk_dist < 1e-10:
            risk_dist = current_atr * 2.0

        if pos.side == PositionSide.LONG:
            mfe = pos.high_since_entry - pos.entry_price
            if mfe > risk_dist * cls.MFE_ACTIVATION_MULT:
                pos.mfe_trail_active = True
                candidate = pos.entry_price + mfe * cls.MFE_TRAIL_PCT
                if candidate > pos.current_sl:
                    pos.current_sl = candidate
                    OrderExecutor.modify_sl(pos.ticket, pos.symbol, pos.current_sl)
        else:
            mfe = pos.entry_price - pos.low_since_entry
            if mfe > risk_dist * cls.MFE_ACTIVATION_MULT:
                pos.mfe_trail_active = True
                candidate = pos.entry_price - mfe * cls.MFE_TRAIL_PCT
                if candidate < pos.current_sl:
                    pos.current_sl = candidate
                    OrderExecutor.modify_sl(pos.ticket, pos.symbol, pos.current_sl)


class SignalArbiter:
    """Fuses signals from MFT and Swing strategies with conflict resolution."""

    @staticmethod
    def arbitrate(mft_signal: int, swing_signal: int) -> int:
        """Resolve conflicting signals between MFT and Swing strategies.

        Priority rules:
            - If both agree: return the shared direction (high confidence)
            - If one is flat: return the active signal
            - If they conflict: return 0 (no trade — conflicting views)
        """
        if mft_signal == swing_signal:
            return mft_signal
        if mft_signal == 0:
            return swing_signal
        if swing_signal == 0:
            return mft_signal
        return 0


class LiveTradingEngine:
    """Main orchestrator for the continuous live trading bot.

    Event-driven architecture:
        1. Check MT5 connection (reconnect with exponential backoff)
        2. For each symbol, detect if a NEW closed bar appeared
        3. Only on new bar: generate signals from closed candle (iloc[-2])
        4. Arbitrate conflicting signals
        5. Check risk circuit breakers
        6. Execute new orders
        7. Update trailing stops on open positions (every cycle)
        8. Log state + heartbeat
        9. Sleep(POLL_INTERVAL)

    Anti-repainting guarantee:
        Signals are NEVER read from the forming candle (iloc[-1]).
        All signal reads use iloc[-2] — the last fully closed bar.
        Signal computation fires ONCE per new closed candle, not per tick.
    """

    def __init__(self):
        self.connection = MT5ConnectionManager()
        self.positions: dict[int, TrackedPosition] = {}
        self.circuit_breaker = RiskCircuitBreaker(DEFAULT_CAPITAL)
        self.executor = OrderExecutor()
        self.trail_manager = TrailingSLManager()
        self.arbiter = SignalArbiter()

        self.mft_strategy = None
        self.swing_strategy = None

        self._last_bar_time: dict[str, datetime] = {}
        self._signal_cache: dict[str, pd.DataFrame] = {}

        self._init_strategies()

        self.running = False
        self.last_heartbeat = 0.0
        self.trade_log: list[dict] = []

    def _init_strategies(self):
        """Initialize the two super-strategies and trigger initial LSTM training."""
        try:
            from quant_engine.strategies.ultimate_mft import UltimateMFTStrategy
            self.mft_strategy = UltimateMFTStrategy()
            logger.info("Ultimate MFT strategy initialized")
        except Exception as exc:
            logger.error("Failed to init MFT strategy: %s", exc)

        try:
            from quant_engine.strategies.ultimate_swing import UltimateSwingStrategy
            self.swing_strategy = UltimateSwingStrategy()
            logger.info("Ultimate Swing strategy initialized")
        except Exception as exc:
            logger.error("Failed to init Swing strategy: %s", exc)

    def _warmup_swing_model(self):
        """Train the Swing LSTM model on startup data and launch background retraining."""
        if self.swing_strategy is None:
            return
        try:
            if SWING_SYMBOLS:
                symbol = SWING_SYMBOLS[0]
                df = self._fetch_bars(symbol, SWING_TIMEFRAME_STR, 500)
                if not df.empty and len(df) >= 200:
                    self.swing_strategy.update_model(df)
                    logger.info("Swing LSTM model warm-up completed on %s", symbol)

            def _fetch_retrain_data():
                """Fetch fresh data for background retraining."""
                if SWING_SYMBOLS:
                    return self._fetch_bars(SWING_SYMBOLS[0], SWING_TIMEFRAME_STR, 500)
                return None

            self.swing_strategy.start_background_retraining(_fetch_retrain_data)
        except Exception as exc:
            logger.error("Swing model warm-up failed: %s", exc)

    def _has_new_closed_bar(self, symbol: str, timeframe_str: str) -> bool:
        """Detect whether a new closed bar has appeared since last check.

        Compares the timestamp of the second-to-last bar (the last CLOSED bar)
        against the stored timestamp. Returns True only on change.
        This is the core anti-repainting mechanism.
        """
        cache_key = f"{symbol}_{timeframe_str}"
        tf = TIMEFRAME_MAP.get(timeframe_str, mt5.TIMEFRAME_M5)
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, 2)
        if rates is None or len(rates) < 2:
            return False

        closed_bar_time = datetime.utcfromtimestamp(rates[-2]["time"] if isinstance(rates[-2], np.void) else rates[-2][0])

        previous = self._last_bar_time.get(cache_key)
        if previous is None or closed_bar_time > previous:
            self._last_bar_time[cache_key] = closed_bar_time
            return True
        return False

    def _fetch_bars(self, symbol: str, timeframe_str: str, count: int = 500) -> pd.DataFrame:
        """Fetch OHLCV bars from MT5 for signal generation."""
        tf = TIMEFRAME_MAP.get(timeframe_str, mt5.TIMEFRAME_M5)
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)
        if rates is None or len(rates) == 0:
            return pd.DataFrame()

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        df.rename(columns={
            "tick_volume": "Volume", "open": "Open",
            "high": "High", "low": "Low", "close": "Close",
        }, inplace=True)
        return df[["Open", "High", "Low", "Close", "Volume"]]

    def _get_latest_signal(self, df: pd.DataFrame, strategy, cache_key: str = "") -> tuple:
        """Run strategy signal generation and return the closed bar's signal.

        Reads from iloc[-2] (last CLOSED bar) to prevent repainting.
        Returns (signal_int, result_dataframe) tuple so the caller can
        reuse the result DataFrame without re-running generate_signals.
        """
        if strategy is None or df.empty or len(df) < 100:
            return 0, None
        try:
            result = strategy.generate_signals(df.copy())
            if cache_key:
                self._signal_cache[cache_key] = result
            if "Signal" in result.columns and len(result) >= 2:
                return int(result["Signal"].iloc[-2]), result
        except Exception as exc:
            logger.warning("Signal generation failed: %s", exc)
        return 0, None

    def _compute_lot_size(self, symbol: str, entry_price: float, sl_price: float) -> float:
        """Compute position size based on risk-per-trade and distance to SL."""
        account = mt5.account_info()
        if account is None:
            return 0.01

        equity = account.equity
        risk_amount = equity * RISK_PER_TRADE_PCT
        sl_distance = abs(entry_price - sl_price)
        if sl_distance < 1e-10:
            sl_distance = entry_price * 0.005

        sym_info = mt5.symbol_info(symbol)
        if sym_info is None:
            return 0.01

        contract_size = sym_info.trade_contract_size
        tick_value = sym_info.trade_tick_value
        tick_size = sym_info.trade_tick_size

        if tick_value > 0 and tick_size > 0:
            sl_ticks = sl_distance / tick_size
            risk_per_lot = sl_ticks * tick_value
            lots = risk_amount / (risk_per_lot + 1e-10)
        else:
            nominal = contract_size * entry_price if contract_size != 100000 else contract_size
            lots = risk_amount / (sl_distance / (entry_price + 1e-10) * nominal + 1e-10)

        lots = max(sym_info.volume_min, min(lots, MAX_ALLOWED_LOTS, sym_info.volume_max))
        lot_step = sym_info.volume_step
        lots = round(lots / lot_step) * lot_step
        return lots

    def _get_sl_tp(
        self, df: pd.DataFrame, signal: int, strategy, cached_result: pd.DataFrame = None,
    ) -> tuple[float, float]:
        """Extract SL and TP from cached strategy output for the closed bar.

        Uses the cached result DataFrame from _get_latest_signal to avoid
        calling generate_signals() twice. Reads iloc[-2] (closed bar).
        Falls back to ATR-based defaults if cached result is unavailable.
        """
        try:
            if cached_result is not None:
                result = cached_result
            else:
                result = strategy.generate_signals(df.copy())
            last = result.iloc[-2]
            sl = last.get("SL_Price", np.nan)
            tp = last.get("TP_Price", np.nan)
            if not np.isnan(sl) and not np.isnan(tp):
                return float(sl), float(tp)
        except Exception:
            pass

        atr_val = calculate_atr(df, 14).iloc[-2]
        price = df["Close"].iloc[-2]
        if signal == 1:
            return price - 2.0 * atr_val, price + 3.5 * atr_val
        else:
            return price + 2.0 * atr_val, price - 3.5 * atr_val

    def _total_exposure_lots(self) -> float:
        """Sum of lots across all tracked positions."""
        return sum(p.lots for p in self.positions.values())

    def _sync_positions_with_mt5(self):
        """Synchronize tracked positions with MT5 terminal state.

        When a position disappears from MT5, query history_deals_get to
        compute the realized PnL (including swap and commission) and feed
        it into the circuit breaker so consecutive-loss and daily-DD
        tracking stays accurate.
        """
        mt5_positions = mt5.positions_get()
        if mt5_positions is None:
            return

        mt5_tickets = {p.ticket for p in mt5_positions}

        closed_tickets = [t for t in self.positions if t not in mt5_tickets]
        for ticket in closed_tickets:
            pos = self.positions.pop(ticket)
            realized_pnl = self._query_deal_pnl(ticket)
            self.circuit_breaker.record_trade(realized_pnl)
            logger.info(
                "Position closed: ticket=%d %s pnl=%.2f (fed to circuit breaker)",
                ticket, pos.symbol, realized_pnl,
            )

    def _query_deal_pnl(self, ticket: int) -> float:
        """Query MT5 deal history for a closed position and return net PnL.

        Sums profit, swap, and commission from all deals associated with
        the position ticket. Returns 0.0 if history is unavailable.
        """
        try:
            now = datetime.utcnow()
            deals = mt5.history_deals_get(
                now - timedelta(days=7), now,
            )
            if deals is None:
                return 0.0
            total = 0.0
            for deal in deals:
                if deal.position_id == ticket:
                    total += deal.profit + deal.swap + deal.commission
            return total
        except Exception as exc:
            logger.warning("Failed to query deal PnL for ticket %d: %s", ticket, exc)
            return 0.0

    def _save_state(self):
        """Persist engine state to disk for crash recovery."""
        state = {
            "positions": {str(k): v.to_dict() for k, v in self.positions.items()},
            "circuit_breaker_tripped": self.circuit_breaker.tripped,
            "trade_log_count": len(self.trade_log),
            "timestamp": datetime.utcnow().isoformat(),
        }
        try:
            with open(STATE_FILE, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
        except Exception as exc:
            logger.error("State save failed: %s", exc)

    def _heartbeat(self):
        """Periodic health check and status logging."""
        now = time.time()
        if now - self.last_heartbeat < HEARTBEAT_INTERVAL:
            return

        self.last_heartbeat = now
        account = mt5.account_info()
        equity_str = f"${account.equity:,.2f}" if account else "N/A"
        logger.info(
            "HEARTBEAT: equity=%s positions=%d exposure=%.2f lots breaker=%s",
            equity_str, len(self.positions), self._total_exposure_lots(),
            "TRIPPED" if self.circuit_breaker.tripped else "OK",
        )

    def _process_symbol(self, symbol: str):
        """Process a single symbol: manage open positions OR open new ones.

        Position management (trailing SL, exits) runs EVERY cycle for live price.
        New signal generation is GATED on _has_new_closed_bar — runs only
        when a new closed candle is detected, reading iloc[-2] throughout.
        """
        open_positions = [
            pos for pos in self.positions.values() if pos.symbol == symbol
        ]

        if open_positions:
            for pos in open_positions:
                active_tf = (
                    MFT_TIMEFRAME_STR if "MFT" in pos.strategy_name
                    else SWING_TIMEFRAME_STR
                )
                df = self._fetch_bars(symbol, active_tf, 300)
                if df.empty:
                    continue

                pos.update_extremes(df["High"].iloc[-1], df["Low"].iloc[-1])
                try:
                    atr_val = calculate_atr(df, 14).iloc[-1]
                    returns = df["Close"].pct_change().fillna(0)
                    garch = fit_garch_volatility(returns)
                    garch_ratio = (
                        garch.iloc[-1] / (garch.rolling(60).mean().iloc[-1] + 1e-10)
                    )
                    self.trail_manager.update_trailing(pos, atr_val, garch_ratio)
                except Exception:
                    pass

                if not self._has_new_closed_bar(symbol, active_tf):
                    continue

                """Check strategy exit signals on the closed bar only."""
                active_strategy = (
                    self.mft_strategy if "MFT" in pos.strategy_name
                    else self.swing_strategy
                )
                if active_strategy is not None and len(df) >= 2:
                    try:
                        result = active_strategy.generate_signals(df.copy())
                        last = result.iloc[-2]
                        should_exit = False

                        if pos.side == PositionSide.LONG and last.get("Exit_Long", False):
                            should_exit = True
                        elif pos.side == PositionSide.SHORT and last.get("Exit_Short", False):
                            should_exit = True

                        current_signal = int(last.get("Signal", 0))
                        if current_signal != 0 and current_signal != pos.side.value:
                            """Strategy wants to reverse — close current position."""
                            should_exit = True

                        if should_exit:
                            logger.info(
                                "STRATEGY EXIT: ticket=%d %s %s",
                                pos.ticket, pos.symbol, pos.side.name,
                            )
                            self.executor.close_position(
                                pos.ticket, pos.symbol, pos.lots, pos.side,
                            )
                    except Exception as exc:
                        logger.warning("Exit signal check failed: %s", exc)
            return

        """Gate new entries on candle change detection."""
        mft_new_bar = (
            symbol in MFT_SYMBOLS
            and self._has_new_closed_bar(symbol, MFT_TIMEFRAME_STR)
        )
        swing_new_bar = (
            symbol in SWING_SYMBOLS
            and self._has_new_closed_bar(symbol, SWING_TIMEFRAME_STR)
        )

        if not mft_new_bar and not swing_new_bar:
            return

        logger.info("New closed bar detected for %s (MFT=%s, Swing=%s)", symbol, mft_new_bar, swing_new_bar)

        if not self.circuit_breaker.check(
            mt5.account_info().equity if mt5.account_info() else DEFAULT_CAPITAL,
            self._total_exposure_lots(),
        ):
            return

        mft_signal = 0
        swing_signal = 0
        mft_result = None
        swing_result = None

        if mft_new_bar and self.mft_strategy is not None:
            df_mft = self._fetch_bars(symbol, MFT_TIMEFRAME_STR, 500)
            mft_signal, mft_result = self._get_latest_signal(
                df_mft, self.mft_strategy, f"{symbol}_MFT"
            )

        if swing_new_bar and self.swing_strategy is not None:
            df_swing = self._fetch_bars(symbol, SWING_TIMEFRAME_STR, 500)
            swing_signal, swing_result = self._get_latest_signal(
                df_swing, self.swing_strategy, f"{symbol}_Swing"
            )

        final_signal = self.arbiter.arbitrate(mft_signal, swing_signal)

        if final_signal == 0:
            return

        active_strategy = self.mft_strategy if mft_signal != 0 else self.swing_strategy
        active_tf = MFT_TIMEFRAME_STR if mft_signal != 0 else SWING_TIMEFRAME_STR
        active_result = mft_result if mft_signal != 0 else swing_result
        df_exec = self._fetch_bars(symbol, active_tf, 500)

        if df_exec.empty or len(df_exec) < 2:
            return

        sl, tp = self._get_sl_tp(df_exec, final_signal, active_strategy, active_result)
        price = df_exec["Close"].iloc[-2]
        side = PositionSide.LONG if final_signal == 1 else PositionSide.SHORT

        lots = self._compute_lot_size(symbol, price, sl)
        if lots < 0.01:
            return

        strategy_name = "MFT" if mft_signal != 0 else "Swing"
        if mft_signal != 0 and swing_signal != 0:
            strategy_name = "MFT+Swing"

        comment = f"{strategy_name}_{side.name}"

        ticket = self.executor.send_market_order(symbol, side, lots, sl, tp, comment)
        if ticket is not None:
            self.positions[ticket] = TrackedPosition(
                ticket=ticket, symbol=symbol, side=side,
                entry_price=price, lots=lots, sl=sl, tp=tp,
                strategy_name=strategy_name,
            )
            self.trade_log.append({
                "time": datetime.utcnow().isoformat(),
                "symbol": symbol, "side": side.name,
                "lots": lots, "price": price,
                "sl": sl, "tp": tp, "strategy": strategy_name,
            })

    def run(self):
        """Main event loop — runs continuously until interrupted."""
        logger.info("=" * 60)
        logger.info("LIVE TRADING ENGINE STARTING")
        logger.info("MFT symbols: %s | Swing symbols: %s", MFT_SYMBOLS, SWING_SYMBOLS)
        logger.info("Risk per trade: %.1f%% | Max DD: %.1f%%", RISK_PER_TRADE_PCT * 100, MAX_DAILY_DRAWDOWN_PCT * 100)
        logger.info("=" * 60)

        if not self.connection.connect():
            logger.critical("Initial MT5 connection failed — exiting")
            return

        account = mt5.account_info()
        if account:
            self.circuit_breaker = RiskCircuitBreaker(account.equity)

        self._warmup_swing_model()

        self.running = True
        all_symbols = list(set(MFT_SYMBOLS + SWING_SYMBOLS))

        try:
            while self.running:
                if not self.connection.ensure_connected():
                    time.sleep(POLL_INTERVAL)
                    continue

                account = mt5.account_info()
                if account:
                    self.circuit_breaker.reset_daily(account.equity)

                self._sync_positions_with_mt5()

                for symbol in all_symbols:
                    try:
                        self._process_symbol(symbol)
                    except Exception as exc:
                        logger.error("Error processing %s: %s", symbol, exc)

                self._heartbeat()
                self._save_state()

                time.sleep(POLL_INTERVAL)

        except KeyboardInterrupt:
            logger.info("Shutdown requested via KeyboardInterrupt")
        except Exception as exc:
            logger.critical("Unhandled exception in main loop: %s", exc, exc_info=True)
        finally:
            self.running = False
            if self.swing_strategy is not None:
                try:
                    self.swing_strategy.stop_background_retraining()
                except Exception:
                    pass
            self._save_state()
            self.connection.shutdown()
            logger.info("Live Trading Engine stopped")


if __name__ == "__main__":
    engine = LiveTradingEngine()
    engine.run()
