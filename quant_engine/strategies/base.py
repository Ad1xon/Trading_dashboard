"""
Base strategy definition — abstract interface for all trading strategies.
"""

import pandas as pd
from abc import ABC, abstractmethod
from config import DEFAULT_MAX_HOLDING, MFE_ACTIVATION_MULTIPLIER, MFE_TRAIL_PCT
from ..macro_filter import MacroFilter


class BaseStrategy(ABC):
    """Abstract base class for trading strategies.

    All strategies expose a ``params`` dict for optimiser introspection.
    The ``MacroFilter`` is initialised here so every subclass inherits
    automatic signal blackout around high-impact macro events.
    """

    params: dict = {}

    def __init__(self):
        self.macro_filter = MacroFilter()

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Produce ``Signal``, ``SL_Price``, ``TP_Price``, ``Max_Hold``
        columns on the input DataFrame."""

    def apply_macro_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Zero-out signals within ±N minutes of high-impact macro releases."""
        return self.macro_filter.apply_blackout_mask(df)

    def get_params(self) -> dict:
        """Return current parameter values as a flat dict."""
        base = {
            'mfe_activation': MFE_ACTIVATION_MULTIPLIER,
            'mfe_trail_pct': MFE_TRAIL_PCT,
            'max_holding': DEFAULT_MAX_HOLDING,
        }
        base.update({k: v[0] for k, v in self.params.items()})
        return base

    def get_param_ranges(self) -> dict:
        """Return optimiser-compatible ``(lo, hi, step)`` tuples."""
        base = {
            'mfe_activation': (0.5, 3.0, 0.5),
            'mfe_trail_pct': (0.1, 0.9, 0.1),
        }
        base.update({k: v[1:] for k, v in self.params.items()})
        return base