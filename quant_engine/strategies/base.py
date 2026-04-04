# quant_engine/strategies/base.py
"""Base strategy definition."""

import pandas as pd
from abc import ABC, abstractmethod
from config import DEFAULT_MAX_HOLDING, MFE_ACTIVATION_MULTIPLIER, MFE_TRAIL_PCT

class BaseStrategy(ABC):
    """All strategies expose *params* dict for optimiser introspection."""

    params: dict = {}

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    def get_params(self) -> dict:
        base = {
            'mfe_activation': MFE_ACTIVATION_MULTIPLIER,
            'mfe_trail_pct': MFE_TRAIL_PCT,
            'max_holding': DEFAULT_MAX_HOLDING
        }
        base.update({k: v[0] for k, v in self.params.items()})
        return base

    def get_param_ranges(self) -> dict:
        base = {
            'mfe_activation': (0.5, 3.0, 0.5),
            'mfe_trail_pct': (0.1, 0.9, 0.1),
        }
        base.update({k: v[1:] for k, v in self.params.items()})
        return base