"""Macroeconomic event filter for volatility blackout periods."""

import pandas as pd
import numpy as np
from datetime import timedelta
import logging
import requests
import xml.etree.ElementTree as ET
from config import MACRO_BLACKOUT_MINUTES

logger = logging.getLogger(__name__)

class MacroFilter:
    """Manages trading blackouts around high-impact economic events."""

    def __init__(self, blackout_minutes: int = MACRO_BLACKOUT_MINUTES):
        self.blackout_minutes = blackout_minutes
        self.high_impact_events = self._fetch_economic_calendar()

    def _fetch_economic_calendar(self) -> pd.DatetimeIndex:
        """Fetch high-impact events from ForexFactory XML feed."""
        url = "https://nfs.faireconomy.media/ff_calendar_thisweek.xml"
        events = []
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                root = ET.fromstring(response.content)
                for event in root.findall('event'):
                    impact = event.find('impact').text
                    if impact is not None and impact.strip() == 'High':
                        date_str = event.find('date').text
                        time_str = event.find('time').text

                        if time_str and "All Day" not in time_str and "Tentative" not in time_str:
                            dt_str = f"{date_str} {time_str}"
                            dt_obj = pd.to_datetime(dt_str, format="%m-%d-%Y %I:%M%p", errors='coerce')
                            if pd.notnull(dt_obj):
                                events.append(dt_obj)

        except Exception as e:
            logger.error(f"Failed to fetch economic calendar: {e}")

        if not events:
            return pd.DatetimeIndex([])

        event_series = pd.Series(events)

        try:
            event_series = event_series.dt.tz_localize('US/Eastern').dt.tz_convert('UTC').dt.tz_localize(None)
        except Exception:
            pass

        return pd.DatetimeIndex(event_series)

    def apply_blackout_mask(self, df: pd.DataFrame) -> pd.DataFrame:
        """Vectorized nullification of signals during blackout windows."""
        df = df.copy()
        if 'Signal' not in df.columns or self.high_impact_events.empty:
            return df

        if not isinstance(df.index, pd.DatetimeIndex):
            return df

        blackout_mask = np.zeros(len(df), dtype=bool)

        safe_index = df.index.tz_localize(None) if df.index.tz is not None else df.index

        for event_time in self.high_impact_events:
            safe_event = event_time.tz_localize(None) if event_time.tz is not None else event_time
            start_window = safe_event - timedelta(minutes=self.blackout_minutes)
            end_window = safe_event + timedelta(minutes=self.blackout_minutes)

            mask = (safe_index >= start_window) & (safe_index <= end_window)
            blackout_mask = blackout_mask | mask

        df.loc[blackout_mask, 'Signal'] = 0
        return df