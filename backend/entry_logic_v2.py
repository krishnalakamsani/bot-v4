from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class EntryDirection(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"


class EntryConfirmation(str, Enum):
    SUPER_TREND_FLIP = "SUPER_TREND_FLIP"
    PRICE_CONFIRMATION = "PRICE_CONFIRMATION"
    MACD_MOMENTUM = "MACD_MOMENTUM"
    CONFIDENCE = "CONFIDENCE"


@dataclass(frozen=True)
class EntryDecision:
    should_enter: bool
    direction: Optional[EntryDirection] = None
    reason: str = ""


class EntryLogic:
    """Implements the user-specified entry rules.

    Mandatory (ALL):
    - SuperTrend flip
    - Candle close relative to SuperTrend line
    - abs(MACD histogram) >= 0.5
    - Entry confidence >= 65
    """

    MACD_HIST_MIN = 0.5

    def check_entry(
        self,
        *,
        market_regime_is_trending: bool,
        st_direction: int,
        prev_st_direction: Optional[int],
        supertrend_value: float,
        candle_close: float,
        macd_histogram: Optional[float],
        entry_confidence: float,
        min_entry_confidence: float = 65.0,
    ) -> EntryDecision:
        if not market_regime_is_trending:
            return EntryDecision(False, reason="SIDEWAYS_REGIME")

        if prev_st_direction is None:
            return EntryDecision(False, reason="NO_PREV_ST")

        if int(st_direction) == int(prev_st_direction):
            return EntryDecision(False, reason="NO_ST_FLIP")

        # Determine flip direction
        direction: Optional[EntryDirection] = None
        if int(prev_st_direction) == -1 and int(st_direction) == 1:
            direction = EntryDirection.BULLISH
        elif int(prev_st_direction) == 1 and int(st_direction) == -1:
            direction = EntryDirection.BEARISH
        else:
            return EntryDecision(False, reason="INVALID_ST_FLIP")

        # Price confirmation vs SuperTrend line
        if direction == EntryDirection.BULLISH and not (float(candle_close) > float(supertrend_value)):
            return EntryDecision(False, reason="PRICE_NOT_ABOVE_ST")
        if direction == EntryDirection.BEARISH and not (float(candle_close) < float(supertrend_value)):
            return EntryDecision(False, reason="PRICE_NOT_BELOW_ST")

        # Momentum confirmation (MACD histogram)
        if macd_histogram is None or abs(float(macd_histogram)) < self.MACD_HIST_MIN:
            return EntryDecision(False, reason="MACD_HIST_TOO_SMALL")

        # Confidence gate
        if float(entry_confidence) < float(min_entry_confidence):
            return EntryDecision(False, reason="LOW_ENTRY_CONFIDENCE")

        return EntryDecision(True, direction=direction, reason="OK")
