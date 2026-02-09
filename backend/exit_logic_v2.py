from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from datetime import time
from typing import Optional


class ExitReason(str, Enum):
    SL_HIT = "SL_HIT"
    TSL_HIT = "TSL_HIT"
    CONFIDENCE_DROP = "CONFIDENCE_DROP"
    ST_REVERSAL = "ST_REVERSAL"
    EOD_EXIT = "EOD_EXIT"


@dataclass(frozen=True)
class ExitSignal:
    should_exit: bool
    reason: Optional[ExitReason] = None


class ExitLogic:
    """Risk-first exits. Exit immediately if ANY condition hits."""

    EOD_CUTOFF = time(hour=15, minute=20)  # Spec: 15:20 IST

    def check_exit(
        self,
        *,
        is_open_position: bool,
        current_option_ltp: float,
        hard_sl: Optional[float],
        trailing_sl: Optional[float],
        runtime_confidence: float,
        min_runtime_confidence: float = 40.0,
        st_direction: Optional[int],
        prev_st_direction: Optional[int],
        position_type: str,
        current_time_ist: time,
        bypass_eod_exit: bool = False,
    ) -> ExitSignal:
        if not is_open_position:
            return ExitSignal(False)

        ltp = float(current_option_ltp or 0.0)

        # 1) Hard stop loss
        if hard_sl is not None and ltp > 0 and ltp <= float(hard_sl):
            return ExitSignal(True, reason=ExitReason.SL_HIT)

        # 2) Trailing stop loss
        if trailing_sl is not None and ltp > 0 and ltp <= float(trailing_sl):
            return ExitSignal(True, reason=ExitReason.TSL_HIT)

        # 3) Confidence-based exit
        if float(runtime_confidence) <= float(min_runtime_confidence):
            return ExitSignal(True, reason=ExitReason.CONFIDENCE_DROP)

        # 4) SuperTrend reversal (lower priority)
        if st_direction is not None and prev_st_direction is not None and int(st_direction) != int(prev_st_direction):
            # Exit only when flip is against the position direction
            pt = (position_type or "").upper()
            if pt == "CE" and int(prev_st_direction) == 1 and int(st_direction) == -1:
                return ExitSignal(True, reason=ExitReason.ST_REVERSAL)
            if pt == "PE" and int(prev_st_direction) == -1 and int(st_direction) == 1:
                return ExitSignal(True, reason=ExitReason.ST_REVERSAL)

        # 5) Time-based safety exit
        if (not bypass_eod_exit) and current_time_ist >= self.EOD_CUTOFF:
            return ExitSignal(True, reason=ExitReason.EOD_EXIT)

        return ExitSignal(False)
