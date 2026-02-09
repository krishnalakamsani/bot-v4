from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple


class ConfidenceLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


@dataclass(frozen=True)
class ConfidenceBreakdown:
    total: float
    st_strength: float
    candle_body_strength: float
    macd_momentum: float
    volatility_expansion: float
    notes: str = ""

    def __str__(self) -> str:  # Used in logs
        return (
            f"Total={self.total:.0f} | ST={self.st_strength:.0f} | Body={self.candle_body_strength:.0f} | "
            f"MACD={self.macd_momentum:.0f} | Vol={self.volatility_expansion:.0f}"
            + (f" | {self.notes}" if self.notes else "")
        )


class ConfidenceCalculator:
    """Confidence scoring for entry and runtime management.

    Matches the user-provided spec:
    - Entry confidence computed once (0-100)
    - Runtime confidence updated every candle while position is open
    """

    MIN_ENTRY_CONFIDENCE = 65.0
    MIN_RUNTIME_CONFIDENCE = 40.0

    # Entry factor thresholds from spec
    _ST_STRONG_THRESHOLD_PCT = 0.25
    _BODY_STRONG_THRESHOLD_PCT = 60.0
    _VOL_EXPANSION_WINDOW = 14

    def calculate_entry_confidence(
        self,
        *,
        supertrend_distance_pct: float,
        candle_open: float,
        candle_high: float,
        candle_low: float,
        candle_close: float,
        macd_histogram: Optional[float],
        recent_ranges: Optional[List[float]] = None,
    ) -> ConfidenceBreakdown:
        st_strength = 30.0 if float(supertrend_distance_pct) > self._ST_STRONG_THRESHOLD_PCT else 0.0

        candle_range = float(candle_high) - float(candle_low)
        body_pct = 0.0
        if candle_range > 0:
            body_pct = abs(float(candle_close) - float(candle_open)) / candle_range * 100.0
        candle_body_strength = 20.0 if body_pct >= self._BODY_STRONG_THRESHOLD_PCT else 0.0

        # MACD momentum score (histogram magnitude); mandatory gate (>=0.5) is enforced in entry_logic.
        macd_momentum = 0.0
        if macd_histogram is not None:
            h = abs(float(macd_histogram))
            if h >= 1.5:
                macd_momentum = 30.0
            elif h >= 1.0:
                macd_momentum = 25.0
            elif h >= 0.75:
                macd_momentum = 20.0
            elif h >= 0.5:
                macd_momentum = 15.0

        volatility_expansion = 0.0
        if candle_range > 0:
            ranges = (recent_ranges or [])
            if len(ranges) >= 3:
                window = ranges[-self._VOL_EXPANSION_WINDOW :]
                avg_range = sum(window) / len(window) if window else 0.0
                if avg_range > 0 and candle_range > avg_range:
                    volatility_expansion = 20.0

        total = float(st_strength + candle_body_strength + macd_momentum + volatility_expansion)
        total = max(0.0, min(100.0, total))
        return ConfidenceBreakdown(
            total=total,
            st_strength=st_strength,
            candle_body_strength=candle_body_strength,
            macd_momentum=macd_momentum,
            volatility_expansion=volatility_expansion,
            notes=f"BodyPct={body_pct:.0f} STDist={supertrend_distance_pct:.3f}%",
        )

    def calculate_runtime_confidence(
        self,
        *,
        previous_confidence: float,
        candle_body_pct: Optional[float],
        macd_histogram: Optional[float],
        prev_macd_histogram: Optional[float],
        candles_held: int,
        decay_after_candles: int = 10,
    ) -> Tuple[float, str]:
        conf = float(previous_confidence)
        reasons: List[str] = []

        # Decay: weak candle body
        if candle_body_pct is not None and float(candle_body_pct) < 30.0:
            conf -= 10.0
            reasons.append("WEAK_BODY")

        # Decay: MACD histogram shrinking (momentum fading)
        if macd_histogram is not None and prev_macd_histogram is not None:
            if abs(float(macd_histogram)) < abs(float(prev_macd_histogram)):
                conf -= 10.0
                reasons.append("MACD_SHRINK")

        # Time-based decay after N candles
        if int(candles_held) > int(decay_after_candles):
            conf -= 5.0
            reasons.append("TIME_DECAY")

        # Optional boost: strong continuation candle (cap at 100)
        if candle_body_pct is not None and float(candle_body_pct) >= 60.0:
            conf += 5.0
            reasons.append("CONTINUATION")

        conf = max(0.0, min(100.0, conf))
        return conf, "+".join(reasons) if reasons else "NO_CHANGE"
