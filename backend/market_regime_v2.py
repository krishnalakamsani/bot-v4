from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple


class MarketRegime(str, Enum):
    SIDEWAYS = "SIDEWAYS"
    TRENDING = "TRENDING"


@dataclass(frozen=True)
class RegimeInfo:
    method: str
    adx: Optional[float]
    adx_threshold: float
    st_distance_pct: Optional[float]
    st_distance_threshold_pct: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "adx": None if self.adx is None else float(self.adx),
            "adx_threshold": float(self.adx_threshold),
            "st_distance_pct": None if self.st_distance_pct is None else float(self.st_distance_pct),
            "st_distance_threshold_pct": float(self.st_distance_threshold_pct),
        }


class MarketRegimeDetector:
    """Gatekeeper: do not allow entries in sideways / low-strength markets."""

    # Spec default: 0.15% ST distance => sideways
    DEFAULT_ST_DISTANCE_SIDEWAYS_THRESHOLD_PCT = 0.15

    def __init__(
        self,
        *,
        use_adx: bool = False,
        adx_threshold: float = 20.0,
        st_distance_sideways_threshold_pct: float = DEFAULT_ST_DISTANCE_SIDEWAYS_THRESHOLD_PCT,
    ) -> None:
        self.use_adx = bool(use_adx)
        self.adx_threshold = float(adx_threshold)
        self.st_distance_sideways_threshold_pct = float(st_distance_sideways_threshold_pct)

    def detect(
        self,
        *,
        current_price: float,
        supertrend_value: float,
        adx: Optional[float],
    ) -> Tuple[MarketRegime, Dict[str, Any]]:
        st_distance_pct: Optional[float] = None
        if current_price and current_price > 0 and supertrend_value is not None:
            st_distance_pct = abs(float(current_price) - float(supertrend_value)) / float(current_price) * 100.0

        if self.use_adx and adx is not None:
            regime = MarketRegime.SIDEWAYS if float(adx) < self.adx_threshold else MarketRegime.TRENDING
            info = RegimeInfo(
                method="ADX",
                adx=float(adx),
                adx_threshold=float(self.adx_threshold),
                st_distance_pct=st_distance_pct,
                st_distance_threshold_pct=float(self.st_distance_sideways_threshold_pct),
            )
            return regime, info.as_dict()

        # Fallback: ST distance % gate
        if st_distance_pct is None:
            # If we cannot compute distance, be conservative: treat as sideways.
            regime = MarketRegime.SIDEWAYS
        else:
            regime = (
                MarketRegime.SIDEWAYS
                if float(st_distance_pct) < self.st_distance_sideways_threshold_pct
                else MarketRegime.TRENDING
            )

        info = RegimeInfo(
            method="ST_DISTANCE",
            adx=None if adx is None else float(adx),
            adx_threshold=float(self.adx_threshold),
            st_distance_pct=st_distance_pct,
            st_distance_threshold_pct=float(self.st_distance_sideways_threshold_pct),
        )
        return regime, info.as_dict()
