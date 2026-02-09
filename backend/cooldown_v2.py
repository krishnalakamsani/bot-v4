from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class CooldownReason(str, Enum):
    EXIT = "EXIT"


@dataclass
class CooldownState:
    active: bool
    candles_remaining: int
    reason: Optional[CooldownReason] = None

    # Backward-compatible alias used by older call sites
    @property
    def is_active(self) -> bool:
        return bool(self.active)


class CooldownManager:
    """Mandatory cooldown after ANY exit (default 3 candles)."""

    def __init__(self, *, cooldown_candles: int = 3) -> None:
        self.cooldown_duration = int(cooldown_candles)
        self._remaining = 0
        self._reason: Optional[CooldownReason] = None

    def reset(self) -> None:
        self._remaining = 0
        self._reason = None

    def activate(self, *, reason: CooldownReason = CooldownReason.EXIT, cooldown_candles: Optional[int] = None) -> None:
        duration = self.cooldown_duration if cooldown_candles is None else int(cooldown_candles)
        self._remaining = max(0, duration)
        self._reason = reason

    def advance(self) -> None:
        if self._remaining > 0:
            self._remaining -= 1
            if self._remaining <= 0:
                self._remaining = 0
                self._reason = None

    def can_enter(self) -> bool:
        return self._remaining <= 0

    def get_state(self) -> CooldownState:
        return CooldownState(active=self._remaining > 0, candles_remaining=int(self._remaining), reason=self._reason)
