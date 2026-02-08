"""Symbol metadata for Dhan bulk historical candles.

Your Dhan bulk request needs:
- securityId
- exchangeSegment
- instrument
- expiryCode

For indices, we reuse the IDs already present in the trading bot repo.
If your Dhan bulk endpoint expects different values (e.g., exchangeSegment='NSE_EQ'),
adjust these mappings.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os


@dataclass(frozen=True)
class DhanInstrument:
    security_id: str
    exchange_segment: str
    instrument: str
    expiry_code: int = 0
    oi: bool = False


# Default mapping based on backend/indices.py
# NOTE: Some Dhan endpoints use different exchangeSegment/instrument strings.
SYMBOLS: dict[str, DhanInstrument] = {
    "NIFTY": DhanInstrument(security_id="13", exchange_segment="IDX_I", instrument="INDEX"),
    "BANKNIFTY": DhanInstrument(security_id="25", exchange_segment="IDX_I", instrument="INDEX"),
    "FINNIFTY": DhanInstrument(security_id="27", exchange_segment="IDX_I", instrument="INDEX"),
    "SENSEX": DhanInstrument(security_id="51", exchange_segment="IDX_I", instrument="INDEX"),
}


def _load_overrides() -> dict[str, DhanInstrument]:
    raw = os.getenv("DHAN_SYMBOL_MAP_JSON", "").strip()
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}

    out: dict[str, DhanInstrument] = {}
    for sym, cfg in payload.items():
        if not isinstance(cfg, dict):
            continue
        out[str(sym).strip().upper()] = DhanInstrument(
            security_id=str(cfg.get("securityId") or cfg.get("security_id") or "0"),
            exchange_segment=str(cfg.get("exchangeSegment") or cfg.get("exchange_segment") or "IDX_I"),
            instrument=str(cfg.get("instrument") or "INDEX"),
            expiry_code=int(cfg.get("expiryCode") or cfg.get("expiry_code") or 0),
            oi=bool(cfg.get("oi") or False),
        )
    return out


def get_dhan_instrument(symbol: str) -> DhanInstrument:
    sym = str(symbol or "").strip().upper()

    overrides = _load_overrides()
    if sym in overrides:
        return overrides[sym]

    if sym in SYMBOLS:
        return SYMBOLS[sym]
    # fallback: treat as index-ish
    return DhanInstrument(security_id="0", exchange_segment="IDX_I", instrument="INDEX")
