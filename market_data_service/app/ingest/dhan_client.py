"""Dhan client adapter.

This module intentionally isolates Dhan SDK/API specifics.
In production you should implement:
- get_index_ltp(symbol) -> float
- get_historical_candles(symbol, timeframe_seconds, start_ts, end_ts) -> list[dict]

Notes:
- Dhan SDK surface varies by version; treat historical endpoints as unstable.
- Keep a strict rate limiter + retry policy around all calls.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
import json
from datetime import datetime, timezone
from typing import Any

import httpx
from tenacity import retry, wait_exponential, stop_after_attempt

from ..settings import settings
from .symbol_map import get_dhan_instrument

logger = logging.getLogger(__name__)


@dataclass
class CandleRow:
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float | None = None


class DhanClient:
    def __init__(self, client_id: str, access_token: str):
        self.client_id = client_id
        self.access_token = access_token

        self._sdk = None
        try:
            from dhanhq import dhanhq  # type: ignore

            self._sdk = dhanhq(client_id, access_token)
        except Exception as e:
            self._sdk = None
            logger.warning(f"[DHAN] SDK not available: {e}")

    def ready(self) -> bool:
        # Streaming may use SDK; historical uses HTTP endpoint.
        return bool(self.client_id and self.access_token)

    def _historical_url(self) -> str:
        if settings.dhan_historical_url:
            return settings.dhan_historical_url
        if settings.dhan_base_url:
            return settings.dhan_base_url.rstrip("/") + "/charts/intraday"
        return ""

    def _historical_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}

        # Prefer explicit JSON headers if provided.
        if settings.dhan_historical_headers_json:
            try:
                raw = json.loads(settings.dhan_historical_headers_json)
                if isinstance(raw, dict):
                    headers.update({str(k): str(v) for k, v in raw.items() if v is not None})
            except Exception:
                pass

        # Common fallbacks (varies by Dhan integration).
        headers.setdefault("access-token", self.access_token)
        headers.setdefault("client-id", self.client_id)
        return headers

    def _parse_ts(self, value: Any) -> datetime:
        if isinstance(value, datetime):
            ts = value
        elif isinstance(value, (int, float)):
            # epoch seconds or ms
            iv = float(value)
            if iv > 10_000_000_000:  # ms
                iv = iv / 1000.0
            # Dhan bulk API provides IST-epoch (epoch computed in IST as if it were UTC).
            # Requirement: persist IST-epoch as-is (no -5:30 normalization).
            # If you ever want to normalize to real UTC, set DHAN_EPOCH_IS_IST=false.
            if not settings.dhan_epoch_is_ist:
                iv = iv - 19_800
            ts = datetime.fromtimestamp(iv, tz=timezone.utc)
        else:
            s = str(value)
            ts = datetime.fromisoformat(s.replace("Z", "+00:00"))

        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(timezone.utc)

    def _parse_candles_payload(self, payload: Any) -> list[CandleRow]:
        """Parse multiple likely candle shapes.

        Supported shapes (best-effort):
        - {"data": [{"timestamp":...,"open":...,"high":...,"low":...,"close":...,"volume":...}, ...]}
        - {"candles": [[ts, o, h, l, c, v?], ...]}
        - [[ts, o, h, l, c, v?], ...]
        """

        if payload is None:
            return []

        data = payload
        if isinstance(payload, dict):
            # Dhan APIs often wrap under status/data
            if "data" in payload:
                data = payload.get("data")
            if isinstance(data, dict) and "candles" in data:
                data = data.get("candles")
            if isinstance(payload.get("candles"), list):
                data = payload.get("candles")

        # Columnar-array format (your provided structure)
        # {
        #   "open": [...], "high": [...], "low": [...], "close": [...],
        #   "volume": [...], "timestamp": [epochSeconds...], "open_interest": [...]
        # }
        if isinstance(data, dict) and all(k in data for k in ("open", "high", "low", "close", "timestamp")):
            try:
                opens = list(data.get("open") or [])
                highs = list(data.get("high") or [])
                lows = list(data.get("low") or [])
                closes = list(data.get("close") or [])
                tss = list(data.get("timestamp") or [])
                vols = list(data.get("volume") or [])

                n = min(len(opens), len(highs), len(lows), len(closes), len(tss))
                candles: list[CandleRow] = []
                for i in range(n):
                    try:
                        ts = self._parse_ts(tss[i])
                        o = float(opens[i])
                        h = float(highs[i])
                        l = float(lows[i])
                        c = float(closes[i])
                        v = None
                        if i < len(vols) and vols[i] is not None:
                            v = float(vols[i])
                        candles.append(CandleRow(ts=ts, open=o, high=h, low=l, close=c, volume=v))
                    except Exception:
                        continue

                candles.sort(key=lambda x: x.ts)
                return candles
            except Exception:
                # Fall through to other parsing modes.
                pass

        candles: list[CandleRow] = []

        if isinstance(data, list):
            for row in data:
                try:
                    if isinstance(row, dict):
                        ts = self._parse_ts(row.get("timestamp") or row.get("ts") or row.get("time"))
                        o = float(row.get("open"))
                        h = float(row.get("high"))
                        l = float(row.get("low"))
                        c = float(row.get("close"))
                        v = row.get("volume")
                        candles.append(CandleRow(ts=ts, open=o, high=h, low=l, close=c, volume=float(v) if v is not None else None))
                    elif isinstance(row, (list, tuple)) and len(row) >= 5:
                        ts = self._parse_ts(row[0])
                        o = float(row[1])
                        h = float(row[2])
                        l = float(row[3])
                        c = float(row[4])
                        v = float(row[5]) if len(row) > 5 and row[5] is not None else None
                        candles.append(CandleRow(ts=ts, open=o, high=h, low=l, close=c, volume=v))
                except Exception:
                    continue

        candles.sort(key=lambda x: x.ts)
        return candles

    @retry(wait=wait_exponential(multiplier=0.5, min=0.5, max=10), stop=stop_after_attempt(6))
    def get_historical_1m_bulk(self, symbol: str, from_date: str, to_date: str) -> list[CandleRow]:
        """Fetch 1-minute candles using Dhan bulk request format.

        Args:
            symbol: e.g. NIFTY
            from_date/to_date: YYYY-MM-DD
        """

        url = self._historical_url()
        if not url:
            raise RuntimeError("Configure DHAN_BASE_URL or DHAN_HISTORICAL_URL for /charts/intraday")

        inst = get_dhan_instrument(symbol)
        body = {
            "securityId": str(inst.security_id),
            "exchangeSegment": str(inst.exchange_segment),
            "instrument": str(inst.instrument),
            "expiryCode": int(inst.expiry_code),
            "oi": bool(inst.oi),
            "fromDate": str(from_date),
            "toDate": str(to_date),
        }

        headers = self._historical_headers()
        with httpx.Client(timeout=30) as client:
            resp = client.post(url, json=body, headers=headers)
            resp.raise_for_status()
            payload = resp.json()

        candles = self._parse_candles_payload(payload)
        return candles

    def get_index_ltp(self, symbol: str) -> float | None:
        """Return current index LTP.

        Uses dhanhq SDK quote_data.
        """
        if not self.ready():
            return None

        if self._sdk is None:
            return None

        inst = get_dhan_instrument(symbol)
        security_id = str(inst.security_id or "").strip()
        if not security_id or security_id == "0":
            return None

        # Some symbols (e.g. SENSEX) can be exposed under different segments.
        segments_to_try = [str(inst.exchange_segment or "IDX_I")]
        if str(symbol or "").strip().upper() == "SENSEX":
            for seg in ("IDX_I", "BSE_INDEX", "BSE"):
                if seg not in segments_to_try:
                    segments_to_try.append(seg)

        try:
            sid_int = int(float(security_id))
        except Exception:
            return None

        for seg in segments_to_try:
            try:
                resp = self._sdk.quote_data({seg: [sid_int]})
            except Exception:
                continue

            if not resp or resp.get("status") != "success":
                continue

            data = resp.get("data", {})
            if isinstance(data, dict) and "data" in data and isinstance(data.get("data"), dict):
                data = data.get("data", {})

            if not isinstance(data, dict):
                continue

            seg_data = data.get(seg, {})
            if not isinstance(seg_data, dict):
                continue

            row = seg_data.get(str(sid_int), {})
            if not isinstance(row, dict) or not row:
                continue

            ltp = row.get("last_price")
            try:
                if ltp is not None and float(ltp) > 0:
                    return float(ltp)
            except Exception:
                pass

            ohlc = row.get("ohlc", {})
            if isinstance(ohlc, dict):
                close = ohlc.get("close")
                try:
                    if close is not None and float(close) > 0:
                        return float(close)
                except Exception:
                    pass

        return None

    def get_historical_candles(
        self,
        symbol: str,
        timeframe_seconds: int,
        start_ts: datetime,
        end_ts: datetime,
    ) -> list[CandleRow]:
        """Fetch historical candles from Dhan (backfill/gap repair).

        MUST obey:
        - strict rate limits
        - pagination/windowing
        - deterministic timestamp alignment
        """
        if not self.ready():
            return []

        # Dhan provides 1-minute only for history in this deployment.
        if int(timeframe_seconds) != 60:
            raise ValueError("Historical timeframe_seconds must be 60 (1-minute)")

        from_date = start_ts.astimezone(timezone.utc).date().isoformat()
        to_date = end_ts.astimezone(timezone.utc).date().isoformat()
        return self.get_historical_1m_bulk(symbol=symbol, from_date=from_date, to_date=to_date)
