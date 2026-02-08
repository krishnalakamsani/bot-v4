from __future__ import annotations

import argparse
import asyncio
import logging
import sqlite3
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from ..db import init_db, upsert_candles_bulk
from ..settings import settings
from .dhan_client import CandleRow


logger = logging.getLogger(__name__)


IST = ZoneInfo("Asia/Kolkata")


def _parse_ts(value: str) -> datetime:
    # candle_data.timestamp is expected to be an ISO string.
    # If it is naive, assume UTC.
    dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _ist_day_bounds_utc(date_ist: str) -> tuple[datetime, datetime]:
    d = datetime.fromisoformat(date_ist).date()
    start_ist = datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=IST)
    end_ist = start_ist + timedelta(days=1) - timedelta(microseconds=1)
    return start_ist.astimezone(timezone.utc), end_ist.astimezone(timezone.utc)


def _load_sqlite_candle_data(
    *,
    sqlite_path: str,
    symbol: str,
    timeframe_seconds: int,
    start_utc: datetime,
    end_utc: datetime,
    limit: int,
) -> list[CandleRow]:
    con = sqlite3.connect(sqlite_path)
    cur = con.cursor()

    # Best-effort: interval_seconds may be NULL for older rows.
    start_s = start_utc.isoformat()
    end_s = end_utc.isoformat()

    try:
        cur.execute(
            """
            SELECT timestamp, high, low, close
            FROM candle_data
            WHERE index_name = ?
              AND (interval_seconds = ? OR interval_seconds IS NULL)
              AND timestamp >= ?
              AND timestamp <= ?
            ORDER BY timestamp ASC
            LIMIT ?
            """,
            (symbol, int(timeframe_seconds), start_s, end_s, int(limit)),
        )
        rows = cur.fetchall()
    finally:
        con.close()

    if not rows:
        return []

    out: list[CandleRow] = []
    prev_close: float | None = None

    for ts_s, high, low, close in rows:
        try:
            ts = _parse_ts(ts_s)
            if not (start_utc <= ts <= end_utc):
                continue
            c = float(close or 0.0)
            h = float(high or c)
            l = float(low or c)
            if c <= 0:
                continue

            o = float(prev_close) if prev_close is not None else c
            prev_close = c

            out.append(CandleRow(ts=ts, open=o, high=h, low=l, close=c, volume=None))
        except Exception:
            continue

    return out


async def _main() -> int:
    parser = argparse.ArgumentParser(description="Import backend SQLite candle_data into Timescale/MDS")
    parser.add_argument("--sqlite", dest="sqlite_path", default=settings.backend_sqlite_path)
    parser.add_argument("--symbol", dest="symbol", default="NIFTY")
    parser.add_argument("--tf", dest="timeframe_seconds", type=int, default=5)
    parser.add_argument("--date-ist", dest="date_ist", required=True, help="YYYY-MM-DD (IST)")
    parser.add_argument("--limit", dest="limit", type=int, default=200000)
    parser.add_argument("--source", dest="source", default="sqlite")
    args = parser.parse_args()

    symbol = str(args.symbol).strip().upper()
    tf = int(args.timeframe_seconds)
    sqlite_path = str(args.sqlite_path)

    start_utc, end_utc = _ist_day_bounds_utc(str(args.date_ist))
    logger.info(
        f"[IMPORT] Loading SQLite candles | DB={sqlite_path} | Symbol={symbol} TF={tf}s | "
        f"DateIST={args.date_ist} UTC={start_utc.isoformat()}..{end_utc.isoformat()}"
    )

    candles = _load_sqlite_candle_data(
        sqlite_path=sqlite_path,
        symbol=symbol,
        timeframe_seconds=tf,
        start_utc=start_utc,
        end_utc=end_utc,
        limit=int(args.limit),
    )

    logger.info(f"[IMPORT] Loaded {len(candles)} rows from SQLite")
    if not candles:
        return 2

    await init_db()
    inserted = await upsert_candles_bulk(
        symbol=symbol,
        timeframe_seconds=tf,
        candles=candles,
        source=str(args.source),
        watermark_kind="import",
    )

    logger.info(f"[IMPORT] Upserted {inserted} rows into Timescale")
    # Print a small summary for CLI usage.
    print(
        {
            "sqlite_path": sqlite_path,
            "symbol": symbol,
            "timeframe_seconds": tf,
            "date_ist": str(args.date_ist),
            "loaded": len(candles),
            "upserted": inserted,
            "first": asdict(candles[0]),
            "last": asdict(candles[-1]),
        }
    )
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    raise SystemExit(asyncio.run(_main()))
