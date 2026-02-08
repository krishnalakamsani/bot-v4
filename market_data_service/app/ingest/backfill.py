import asyncio
import logging
from datetime import datetime, timezone, timedelta, date

from ..settings import settings
from ..db import get_watermark, upsert_candles_bulk, set_watermark
from .dhan_client import DhanClient

logger = logging.getLogger(__name__)


def _to_yyyy_mm_dd(d: date) -> str:
    return d.isoformat()


def _utc_today() -> date:
    return datetime.now(timezone.utc).date()


async def backfill_symbol_1m(symbol: str) -> None:
    """Backfill 1-minute candles for a symbol for the last N years.

    Uses a date-windowed approach compatible with Dhan bulk intraday charts endpoint:
    POST /charts/intraday
      { securityId, exchangeSegment, instrument, expiryCode, oi, fromDate, toDate }

    Resume mechanism:
    - reads ingest_watermarks(kind='backfill')
    - continues from last_ts date (idempotent upserts)
    """

    tf = 60
    client = DhanClient(settings.dhan_client_id, settings.dhan_access_token)
    if not client.ready():
        raise RuntimeError("Dhan credentials missing for backfill")

    end_d = _utc_today()
    start_d = (datetime.now(timezone.utc) - timedelta(days=int(settings.backfill_years) * 365)).date()

    wm = await get_watermark(symbol, tf, "backfill")
    if wm and wm.get("last_ts"):
        try:
            last_ts = wm["last_ts"]
            # asyncpg returns datetime for timestamptz
            if isinstance(last_ts, datetime):
                resume_d = last_ts.astimezone(timezone.utc).date()
            else:
                resume_d = start_d
        except Exception:
            resume_d = start_d
    else:
        resume_d = start_d

    # Always clamp
    if resume_d < start_d:
        resume_d = start_d
    if resume_d > end_d:
        logger.info(f"[BACKFILL] {symbol} already up to date")
        return

    window_days = max(1, min(30, int(settings.backfill_window_days)))

    cur = resume_d
    total = 0
    while cur <= end_d:
        to_d = min(end_d, cur + timedelta(days=window_days))

        from_s = _to_yyyy_mm_dd(cur)
        to_s = _to_yyyy_mm_dd(to_d)

        logger.info(f"[BACKFILL] {symbol} 1m window {from_s} -> {to_s}")
        candles = await asyncio.to_thread(client.get_historical_1m_bulk, symbol, from_s, to_s)

        # Upsert is idempotent. Watermark kind stays 'backfill'.
        inserted = await upsert_candles_bulk(
            symbol=symbol,
            timeframe_seconds=tf,
            candles=candles,
            source="backfill",
            watermark_kind="backfill",
        )
        total += inserted

        # Persist progress even if API returns empty on holidays.
        await set_watermark(
            symbol=symbol,
            timeframe_seconds=tf,
            kind="backfill",
            last_ts=datetime(to_d.year, to_d.month, to_d.day, 23, 59, tzinfo=timezone.utc),
            details={"from": from_s, "to": to_s, "inserted": inserted},
        )

        # Advance to next window
        cur = to_d + timedelta(days=1)

    logger.info(f"[BACKFILL] {symbol} complete. Rows upserted: {total}")


async def run_backfill_all() -> None:
    # Hard constraint: Dhan bulk endpoints are often rate-limited. Keep concurrency conservative.
    max_c = max(1, min(4, int(settings.backfill_max_concurrency)))
    sem = asyncio.Semaphore(max_c)

    async def _run(sym: str):
        async with sem:
            await backfill_symbol_1m(sym)

    tasks = [asyncio.create_task(_run(sym)) for sym in settings.symbols]
    await asyncio.gather(*tasks)
