import asyncpg
import logging

from .ingest.dhan_client import CandleRow

from .settings import settings

logger = logging.getLogger(__name__)

_pool: asyncpg.Pool | None = None


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(dsn=settings.postgres_dsn, min_size=1, max_size=10)
    return _pool


async def init_db() -> None:
    pool = await get_pool()
    async with pool.acquire() as conn:
        # Extensions
        await conn.execute("CREATE EXTENSION IF NOT EXISTS timescaledb")

        # Candle table (idempotent writes enforced by UNIQUE constraint)
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS candles (
              ts timestamptz NOT NULL,
              symbol text NOT NULL,
              timeframe_seconds int NOT NULL,
              open double precision NOT NULL,
              high double precision NOT NULL,
              low double precision NOT NULL,
              close double precision NOT NULL,
              volume double precision NULL,
              source text NOT NULL DEFAULT 'stream',
              ingested_at timestamptz NOT NULL DEFAULT now(),
              PRIMARY KEY (symbol, timeframe_seconds, ts)
            );
            """
        )

        # Watermarks (resume points, lag tracking)
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ingest_watermarks (
              symbol text NOT NULL,
              timeframe_seconds int NOT NULL,
              kind text NOT NULL, -- 'backfill' | 'stream'
              last_ts timestamptz NULL,
              updated_at timestamptz NOT NULL DEFAULT now(),
              details jsonb NULL,
              PRIMARY KEY (symbol, timeframe_seconds, kind)
            );
            """
        )

        # Convert to hypertable if not already
        await conn.execute(
            """
            SELECT create_hypertable('candles', 'ts', if_not_exists => TRUE);
            """
        )

        # Helpful indexes
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS candles_symbol_tf_ts_desc ON candles (symbol, timeframe_seconds, ts DESC)"
        )

        logger.info("[DB] Schema ready")


async def get_watermark(symbol: str, timeframe_seconds: int, kind: str) -> dict | None:
    pool = await get_pool()
    sym = str(symbol).strip().upper()
    tf = int(timeframe_seconds)
    k = str(kind)
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT symbol, timeframe_seconds, kind, last_ts, updated_at, details
            FROM ingest_watermarks
            WHERE symbol=$1 AND timeframe_seconds=$2 AND kind=$3
            """,
            sym,
            tf,
            k,
        )
    return dict(row) if row else None


async def set_watermark(symbol: str, timeframe_seconds: int, kind: str, last_ts, details: dict | None = None) -> None:
    pool = await get_pool()
    sym = str(symbol).strip().upper()
    tf = int(timeframe_seconds)
    k = str(kind)
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO ingest_watermarks(symbol, timeframe_seconds, kind, last_ts, details)
            VALUES ($1,$2,$3,$4,$5)
            ON CONFLICT(symbol, timeframe_seconds, kind)
            DO UPDATE SET last_ts=EXCLUDED.last_ts, updated_at=now(), details=EXCLUDED.details;
            """,
            sym,
            tf,
            k,
            last_ts,
            details,
        )


async def upsert_candles_bulk(
    symbol: str,
    timeframe_seconds: int,
    candles: list[CandleRow],
    source: str,
    watermark_kind: str,
) -> int:
    if not candles:
        return 0

    pool = await get_pool()
    sym = str(symbol).strip().upper()
    tf = int(timeframe_seconds)

    rows = [
        (
            c.ts,
            sym,
            tf,
            float(c.open),
            float(c.high),
            float(c.low),
            float(c.close),
            float(c.volume) if c.volume is not None else None,
            str(source),
        )
        for c in candles
    ]

    async with pool.acquire() as conn:
        await conn.executemany(
            """
            INSERT INTO candles (ts, symbol, timeframe_seconds, open, high, low, close, volume, source)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
            ON CONFLICT (symbol, timeframe_seconds, ts)
            DO UPDATE SET
              open=EXCLUDED.open,
              high=EXCLUDED.high,
              low=EXCLUDED.low,
              close=EXCLUDED.close,
              volume=EXCLUDED.volume,
              source=EXCLUDED.source,
              ingested_at=now();
            """,
            rows,
        )

        # Update watermark to newest candle
        last_ts = max(c.ts for c in candles)
        await conn.execute(
            """
            INSERT INTO ingest_watermarks(symbol, timeframe_seconds, kind, last_ts)
            VALUES ($1,$2,$3,$4)
            ON CONFLICT(symbol, timeframe_seconds, kind)
            DO UPDATE SET last_ts=EXCLUDED.last_ts, updated_at=now();
            """,
            sym,
            tf,
            str(watermark_kind),
            last_ts,
        )

    return len(rows)
