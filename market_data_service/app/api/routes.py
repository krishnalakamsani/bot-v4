from fastapi import APIRouter, HTTPException, Query
from datetime import datetime

from ..db import get_pool
from ..settings import settings

router = APIRouter()


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/lag")
async def lag():
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT symbol, timeframe_seconds, kind, last_ts, updated_at
            FROM ingest_watermarks
            ORDER BY symbol, timeframe_seconds, kind
            """
        )
    return {
        "symbols": settings.symbols,
        "watermarks": [dict(r) for r in rows],
    }


@router.get("/candles/last")
async def candles_last(
    symbol: str = Query(...),
    timeframe_seconds: int = Query(..., ge=1),
    limit: int = Query(500, ge=1, le=20000),
):
    symbol = symbol.strip().upper()
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT ts, open, high, low, close, volume
            FROM candles
            WHERE symbol=$1 AND timeframe_seconds=$2
            ORDER BY ts DESC
            LIMIT $3
            """,
            symbol,
            int(timeframe_seconds),
            int(limit),
        )

    # Return ascending for indicator consumption
    out = [dict(r) for r in reversed(rows)]
    return {"symbol": symbol, "timeframe_seconds": int(timeframe_seconds), "candles": out}


@router.get("/candles/range")
async def candles_range(
    symbol: str = Query(...),
    timeframe_seconds: int = Query(..., ge=1),
    start: str = Query(..., description="ISO-8601 timestamptz"),
    end: str = Query(..., description="ISO-8601 timestamptz"),
    limit: int = Query(20000, ge=1, le=200000),
):
    symbol = symbol.strip().upper()
    try:
        start_dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
        end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid start/end timestamp")

    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT ts, open, high, low, close, volume
            FROM candles
            WHERE symbol=$1 AND timeframe_seconds=$2 AND ts >= $3 AND ts <= $4
            ORDER BY ts ASC
            LIMIT $5
            """,
            symbol,
            int(timeframe_seconds),
            start_dt,
            end_dt,
            int(limit),
        )

    return {"symbol": symbol, "timeframe_seconds": int(timeframe_seconds), "candles": [dict(r) for r in rows]}
