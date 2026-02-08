import asyncio
import logging
from datetime import datetime, timezone

from tenacity import retry, wait_exponential, stop_after_attempt

from ..settings import settings
from ..db import get_pool, upsert_candles_bulk
from .dhan_client import CandleRow
from .dhan_client import DhanClient
from .creds_sync import creds_refresh_loop, refresh_dhan_credentials_from_backend

logger = logging.getLogger(__name__)


def _floor_ts(ts: datetime, step_seconds: int) -> datetime:
    epoch = int(ts.timestamp())
    floored = epoch - (epoch % step_seconds)
    return datetime.fromtimestamp(floored, tz=timezone.utc)


class StreamingSupervisor:
    """Polling-based streaming: quotes -> base timeframe candles -> DB.

    Opinionated design choice:
    - build and persist *base* timeframe (default 5s)
    - derive 15s/30s/60s via DB continuous aggregates (recommended)

    This avoids multi-timeframe duplication and prevents overlap issues.
    """

    def __init__(self):
        self._running = False
        self._task: asyncio.Task | None = None
        self._dhan = DhanClient(settings.dhan_client_id, settings.dhan_access_token)
        self._cred_key: tuple[str, str] = (str(settings.dhan_client_id), str(settings.dhan_access_token))
        self._creds_stop = asyncio.Event()
        self._creds_task: asyncio.Task | None = None

        self._cur = {}  # (symbol) -> candle builder state

    async def start(self):
        if self._running:
            return
        self._running = True
        self._creds_stop.clear()
        self._creds_task = asyncio.create_task(creds_refresh_loop(self._creds_stop))
        self._task = asyncio.create_task(self._loop())

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None

        if self._creds_task:
            self._creds_stop.set()
            self._creds_task.cancel()
            self._creds_task = None

    async def _upsert_candle(self, symbol: str, timeframe_seconds: int, ts: datetime, o: float, h: float, l: float, c: float):
        await upsert_candles_bulk(
            symbol=symbol,
            timeframe_seconds=timeframe_seconds,
            candles=[CandleRow(ts=ts, open=o, high=h, low=l, close=c)],
            source="stream",
            watermark_kind="stream",
        )

    async def _loop(self):
        base_tf = int(settings.candle_base_seconds)
        poll = float(settings.poll_seconds)

        # Attempt a first refresh (in case env creds are not set but backend has them)
        try:
            await refresh_dhan_credentials_from_backend()
        except Exception:
            pass

        while self._running:
            try:
                # Rebuild Dhan client if credentials changed
                new_key = (str(settings.dhan_client_id), str(settings.dhan_access_token))
                if new_key != self._cred_key:
                    self._dhan = DhanClient(settings.dhan_client_id, settings.dhan_access_token)
                    self._cred_key = new_key

                if not self._dhan.ready():
                    await asyncio.sleep(max(1.0, poll))
                    continue

                now = datetime.now(timezone.utc)
                bucket = _floor_ts(now, base_tf)

                for symbol in settings.symbols:
                    ltp = None
                    try:
                        ltp = await asyncio.to_thread(self._dhan.get_index_ltp, symbol)
                    except Exception:
                        ltp = None

                    if ltp is None or ltp <= 0:
                        continue

                    st = self._cur.get(symbol)
                    if not st or st["bucket"] != bucket:
                        # flush previous
                        if st and st.get("count", 0) > 0:
                            await self._upsert_candle(
                                symbol=symbol,
                                timeframe_seconds=base_tf,
                                ts=st["bucket"],
                                o=st["open"],
                                h=st["high"],
                                l=st["low"],
                                c=st["close"],
                            )

                        self._cur[symbol] = {
                            "bucket": bucket,
                            "open": float(ltp),
                            "high": float(ltp),
                            "low": float(ltp),
                            "close": float(ltp),
                            "count": 1,
                        }
                    else:
                        st["high"] = max(st["high"], float(ltp))
                        st["low"] = min(st["low"], float(ltp))
                        st["close"] = float(ltp)
                        st["count"] += 1

                await asyncio.sleep(poll)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[STREAM] Error: {e}", exc_info=True)
                await asyncio.sleep(1)
