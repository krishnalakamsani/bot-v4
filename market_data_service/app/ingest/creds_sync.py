import asyncio
import logging
from typing import Tuple

import sqlite3
from pathlib import Path

import httpx

from ..settings import settings

logger = logging.getLogger(__name__)


_last: Tuple[str, str] | None = None


async def refresh_dhan_credentials_from_backend() -> bool:
    """Fetch Dhan credentials from backend (secret-protected).

    Returns True if credentials changed.
    """
    global _last

    if not settings.fetch_creds_from_backend:
        return False

    # Option A (preferred): call backend internal endpoint with a shared secret.

    base = str(settings.backend_base_url or "").strip()
    secret = str(settings.internal_api_secret or "").strip()
    payload = None

    if base and secret:
        url = base.rstrip("/") + "/internal/dhan-credentials"
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(2.5, connect=2.0)) as client:
                resp = await client.get(url, headers={"X-Internal-Secret": secret})
                if resp.status_code == 200:
                    payload = resp.json() if resp.content else {}
        except Exception as e:
            logger.debug(f"[CREDS] http refresh failed: {e}")

    # Option B (fallback): read backend SQLite config table via shared volume.
    if payload is None:
        sqlite_path = str(getattr(settings, "backend_sqlite_path", "") or "").strip()
        if sqlite_path:
            try:
                payload = await asyncio.to_thread(_read_sqlite_creds, sqlite_path)
            except Exception as e:
                logger.debug(f"[CREDS] sqlite refresh failed: {e}")
                payload = None

    if not payload:
        return False

    client_id = str(payload.get("dhan_client_id") or "").strip()
    token = str(payload.get("dhan_access_token") or "").strip()

    if not client_id or not token:
        return False

    cur = (client_id, token)
    if _last == cur:
        return False

    _last = cur

    # Update settings in-process so streaming/backfill can pick it up.
    settings.dhan_client_id = client_id
    settings.dhan_access_token = token

    logger.info("[CREDS] Loaded Dhan credentials from backend")
    return True


def _read_sqlite_creds(db_path: str) -> dict:
    path = Path(db_path)
    if not path.exists() or not path.is_file():
        return {}

    out: dict[str, str] = {}
    con = sqlite3.connect(str(path))
    try:
        cur = con.execute("SELECT key, value FROM config WHERE key IN ('dhan_client_id', 'dhan_access_token')")
        for k, v in cur.fetchall():
            out[str(k)] = str(v)
    finally:
        try:
            con.close()
        except Exception:
            pass
    return out


async def creds_refresh_loop(stop_event: asyncio.Event):
    """Background refresher to keep Dhan token updated (daily)."""
    while not stop_event.is_set():
        try:
            await refresh_dhan_credentials_from_backend()
        except Exception:
            pass

        delay = float(getattr(settings, "creds_refresh_seconds", 30.0) or 30.0)
        delay = max(5.0, min(300.0, delay))
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=delay)
        except asyncio.TimeoutError:
            continue
