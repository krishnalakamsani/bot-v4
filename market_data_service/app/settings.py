from pydantic import BaseModel
import os


class Settings(BaseModel):
    postgres_dsn: str = os.getenv(
        "POSTGRES_DSN", "postgresql://market:market@localhost:5432/marketdata"
    )

    enable_streaming: bool = os.getenv("ENABLE_STREAMING", "true").lower() in ("1", "true", "yes")

    # polling-based streaming (ticks -> candles)
    poll_seconds: float = float(os.getenv("POLL_SECONDS", "1"))
    candle_base_seconds: int = int(os.getenv("CANDLE_BASE_SECONDS", "5"))

    # symbols
    symbols: list[str] = [s.strip().upper() for s in os.getenv("SYMBOLS", "NIFTY,BANKNIFTY,FINNIFTY,SENSEX").split(",") if s.strip()]

    # Dhan credentials
    dhan_client_id: str = os.getenv("DHAN_CLIENT_ID", "")
    dhan_access_token: str = os.getenv("DHAN_ACCESS_TOKEN", "")

    # Optional: pull credentials from backend (frontend updates backend daily)
    fetch_creds_from_backend: bool = os.getenv("FETCH_CREDS_FROM_BACKEND", "true").lower() in (
        "1",
        "true",
        "yes",
    )
    backend_base_url: str = os.getenv("BACKEND_BASE_URL", "")  # e.g. http://backend:8001/api
    internal_api_secret: str = os.getenv("INTERNAL_API_SECRET", "")
    creds_refresh_seconds: float = float(os.getenv("CREDS_REFRESH_SECONDS", "30"))
    backend_sqlite_path: str = os.getenv("BACKEND_SQLITE_PATH", "")  # e.g. /shared/backend_data/trading.db

    # Dhan historical bulk candles endpoint (1-minute only)
    dhan_base_url: str = os.getenv("DHAN_BASE_URL", "")
    dhan_historical_url: str = os.getenv("DHAN_HISTORICAL_URL", "")
    # Dhan returns epoch timestamps in IST (naive-local epoch).
    # When true (default), persist IST-epoch as-is. Set false to normalize to real UTC (-5h30).
    dhan_epoch_is_ist: bool = os.getenv("DHAN_EPOCH_IS_IST", "true").lower() in ("1", "true", "yes")
    # Optional JSON dict of extra headers required by your Dhan endpoint
    # Example: {"access-token": "...", "client-id": "..."}
    dhan_historical_headers_json: str = os.getenv("DHAN_HISTORICAL_HEADERS_JSON", "")

    # Backfill controls
    backfill_years: int = int(os.getenv("BACKFILL_YEARS", "5"))
    backfill_window_days: int = int(os.getenv("BACKFILL_WINDOW_DAYS", "25"))
    backfill_max_concurrency: int = int(os.getenv("BACKFILL_MAX_CONCURRENCY", "1"))

    # operational
    pause_when_market_closed: bool = os.getenv("PAUSE_WHEN_CLOSED", "true").lower() in ("1", "true", "yes")


settings = Settings()
