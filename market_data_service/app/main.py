import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .settings import settings
from .db import init_db
from .api.routes import router as api_router
from .ingest.streaming import StreamingSupervisor

logger = logging.getLogger(__name__)

_streaming: StreamingSupervisor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _streaming

    await init_db()

    if settings.enable_streaming:
        _streaming = StreamingSupervisor()
        await _streaming.start()
        logger.info("[MDS] Streaming supervisor started")

    yield

    if _streaming:
        await _streaming.stop()
        _streaming = None
        logger.info("[MDS] Streaming supervisor stopped")


app = FastAPI(title="Market Data Service", version="1.0", lifespan=lifespan)
app.include_router(api_router, prefix="/v1")
