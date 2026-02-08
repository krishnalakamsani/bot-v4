import asyncio
import logging

from .db import init_db
from .ingest.backfill import run_backfill_all


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


async def _main():
    await init_db()
    await run_backfill_all()


if __name__ == "__main__":
    asyncio.run(_main())
