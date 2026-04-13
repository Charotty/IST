from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import List

from its_project.common.types import MarketData

logger = logging.getLogger(__name__)


async def batch_writer_task(
    *,
    name: str,
    storage,
    batch_size: int,
    max_interval_s: float,
    in_queue: asyncio.Queue[MarketData],
    stop_event: asyncio.Event,
) -> None:
    """Collect items from in_queue and write to storage in batches."""
    buffer: List[MarketData] = []
    last_flush = asyncio.get_event_loop().time()

    async def flush() -> None:
        if not buffer:
            return
        try:
            written = await storage.write_batch(buffer)
            logger.debug("%s batch write: %d records", name, written)
        except Exception:
            logger.exception("%s batch write error", name)
        finally:
            buffer.clear()
            nonlocal last_flush
            last_flush = asyncio.get_event_loop().time()

    while not stop_event.is_set():
        try:
            timeout = max(0.0, max_interval_s - (asyncio.get_event_loop().time() - last_flush))
            md = await asyncio.wait_for(in_queue.get(), timeout=timeout)
            buffer.append(md)
            if len(buffer) >= batch_size:
                await flush()
        except asyncio.TimeoutError:
            await flush()
        except Exception:
            logger.exception("%s writer task error", name)

    await flush()
    # Drain remaining items if any
    while True:
        try:
            md = in_queue.get_nowait()
            buffer.append(md)
        except asyncio.QueueEmpty:
            break
    await flush()
    logger.info("%s writer task stopped", name)
