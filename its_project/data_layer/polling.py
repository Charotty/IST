from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

from its_project.common.types import MarketData

logger = logging.getLogger(__name__)


async def polling_task(
    *,
    name: str,
    interval_s: float,
    fetch: Callable[[], Awaitable[MarketData | None]],
    out_queue: asyncio.Queue[MarketData],
    stop_event: asyncio.Event,
) -> None:
    while not stop_event.is_set():
        try:
            md = await fetch()
            if md is not None:
                await out_queue.put(md)
        except Exception:
            logger.exception("Polling task error: %s", name)

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval_s)
        except TimeoutError:
            continue
