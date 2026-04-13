from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)


async def reconnect_with_backoff(
    *,
    disconnect_coro,
    connect_coro,
    max_retries: int = 5,
    base_delay_s: float = 1.0,
) -> bool:
    for attempt in range(max_retries):
        try:
            try:
                await disconnect_coro()
            except Exception:
                pass

            await asyncio.sleep(base_delay_s * (2**attempt))
            await connect_coro()
            logger.info("Reconnect succeeded (attempt %s)", attempt + 1)
            return True
        except Exception as e:
            logger.warning("Reconnect attempt %s failed: %s", attempt + 1, e)

    return False
