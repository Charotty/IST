from __future__ import annotations

import asyncio
import time


class RateLimiter:
    def __init__(self, max_calls: int, period_s: float) -> None:
        self._max_calls = max_calls
        self._period_s = period_s
        self._calls: list[float] = []
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.time()
            self._calls = [t for t in self._calls if now - t < self._period_s]

            if len(self._calls) >= self._max_calls:
                sleep_time = self._period_s - (now - self._calls[0])
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

            self._calls.append(time.time())
