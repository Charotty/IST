from __future__ import annotations

import logging
from datetime import datetime
from typing import List, Optional

from its_project.storage.base import BaseStorage
from its_project.common.types import MarketData

logger = logging.getLogger(__name__)


class StorageReadAPI:
    """Simple read API that tries warm storage first, falls back to cold storage."""

    def __init__(self, warm: BaseStorage, cold: BaseStorage) -> None:
        self._warm = warm
        self._cold = cold

    async def read(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        data_type: Optional[str] = None,
        prefer_warm: bool = True,
    ) -> List[MarketData]:
        if prefer_warm:
            try:
                return await self._warm.read(symbol, start, end, data_type)
            except Exception:
                logger.warning("Warm storage read failed, falling back to cold")
        # fallback to cold
        try:
            return await self._cold.read(symbol, start, end, data_type)
        except Exception:
            logger.exception("Cold storage read failed")
            return []

    async def get_latest(
        self,
        symbol: str,
        limit: int = 1,
        prefer_warm: bool = True,
    ) -> List[MarketData]:
        if prefer_warm:
            try:
                return await self._warm.get_latest(symbol, limit)
            except Exception:
                logger.warning("Warm storage get_latest failed, falling back to cold")
        try:
            return await self._cold.get_latest(symbol, limit)
        except Exception:
            logger.exception("Cold storage get_latest failed")
            return []
