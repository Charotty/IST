from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional

from its_project.common.types import MarketData


class BaseStorage(ABC):
    @abstractmethod
    async def write(self, data: MarketData) -> bool:
        """Write a single record."""
        ...

    @abstractmethod
    async def write_batch(self, data: List[MarketData]) -> int:
        """Write a batch of records (preferred). Return number written."""
        ...

    @abstractmethod
    async def read(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        data_type: Optional[str] = None,
    ) -> List[MarketData]:
        """Read records for a symbol in a time window."""
        ...

    @abstractmethod
    async def get_latest(self, symbol: str, limit: int = 1) -> List[MarketData]:
        """Get the latest N records for a symbol."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Cleanup resources (connections, etc.)."""
        ...
