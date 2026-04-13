from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator

from its_project.common.types import MarketData


class BaseDataSource(ABC):
    def __init__(self) -> None:
        self._connected = False

    @abstractmethod
    async def connect(self) -> None: ...

    @abstractmethod
    async def disconnect(self) -> None: ...

    @abstractmethod
    async def subscribe(self, symbols: list[str] | None = None) -> AsyncIterator[MarketData]: ...

    @abstractmethod
    async def fetch(self, symbol: str, **params) -> MarketData: ...

    @abstractmethod
    async def is_alive(self) -> bool: ...
