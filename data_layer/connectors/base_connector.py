"""
Base Connector

Абстрактный базовый класс для всех коннекторов данных.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, AsyncGenerator
import asyncio
import logging
from datetime import datetime


class BaseConnector(ABC):
    """Базовый класс для всех коннекторов"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Инициализация коннектора
        
        Args:
            config: Конфигурация коннектора
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self._connected = False
        self._rate_limiter = None
        
    @abstractmethod
    async def connect(self) -> bool:
        """
        Установление соединения с источником данных
        
        Returns:
            bool: True если соединение успешно
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Разрыв соединения"""
        pass
    
    @abstractmethod
    async def get_historical_data(
        self, 
        symbol: str, 
        timeframe: str, 
        start_time: datetime, 
        end_time: datetime
    ) -> Dict[str, Any]:
        """
        Получение исторических данных
        
        Args:
            symbol: Торговая пара
            timeframe: Таймфрейм
            start_time: Время начала
            end_time: Время окончания
            
        Returns:
            Dict: Исторические данные
        """
        pass
    
    @abstractmethod
    async def stream_data(
        self, 
        symbol: str, 
        data_type: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Потоковая передача данных
        
        Args:
            symbol: Торговая пара
            data_type: Тип данных (trades, orderbook, ohlcv)
            
        Yields:
            Dict: Потоковые данные
        """
        pass
    
    async def reconnect(self) -> bool:
        """
        Переподключение к источнику данных
        
        Returns:
            bool: True если переподключение успешно
        """
        self.logger.info("Attempting to reconnect...")
        await self.disconnect()
        
        # Экспоненциальная задержка
        delay = 1
        max_delay = 30
        
        while delay <= max_delay:
            try:
                success = await self.connect()
                if success:
                    self.logger.info("Reconnection successful")
                    return True
            except Exception as e:
                self.logger.warning(f"Reconnection attempt failed: {e}")
            
            await asyncio.sleep(delay)
            delay *= 2
        
        self.logger.error("Reconnection failed after maximum attempts")
        return False
    
    @property
    def is_connected(self) -> bool:
        """Проверка статуса соединения"""
        return self._connected
    
    def validate_symbol(self, symbol: str) -> bool:
        """
        Валидация символа
        
        Args:
            symbol: Торговая пара
            
        Returns:
            bool: True если символ валидный
        """
        # Базовая валидация - можно переопределить в дочерних классах
        return bool(symbol and '/' in symbol)
    
    def validate_timeframe(self, timeframe: str) -> bool:
        """
        Валидация таймфрейма
        
        Args:
            timeframe: Таймфрейм
            
        Returns:
            bool: True если таймфрейм валидный
        """
        valid_timeframes = ['1s', '5s', '1m', '5m', '15m', '1h', '4h', '1d']
        return timeframe in valid_timeframes
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Проверка здоровья коннектора
        
        Returns:
            Dict: Статус здоровья
        """
        return {
            'connector': self.__class__.__name__,
            'connected': self.is_connected,
            'timestamp': datetime.utcnow().isoformat(),
            'config_valid': bool(self.config)
        }
