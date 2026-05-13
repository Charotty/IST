"""
CCXT Connector для получения данных с криптовалютных бирж
"""
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, List
import ccxt
import pandas as pd

from .base_connector import BaseConnector
from utils.logger import LoggerConfig

logger = LoggerConfig.get_logger(__name__)


class CCXTConnector(BaseConnector):
    """Коннектор для работы с биржами через CCXT библиотеку"""
    
    def __init__(self, exchange_name: str = 'binance', config: Optional[Dict] = None):
        """
        Инициализация CCXT коннектора
        
        Args:
            exchange_name: Название биржи (binance, okx, kucoin и т.д.)
            config: Конфигурация для биржи (API ключи и т.д.)
        """
        # Объединяем конфигурацию с названием биржи
        full_config = config or {}
        full_config['exchange'] = exchange_name
        
        super().__init__(full_config)
        self.exchange_name = exchange_name
        self.exchange = None
        self._connected = False
        
    async def connect(self) -> bool:
        """Подключение к бирже"""
        try:
            # Создаем экземпляр биржи
            exchange_class = getattr(ccxt, self.exchange_name)
            self.exchange = exchange_class(self.config)
            
            # Проверяем подключение
            await asyncio.get_event_loop().run_in_executor(
                None, 
                self.exchange.load_markets
            )
            
            self._connected = True
            logger.info(f"Connected to {self.exchange_name} via CCXT")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to {self.exchange_name}: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Отключение от биржи"""
        try:
            if self.exchange:
                self.exchange.close()
                self._connected = False
                logger.info(f"Disconnected from {self.exchange_name}")
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")
    
    async def get_historical_data(
        self, 
        symbol: str, 
        timeframe: str, 
        start_time: datetime, 
        end_time: datetime,
        limit: int = 1000,
        since: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Получение исторических OHLCV данных с пагинацией
        
        Args:
            symbol: Торговая пара (например, BTC/USDT)
            timeframe: Таймфрейм (1m, 5m, 1h, 1d и т.д.)
            start_time: Время начала
            end_time: Время окончания
            limit: Лимит записей за один запрос
            since: Timestamp для пагинации (в миллисекундах)
            
        Returns:
            Dict: Исторические данные
        """
        try:
            if not self.exchange or not self._connected:
                raise Exception("Not connected to exchange")
            
            # Конвертируем символ в формат CCXT
            ccxt_symbol = self._convert_symbol(symbol)
            
            # Конвертируем таймфрейм
            ccxt_timeframe = self._convert_timeframe(timeframe)
            
            # Если since не указан, используем start_time
            if since is None:
                since = int(start_time.timestamp() * 1000)
            
            # Получаем данные
            loop = asyncio.get_event_loop()
            ohlcv = await loop.run_in_executor(
                None,
                lambda: self.exchange.fetch_ohlcv(
                    ccxt_symbol,
                    ccxt_timeframe,
                    since=since,
                    limit=limit
                )
            )
            
            # Конвертируем данные
            data = []
            for candle in ohlcv:
                candle_time = datetime.fromtimestamp(candle[0] / 1000)
                # Фильтруем по end_time
                if candle_time <= end_time:
                    data.append({
                        'timestamp': candle_time,
                        'open': float(candle[1]),
                        'high': float(candle[2]),
                        'low': float(candle[3]),
                        'close': float(candle[4]),
                        'volume': float(candle[5]),
                        'symbol': symbol
                    })
            
            return {
                'data': data,
                'symbol': symbol,
                'timeframe': timeframe,
                'start_time': start_time,
                'end_time': end_time,
                'source': f'ccxt_{self.exchange_name}'
            }
            
        except Exception as e:
            logger.error(f"Failed to get historical data: {e}")
            raise
    
    async def get_full_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime,
        limit: int = 1000
    ) -> Dict[str, Any]:
        """
        Получение полной исторической данных с автоматической пагинацией
        
        Args:
            symbol: Торговая пара
            timeframe: Таймфрейм
            start_time: Время начала
            end_time: Время окончания
            limit: Лимит записей за один запрос
            
        Returns:
            Dict: Все исторические данные
        """
        try:
            all_data = []
            since = int(start_time.timestamp() * 1000)
            end_timestamp = int(end_time.timestamp() * 1000)
            
            while True:
                # Получаем данные
                result = await self.get_historical_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_time=start_time,
                    end_time=end_time,
                    limit=limit,
                    since=since
                )
                
                if not result['data']:
                    logger.info("No more data available")
                    break
                
                all_data.extend(result['data'])
                
                # Обновляем since для следующего запроса
                last_timestamp = int(result['data'][-1]['timestamp'].timestamp() * 1000)
                
                # Проверяем, достигли ли конца
                if last_timestamp >= end_timestamp:
                    logger.info(f"Reached end time: {end_time}")
                    break
                
                since = last_timestamp + 1  # +1 чтобы избежать дубликатов
                
                logger.info(f"Loaded {len(result['data'])} candles, total: {len(all_data)}")
                
                # Задержка между запросами
                await asyncio.sleep(0.5)
                
                # Защита от бесконечного цикла
                if len(all_data) > 100000:
                    logger.warning("Reached maximum limit of 100,000 candles")
                    break
            
            # Удаляем дубликаты и сортируем
            unique_data = []
            seen_timestamps = set()
            for item in all_data:
                ts = item['timestamp']
                if ts not in seen_timestamps:
                    seen_timestamps.add(ts)
                    unique_data.append(item)
            
            unique_data.sort(key=lambda x: x['timestamp'])
            
            logger.info(f"Total unique candles loaded: {len(unique_data)}")
            
            return {
                'data': unique_data,
                'symbol': symbol,
                'timeframe': timeframe,
                'start_time': start_time,
                'end_time': end_time,
                'source': f'ccxt_{self.exchange_name}_full'
            }
            
        except Exception as e:
            logger.error(f"Failed to get full historical data: {e}")
            raise
    
    def _convert_symbol(self, symbol: str) -> str:
        """Конвертация символа в формат CCXT"""
        # BTC-USDT -> BTC/USDT
        if '-' in symbol:
            return symbol.replace('-', '/')
        return symbol
    
    def _convert_timeframe(self, timeframe: str) -> str:
        """Конвертация таймфрейма в формат CCXT"""
        # CCXT использует те же обозначения: 1m, 5m, 1h, 1d и т.д.
        return timeframe
    
    async def stream_data(self, symbol: str, data_type: str):
        """Потоковая передача данных (не реализовано для CCXT)"""
        raise NotImplementedError("Streaming not implemented for CCXT connector")
