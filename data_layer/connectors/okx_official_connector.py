"""
OKX Official Connector

Коннектор для OKX API с использованием официальной библиотеки python-okx.
"""

import asyncio
from typing import Dict, Any, Optional, AsyncGenerator
from datetime import datetime, timedelta
import logging

from okx import MarketData, Account, Trade, PublicData

from .base_connector import BaseConnector


class OKXOfficialConnector(BaseConnector):
    """OKX API коннектор на базе официальной библиотеки python-okx"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.api_key = config.get('api_key')
        self.secret_key = config.get('secret_key')
        self.passphrase = config.get('passphrase')
        self.sandbox = config.get('sandbox', True)
        self.flag = "1" if self.sandbox else "0"  # 1 = demo, 0 = live
        self.base_url = config.get('base_url', 'https://www.okx.cab/api/v5')
        
        # Инициализация API клиентов
        self.market_api = None
        self.account_api = None
        self.trade_api = None
        self.public_api = None
        
        # WebSocket клиенты (пока не используем)
        self.public_ws = None
        self.private_ws = None
        
        # Mock режим для демонстрации
        self.mock_mode = False
        
    async def connect(self) -> bool:
        """Установление соединения с OKX"""
        try:
            # Инициализация API клиентов
            self.market_api = MarketData.MarketAPI(
                api_key=self.api_key,
                api_secret_key=self.secret_key,
                passphrase=self.passphrase,
                flag=self.flag,
                debug=False
            )
            
            self.account_api = Account.AccountAPI(
                api_key=self.api_key,
                api_secret_key=self.secret_key,
                passphrase=self.passphrase,
                flag=self.flag,
                debug=False
            )
            
            self.trade_api = Trade.TradeAPI(
                api_key=self.api_key,
                api_secret_key=self.secret_key,
                passphrase=self.passphrase,
                flag=self.flag,
                debug=False
            )
            
            self.public_api = PublicData.PublicAPI(
                debug=False
            )
            
            # Тест соединения
            try:
                # Проверка публичного API (синхронный вызов)
                import asyncio
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, lambda: self.market_api.get_tickers(instType="SPOT"))
                
                if result.get('code') == '0':
                    self.logger.info("Connected to OKX API successfully")
                    self._connected = True
                    return True
                else:
                    raise Exception(f"API error: {result}")
                    
            except Exception as e:
                self.logger.warning(f"Failed to connect to real OKX API: {e}")
                self.logger.info("Using mock mode for demonstration")
                self.mock_mode = True
                self._connected = True
                return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to OKX: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Разрыв соединения"""
        try:
            # Закрытие WebSocket соединений
            if self.public_ws:
                await self.public_ws.close()
            if self.private_ws:
                await self.private_ws.close()
            
            self._connected = False
            self.logger.info("Disconnected from OKX")
            
        except Exception as e:
            self.logger.error(f"Error during disconnect: {e}")
    
    async def get_historical_data(
        self, 
        symbol: str, 
        timeframe: str, 
        start_time: datetime, 
        end_time: datetime
    ) -> Dict[str, Any]:
        """Получение исторических OHLCV данных"""
        try:
            if self.mock_mode:
                return await self._get_mock_historical_data(symbol, timeframe, start_time, end_time)
            
            if not self.market_api or not self._connected:
                raise Exception("Not connected to exchange")
            
            # Конвертация таймфрейма
            okx_timeframe = self._convert_timeframe(timeframe)
            
            # Сначала пробуем исторические данные
            try:
                # Конвертация времени
                since = str(int(start_time.timestamp() * 1000))
                until = str(int(end_time.timestamp() * 1000))
                
                # Получение исторических данных (синхронный вызов)
                import asyncio
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, 
                    lambda: self.market_api.get_history_candlesticks(
                        instId=symbol,
                        bar=okx_timeframe,
                        after=since,
                        before=until,
                        limit=100
                    )
                )
                
                if result.get('code') == '0' and result.get('data'):
                    # Конвертация данных
                    data = []
                    for candle in result.get('data', []):
                        data.append({
                            'timestamp': datetime.fromtimestamp(int(candle[0]) / 1000),
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
                        'source': 'okx_official'
                    }
                    
            except Exception as e:
                self.logger.warning(f"Failed to get historical data, trying latest candles: {e}")
            
            # Fallback: получаем последние свечи как исторические данные
            self.logger.info(f"Using latest candles as historical data for {symbol}")
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                lambda: self.market_api.get_candlesticks(
                    instId=symbol,
                    bar=okx_timeframe,
                    limit=100
                )
            )
            
            if result.get('code') != '0':
                raise Exception(f"API error: {result}")
            
            # Конвертация данных
            data = []
            for candle in result.get('data', []):
                data.append({
                    'timestamp': datetime.fromtimestamp(int(candle[0]) / 1000),
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
                'source': 'okx_official_latest'
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get historical data: {e}")
            raise
    
    async def stream_data(
        self, 
        symbol: str, 
        data_type: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Потоковая передача данных"""
        try:
            if self.mock_mode:
                self.logger.info(f"Using mock streaming for {data_type} on {symbol}")
                async for data in self._stream_mock_data(symbol, data_type):
                    yield data
                return
            
            # Получаем реальные данные через REST API polling
            self.logger.info(f"Using real data streaming for {data_type} on {symbol}")
            
            import asyncio
            loop = asyncio.get_event_loop()
            
            while self._connected:
                try:
                    if data_type == 'trades':
                        # Получаем реальные сделки
                        result = await loop.run_in_executor(
                            None, 
                            lambda: self.market_api.get_trades(instId=symbol, limit=10)
                        )
                        
                        if result.get('code') == '0' and result.get('data'):
                            for trade in result['data']:
                                yield {
                                    'timestamp': datetime.fromtimestamp(int(trade['ts']) / 1000),
                                    'price': float(trade['px']),
                                    'volume': float(trade['sz']),
                                    'side': trade['side'],
                                    'symbol': symbol,
                                    'trade_id': trade['tradeId'],
                                    'data_type': 'trade'
                                }
                    
                    elif data_type == 'orderbook':
                        # Получаем реальный order book
                        result = await loop.run_in_executor(
                            None, 
                            lambda: self.market_api.get_orderbook(instId=symbol, sz=10)
                        )
                        
                        if result.get('code') == '0' and result.get('data'):
                            book = result['data'][0] if result['data'] else {}
                            yield {
                                'timestamp': datetime.fromtimestamp(int(book.get('ts', 0)) / 1000),
                                'bids': [[float(bid[0]), float(bid[1])] for bid in book.get('bids', [])],
                                'asks': [[float(ask[0]), float(ask[1])] for ask in book.get('asks', [])],
                                'symbol': symbol,
                                'data_type': 'orderbook'
                            }
                    
                    elif data_type == 'ohlcv':
                        # Получаем реальные свечи
                        result = await loop.run_in_executor(
                            None, 
                            lambda: self.market_api.get_candlesticks(instId=symbol, bar='1m', limit=1)
                        )
                        
                        if result.get('code') == '0' and result.get('data'):
                            candle = result['data'][0]
                            yield {
                                'timestamp': datetime.fromtimestamp(int(candle[0]) / 1000),
                                'open': float(candle[1]),
                                'high': float(candle[2]),
                                'low': float(candle[3]),
                                'close': float(candle[4]),
                                'volume': float(candle[5]),
                                'symbol': symbol,
                                'data_type': 'ohlcv'
                            }
                    
                    # Задержка между запросами
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    self.logger.error(f"Error in real streaming loop: {e}")
                    await asyncio.sleep(5)  # Backoff при ошибке
                    
        except Exception as e:
            self.logger.error(f"Error in stream_data: {e}")
            raise
    
    async def get_orderbook_snapshot(self, symbol: str, depth: int = 10) -> Dict[str, Any]:
        """
        Получение снэпшота Order Book
        
        Args:
            symbol: Торговая пара
            depth: Глубина Order Book
            
        Returns:
            Dict: Данные Order Book
        """
        try:
            if self.mock_mode:
                return await self._get_mock_orderbook(symbol, depth)
            
            if not self.market_api or not self._connected:
                raise Exception("Not connected to exchange")
            
            # Получение Order Book
            import asyncio
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                lambda: self.market_api.get_orderbook(instId=symbol, sz=depth)
            )
            
            if result.get('code') != '0':
                raise Exception(f"OKX API error: {result}")
            
            # Форматирование данных
            if result.get('data'):
                book = result['data'][0]
                return {
                    'timestamp': datetime.fromtimestamp(int(book.get('ts', 0)) / 1000),
                    'symbol': symbol,
                    'bids': [[float(bid[0]), float(bid[1])] for bid in book.get('bids', [])],
                    'asks': [[float(ask[0]), float(ask[1])] for ask in book.get('asks', [])],
                    'depth': depth,
                    'source': 'okx_official'
                }
            else:
                return {
                    'timestamp': datetime.utcnow(),
                    'symbol': symbol,
                    'bids': [],
                    'asks': [],
                    'depth': depth,
                    'source': 'okx_official'
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get orderbook snapshot: {e}")
            raise
    
    async def _get_mock_orderbook(self, symbol: str, depth: int) -> Dict[str, Any]:
        """Моковый Order Book для демонстрации"""
        import random
        
        base_price = 80000.0  # Базовая цена для BTC-USDT
        
        # Генерация bids и asks
        bids = []
        asks = []
        
        for i in range(depth):
            bid_price = base_price - (i * 0.1)
            bid_volume = random.uniform(0.1, 2.0)
            bids.append([bid_price, bid_volume])
            
            ask_price = base_price + (i * 0.1)
            ask_volume = random.uniform(0.1, 2.0)
            asks.append([ask_price, ask_volume])
        
        return {
            'timestamp': datetime.utcnow(),
            'symbol': symbol,
            'bids': bids,
            'asks': asks,
            'depth': depth,
            'source': 'mock'
        }
    
    def _convert_timeframe(self, timeframe: str) -> str:
        """Конвертация таймфрейма в формат OKX"""
        timeframe_map = {
            '1s': '1S',
            '5s': '5S',
            '1m': '1m',
            '3m': '3m',
            '5m': '5m',
            '15m': '15m',
            '30m': '30m',
            '1h': '1H',
            '2h': '2H',
            '4h': '4H',
            '6h': '6H',
            '12h': '12H',
            '1d': '1D',
            '1w': '1W',
            '1M': '1M'
        }
        return timeframe_map.get(timeframe, '1m')
    
    async def _get_mock_historical_data(
        self, 
        symbol: str, 
        timeframe: str, 
        start_time: datetime, 
        end_time: datetime
    ) -> Dict[str, Any]:
        """Получение mock исторических данных"""
        import random
        
        # Генерация mock данных
        data = []
        current_time = start_time
        base_price = 45000 if 'BTC' in symbol else 3000
        
        while current_time < end_time:
            price_change = random.uniform(-0.01, 0.01)
            open_price = base_price + random.uniform(-100, 100)
            close_price = open_price * (1 + price_change)
            high_price = max(open_price, close_price) * (1 + random.uniform(0, 0.005))
            low_price = min(open_price, close_price) * (1 - random.uniform(0, 0.005))
            volume = random.uniform(10, 100)
            
            data.append({
                'timestamp': current_time,
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2),
                'volume': round(volume, 2),
                'symbol': symbol
            })
            
            # Следующий временной интервал
            if timeframe == '1m':
                current_time += timedelta(minutes=1)
            elif timeframe == '5m':
                current_time += timedelta(minutes=5)
            elif timeframe == '1h':
                current_time += timedelta(hours=1)
            else:
                current_time += timedelta(minutes=1)
            
            base_price = close_price
        
        return {
            'data': data,
            'symbol': symbol,
            'timeframe': timeframe,
            'start_time': start_time,
            'end_time': end_time,
            'source': 'okx_mock'
        }
    
    async def _stream_mock_data(
        self, 
        symbol: str, 
        data_type: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Потоковая передача mock данных"""
        import random
        
        base_price = 45000 if 'BTC' in symbol else 3000
        
        while self._connected:
            try:
                if data_type == 'trades':
                    yield {
                        'timestamp': datetime.utcnow(),
                        'price': round(base_price + random.uniform(-100, 100), 2),
                        'volume': round(random.uniform(0.1, 10), 4),
                        'side': random.choice(['buy', 'sell']),
                        'symbol': symbol,
                        'trade_id': f"mock_{random.randint(1000000, 9999999)}",
                        'data_type': 'trade'
                    }
                
                elif data_type == 'orderbook':
                    bids = []
                    asks = []
                    for i in range(10):
                        bid_price = base_price - (i * 0.01)
                        ask_price = base_price + (i * 0.01)
                        bid_volume = random.uniform(0.1, 10)
                        ask_volume = random.uniform(0.1, 10)
                        bids.append([round(bid_price, 2), round(bid_volume, 4)])
                        asks.append([round(ask_price, 2), round(ask_volume, 4)])
                    
                    yield {
                        'timestamp': datetime.utcnow(),
                        'bids': bids,
                        'asks': asks,
                        'symbol': symbol,
                        'data_type': 'orderbook'
                    }
                
                elif data_type == 'ohlcv':
                    price_change = random.uniform(-0.01, 0.01)
                    open_price = base_price + random.uniform(-100, 100)
                    close_price = open_price * (1 + price_change)
                    high_price = max(open_price, close_price) * (1 + random.uniform(0, 0.005))
                    low_price = min(open_price, close_price) * (1 - random.uniform(0, 0.005))
                    volume = random.uniform(10, 100)
                    
                    yield {
                        'timestamp': datetime.utcnow(),
                        'open': round(open_price, 2),
                        'high': round(high_price, 2),
                        'low': round(low_price, 2),
                        'close': round(close_price, 2),
                        'volume': round(volume, 2),
                        'symbol': symbol,
                        'data_type': 'ohlcv'
                    }
                
                await asyncio.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error in mock streaming: {e}")
                await asyncio.sleep(5)
    
    def validate_symbol(self, symbol: str) -> bool:
        """Валидация символа"""
        # Mock валидация для популярных пар
        valid_symbols = ['BTC-USDT', 'ETH-USDT', 'BTC-ETH', 'ETH-BTC', 'BTC-USDC', 'ETH-USDC']
        return symbol in valid_symbols
    
    def validate_timeframe(self, timeframe: str) -> bool:
        """Валидация таймфрейма"""
        valid_timeframes = ['1s', '5s', '1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d', '1w', '1M']
        return timeframe in valid_timeframes
    
    async def get_exchange_info(self) -> Dict[str, Any]:
        """Получение информации об бирже"""
        try:
            if self.mock_mode:
                return {
                    'name': 'OKX (Mock)',
                    'id': 'okx',
                    'sandbox': self.sandbox,
                    'mock_mode': True,
                    'supported_timeframes': ['1m', '5m', '15m', '1h', '1d'],
                    'supported_symbols': ['BTC-USDT', 'ETH-USDT', 'BTC-ETH']
                }
            
            if not self.public_api:
                return {}
            
            # Получение информации об инструментах
            result = await self.public_api.get_instruments(instType="SPOT")
            
            return {
                'name': 'OKX',
                'id': 'okx',
                'sandbox': self.sandbox,
                'mock_mode': False,
                'instruments_count': len(result.get('data', [])),
                'supported_symbols': [inst['instId'] for inst in result.get('data', [])[:10]]
            }
            
        except Exception as e:
            self.logger.error(f"Error getting exchange info: {e}")
            return {}
