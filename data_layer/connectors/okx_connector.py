"""
OKX Connector

Коннектор для OKX API - REST и WebSocket.
"""

import asyncio
import json
import websockets
import aiohttp
from typing import Dict, Any, Optional, AsyncGenerator
from datetime import datetime, timedelta
import hmac
import hashlib
import base64
import time

from .base_connector import BaseConnector


class OKXConnector(BaseConnector):
    """OKX API коннектор"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.api_key = config.get('api_key')
        self.secret_key = config.get('secret_key')
        self.passphrase = config.get('passphrase')
        self.sandbox = config.get('sandbox', True)
        
        # Base URLs
        if self.sandbox:
            self.rest_url = "https://www.okx.cab/api/v5"
            self.ws_url = "wss://wspap.okx.com:8443/ws/v5/public"
        else:
            self.rest_url = "https://www.okx.cab/api/v5"
            self.ws_url = "wss://ws.okx.com:8443/ws/v5/public"
        
        self._session = None
        self._ws_connection = None
        
    async def connect(self) -> bool:
        """Установление соединения с OKX"""
        try:
            # Создание HTTP сессии
            self._session = aiohttp.ClientSession()
            
            # Проверка соединения через API
            await self._test_connection()
            
            self._connected = True
            self.logger.info("Connected to OKX API")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to OKX: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Разрыв соединения"""
        try:
            if self._ws_connection:
                await self._ws_connection.close()
                self._ws_connection = None
            
            if self._session:
                await self._session.close()
                self._session = None
            
            self._connected = False
            self.logger.info("Disconnected from OKX")
            
        except Exception as e:
            self.logger.error(f"Error during disconnect: {e}")
    
    async def get_historical_data(
        self, 
        symbol: str, 
        timeframe: str, 
        start_time: datetime, 
        end_time: datetime,
        limit: int = 300
    ) -> Dict[str, Any]:
        """Получение исторических OHLCV данных"""
        try:
            # Конвертация таймфрейма
            okx_timeframe = self._convert_timeframe(timeframe)
            
            # Формирование параметров запроса
            params = {
                'instId': symbol,
                'bar': okx_timeframe,
                'before': str(int(start_time.timestamp() * 1000)),
                'after': str(int(end_time.timestamp() * 1000)),
                'limit': str(limit)  # OKX поддерживает до 300 свечей за запрос
            }
            
            # Запрос данных
            url = f"{self.rest_url}/market/candles"
            async with self._session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get('code') == '0':
                        candles = data.get('data', [])
                        return self._format_ohlcv_data(candles)
                    else:
                        raise Exception(f"OKX API error: {data.get('msg')}")
                else:
                    raise Exception(f"HTTP error: {response.status}")
                    
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
            # Подключение к WebSocket
            self._ws_connection = await websockets.connect(self.ws_url)
            
            # Подписка на данные
            if data_type == 'trades':
                await self._subscribe_trades(symbol)
            elif data_type == 'orderbook':
                await self._subscribe_orderbook(symbol)
            elif data_type == 'ohlcv':
                await self._subscribe_ohlcv(symbol)
            
            # Обработка потока данных
            async for message in self._ws_connection:
                data = json.loads(message)
                
                if data.get('event') == 'trade':
                    yield self._format_trade_data(data)
                elif data.get('event') == 'books':
                    yield self._format_orderbook_data(data)
                elif data.get('event') == 'candle1m':
                    yield self._format_ohlcv_stream_data(data)
                    
        except Exception as e:
            self.logger.error(f"WebSocket streaming error: {e}")
            raise
    
    async def _test_connection(self) -> None:
        """Тестирование соединения"""
        url = f"{self.rest_url}/public/time"
        async with self._session.get(url) as response:
            if response.status != 200:
                raise Exception("Connection test failed")
    
    def _convert_timeframe(self, timeframe: str) -> str:
        """Конвертация таймфрейма в формат OKX"""
        mapping = {
            '1s': '1S',
            '5s': '5S',
            '1m': '1m',
            '5m': '5m',
            '15m': '15m',
            '1h': '1H',
            '4h': '4H',
            '1d': '1D'
        }
        return mapping.get(timeframe, '1m')
    
    def _format_ohlcv_data(self, candles: list) -> Dict[str, Any]:
        """Форматирование OHLCV данных"""
        if not candles:
            return {'data': []}
        
        formatted_data = []
        for candle in candles:
            # OKX формат: [timestamp, open, high, low, close, volume, volume_ccy, volume_ccy_quote, confirm]
            formatted_candle = {
                'timestamp': datetime.fromtimestamp(int(candle[0]) / 1000),
                'open': float(candle[1]),
                'high': float(candle[2]),
                'low': float(candle[3]),
                'close': float(candle[4]),
                'volume': float(candle[5]),
                'symbol': self.config.get('symbol', '')
            }
            formatted_data.append(formatted_candle)
        
        return {'data': formatted_data}
    
    def _format_trade_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Форматирование данных сделок"""
        trade_data = data.get('data', [{}])[0]
        
        return {
            'timestamp': datetime.fromtimestamp(int(trade_data.get('ts', 0)) / 1000),
            'price': float(trade_data.get('px', 0)),
            'volume': float(trade_data.get('sz', 0)),
            'side': trade_data.get('side', ''),
            'symbol': trade_data.get('instId', ''),
            'trade_id': trade_data.get('tradeId', ''),
            'data_type': 'trade'
        }
    
    def _format_orderbook_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Форматирование данных order book"""
        book_data = data.get('data', [{}])[0]
        
        return {
            'timestamp': datetime.fromtimestamp(int(book_data.get('ts', 0)) / 1000),
            'bids': [[float(bid[0]), float(bid[1])] for bid in book_data.get('bids', [])],
            'asks': [[float(ask[0]), float(ask[1])] for ask in book_data.get('asks', [])],
            'symbol': book_data.get('instId', ''),
            'data_type': 'orderbook'
        }
    
    def _format_ohlcv_stream_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Форматирование потоковых OHLCV данных"""
        candle_data = data.get('data', [{}])[0]
        
        return {
            'timestamp': datetime.fromtimestamp(int(candle_data.get('ts', 0)) / 1000),
            'open': float(candle_data.get('o', 0)),
            'high': float(candle_data.get('h', 0)),
            'low': float(candle_data.get('l', 0)),
            'close': float(candle_data.get('c', 0)),
            'volume': float(candle_data.get('vol', 0)),
            'symbol': candle_data.get('instId', ''),
            'data_type': 'ohlcv'
        }
    
    async def _subscribe_trades(self, symbol: str) -> None:
        """Подписка на сделки"""
        subscribe_msg = {
            "op": "subscribe",
            "args": [{
                "channel": "trades",
                "instId": symbol
            }]
        }
        await self._ws_connection.send(json.dumps(subscribe_msg))
    
    async def _subscribe_orderbook(self, symbol: str) -> None:
        """Подписка на order book"""
        subscribe_msg = {
            "op": "subscribe",
            "args": [{
                "channel": "books",
                "instId": symbol
            }]
        }
        await self._ws_connection.send(json.dumps(subscribe_msg))
    
    async def _subscribe_ohlcv(self, symbol: str) -> None:
        """Подписка на OHLCV"""
        subscribe_msg = {
            "op": "subscribe",
            "args": [{
                "channel": "candle1m",
                "instId": symbol
            }]
        }
        await self._ws_connection.send(json.dumps(subscribe_msg))
    
    def _generate_signature(self, timestamp: str, method: str, request_path: str, body: str) -> str:
        """Генерация подписи для аутентификации"""
        message = timestamp + method + request_path + body
        signature = hmac.new(
            self.secret_key.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).digest()
        
        return base64.b64encode(signature).decode('utf-8')
    
    def validate_symbol(self, symbol: str) -> bool:
        """Валидация символа OKX"""
        # OKX использует формат BTC-USDT
        if not symbol:
            return False
        
        # Базовая проверка формата
        return '-' in symbol and len(symbol.split('-')) == 2
