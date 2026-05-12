"""
WebSocket Streamer

Потоковая передача данных через WebSocket.
"""

import asyncio
import json
import websockets
from typing import Dict, Any, AsyncGenerator, Callable, Optional
from datetime import datetime
import logging
import time


class WebSocketStreamer:
    """WebSocket стример для потоковых данных"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Инициализация WebSocket стримера
        
        Args:
            config: Конфигурация стримера
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.url = config.get('url')
        self.subscriptions = []
        self.callbacks = {}
        self._connection = None
        self._running = False
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = config.get('max_reconnect_attempts', 10)
        self._reconnect_delay = config.get('reconnect_delay', 5)
        
        # Буфер для данных
        self.buffer_size = config.get('buffer_size', 10000)
        self._data_buffer = asyncio.Queue(maxsize=self.buffer_size)
        
    async def connect(self) -> bool:
        """Установление WebSocket соединения"""
        try:
            self.logger.info(f"Connecting to WebSocket: {self.url}")
            
            # Установление соединения
            self._connection = await websockets.connect(
                self.url,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=10
            )
            
            self._running = True
            self._reconnect_attempts = 0
            
            # Запуск задачи обработки сообщений
            asyncio.create_task(self._message_handler())
            asyncio.create_task(self._heartbeat())
            
            self.logger.info("WebSocket connection established")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to WebSocket: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Разрыв WebSocket соединения"""
        try:
            self._running = False
            
            if self._connection:
                await self._connection.close()
                self._connection = None
            
            # Очистка буфера
            while not self._data_buffer.empty():
                try:
                    self._data_buffer.get_nowait()
                except asyncio.QueueEmpty:
                    break
            
            self.logger.info("WebSocket connection closed")
            
        except Exception as e:
            self.logger.error(f"Error during disconnect: {e}")
    
    async def subscribe(self, channel: str, params: Dict[str, Any]) -> None:
        """
        Подписка на канал данных
        
        Args:
            channel: Канал подписки
            params: Параметры подписки
        """
        try:
            if not self._connection or self._connection.closed:
                raise ConnectionError("WebSocket not connected")
            
            # Формирование сообщения подписки
            subscribe_msg = {
                "op": "subscribe",
                "args": [{
                    "channel": channel,
                    **params
                }]
            }
            
            # Отправка сообщения
            await self._connection.send(json.dumps(subscribe_msg))
            
            # Сохранение подписки
            self.subscriptions.append({
                'channel': channel,
                'params': params,
                'timestamp': datetime.utcnow()
            })
            
            self.logger.info(f"Subscribed to channel: {channel}")
            
        except Exception as e:
            self.logger.error(f"Failed to subscribe to {channel}: {e}")
            raise
    
    async def unsubscribe(self, channel: str) -> None:
        """
        Отписка от канала
        
        Args:
            channel: Канал для отписки
        """
        try:
            if not self._connection or self._connection.closed:
                return
            
            # Формирование сообщения отписки
            unsubscribe_msg = {
                "op": "unsubscribe",
                "args": [{
                    "channel": channel
                }]
            }
            
            # Отправка сообщения
            await self._connection.send(json.dumps(unsubscribe_msg))
            
            # Удаление подписки
            self.subscriptions = [
                sub for sub in self.subscriptions 
                if sub['channel'] != channel
            ]
            
            self.logger.info(f"Unsubscribed from channel: {channel}")
            
        except Exception as e:
            self.logger.error(f"Failed to unsubscribe from {channel}: {e}")
    
    async def stream_data(self) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Потоковая передача данных
        
        Yields:
            Dict: Данные из WebSocket
        """
        try:
            while self._running and self._connection:
                try:
                    # Получение данных из буфера с таймаутом
                    data = await asyncio.wait_for(
                        self._data_buffer.get(), 
                        timeout=1.0
                    )
                    yield data
                    
                except asyncio.TimeoutError:
                    continue
                except asyncio.QueueEmpty:
                    continue
                    
        except Exception as e:
            self.logger.error(f"Error in stream_data: {e}")
            raise
    
    def set_callback(self, channel: str, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Установка callback функции для канала
        
        Args:
            channel: Канал
            callback: Функция обратного вызова
        """
        self.callbacks[channel] = callback
    
    async def _message_handler(self) -> None:
        """Обработчик входящих сообщений"""
        try:
            async for message in self._connection:
                if not self._running:
                    break
                
                try:
                    data = json.loads(message)
                    
                    # Обработка сообщения
                    await self._process_message(data)
                    
                except json.JSONDecodeError as e:
                    self.logger.error(f"Failed to decode JSON message: {e}")
                except Exception as e:
                    self.logger.error(f"Error processing message: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error in message handler: {e}")
    
    async def _process_message(self, data: Dict[str, Any]) -> None:
        """
        Обработка входящего сообщения
        
        Args:
            data: Данные сообщения
        """
        try:
            # Определение типа сообщения
            event = data.get('event', '')
            channel = data.get('arg', {}).get('channel', '')
            
            # Добавление временной метки
            data['received_at'] = datetime.utcnow().isoformat()
            
            # Помещение в буфер
            try:
                self._data_buffer.put_nowait(data)
            except asyncio.QueueFull:
                # Буфер полон - удаляем старые данные
                try:
                    self._data_buffer.get_nowait()
                    self._data_buffer.put_nowait(data)
                except asyncio.QueueEmpty:
                    pass
            
            # Вызов callback если есть
            if channel in self.callbacks:
                try:
                    self.callbacks[channel](data)
                except Exception as e:
                    self.logger.error(f"Error in callback for {channel}: {e}")
            
            # Логирование специальных событий
            if event == 'error':
                self.logger.error(f"WebSocket error: {data}")
            elif event == 'subscribe':
                self.logger.info(f"Subscription confirmed: {data}")
            elif event == 'unsubscribe':
                self.logger.info(f"Unsubscription confirmed: {data}")
                
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
    
    async def _heartbeat(self) -> None:
        """Проверка соединения (heartbeat)"""
        try:
            while self._running:
                await asyncio.sleep(30)  # 30 секунд
                
                if self._connection and not self._connection.closed:
                    # Проверка соединения ping/pong
                    try:
                        await self._connection.ping()
                    except Exception as e:
                        self.logger.warning(f"Heartbeat failed: {e}")
                        await self._reconnect()
                else:
                    self.logger.warning("Connection lost, attempting reconnect")
                    await self._reconnect()
                    
        except Exception as e:
            self.logger.error(f"Error in heartbeat: {e}")
    
    async def _reconnect(self) -> None:
        """Переподключение к WebSocket"""
        if self._reconnect_attempts >= self._max_reconnect_attempts:
            self.logger.error("Max reconnect attempts reached")
            self._running = False
            return
        
        self._reconnect_attempts += 1
        
        try:
            # Закрытие текущего соединения
            if self._connection:
                await self._connection.close()
            
            # Ожидание перед переподключением
            await asyncio.sleep(self._reconnect_delay * self._reconnect_attempts)
            
            # Попытка переподключения
            if await self.connect():
                # Восстановление подписок
                for subscription in self.subscriptions:
                    await self.subscribe(
                        subscription['channel'], 
                        subscription['params']
                    )
                
                self.logger.info("Reconnection successful")
            else:
                self.logger.warning(f"Reconnection attempt {self._reconnect_attempts} failed")
                
        except Exception as e:
            self.logger.error(f"Error during reconnect: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Получение статуса стримера
        
        Returns:
            Dict: Статус стримера
        """
        return {
            'connected': self._connection is not None and not self._connection.closed,
            'running': self._running,
            'url': self.url,
            'subscriptions': len(self.subscriptions),
            'buffer_size': self._data_buffer.qsize(),
            'reconnect_attempts': self._reconnect_attempts,
            'last_activity': datetime.utcnow().isoformat() if self._running else None
        }
    
    def get_buffer_stats(self) -> Dict[str, Any]:
        """
        Получение статистики буфера
        
        Returns:
            Dict: Статистика буфера
        """
        return {
            'current_size': self._data_buffer.qsize(),
            'max_size': self.buffer_size,
            'utilization': round(self._data_buffer.qsize() / self.buffer_size * 100, 2),
            'is_full': self._data_buffer.full(),
            'is_empty': self._data_buffer.empty()
        }
    
    async def clear_buffer(self) -> None:
        """Очистка буфера"""
        while not self._data_buffer.empty():
            try:
                self._data_buffer.get_nowait()
            except asyncio.QueueEmpty:
                break
        
        self.logger.info("Buffer cleared")
    
    def set_buffer_size(self, size: int) -> None:
        """
        Изменение размера буфера
        
        Args:
            size: Новый размер буфера
        """
        self.buffer_size = size
        
        # Создание нового буфера с нужным размером
        old_buffer = self._data_buffer
        self._data_buffer = asyncio.Queue(maxsize=size)
        
        # Перенос данных из старого буфера
        try:
            while not old_buffer.empty():
                data = old_buffer.get_nowait()
                self._data_buffer.put_nowait(data)
        except asyncio.QueueEmpty:
            pass
        
        self.logger.info(f"Buffer size changed to {size}")
