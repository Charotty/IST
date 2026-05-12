"""
REST Streamer

Потоковая передача данных через REST API с периодическими запросами.
"""

import asyncio
import aiohttp
from typing import Dict, Any, AsyncGenerator, Optional, Callable
from datetime import datetime, timedelta
import logging
import time


class RESTStreamer:
    """REST стример для периодических запросов"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Инициализация REST стримера
        
        Args:
            config: Конфигурация стримера
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.base_url = config.get('base_url')
        self.api_key = config.get('api_key')
        self.secret_key = config.get('secret_key')
        
        # Параметры стриминга
        self.poll_interval = config.get('poll_interval', 1.0)  # seconds
        self.batch_size = config.get('batch_size', 100)
        self.max_retries = config.get('max_retries', 3)
        
        # Состояние
        self._running = False
        self._session = None
        self._last_request_time = {}
        self._rate_limit = config.get('rate_limit', 10)  # requests per minute
        
        # Буфер для данных
        self.buffer_size = config.get('buffer_size', 1000)
        self._data_buffer = asyncio.Queue(maxsize=self.buffer_size)
        
        # Callback функции
        self.callbacks = {}
        
    async def connect(self) -> bool:
        """Установление соединения"""
        try:
            # Создание HTTP сессии
            self._session = aiohttp.ClientSession()
            
            # Тест соединения
            await self._test_connection()
            
            self.logger.info(f"Connected to REST API: {self.base_url}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to REST API: {e}")
            return False
    
    async def disconnect(self) -> None:
        """Разрыв соединения"""
        try:
            self._running = False
            
            if self._session:
                await self._session.close()
                self._session = None
            
            # Очистка буфера
            while not self._data_buffer.empty():
                try:
                    self._data_buffer.get_nowait()
                except asyncio.QueueEmpty:
                    break
            
            self.logger.info("Disconnected from REST API")
            
        except Exception as e:
            self.logger.error(f"Error during disconnect: {e}")
    
    async def stream_data(
        self,
        endpoint: str,
        params: Dict[str, Any],
        data_type: str = 'ohlcv'
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Потоковая передача данных через REST
        
        Args:
            endpoint: API эндпоинт
            params: Параметры запроса
            data_type: Тип данных
            
        Yields:
            Dict: Данные из API
        """
        try:
            self._running = True
            last_timestamp = None
            
            while self._running:
                try:
                    # Rate limiting
                    await self._rate_limit_check()
                    
                    # Получение данных
                    data = await self._fetch_data(endpoint, params, last_timestamp)
                    
                    if data and data.get('data'):
                        # Обработка данных
                        processed_data = await self._process_data(data, data_type)
                        
                        # Помещение в буфер
                        await self._data_buffer.put(processed_data)
                        
                        # Обновление временной метки
                        if data.get('timestamp'):
                            last_timestamp = data['timestamp']
                        
                        yield processed_data
                    
                    # Ожидание следующего запроса
                    await asyncio.sleep(self.poll_interval)
                    
                except Exception as e:
                    self.logger.error(f"Error in streaming loop: {e}")
                    await asyncio.sleep(self.poll_interval * 2)  # Backoff
                    
        except Exception as e:
            self.logger.error(f"Error in stream_data: {e}")
            raise
    
    async def get_latest_data(
        self,
        endpoint: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Получение последних данных
        
        Args:
            endpoint: API эндпоинт
            params: Параметры запроса
            
        Returns:
            Dict: Последние данные
        """
        try:
            await self._rate_limit_check()
            return await self._fetch_data(endpoint, params)
            
        except Exception as e:
            self.logger.error(f"Error getting latest data: {e}")
            raise
    
    def set_callback(self, data_type: str, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Установка callback функции
        
        Args:
            data_type: Тип данных
            callback: Функция обратного вызова
        """
        self.callbacks[data_type] = callback
    
    async def _test_connection(self) -> None:
        """Тест соединения"""
        if not self._session:
            raise Exception("Session not initialized")
        
        # Простой запрос для проверки соединения
        test_url = f"{self.base_url}/public/time"
        
        async with self._session.get(test_url) as response:
            if response.status != 200:
                raise Exception(f"Connection test failed: {response.status}")
    
    async def _rate_limit_check(self) -> None:
        """Проверка rate limiting"""
        current_time = time.time()
        
        # Очистка старых записей
        cutoff_time = current_time - 60  # 1 минута
        self._last_request_time = {
            k: v for k, v in self._last_request_time.items()
            if v > cutoff_time
        }
        
        # Проверка лимита
        recent_requests = len(self._last_request_time)
        if recent_requests >= self._rate_limit:
            # Расчет времени ожидания
            oldest_request = min(self._last_request_time.values()) if self._last_request_time else current_time
            wait_time = 60 - (current_time - oldest_request)
            
            if wait_time > 0:
                self.logger.debug(f"Rate limit reached, waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
        
        # Регистрация текущего запроса
        self._last_request_time[f"request_{len(self._last_request_time)}"] = current_time
    
    async def _fetch_data(
        self,
        endpoint: str,
        params: Dict[str, Any],
        since_timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Получение данных из API
        
        Args:
            endpoint: API эндпоинт
            params: Параметры запроса
            since_timestamp: Получение данных с указанного времени
            
        Returns:
            Dict: Данные из API
        """
        try:
            # Формирование URL
            url = f"{self.base_url}/{endpoint}"
            
            # Добавление временной метки если нужно
            if since_timestamp:
                params['since'] = int(since_timestamp.timestamp() * 1000)
            
            # Заголовки
            headers = {
                'Content-Type': 'application/json'
            }
            
            if self.api_key:
                headers['Authorization'] = f'Bearer {self.api_key}'
            
            # Выполнение запроса с retry
            for attempt in range(self.max_retries):
                try:
                    async with self._session.get(url, params=params, headers=headers) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            # Добавление метаданных
                            data['fetched_at'] = datetime.utcnow()
                            data['endpoint'] = endpoint
                            data['params'] = params
                            
                            return data
                        elif response.status == 429:  # Rate limit
                            retry_after = int(response.headers.get('Retry-After', 5))
                            self.logger.warning(f"Rate limited, waiting {retry_after}s")
                            await asyncio.sleep(retry_after)
                            continue
                        else:
                            error_text = await response.text()
                            raise Exception(f"HTTP {response.status}: {error_text}")
                            
                except aiohttp.ClientError as e:
                    if attempt == self.max_retries - 1:
                        raise
                    self.logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
            
            raise Exception(f"All {self.max_retries} attempts failed")
            
        except Exception as e:
            self.logger.error(f"Error fetching data: {e}")
            raise
    
    async def _process_data(
        self,
        raw_data: Dict[str, Any],
        data_type: str
    ) -> Dict[str, Any]:
        """
        Обработка полученных данных
        
        Args:
            raw_data: Сырые данные
            data_type: Тип данных
            
        Returns:
            Dict: Обработанные данные
        """
        try:
            processed_data = {
                'data_type': data_type,
                'raw_data': raw_data,
                'processed_at': datetime.utcnow(),
                'source': 'rest_api'
            }
            
            # Вызов callback если есть
            if data_type in self.callbacks:
                try:
                    self.callbacks[data_type](processed_data)
                except Exception as e:
                    self.logger.error(f"Error in callback for {data_type}: {e}")
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Error processing data: {e}")
            return raw_data
    
    async def get_buffer_data(self) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Получение данных из буфера
        
        Yields:
            Dict: Данные из буфера
        """
        try:
            while self._running:
                try:
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
            self.logger.error(f"Error getting buffer data: {e}")
            raise
    
    def get_status(self) -> Dict[str, Any]:
        """
        Получение статуса стримера
        
        Returns:
            Dict: Статус стримера
        """
        return {
            'connected': self._session is not None,
            'running': self._running,
            'base_url': self.base_url,
            'poll_interval': self.poll_interval,
            'buffer_size': self._data_buffer.qsize(),
            'rate_limit': self._rate_limit,
            'last_requests': len(self._last_request_time),
            'callbacks': list(self.callbacks.keys())
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
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        Обновление конфигурации
        
        Args:
            new_config: Новая конфигурация
        """
        self.config.update(new_config)
        
        # Обновление параметров
        if 'poll_interval' in new_config:
            self.poll_interval = new_config['poll_interval']
        if 'rate_limit' in new_config:
            self._rate_limit = new_config['rate_limit']
        if 'buffer_size' in new_config:
            self.buffer_size = new_config['buffer_size']
        
        self.logger.info(f"Configuration updated: {new_config}")
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Проверка здоровья стримера
        
        Returns:
            Dict: Статус здоровья
        """
        try:
            if not self._session:
                return {
                    'healthy': False,
                    'reason': 'No active session'
                }
            
            # Тестовый запрос
            start_time = time.time()
            await self._test_connection()
            response_time = time.time() - start_time
            
            return {
                'healthy': True,
                'response_time': response_time,
                'last_request': max(self._last_request_time.values()) if self._last_request_time else None,
                'buffer_utilization': self._data_buffer.qsize() / self.buffer_size,
                'active_callbacks': len(self.callbacks)
            }
            
        except Exception as e:
            return {
                'healthy': False,
                'reason': str(e),
                'error_time': datetime.utcnow().isoformat()
            }
