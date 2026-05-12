"""
Data Manager

Главный менеджер данных для координации всех коннекторов и обработки данных.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, AsyncGenerator
from datetime import datetime, timedelta
from dataclasses import dataclass

from .connectors.base_connector import BaseConnector
from .connectors.okx_official_connector import OKXOfficialConnector
from .processors.ohlcv_processor import OHLCVProcessor
from .storage.parquet_storage import ParquetStorage
from .streamers.websocket_streamer import WebSocketStreamer

# Импорт Synchronization Layer
try:
    from ..synchronization.sync_manager import SyncManager
except ImportError:
    # Fallback для случаев, когда Synchronization Layer не в той же директории
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from synchronization.sync_manager import SyncManager


@dataclass
class DataSource:
    """Класс для описания источника данных"""
    name: str
    connector: BaseConnector
    data_types: List[str]
    priority: int = 1
    enabled: bool = True


class DataManager:
    """Главный менеджер данных"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Инициализация менеджера данных
        
        Args:
            config: Конфигурация системы
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Инициализация компонентов
        self._connectors = {}
        self._processors = {}
        self._storage = None
        self._streamers = {}
        self._sync_manager = None
        
        # Состояние системы
        self._running = False
        self._data_sources = []
        self._active_subscriptions = {}
        
        # Метрики
        self._metrics = {
            'messages_processed': 0,
            'errors_count': 0,
            'last_update': None,
            'connected_sources': 0
        }
        
        # Инициализация
        self._initialize_components()
    
    async def start(self) -> None:
        """Запуск менеджера данных"""
        try:
            self.logger.info("Starting Data Manager...")
            
            # Подключение всех коннекторов
            await self._connect_all_sources()
            
            # Запуск Synchronization Manager
            if self._sync_manager:
                await self._sync_manager.start()
                
                # Добавление источников данных в синхронизацию
                await self._setup_synchronization_sources()
            
            # Инициализация хранилища
            await self._initialize_storage()
            
            # Запуск потоковой обработки
            await self._start_streaming()
            
            self._running = True
            self.logger.info("Data Manager started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start Data Manager: {e}")
            raise
    
    async def stop(self) -> None:
        """Остановка менеджера данных"""
        try:
            self.logger.info("Stopping Data Manager...")
            
            self._running = False
            
            # Отписка от всех потоков
            await self._unsubscribe_all()
            
            # Остановка Synchronization Manager
            if self._sync_manager:
                await self._sync_manager.stop()
            
            # Отключение всех коннекторов
            await self._disconnect_all_sources()
            
            self.logger.info("Data Manager stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping Data Manager: {e}")
    
    async def get_historical_data(
        self,
        symbol: str,
        data_type: str,
        timeframe: str,
        start_time: datetime,
        end_time: datetime,
        source: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Получение исторических данных
        
        Args:
            symbol: Торговая пара
            data_type: Тип данных (ohlcv, trades, onchain)
            timeframe: Таймфрейм
            start_time: Время начала
            end_time: Время окончания
            source: Источник данных (если None, выбирается автоматически)
            
        Returns:
            Dict: Исторические данные
        """
        try:
            # Выбор источника данных
            connector = self._get_best_source(data_type, source)
            
            if not connector:
                raise ValueError(f"No available source for data type: {data_type}")
            
            # Получение данных
            data = await connector.get_historical_data(
                symbol, timeframe, start_time, end_time
            )
            
            # Обработка данных
            processed_data = await self._process_data(data, data_type)
            
            # Сохранение в хранилище
            if self._storage:
                await self._save_data(processed_data, symbol, data_type, timeframe)
            
            # Обновление метрик
            self._update_metrics('historical_request')
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Error getting historical data: {e}")
            self._update_metrics('error')
            raise
    
    async def stream_data(
        self,
        symbol: str,
        data_type: str,
        source: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Потоковая передача данных
        
        Args:
            symbol: Торговая пара
            data_type: Тип данных
            source: Источник данных
            
        Yields:
            Dict: Потоковые данные
        """
        try:
            # Выбор источника данных
            connector = self._get_best_source(data_type, source)
            
            if not connector:
                raise ValueError(f"No available source for data type: {data_type}")
            
            # Проверка подписки
            subscription_key = f"{symbol}_{data_type}"
            if subscription_key in self._active_subscriptions:
                self.logger.warning(f"Already subscribed to {subscription_key}")
                return
            
            # Потоковая передача данных
            async for data in connector.stream_data(symbol, data_type):
                # Синхронизация данных (если включена)
                if self._sync_manager:
                    # Используем имя коннектора из конфигурации
                    connector_name = 'okx'  # или можно получить из data_sources
                    source_name = f"{connector_name}_{data_type}"
                    synchronized_data = await self._sync_manager.synchronize_data(data, source_name)
                    if synchronized_data:
                        processed_data = await self._process_data(synchronized_data, data_type)
                    else:
                        processed_data = await self._process_data(data, data_type)
                else:
                    processed_data = await self._process_data(data, data_type)
                
                # Сохранение в хранилище (асинхронно)
                if self._storage:
                    asyncio.create_task(
                        self._save_stream_data(processed_data, symbol, data_type)
                    )
                
                # Обновление метрик
                self._update_metrics('stream_message')
                
                yield processed_data
            
            # Регистрация подписки
            self._active_subscriptions[subscription_key] = {
                'symbol': symbol,
                'data_type': data_type,
                'source': connector.__class__.__name__,
                'started_at': datetime.utcnow()
            }
            
        except Exception as e:
            self.logger.error(f"Error in stream_data: {e}")
            self._update_metrics('error')
            raise
    
    async def get_available_symbols(self, data_type: str = 'ohlcv') -> List[str]:
        """
        Получение списка доступных символов
        
        Args:
            data_type: Тип данных
            
        Returns:
            List[str]: Список символов
        """
        try:
            symbols = set()
            
            # Получение символов от всех источников
            for source in self._data_sources:
                if source.enabled and data_type in source.data_types:
                    # Для OHLCV и trades используем коннекторы бирж
                    if data_type in ['ohlcv', 'trades']:
                        # Здесь можно добавить логику получения символов
                        # из конфигурации или API
                        pass
                    # Для on-chain используем Glassnode
                    elif data_type == 'onchain':
                        if hasattr(source.connector, 'validate_symbol'):
                            # Добавляем базовые активы
                            base_assets = ['BTC', 'ETH', 'LTC']
                            symbols.update(base_assets)
            
            return sorted(list(symbols))
            
        except Exception as e:
            self.logger.error(f"Error getting available symbols: {e}")
            return []
    
    async def get_data_status(self) -> Dict[str, Any]:
        """
        Получение статуса данных
        
        Returns:
            Dict: Статус системы данных
        """
        try:
            status = {
                'manager': {
                    'running': self._running,
                    'total_sources': len(self._data_sources),
                    'active_sources': self._metrics['connected_sources'],
                    'active_subscriptions': len(self._active_subscriptions)
                },
                'sources': [],
                'metrics': self._metrics.copy(),
                'storage': {
                    'available': self._storage is not None,
                    'type': 'parquet' if self._storage else None
                }
            }
            
            # Статус каждого источника
            for source in self._data_sources:
                source_status = {
                    'name': source.name,
                    'enabled': source.enabled,
                    'connected': source.connector.is_connected if source.connector else False,
                    'data_types': source.data_types,
                    'priority': source.priority
                }
                status['sources'].append(source_status)
            
            return status
            
        except Exception as e:
            self.logger.error(f"Error getting data status: {e}")
            return {'error': str(e)}
    
    def _initialize_components(self) -> None:
        """Инициализация компонентов"""
        try:
            # Инициализация коннекторов
            self._initialize_connectors()
            
            # Инициализация процессоров
            self._initialize_processors()
            
            # Инициализация Synchronization Manager
            self._initialize_synchronization()
            
            # Инициализация хранилища
            storage_config = self.config.get('data_layer', {}).get('storage', {})
            if storage_config:
                self._storage = ParquetStorage(storage_config)
            
        except Exception as e:
            self.logger.error(f"Error initializing components: {e}")
    
    async def _setup_synchronization_sources(self) -> None:
        """Настройка источников данных для синхронизации"""
        try:
            # Добавление источников данных в SyncManager
            for source in self._data_sources:
                if source.enabled:
                    for data_type in source.data_types:
                        source_name = f"{source.name}_{data_type}"
                        await self._sync_manager.add_data_source(
                            source_name, data_type, data_type
                        )
                        self.logger.info(f"Added sync source: {source_name}")
            
        except Exception as e:
            self.logger.error(f"Error setting up synchronization sources: {e}")
    
    def _initialize_synchronization(self) -> None:
        """Инициализация Synchronization Manager"""
        try:
            sync_config = self.config.get('synchronization', {})
            if sync_config:
                self._sync_manager = SyncManager(sync_config)
                self.logger.info("Synchronization Manager initialized")
            else:
                self.logger.info("Synchronization disabled in config")
        except Exception as e:
            self.logger.error(f"Error initializing synchronization: {e}")
            self._sync_manager = None
    
    def _initialize_connectors(self) -> None:
        """Инициализация коннекторов"""
        connectors_config = self.config.get('data_layer', {}).get('connectors', {})
        
        # OKX Official коннектор
        if 'okx' in connectors_config:
            okx_connector = OKXOfficialConnector(connectors_config['okx'])
            self._connectors['okx'] = okx_connector
            
            # Добавление в источники данных
            self._data_sources.append(DataSource(
                name='okx',
                connector=okx_connector,
                data_types=['ohlcv', 'trades', 'orderbook'],
                priority=1,
                enabled=True
            ))
        
        # Сортировка по приоритету
        self._data_sources.sort(key=lambda x: x.priority)
    
    def _initialize_processors(self) -> None:
        """Инициализация процессоров"""
        processor_config = self.config.get('data_layer', {})
        
        # OHLCV процессор
        self._processors['ohlcv'] = OHLCVProcessor(processor_config)
    
    async def _connect_all_sources(self) -> None:
        """Подключение всех источников данных"""
        for source in self._data_sources:
            if source.enabled:
                try:
                    success = await source.connector.connect()
                    if success:
                        self._metrics['connected_sources'] += 1
                        self.logger.info(f"Connected to source: {source.name}")
                    else:
                        self.logger.warning(f"Failed to connect to source: {source.name}")
                except Exception as e:
                    self.logger.error(f"Error connecting to {source.name}: {e}")
    
    async def _disconnect_all_sources(self) -> None:
        """Отключение всех источников данных"""
        for source in self._data_sources:
            if source.connector:
                try:
                    await source.connector.disconnect()
                    self.logger.info(f"Disconnected from source: {source.name}")
                except Exception as e:
                    self.logger.error(f"Error disconnecting from {source.name}: {e}")
    
    async def _initialize_storage(self) -> None:
        """Инициализация хранилища"""
        # Хранилище уже инициализировано в _initialize_components
        pass
    
    async def _start_streaming(self) -> None:
        """Запуск потоковой обработки"""
        # Здесь можно добавить логику автоматической подписки
        # на определенные потоки данных
        pass
    
    def _get_best_source(self, data_type: str, preferred_source: Optional[str] = None) -> Optional[BaseConnector]:
        """
        Выбор лучшего источника данных
        
        Args:
            data_type: Тип данных
            preferred_source: Предпочитаемый источник
            
        Returns:
            BaseConnector: Коннектор
        """
        # Если указан предпочтительный источник
        if preferred_source and preferred_source in self._connectors:
            connector = self._connectors[preferred_source]
            if connector.is_connected:
                return connector
        
        # Выбор по приоритету
        for source in self._data_sources:
            if (source.enabled and 
                data_type in source.data_types and 
                source.connector.is_connected):
                return source.connector
        
        return None
    
    async def _process_data(self, data: Dict[str, Any], data_type: str) -> Dict[str, Any]:
        """
        Обработка данных
        
        Args:
            data: Сырые данные
            data_type: Тип данных
            
        Returns:
            Dict: Обработанные данные
        """
        try:
            if data_type == 'ohlcv' and 'ohlcv' in self._processors:
                raw_data = data.get('data', [])
                processed_df = self._processors['ohlcv'].process_raw_data(raw_data)
                
                return {
                    'data_type': 'ohlcv',
                    'data': processed_df,
                    'processed_at': datetime.utcnow(),
                    'source': data.get('source', 'unknown')
                }
            
            # Для других типов данных возвращаем как есть
            return data
            
        except Exception as e:
            self.logger.error(f"Error processing {data_type} data: {e}")
            return data
    
    async def _save_data(
        self, 
        data: Dict[str, Any], 
        symbol: str, 
        data_type: str, 
        timeframe: str = '1m'
    ) -> None:
        """
        Сохранение данных
        
        Args:
            data: Данные для сохранения
            symbol: Символ
            data_type: Тип данных
            timeframe: Таймфрейм
        """
        try:
            if not self._storage:
                return
            
            if data_type == 'ohlcv':
                df = data.get('data')
                if df is not None and not df.empty:
                    await self._storage.save_ohlcv_data(df, symbol, timeframe)
            
        except Exception as e:
            self.logger.error(f"Error saving data: {e}")
    
    async def _save_stream_data(
        self, 
        data: Dict[str, Any], 
        symbol: str, 
        data_type: str
    ) -> None:
        """
        Сохранение потоковых данных
        
        Args:
            data: Данные для сохранения
            symbol: Символ
            data_type: Тип данных
        """
        try:
            if not self._storage:
                return
            
            # Здесь можно добавить логику для сохранения потоковых данных
            # в реальном времени или батчами
            pass
            
        except Exception as e:
            self.logger.error(f"Error saving stream data: {e}")
    
    async def _unsubscribe_all(self) -> None:
        """Отписка от всех потоков"""
        for subscription_key in list(self._active_subscriptions.keys()):
            try:
                symbol, data_type = subscription_key.split('_')
                connector = self._get_best_source(data_type)
                
                if connector and hasattr(connector, 'stream_data'):
                    # Здесь можно добавить логику отписки
                    pass
                
                del self._active_subscriptions[subscription_key]
                
            except Exception as e:
                self.logger.error(f"Error unsubscribing from {subscription_key}: {e}")
    
    def _update_metrics(self, event_type: str) -> None:
        """
        Обновление метрик
        
        Args:
            event_type: Тип события
        """
        if event_type == 'stream_message':
            self._metrics['messages_processed'] += 1
            self._metrics['last_update'] = datetime.utcnow()
        elif event_type == 'error':
            self._metrics['errors_count'] += 1
        elif event_type == 'historical_request':
            self._metrics['last_update'] = datetime.utcnow()
    
    async def get_synchronized_data(
        self,
        data_type: str,
        timestamp: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Получение синхронизированных данных
        
        Args:
            data_type: Тип данных
            timestamp: Временная метка
            
        Returns:
            List[Dict]: Синхронизированные данные
        """
        if not self._sync_manager:
            return []
        
        return await self._sync_manager.get_aggregated_data(data_type, timestamp)
    
    async def get_synchronized_stream(self) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Получение синхронизированного потока данных
        
        Yields:
            Dict: Синхронизированные данные
        """
        if not self._sync_manager:
            return
        
        async for data in self._sync_manager.get_synchronized_streams():
            yield data
    
    def get_synchronization_metrics(self) -> Dict[str, Any]:
        """
        Получение метрик синхронизации
        
        Returns:
            Dict: Метрики синхронизации
        """
        if not self._sync_manager:
            return {'synchronization': 'disabled'}
        
        return self._sync_manager.get_metrics()
    
    async def get_data_status(self) -> Dict[str, Any]:
        """
        Получение статуса данных
        
        Returns:
            Dict: Статус данных
        """
        status = {
            'data_manager': {
                'running': self._running,
                'connected_sources': self._metrics['connected_sources'],
                'messages_processed': self._metrics['messages_processed'],
                'errors_count': self._metrics['errors_count'],
                'last_update': self._metrics['last_update']
            },
            'connectors': {}
        }
        
        # Статус коннекторов
        for name, connector in self._connectors.items():
            status['connectors'][name] = {
                'connected': connector.is_connected,
                'mock_mode': getattr(connector, 'mock_mode', False)
            }
        
        # Метрики синхронизации
        if self._sync_manager:
            status['synchronization'] = self.get_synchronization_metrics()
        else:
            status['synchronization'] = {'status': 'disabled'}
        
        return status
