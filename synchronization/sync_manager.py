"""
Sync Manager

Главный менеджер синхронизации данных.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, AsyncGenerator
from datetime import datetime, timedelta
from dataclasses import dataclass

from .synchronizers.base_synchronizer import BaseSynchronizer
from .synchronizers.ohlcv_synchronizer import OHLCVSynchronizer
from .synchronizers.trades_synchronizer import TradesSynchronizer
from .synchronizers.orderbook_synchronizer import OrderBookSynchronizer
from .aggregators.time_aggregator import TimeAggregator
from .handlers.gap_handler import GapHandler


@dataclass
class DataSource:
    """Описание источника данных"""
    name: str
    data_type: str
    synchronizer: BaseSynchronizer
    enabled: bool = True


class SyncManager:
    """Главный менеджер синхронизации"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Инициализация менеджера синхронизации
        
        Args:
            config: Конфигурация системы
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Инициализация компонентов
        self.synchronizers = {}
        self.data_sources = {}
        self.aggregators = {}
        self.handlers = {}
        
        # Состояние системы
        self._running = False
        self._start_time = None
        
        # Метрики
        self.metrics = {
            'total_messages_processed': 0,
            'total_gaps_detected': 0,
            'total_gaps_filled': 0,
            'data_sources_active': 0,
            'sync_quality_score': 0.0,
            'last_update': None
        }
        
        # Инициализация синхронизаторов
        self._initialize_synchronizers()
        
        # Инициализация агрегаторов
        self._initialize_aggregators()
        
        # Инициализация обработчиков
        self._initialize_handlers()
    
    def _initialize_synchronizers(self) -> None:
        """Инициализация синхронизаторов"""
        sync_config = self.config.get('synchronization', {})
        
        # OHLCV синхронизатор
        ohlcv_config = sync_config.get('ohlcv', {})
        self.synchronizers['ohlcv'] = OHLCVSynchronizer(ohlcv_config)
        
        # Trades синхронизатор
        trades_config = sync_config.get('trades', {})
        self.synchronizers['trades'] = TradesSynchronizer(trades_config)
        
        # Order Book синхронизатор
        orderbook_config = sync_config.get('orderbook', {})
        self.synchronizers['orderbook'] = OrderBookSynchronizer(orderbook_config)
        
        self.logger.info(f"Initialized {len(self.synchronizers)} synchronizers")
    
    def _initialize_aggregators(self) -> None:
        """Инициализация агрегаторов"""
        agg_config = self.config.get('aggregation', {})
        
        # Временной агрегатор
        self.aggregators['time'] = TimeAggregator(agg_config)
        
        self.logger.info(f"Initialized {len(self.aggregators)} aggregators")
    
    def _initialize_handlers(self) -> None:
        """Инициализация обработчиков"""
        handler_config = self.config.get('gap_handling', {})
        
        # Обработчик пропусков
        self.handlers['gap'] = GapHandler(handler_config)
        
        self.logger.info(f"Initialized {len(self.handlers)} handlers")
    
    async def start(self) -> None:
        """Запуск менеджера синхронизации"""
        try:
            self._running = True
            self._start_time = datetime.utcnow()
            
            # Запуск всех синхронизаторов
            for name, synchronizer in self.synchronizers.items():
                await synchronizer.start()
                self.logger.info(f"Started {name} synchronizer")
            
            self.logger.info("Sync Manager started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start Sync Manager: {e}")
            await self.stop()
            raise
    
    async def stop(self) -> None:
        """Остановка менеджера синхронизации"""
        try:
            self._running = False
            
            # Остановка всех синхронизаторов
            for name, synchronizer in self.synchronizers.items():
                await synchronizer.stop()
                self.logger.info(f"Stopped {name} synchronizer")
            
            self.logger.info("Sync Manager stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping Sync Manager: {e}")
    
    async def add_data_source(self, name: str, data_type: str, synchronizer_name: str) -> bool:
        """
        Добавление источника данных
        
        Args:
            name: Имя источника
            data_type: Тип данных
            synchronizer_name: Имя синхронизатора
            
        Returns:
            bool: True если успешно добавлен
        """
        try:
            if synchronizer_name not in self.synchronizers:
                self.logger.error(f"Synchronizer {synchronizer_name} not found")
                return False
            
            data_source = DataSource(
                name=name,
                data_type=data_type,
                synchronizer=self.synchronizers[synchronizer_name]
            )
            
            self.data_sources[name] = data_source
            self.logger.info(f"Added data source: {name} ({data_type})")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add data source {name}: {e}")
            return False
    
    async def synchronize_data(self, data: Dict[str, Any], source_name: str) -> Optional[Dict[str, Any]]:
        """
        Синхронизация данных от источника
        
        Args:
            data: Данные для синхронизации
            source_name: Имя источника
            
        Returns:
            Dict: Синхронизированные данные или None
        """
        try:
            if source_name not in self.data_sources:
                self.logger.error(f"Data source {source_name} not found")
                return None
            
            data_source = self.data_sources[source_name]
            
            if not data_source.enabled:
                return None
            
            # Синхронизация через соответствующий синхронизатор
            synchronized_data = await data_source.synchronizer.synchronize(data)
            
            # Добавление во временной агрегатор
            if synchronized_data:
                await self.aggregators['time'].add_data(synchronized_data)
            
            # Обновление метрик
            self._update_metrics('message')
            
            return synchronized_data
            
        except Exception as e:
            self.logger.error(f"Error synchronizing data from {source_name}: {e}")
            return None
    
    async def get_synchronized_streams(self) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Получение синхронизированных потоков данных
        
        Yields:
            Dict: Синхронизированные данные
        """
        if not self._running:
            return
        
        # Создаем задачи для каждого синхронизатора
        stream_tasks = []
        
        for name, synchronizer in self.synchronizers.items():
            if synchronizer.is_running():
                task = asyncio.create_task(
                    self._stream_synchronizer_data(name, synchronizer)
                )
                stream_tasks.append(task)
        
        # Объединяем потоки
        if stream_tasks:
            done, pending = await asyncio.wait(
                stream_tasks,
                return_when=asyncio.FIRST_COMPLETED
            )
            
            for task in done:
                try:
                    result = task.result()
                    if result:
                        yield result
                except Exception as e:
                    self.logger.error(f"Error in stream task: {e}")
            
            # Перезапуск завершенных задач
            for task in pending:
                task.cancel()
    
    async def _stream_synchronizer_data(self, name: str, synchronizer: BaseSynchronizer) -> Optional[Dict[str, Any]]:
        """
        Потоковая передача данных от синхронизатора
        
        Args:
            name: Имя синхронизатора
            synchronizer: Синхронизатор
            
        Returns:
            Dict: Данные или None
        """
        try:
            async for data in synchronizer.get_synchronized_stream():
                if data:
                    # Добавление метаданных
                    data['synchronizer'] = name
                    data['sync_timestamp'] = datetime.utcnow()
                    return data
        except Exception as e:
            self.logger.error(f"Error in {name} stream: {e}")
        
        return None
    
    async def get_aggregated_data(self, data_type: str, timestamp: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Получение агрегированных данных
        
        Args:
            data_type: Тип данных
            timestamp: Временная метка
            
        Returns:
            List[Dict]: Агрегированные данные
        """
        try:
            # Очистка завершенных timesteps
            aggregated = await self.aggregators['time'].cleanup_completed_timesteps(timestamp)
            
            # Фильтрация по типу данных
            if data_type:
                aggregated = [data for data in aggregated if data.get('data_type') == data_type]
            
            return aggregated
            
        except Exception as e:
            self.logger.error(f"Error getting aggregated data: {e}")
            return []
    
    async def handle_gaps_in_series(self, data_series: List[Dict[str, Any]], data_type: str) -> List[Dict[str, Any]]:
        """
        Обработка пропусков в временном ряду
        
        Args:
            data_series: Временной ряд
            data_type: Тип данных
            
        Returns:
            List[Dict]: Данные с заполненными пропусками
        """
        try:
            if 'gap' not in self.handlers:
                return data_series
            
            gap_handler = self.handlers['gap']
            filled_series = await gap_handler.fill_gaps_in_series(data_series)
            
            # Обновление метрик
            gap_metrics = gap_handler.get_metrics()
            self.metrics['total_gaps_detected'] += gap_metrics['gaps_detected']
            self.metrics['total_gaps_filled'] += gap_metrics['gaps_filled']
            
            return filled_series
            
        except Exception as e:
            self.logger.error(f"Error handling gaps: {e}")
            return data_series
    
    def _update_metrics(self, event_type: str) -> None:
        """
        Обновление метрик
        
        Args:
            event_type: Тип события
        """
        self.metrics['total_messages_processed'] += 1
        self.metrics['last_update'] = datetime.utcnow()
        
        # Обновление активных источников
        active_sources = sum(1 for ds in self.data_sources.values() if ds.enabled)
        self.metrics['data_sources_active'] = active_sources
        
        # Расчет качества синхронизации
        self._calculate_sync_quality()
    
    def _calculate_sync_quality(self) -> None:
        """Расчет качества синхронизации"""
        try:
            # Базовый score на основе активности источников
            total_sources = len(self.data_sources)
            active_sources = self.metrics['data_sources_active']
            
            if total_sources == 0:
                source_score = 0.0
            else:
                source_score = active_sources / total_sources
            
            # Score на основе обработки пропусков
            if self.metrics['total_gaps_detected'] > 0:
                gap_score = self.metrics['total_gaps_filled'] / self.metrics['total_gaps_detected']
            else:
                gap_score = 1.0
            
            # Комбинированный score
            self.metrics['sync_quality_score'] = (source_score + gap_score) / 2
            
        except Exception as e:
            self.logger.error(f"Error calculating sync quality: {e}")
            self.metrics['sync_quality_score'] = 0.0
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Получение метрик менеджера синхронизации
        
        Returns:
            Dict: Метрики
        """
        # Добавление метрик от компонентов
        synchronizer_metrics = {}
        for name, synchronizer in self.synchronizers.items():
            synchronizer_metrics[name] = synchronizer.get_metrics()
        
        aggregator_metrics = {}
        for name, aggregator in self.aggregators.items():
            aggregator_metrics[name] = aggregator.get_statistics()
        
        handler_metrics = {}
        for name, handler in self.handlers.items():
            handler_metrics[name] = handler.get_metrics()
        
        return {
            'sync_manager': self.metrics.copy(),
            'synchronizers': synchronizer_metrics,
            'aggregators': aggregator_metrics,
            'handlers': handler_metrics,
            'uptime': (datetime.utcnow() - self._start_time).total_seconds() if self._start_time else 0
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Проверка здоровья системы синхронизации
        
        Returns:
            Dict: Статус здоровья
        """
        health = {
            'sync_manager': {
                'running': self._running,
                'uptime': (datetime.utcnow() - self._start_time).total_seconds() if self._start_time else 0,
                'data_sources': len(self.data_sources),
                'active_sources': self.metrics['data_sources_active']
            }
        }
        
        # Проверка здоровья синхронизаторов
        synchronizer_health = {}
        for name, synchronizer in self.synchronizers.items():
            synchronizer_health[name] = await synchronizer.health_check()
        
        health['synchronizers'] = synchronizer_health
        
        return health
    
    def is_running(self) -> bool:
        """Проверка состояния работы"""
        return self._running
    
    def get_data_sources(self) -> Dict[str, DataSource]:
        """Получение списка источников данных"""
        return self.data_sources.copy()
    
    def enable_data_source(self, source_name: str) -> bool:
        """
        Включение источника данных
        
        Args:
            source_name: Имя источника
            
        Returns:
            bool: True если успешно
        """
        if source_name in self.data_sources:
            self.data_sources[source_name].enabled = True
            self.logger.info(f"Enabled data source: {source_name}")
            return True
        return False
    
    def disable_data_source(self, source_name: str) -> bool:
        """
        Отключение источника данных
        
        Args:
            source_name: Имя источника
            
        Returns:
            bool: True если успешно
        """
        if source_name in self.data_sources:
            self.data_sources[source_name].enabled = False
            self.logger.info(f"Disabled data source: {source_name}")
            return True
        return False
