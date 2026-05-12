"""
Base Synchronizer

Абстрактный базовый класс для всех синхронизаторов данных.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, AsyncGenerator
from datetime import datetime, timedelta
import pandas as pd
import numpy as np


class BaseSynchronizer(ABC):
    """Базовый класс синхронизатора"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Инициализация базового синхронизатора
        
        Args:
            config: Конфигурация синхронизатора
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Базовый timestep (минимальный интервал синхронизации)
        self.base_timestep = config.get('base_timestep', '1s')
        self.timestep_seconds = self._parse_timestep(self.base_timestep)
        
        # Буфер для данных
        self.buffer = []
        self.buffer_size = config.get('buffer_size', 10000)
        
        # Метрики синхронизации
        self.metrics = {
            'messages_processed': 0,
            'gaps_detected': 0,
            'gaps_filled': 0,
            'anomalies_detected': 0,
            'last_update': None
        }
        
        # Состояние синхронизатора
        self._running = False
        self._last_timestamp = None
    
    @abstractmethod
    async def synchronize(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Синхронизация одного элемента данных
        
        Args:
            data: Входные данные
            
        Returns:
            Dict: Синхронизированные данные или None если нужно пропустить
        """
        pass
    
    @abstractmethod
    async def get_synchronized_stream(self) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Получение синхронизированного потока данных
        
        Yields:
            Dict: Синхронизированные данные
        """
        pass
    
    def _parse_timestep(self, timestep: str) -> int:
        """
        Парсинг timestep в секунды
        
        Args:
            timestep: Строка timestep ('1s', '5s', '1m', '5m', '15m')
            
        Returns:
            int: Количество секунд
        """
        timestep_map = {
            '1s': 1,
            '5s': 5,
            '1m': 60,
            '5m': 300,
            '15m': 900
        }
        return timestep_map.get(timestep, 1)
    
    def _round_timestamp(self, timestamp: datetime) -> datetime:
        """
        Округление временной метки до базового timestep
        
        Args:
            timestamp: Исходная временная метка
            
        Returns:
            datetime: Округленная временная метка
        """
        timestamp_seconds = int(timestamp.timestamp())
        rounded_seconds = (timestamp_seconds // self.timestep_seconds) * self.timestep_seconds
        return datetime.fromtimestamp(rounded_seconds)
    
    def _detect_gap(self, current_timestamp: datetime, previous_timestamp: datetime) -> bool:
        """
        Детекция пропуска в данных
        
        Args:
            current_timestamp: Текущая временная метка
            previous_timestamp: Предыдущая временная метка
            
        Returns:
            bool: True если есть пропуск
        """
        if previous_timestamp is None:
            return False
        
        gap_seconds = (current_timestamp - previous_timestamp).total_seconds()
        max_gap = self.config.get('max_gap_size', 60)  # Максимальный размер пропуска в секундах
        
        return gap_seconds > max_gap
    
    def _validate_timestamp(self, timestamp: datetime) -> bool:
        """
        Валидация временной метки
        
        Args:
            timestamp: Временная метка для валидации
            
        Returns:
            bool: True если валидная
        """
        # Минимальная валидация - только проверка на очень экстремальные значения
        try:
            # Проверка что это datetime объект
            if not isinstance(timestamp, datetime):
                return False
            
            # Проверка на очень старые данные (более года)
            now = datetime.utcnow()
            if timestamp < now - timedelta(days=365):
                return False
            
            # Проверка на очень далекое будущее (более часа)
            if timestamp > now + timedelta(hours=1):
                return False
            
            return True
            
        except Exception:
            return False
    
    def _add_to_buffer(self, data: Dict[str, Any]) -> None:
        """
        Добавление данных в буфер
        
        Args:
            data: Данные для добавления
        """
        self.buffer.append(data)
        
        # Ограничение размера буфера
        if len(self.buffer) > self.buffer_size:
            self.buffer = self.buffer[-self.buffer_size:]
    
    def _get_buffer_data(self, since: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Получение данных из буфера
        
        Args:
            since: Начальная временная метка (опционально)
            
        Returns:
            List[Dict]: Данные из буфера
        """
        if since is None:
            return self.buffer.copy()
        
        return [data for data in self.buffer if data.get('timestamp', datetime.min) >= since]
    
    def _update_metrics(self, event_type: str) -> None:
        """
        Обновление метрик
        
        Args:
            event_type: Тип события ('message', 'gap', 'anomaly')
        """
        self.metrics['messages_processed'] += 1
        self.metrics['last_update'] = datetime.utcnow()
        
        if event_type == 'gap':
            self.metrics['gaps_detected'] += 1
        elif event_type == 'gap_filled':
            self.metrics['gaps_filled'] += 1
        elif event_type == 'anomaly':
            self.metrics['anomalies_detected'] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Получение метрик синхронизатора
        
        Returns:
            Dict: Метрики
        """
        return self.metrics.copy()
    
    async def start(self) -> None:
        """Запуск синхронизатора"""
        self._running = True
        self.logger.info(f"Started {self.__class__.__name__}")
    
    async def stop(self) -> None:
        """Остановка синхронизатора"""
        self._running = False
        self.logger.info(f"Stopped {self.__class__.__name__}")
    
    def is_running(self) -> bool:
        """Проверка состояния работы"""
        return self._running
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Проверка здоровья синхронизатора
        
        Returns:
            Dict: Статус здоровья
        """
        return {
            'synchronizer': self.__class__.__name__,
            'running': self._running,
            'base_timestep': self.base_timestep,
            'buffer_size': len(self.buffer),
            'metrics': self.get_metrics()
        }
    
    def reset_metrics(self) -> None:
        """Сброс метрик"""
        self.metrics = {
            'messages_processed': 0,
            'gaps_detected': 0,
            'gaps_filled': 0,
            'anomalies_detected': 0,
            'last_update': None
        }
    
    def clear_buffer(self) -> None:
        """Очистка буфера"""
        self.buffer.clear()
        self.logger.info("Buffer cleared")
