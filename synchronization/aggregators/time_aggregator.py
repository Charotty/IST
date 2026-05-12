"""
Time Aggregator

Агрегация данных в единые временные интервалы.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np


class TimeAggregator:
    """Агрегатор данных по времени"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Инициализация временного агрегатора
        
        Args:
            config: Конфигурация агрегатора
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Базовый timestep
        self.base_timestep = config.get('base_timestep', '1s')
        self.timestep_seconds = self._parse_timestep(self.base_timestep)
        
        # Методы агрегации
        self.aggregation_method = config.get('aggregation_method', 'standard')
        
        # Буферы для каждого timestep
        self.time_buffers = {}
        
        # Максимальный размер буфера
        self.max_buffer_size = config.get('max_buffer_size', 10000)
    
    def _parse_timestep(self, timestep: str) -> int:
        """
        Парсинг timestep в секунды
        
        Args:
            timestep: Строка timestep
            
        Returns:
            int: Количество секунд
        """
        timestep_map = {
            '1s': 1,
            '5s': 5,
            '1m': 60,
            '5m': 300,
            '15m': 900,
            '1h': 3600
        }
        return timestep_map.get(timestep, 1)
    
    def _round_timestamp(self, timestamp: datetime) -> datetime:
        """
        Округление временной метки до timestep
        
        Args:
            timestamp: Исходная временная метка
            
        Returns:
            datetime: Округленная временная метка
        """
        timestamp_seconds = int(timestamp.timestamp())
        rounded_seconds = (timestamp_seconds // self.timestep_seconds) * self.timestep_seconds
        return datetime.fromtimestamp(rounded_seconds)
    
    async def add_data(self, data: Dict[str, Any]) -> None:
        """
        Добавление данных в временной буфер
        
        Args:
            data: Данные для агрегации
        """
        if 'timestamp' not in data:
            self.logger.warning("Data missing timestamp")
            return
        
        timestamp = data['timestamp']
        rounded_timestamp = self._round_timestamp(timestamp)
        
        if rounded_timestamp not in self.time_buffers:
            self.time_buffers[rounded_timestamp] = []
        
        self.time_buffers[rounded_timestamp].append(data)
        
        # Ограничение размера буфера
        if len(self.time_buffers[rounded_timestamp]) > 100:
            self.time_buffers[rounded_timestamp] = self.time_buffers[rounded_timestamp][-100:]
        
        # Ограничение общего количества timesteps
        if len(self.time_buffers) > self.max_buffer_size:
            oldest_timestamp = min(self.time_buffers.keys())
            del self.time_buffers[oldest_timestamp]
    
    async def get_aggregated_data(self, timestamp: Optional[datetime] = None) -> Optional[Dict[str, Any]]:
        """
        Получение агрегированных данных для timestep
        
        Args:
            timestamp: Временная метка (если None, используется текущая)
            
        Returns:
            Dict: Агрегированные данные или None
        """
        if timestamp is None:
            timestamp = self._round_timestamp(datetime.utcnow())
        else:
            timestamp = self._round_timestamp(timestamp)
        
        buffer = self.time_buffers.get(timestamp, [])
        
        if not buffer:
            return None
        
        # Агрегация данных
        if self.aggregation_method == 'standard':
            return await self._standard_aggregation(buffer, timestamp)
        elif self.aggregation_method == 'vwap':
            return await self._vwap_aggregation(buffer, timestamp)
        elif self.aggregation_method == 'twap':
            return await self._twap_aggregation(buffer, timestamp)
        else:
            return await self._standard_aggregation(buffer, timestamp)
    
    async def _standard_aggregation(self, data_list: List[Dict[str, Any]], timestamp: datetime) -> Dict[str, Any]:
        """
        Стандартная агрегация данных
        
        Args:
            data_list: Список данных
            timestamp: Временная метка
            
        Returns:
            Dict: Агрегированные данные
        """
        if not data_list:
            return {}
        
        # Базовая статистика
        result = {
            'timestamp': timestamp,
            'count': len(data_list),
            'aggregation_method': 'standard'
        }
        
        # Агрегация числовых полей
        numeric_fields = set()
        for data in data_list:
            for key, value in data.items():
                if key != 'timestamp' and isinstance(value, (int, float)):
                    numeric_fields.add(key)
        
        for field in numeric_fields:
            values = [float(data[field]) for data in data_list if field in data]
            
            if values:
                result[f'{field}_min'] = min(values)
                result[f'{field}_max'] = max(values)
                result[f'{field}_mean'] = np.mean(values)
                result[f'{field}_sum'] = sum(values)
                result[f'{field}_std'] = np.std(values)
                result[f'{field}_first'] = values[0]
                result[f'{field}_last'] = values[-1]
        
        return result
    
    async def _vwap_aggregation(self, data_list: List[Dict[str, Any]], timestamp: datetime) -> Dict[str, Any]:
        """
        VWAP агрегация (Volume Weighted Average Price)
        
        Args:
            data_list: Список данных (должен содержать price и volume)
            timestamp: Временная метка
            
        Returns:
            Dict: VWAP агрегированные данные
        """
        if not data_list:
            return {}
        
        prices = []
        volumes = []
        
        for data in data_list:
            if 'price' in data and 'volume' in data:
                try:
                    price = float(data['price'])
                    volume = float(data['volume'])
                    if price > 0 and volume > 0:
                        prices.append(price)
                        volumes.append(volume)
                except (ValueError, TypeError):
                    continue
        
        if not prices or not volumes:
            return await self._standard_aggregation(data_list, timestamp)
        
        # VWAP расчет
        total_volume = sum(volumes)
        vwap = sum(p * v for p, v in zip(prices, volumes)) / total_volume if total_volume > 0 else 0
        
        result = {
            'timestamp': timestamp,
            'count': len(data_list),
            'aggregation_method': 'vwap',
            'vwap': vwap,
            'total_volume': total_volume,
            'price_min': min(prices),
            'price_max': max(prices),
            'price_mean': np.mean(prices),
            'volume_mean': np.mean(volumes)
        }
        
        return result
    
    async def _twap_aggregation(self, data_list: List[Dict[str, Any]], timestamp: datetime) -> Dict[str, Any]:
        """
        TWAP агрегация (Time Weighted Average Price)
        
        Args:
            data_list: Список данных
            timestamp: Временная метка
            
        Returns:
            Dict: TWAP агрегированные данные
        """
        if not data_list:
            return {}
        
        prices = []
        
        for data in data_list:
            if 'price' in data:
                try:
                    price = float(data['price'])
                    if price > 0:
                        prices.append(price)
                except (ValueError, TypeError):
                    continue
        
        if not prices:
            return await self._standard_aggregation(data_list, timestamp)
        
        # TWAP расчет (простое среднее)
        twap = np.mean(prices)
        
        result = {
            'timestamp': timestamp,
            'count': len(data_list),
            'aggregation_method': 'twap',
            'twap': twap,
            'price_min': min(prices),
            'price_max': max(prices),
            'price_std': np.std(prices)
        }
        
        return result
    
    async def get_completed_timesteps(self, current_time: Optional[datetime] = None) -> List[datetime]:
        """
        Получение завершенных timesteps
        
        Args:
            current_time: Текущее время
            
        Returns:
            List[datetime]: Завершенные timesteps
        """
        if current_time is None:
            current_time = datetime.utcnow()
        
        current_rounded = self._round_timestamp(current_time)
        completed = []
        
        for timestamp in sorted(self.time_buffers.keys()):
            next_timestep = timestamp + timedelta(seconds=self.timestep_seconds)
            if current_rounded >= next_timestep:
                completed.append(timestamp)
        
        return completed
    
    async def cleanup_completed_timesteps(self, current_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Очистка завершенных timesteps и возврат агрегированных данных
        
        Args:
            current_time: Текущее время
            
        Returns:
            List[Dict]: Агрегированные данные для завершенных timesteps
        """
        completed_timesteps = await self.get_completed_timesteps(current_time)
        aggregated_data = []
        
        for timestamp in completed_timesteps:
            data = await self.get_aggregated_data(timestamp)
            if data:
                aggregated_data.append(data)
            
            # Удаление из буфера
            if timestamp in self.time_buffers:
                del self.time_buffers[timestamp]
        
        return aggregated_data
    
    def get_buffer_status(self) -> Dict[str, Any]:
        """
        Получение статуса буфера
        
        Returns:
            Dict: Статус буфера
        """
        total_items = sum(len(buffer) for buffer in self.time_buffers.values())
        
        return {
            'active_timesteps': len(self.time_buffers),
            'total_items': total_items,
            'oldest_timestamp': min(self.time_buffers.keys()) if self.time_buffers else None,
            'newest_timestamp': max(self.time_buffers.keys()) if self.time_buffers else None,
            'timestep_seconds': self.timestep_seconds,
            'aggregation_method': self.aggregation_method
        }
    
    def clear_buffer(self) -> None:
        """Очистка буфера"""
        self.time_buffers.clear()
        self.logger.info("Time aggregator buffer cleared")
    
    async def get_statistics(self) -> Dict[str, Any]:
        """
        Получение статистики агрегации
        
        Returns:
            Dict: Статистика
        """
        if not self.time_buffers:
            return {}
        
        all_counts = [len(buffer) for buffer in self.time_buffers.values()]
        
        return {
            'buffer_status': self.get_buffer_status(),
            'avg_items_per_timestep': np.mean(all_counts) if all_counts else 0,
            'max_items_per_timestep': max(all_counts) if all_counts else 0,
            'total_processed_items': sum(all_counts)
        }
