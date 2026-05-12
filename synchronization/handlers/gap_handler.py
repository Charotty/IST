"""
Gap Handler

Интеллектуальная обработка пропусков в данных.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from scipy import interpolate


class GapHandler:
    """Обработчик пропусков в данных"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Инициализация обработчика пропусков
        
        Args:
            config: Конфигурация обработчика
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Методы обработки пропусков
        self.default_method = config.get('method', 'adaptive')
        self.max_gap_size = config.get('max_gap_size', 60)  # секунд
        
        # Пороги для разных методов
        self.forward_fill_threshold = config.get('forward_fill_threshold', 5)  # секунд
        self.linear_threshold = config.get('linear_threshold', 30)  # секунд
        self.interpolation_threshold = config.get('interpolation_threshold', 60)  # секунд
        
        # Метрики обработки
        self.metrics = {
            'gaps_detected': 0,
            'gaps_filled': 0,
            'forward_fill_used': 0,
            'linear_used': 0,
            'interpolation_used': 0,
            'too_large_gaps': 0
        }
    
    async def detect_gaps(self, timestamps: List[datetime], max_gap_seconds: Optional[int] = None) -> List[Tuple[datetime, datetime]]:
        """
        Детекция пропусков в последовательности временных меток
        
        Args:
            timestamps: Список временных меток
            max_gap_seconds: Максимальный размер пропуска для детекции
            
        Returns:
            List[Tuple]: Список пропусков [(start, end), ...]
        """
        if len(timestamps) < 2:
            return []
        
        if max_gap_seconds is None:
            max_gap_seconds = self.max_gap_size
        
        gaps = []
        sorted_timestamps = sorted(timestamps)
        
        for i in range(len(sorted_timestamps) - 1):
            current = sorted_timestamps[i]
            next_time = sorted_timestamps[i + 1]
            gap_seconds = (next_time - current).total_seconds()
            
            if gap_seconds > max_gap_seconds:
                gaps.append((current, next_time))
        
        self.metrics['gaps_detected'] += len(gaps)
        return gaps
    
    async def fill_gap(self, data_before: Dict[str, Any], data_after: Dict[str, Any], 
                      gap_start: datetime, gap_end: datetime, 
                      method: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Заполнение пропуска в данных
        
        Args:
            data_before: Данные перед пропуском
            data_after: Данные после пропуска
            gap_start: Начало пропуска
            gap_end: Конец пропуска
            method: Метод заполнения
            
        Returns:
            List[Dict]: Заполненные данные
        """
        if method is None:
            method = self.default_method
        
        gap_seconds = (gap_end - gap_start).total_seconds()
        
        # Проверка размера пропуска
        if gap_seconds > self.interpolation_threshold:
            self.metrics['too_large_gaps'] += 1
            self.logger.warning(f"Gap too large ({gap_seconds}s), skipping fill")
            return []
        
        # Выбор метода заполнения
        if method == 'adaptive':
            method = self._choose_fill_method(gap_seconds)
        
        # Заполнение пропуска
        if method == 'forward_fill':
            return await self._forward_fill(data_before, gap_start, gap_end)
        elif method == 'linear':
            return await self._linear_interpolation(data_before, data_after, gap_start, gap_end)
        elif method == 'cubic':
            return await self._cubic_interpolation(data_before, data_after, gap_start, gap_end)
        else:
            return await self._forward_fill(data_before, gap_start, gap_end)
    
    def _choose_fill_method(self, gap_seconds: float) -> str:
        """
        Выбор метода заполнения на основе размера пропуска
        
        Args:
            gap_seconds: Размер пропуска в секундах
            
        Returns:
            str: Метод заполнения
        """
        if gap_seconds <= self.forward_fill_threshold:
            return 'forward_fill'
        elif gap_seconds <= self.linear_threshold:
            return 'linear'
        else:
            return 'cubic'
    
    async def _forward_fill(self, data_before: Dict[str, Any], gap_start: datetime, gap_end: datetime) -> List[Dict[str, Any]]:
        """
        Forward fill заполнение
        
        Args:
            data_before: Данные перед пропуском
            gap_start: Начало пропуска
            gap_end: Конец пропуска
            
        Returns:
            List[Dict]: Заполненные данные
        """
        self.metrics['forward_fill_used'] += 1
        self.metrics['gaps_filled'] += 1
        
        # Генерация timestep'ов в пропуске
        timestep_seconds = (gap_end - gap_start).total_seconds() / 10  # 10 точек
        filled_data = []
        
        current_time = gap_start
        while current_time < gap_end:
            filled_point = data_before.copy()
            filled_point['timestamp'] = current_time
            filled_point['gap_filled'] = True
            filled_point['fill_method'] = 'forward_fill'
            
            # Обнуление объемов для OHLCV данных
            if 'volume' in filled_point:
                filled_point['volume'] = 0.0
            
            filled_data.append(filled_point)
            current_time += timedelta(seconds=timestep_seconds)
        
        return filled_data
    
    async def _linear_interpolation(self, data_before: Dict[str, Any], data_after: Dict[str, Any],
                                   gap_start: datetime, gap_end: datetime) -> List[Dict[str, Any]]:
        """
        Линейная интерполяция
        
        Args:
            data_before: Данные перед пропуском
            data_after: Данные после пропуска
            gap_start: Начало пропуска
            gap_end: Конец пропуска
            
        Returns:
            List[Dict]: Заполненные данные
        """
        self.metrics['linear_used'] += 1
        self.metrics['gaps_filled'] += 1
        
        # Числовые поля для интерполяции
        numeric_fields = []
        for field in data_before.keys():
            if field != 'timestamp' and isinstance(data_before[field], (int, float)):
                if field in data_after and isinstance(data_after[field], (int, float)):
                    numeric_fields.append(field)
        
        if not numeric_fields:
            return await self._forward_fill(data_before, gap_start, gap_end)
        
        # Генерация точек
        num_points = 10
        filled_data = []
        
        for i in range(num_points):
            alpha = i / (num_points - 1)  # Вес для интерполяции
            current_time = gap_start + timedelta(seconds=(gap_end - gap_start).total_seconds() * alpha)
            
            interpolated_point = {'timestamp': current_time, 'gap_filled': True, 'fill_method': 'linear'}
            
            # Линейная интерполяция для каждого числового поля
            for field in numeric_fields:
                before_value = float(data_before[field])
                after_value = float(data_after[field])
                interpolated_value = before_value + alpha * (after_value - before_value)
                interpolated_point[field] = interpolated_value
            
            # Обнуление объемов
            if 'volume' in interpolated_point:
                interpolated_point['volume'] = 0.0
            
            filled_data.append(interpolated_point)
        
        return filled_data
    
    async def _cubic_interpolation(self, data_before: Dict[str, Any], data_after: Dict[str, Any],
                                  gap_start: datetime, gap_end: datetime) -> List[Dict[str, Any]]:
        """
        Кубическая интерполяция
        
        Args:
            data_before: Данные перед пропуском
            data_after: Данные после пропуска
            gap_start: Начало пропуска
            gap_end: Конец пропуска
            
        Returns:
            List[Dict]: Заполненные данные
        """
        self.metrics['interpolation_used'] += 1
        self.metrics['gaps_filled'] += 1
        
        # Числовые поля для интерполяции
        numeric_fields = []
        for field in data_before.keys():
            if field != 'timestamp' and isinstance(data_before[field], (int, float)):
                if field in data_after and isinstance(data_after[field], (int, float)):
                    numeric_fields.append(field)
        
        if not numeric_fields:
            return await self._linear_interpolation(data_before, data_after, gap_start, gap_end)
        
        # Создание точек для интерполяции
        num_points = 10
        filled_data = []
        
        # Временные точки для интерполяции
        time_points = [0, 1]  # 0 = before, 1 = after
        new_time_points = [i / (num_points - 1) for i in range(num_points)]
        
        for i, alpha in enumerate(new_time_points):
            current_time = gap_start + timedelta(seconds=(gap_end - gap_start).total_seconds() * alpha)
            
            interpolated_point = {'timestamp': current_time, 'gap_filled': True, 'fill_method': 'cubic'}
            
            # Кубическая интерполяция для каждого поля
            for field in numeric_fields:
                try:
                    values = [float(data_before[field]), float(data_after[field])]
                    
                    # Для кубической интерполяции нужны минимум 4 точки, используем линейную
                    if len(values) < 4:
                        interpolated_value = values[0] + alpha * (values[1] - values[0])
                    else:
                        # Простая кубическая интерполяция
                        f = interpolate.interp1d(time_points, values, kind='cubic', fill_value='extrapolate')
                        interpolated_value = float(f(alpha))
                    
                    interpolated_point[field] = interpolated_value
                    
                except Exception as e:
                    # Fallback к линейной интерполяции
                    before_value = float(data_before[field])
                    after_value = float(data_after[field])
                    interpolated_value = before_value + alpha * (after_value - before_value)
                    interpolated_point[field] = interpolated_value
            
            # Обнуление объемов
            if 'volume' in interpolated_point:
                interpolated_point['volume'] = 0.0
            
            filled_data.append(interpolated_point)
        
        return filled_data
    
    async def fill_gaps_in_series(self, data_series: List[Dict[str, Any]], 
                                 timestamp_field: str = 'timestamp') -> List[Dict[str, Any]]:
        """
        Заполнение всех пропусков в временном ряду
        
        Args:
            data_series: Временной ряд данных
            timestamp_field: Имя поля с временной меткой
            
        Returns:
            List[Dict]: Данные с заполненными пропусками
        """
        if len(data_series) < 2:
            return data_series
        
        # Сортировка по времени
        sorted_data = sorted(data_series, key=lambda x: x[timestamp_field])
        
        # Детекция пропусков
        timestamps = [data[timestamp_field] for data in sorted_data]
        gaps = await self.detect_gaps(timestamps)
        
        if not gaps:
            return sorted_data
        
        # Заполнение пропусков
        filled_data = [sorted_data[0]]  # Начинаем с первой точки
        
        for i, (gap_start, gap_end) in enumerate(gaps):
            # Найти данные до и после пропуска
            before_data = None
            after_data = None
            
            for data in sorted_data:
                if data[timestamp_field] == gap_start:
                    before_data = data
                elif data[timestamp_field] == gap_end:
                    after_data = data
            
            if before_data and after_data:
                # Заполнение пропуска
                gap_fill = await self.fill_gap(before_data, after_data, gap_start, gap_end)
                filled_data.extend(gap_fill)
            
            # Добавление данных после пропуска
            if after_data:
                filled_data.append(after_data)
        
        # Сортировка результата
        filled_data.sort(key=lambda x: x[timestamp_field])
        
        return filled_data
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Получение метрик обработки пропусков
        
        Returns:
            Dict: Метрики
        """
        return self.metrics.copy()
    
    def reset_metrics(self) -> None:
        """Сброс метрик"""
        self.metrics = {
            'gaps_detected': 0,
            'gaps_filled': 0,
            'forward_fill_used': 0,
            'linear_used': 0,
            'interpolation_used': 0,
            'too_large_gaps': 0
        }
    
    def get_fill_quality_score(self) -> float:
        """
        Расчет качества заполнения пропусков
        
        Returns:
            float: Оценка качества (0-1)
        """
        if self.metrics['gaps_detected'] == 0:
            return 1.0
        
        fill_ratio = self.metrics['gaps_filled'] / self.metrics['gaps_detected']
        
        # Бонус за использование более сложных методов
        method_bonus = 0
        if self.metrics['linear_used'] > 0:
            method_bonus += 0.1
        if self.metrics['interpolation_used'] > 0:
            method_bonus += 0.2
        
        # Штраф за слишком большие пропуски
        large_gap_penalty = self.metrics['too_large_gaps'] / max(self.metrics['gaps_detected'], 1) * 0.3
        
        score = min(1.0, fill_ratio + method_bonus - large_gap_penalty)
        return max(0.0, score)
