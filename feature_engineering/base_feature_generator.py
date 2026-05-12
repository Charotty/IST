"""
Base Feature Generator

Абстрактный базовый класс для генераторов признаков.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import pandas as pd
import numpy as np


class BaseFeatureGenerator(ABC):
    """Базовый класс для генераторов признаков"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Инициализация генератора признаков
        
        Args:
            config: Конфигурация генератора
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Кэш для вычисленных признаков
        self._feature_cache = {}
        
        # Метрики производительности
        self._metrics = {
            'features_generated': 0,
            'cache_hits': 0,
            'computation_time': 0.0,
            'errors_count': 0
        }
        
        # Валидация конфигурации
        self._validate_config()
    
    @abstractmethod
    def _validate_config(self) -> None:
        """
        Валидация конфигурации генератора
        
        Raises:
            ValueError: Если конфигурация невалидна
        """
        pass
    
    @abstractmethod
    async def generate_features(self, data: Union[pd.DataFrame, Dict[str, Any]]) -> pd.DataFrame:
        """
        Генерация признаков
        
        Args:
            data: Входные данные
            
        Returns:
            pd.DataFrame: DataFrame с признаками
        """
        pass
    
    def _get_cache_key(self, data_hash: str, feature_name: str) -> str:
        """
        Генерация ключа для кэша
        
        Args:
            data_hash: Хэш данных
            feature_name: Имя признака
            
        Returns:
            str: Ключ для кэша
        """
        return f"{self.__class__.__name__}_{feature_name}_{data_hash}"
    
    def _get_data_hash(self, data: Union[pd.DataFrame, Dict[str, Any]]) -> str:
        """
        Получение хэша данных для кэширования
        
        Args:
            data: Данные для хэширования
            
        Returns:
            str: Хэш данных
        """
        if isinstance(data, pd.DataFrame):
            # Используем последние несколько строк для хэша
            tail_data = data.tail(10) if len(data) > 10 else data
            return str(hash(str(tail_data.values.tobytes())))
        else:
            return str(hash(str(data)))
    
    def _get_cached_feature(self, cache_key: str) -> Optional[Any]:
        """
        Получение признака из кэша
        
        Args:
            cache_key: Ключ кэша
            
        Returns:
            Any: Признак из кэша или None
        """
        if cache_key in self._feature_cache:
            self._metrics['cache_hits'] += 1
            return self._feature_cache[cache_key]
        return None
    
    def _cache_feature(self, cache_key: str, feature: Any) -> None:
        """
        Сохранение признака в кэш
        
        Args:
            cache_key: Ключ кэша
            feature: Признак для сохранения
        """
        # Ограничение размера кэша
        if len(self._feature_cache) > 1000:
            # Удаляем самые старые записи
            oldest_keys = list(self._feature_cache.keys())[:100]
            for key in oldest_keys:
                del self._feature_cache[key]
        
        self._feature_cache[cache_key] = feature
    
    def _validate_input_data(self, data: Union[pd.DataFrame, Dict[str, Any]]) -> bool:
        """
        Валидация входных данных
        
        Args:
            data: Входные данные
            
        Returns:
            bool: True если данные валидны
        """
        if data is None:
            return False
        
        if isinstance(data, pd.DataFrame):
            return not data.empty
        elif isinstance(data, dict):
            return len(data) > 0
        
        return False
    
    def _handle_error(self, error: Exception, context: str) -> None:
        """
        Обработка ошибок
        
        Args:
            error: Исключение
            context: Контекст ошибки
        """
        self._metrics['errors_count'] += 1
        self.logger.error(f"Error in {context}: {error}")
    
    def _update_metrics(self, features_count: int, computation_time: float) -> None:
        """
        Обновление метрик
        
        Args:
            features_count: Количество сгенерированных признаков
            computation_time: Время вычисления
        """
        self._metrics['features_generated'] += features_count
        self._metrics['computation_time'] += computation_time
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Получение метрик производительности
        
        Returns:
            Dict: Метрики
        """
        metrics = self._metrics.copy()
        
        # Добавление производных метрик
        if metrics['computation_time'] > 0:
            metrics['features_per_second'] = metrics['features_generated'] / metrics['computation_time']
        else:
            metrics['features_per_second'] = 0
        
        if metrics['features_generated'] > 0:
            metrics['cache_hit_rate'] = metrics['cache_hits'] / metrics['features_generated']
        else:
            metrics['cache_hit_rate'] = 0
        
        metrics['cache_size'] = len(self._feature_cache)
        
        return metrics
    
    def reset_metrics(self) -> None:
        """Сброс метрик"""
        self._metrics = {
            'features_generated': 0,
            'cache_hits': 0,
            'computation_time': 0.0,
            'errors_count': 0
        }
    
    def clear_cache(self) -> None:
        """Очистка кэша"""
        self._feature_cache.clear()
        self.logger.info("Feature cache cleared")
    
    async def generate_features_with_cache(self, data: Union[pd.DataFrame, Dict[str, Any]]) -> pd.DataFrame:
        """
        Генерация признаков с использованием кэша
        
        Args:
            data: Входные данные
            
        Returns:
            pd.DataFrame: DataFrame с признаками
        """
        try:
            # Валидация данных
            if not self._validate_input_data(data):
                self.logger.warning("Invalid input data")
                return pd.DataFrame()
            
            # Получение хэша данных
            data_hash = self._get_data_hash(data)
            
            # Генерация признаков
            start_time = asyncio.get_event_loop().time()
            features = await self.generate_features(data)
            computation_time = asyncio.get_event_loop().time() - start_time
            
            # Обновление метрик
            self._update_metrics(len(features.columns) if not features.empty else 0, computation_time)
            
            return features
            
        except Exception as e:
            self._handle_error(e, "generate_features_with_cache")
            return pd.DataFrame()
    
    def get_feature_names(self) -> List[str]:
        """
        Получение списка имен признаков
        
        Returns:
            List[str]: Имена признаков
        """
        return list(self.config.get('features', []))
    
    def is_enabled(self) -> bool:
        """
        Проверка включен ли генератор
        
        Returns:
            bool: True если включен
        """
        return self.config.get('enabled', True)
    
    def get_required_data_types(self) -> List[str]:
        """
        Получение требуемых типов данных
        
        Returns:
            List[str]: Типы данных
        """
        return self.config.get('required_data_types', ['ohlcv'])
    
    def __str__(self) -> str:
        """Строковое представление"""
        return f"{self.__class__.__name__}(features={len(self.get_feature_names())})"
    
    def __repr__(self) -> str:
        """Подробное строковое представление"""
        return (f"{self.__class__.__name__}("
                f"features={self.get_feature_names()}, "
                f"enabled={self.is_enabled()}, "
                f"cache_size={len(self._feature_cache)})")
