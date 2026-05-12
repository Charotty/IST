"""
Feature Pipeline

Пайплайн для обработки и генерации признаков.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from .technical.moving_averages import MovingAverages
from .technical.momentum import MomentumIndicators
from .technical.volatility import VolatilityIndicators
from .orderbook.imbalance import OrderBookImbalance
from .orderbook.spread import SpreadFeatures
from .orderbook.depth import DepthFeatures
from .temporal.returns import ReturnFeatures
from .temporal.time_features import TimeBasedFeatures


class FeaturePipeline:
    """Пайплайн генерации признаков"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Инициализация пайплайна
        
        Args:
            config: Конфигурация пайплайна
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Инициализация генераторов признаков
        self.feature_generators = {}
        self._initialize_generators()
        
        # Метрики производительности
        self._metrics = {
            'total_features_generated': 0,
            'pipeline_runs': 0,
            'errors_count': 0,
            'processing_time': 0.0
        }
        
        # Пул потоков для параллельной обработки
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def _initialize_generators(self) -> None:
        """Инициализация генераторов признаков"""
        try:
            feature_config = self.config.get('features', {})
            
            # Технические индикаторы
            if 'technical' in feature_config:
                tech_config = feature_config['technical']
                
                if 'moving_averages' in tech_config:
                    self.feature_generators['moving_averages'] = MovingAverages(
                        tech_config['moving_averages']
                    )
                
                if 'momentum' in tech_config:
                    self.feature_generators['momentum'] = MomentumIndicators(
                        tech_config['momentum']
                    )
                
                if 'volatility' in tech_config:
                    self.feature_generators['volatility'] = VolatilityIndicators(
                        tech_config['volatility']
                    )
            
            # Order Book признаки
            if 'orderbook' in feature_config:
                ob_config = feature_config['orderbook']
                
                if 'imbalance' in ob_config:
                    self.feature_generators['imbalance'] = OrderBookImbalance(
                        ob_config['imbalance']
                    )
                
                if 'spread' in ob_config:
                    self.feature_generators['spread'] = SpreadFeatures(
                        ob_config['spread']
                    )
                
                if 'depth' in ob_config:
                    self.feature_generators['depth'] = DepthFeatures(
                        ob_config['depth']
                    )
            
            # Временные признаки
            if 'temporal' in feature_config:
                temp_config = feature_config['temporal']
                
                if 'returns' in temp_config:
                    self.feature_generators['returns'] = ReturnFeatures(
                        temp_config['returns']
                    )
                
                if 'time_features' in temp_config:
                    self.feature_generators['time_features'] = TimeBasedFeatures(
                        temp_config['time_features']
                    )
            
            self.logger.info(f"Initialized {len(self.feature_generators)} feature generators")
            
        except Exception as e:
            self.logger.error(f"Error initializing feature generators: {e}")
    
    async def process_data(self, data: Union[pd.DataFrame, Dict[str, Any]], 
                          data_type: str) -> pd.DataFrame:
        """
        Обработка данных и генерация признаков
        
        Args:
            data: Входные данные
            data_type: Тип данных
            
        Returns:
            pd.DataFrame: DataFrame с признаками
        """
        try:
            start_time = asyncio.get_event_loop().time()
            
            # Валидация входных данных
            if not self._validate_input_data(data):
                self.logger.warning("Invalid input data")
                return pd.DataFrame()
            
            # Определение подходящих генераторов
            relevant_generators = self._get_relevant_generators(data_type)
            
            if not relevant_generators:
                self.logger.warning(f"No relevant generators for data type: {data_type}")
                return pd.DataFrame()
            
            # Параллельная генерация признаков
            feature_tasks = []
            
            for name, generator in relevant_generators.items():
                if generator.is_enabled():
                    task = asyncio.create_task(
                        self._generate_features_safe(generator, data, name)
                    )
                    feature_tasks.append(task)
            
            # Ожидание завершения всех задач
            feature_results = await asyncio.gather(*feature_tasks, return_exceptions=True)
            
            # Объединение результатов
            combined_features = self._combine_feature_results(feature_results)
            
            # Обновление метрик
            processing_time = asyncio.get_event_loop().time() - start_time
            self._update_metrics(combined_features, processing_time)
            
            return combined_features
            
        except Exception as e:
            self.logger.error(f"Error in feature pipeline: {e}")
            self._metrics['errors_count'] += 1
            return pd.DataFrame()
    
    async def _generate_features_safe(self, generator: Any, data: Union[pd.DataFrame, Dict[str, Any]], 
                                    name: str) -> pd.DataFrame:
        """
        Безопасная генерация признаков с обработкой ошибок
        
        Args:
            generator: Генератор признаков
            data: Входные данные
            name: Имя генератора
            
        Returns:
            pd.DataFrame: DataFrame с признаками
        """
        try:
            return await generator.generate_features_with_cache(data)
        except Exception as e:
            self.logger.error(f"Error in generator {name}: {e}")
            return pd.DataFrame()
    
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
    
    def _get_relevant_generators(self, data_type: str) -> Dict[str, Any]:
        """
        Получение релевантных генераторов для типа данных
        
        Args:
            data_type: Тип данных
            
        Returns:
            Dict: Релевантные генераторы
        """
        relevant = {}
        
        for name, generator in self.feature_generators.items():
            required_types = generator.get_required_data_types()
            
            if data_type in required_types or 'all' in required_types:
                relevant[name] = generator
        
        return relevant
    
    def _combine_feature_results(self, feature_results: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Объединение результатов генерации признаков
        
        Args:
            feature_results: Список DataFrame с признаками
            
        Returns:
            pd.DataFrame: Объединенный DataFrame
        """
        try:
            # Фильтрация пустых результатов
            valid_results = [df for df in feature_results if isinstance(df, pd.DataFrame) and not df.empty]
            
            if not valid_results:
                return pd.DataFrame()
            
            # Объединение по индексу
            combined = pd.concat(valid_results, axis=1, join='outer')
            
            # Удаление дубликатов столбцов
            combined = combined.loc[:, ~combined.columns.duplicated()]
            
            # Заполнение NaN значений
            combined = combined.fillna(0)
            
            return combined
            
        except Exception as e:
            self.logger.error(f"Error combining feature results: {e}")
            return pd.DataFrame()
    
    def _update_metrics(self, features: pd.DataFrame, processing_time: float) -> None:
        """
        Обновление метрик производительности
        
        Args:
            features: DataFrame с признаками
            processing_time: Время обработки
        """
        self._metrics['pipeline_runs'] += 1
        self._metrics['processing_time'] += processing_time
        
        if not features.empty:
            self._metrics['total_features_generated'] += len(features.columns)
    
    async def process_batch(self, data_batch: List[Dict[str, Any]], 
                          data_type: str) -> pd.DataFrame:
        """
        Пакетная обработка данных
        
        Args:
            data_batch: Пакет данных
            data_type: Тип данных
            
        Returns:
            pd.DataFrame: DataFrame с признаками
        """
        try:
            if not data_batch:
                return pd.DataFrame()
            
            # Конвертация в DataFrame
            df = pd.DataFrame(data_batch)
            
            # Обработка через основной пайплайн
            return await self.process_data(df, data_type)
            
        except Exception as e:
            self.logger.error(f"Error in batch processing: {e}")
            return pd.DataFrame()
    
    def get_feature_names(self, data_type: str) -> List[str]:
        """
        Получение списка имен признаков для типа данных
        
        Args:
            data_type: Тип данных
            
        Returns:
            List[str]: Имена признаков
        """
        relevant_generators = self._get_relevant_generators(data_type)
        feature_names = []
        
        for generator in relevant_generators.values():
            if generator.is_enabled():
                feature_names.extend(generator.get_feature_names())
        
        return list(set(feature_names))  # Удаление дубликатов
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Получение метрик производительности
        
        Returns:
            Dict: Метрики
        """
        metrics = self._metrics.copy()
        
        # Добавление производных метрик
        if metrics['pipeline_runs'] > 0:
            metrics['avg_processing_time'] = metrics['processing_time'] / metrics['pipeline_runs']
            metrics['avg_features_per_run'] = metrics['total_features_generated'] / metrics['pipeline_runs']
        else:
            metrics['avg_processing_time'] = 0.0
            metrics['avg_features_per_run'] = 0.0
        
        # Метрики генераторов
        metrics['generators'] = {}
        for name, generator in self.feature_generators.items():
            metrics['generators'][name] = generator.get_metrics()
        
        return metrics
    
    def reset_metrics(self) -> None:
        """Сброс метрик"""
        self._metrics = {
            'total_features_generated': 0,
            'pipeline_runs': 0,
            'errors_count': 0,
            'processing_time': 0.0
        }
        
        # Сброс метрик генераторов
        for generator in self.feature_generators.values():
            generator.reset_metrics()
    
    def clear_caches(self) -> None:
        """Очистка кэшей всех генераторов"""
        for generator in self.feature_generators.values():
            generator.clear_cache()
        
        self.logger.info("All feature generator caches cleared")
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        # Очистка ресурсов
        self.executor.shutdown(wait=True)
        self.clear_caches()
