"""
Feature Manager

Главный менеджер признаков для координации всех компонентов.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import pandas as pd
import numpy as np

from .feature_pipeline import FeaturePipeline
from .scaling.scaler import FeatureScaler
from .scaling.normalizer import FeatureNormalizer


class FeatureManager:
    """Главный менеджер признаков"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Инициализация менеджера признаков
        
        Args:
            config: Конфигурация
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Инициализация компонентов
        self.pipeline = None
        self.scaler = None
        self.normalizer = None
        
        # Состояние системы
        self._running = False
        self._start_time = None
        
        # Метрики
        self._metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'errors_count': 0,
            'features_generated': 0,
            'processing_time': 0.0
        }
        
        # Кэш для сгенерированных признаков
        self._feature_cache = {}
        
        # Инициализация
        self._initialize_components()
    
    def _initialize_components(self) -> None:
        """Инициализация компонентов"""
        try:
            # Инициализация пайплайна
            feature_config = self.config.get('feature_engineering', self.config)
            self.pipeline = FeaturePipeline(feature_config)
            
            # Инициализация масштабатора
            feature_engineering_config = self.config.get('feature_engineering', {})
            scaling_config = feature_engineering_config.get('scaling', {})
            if scaling_config:
                self.scaler = FeatureScaler(scaling_config)
            
            # Инициализация нормализатора
            normalization_config = feature_engineering_config.get('normalization', {})
            if normalization_config:
                self.normalizer = FeatureNormalizer(normalization_config)
            
            self.logger.info("Feature Manager initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing Feature Manager: {e}")
            raise
    
    async def start(self) -> None:
        """Запуск менеджера признаков"""
        try:
            self._running = True
            self._start_time = datetime.utcnow()
            
            # Загрузка сохраненных масштабаторов
            if self.scaler:
                await self.scaler.load_scalers()
            
            self.logger.info("Feature Manager started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start Feature Manager: {e}")
            await self.stop()
            raise
    
    async def stop(self) -> None:
        """Остановка менеджера признаков"""
        try:
            self._running = False
            
            # Очистка кэшей
            self._feature_cache.clear()
            
            # Очистка компонентов
            if self.pipeline:
                self.pipeline.clear_caches()
            
            self.logger.info("Feature Manager stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping Feature Manager: {e}")
    
    async def generate_features(self, data: Union[pd.DataFrame, Dict[str, Any]], 
                              data_type: str, apply_scaling: bool = True,
                              apply_normalization: bool = False) -> pd.DataFrame:
        """
        Генерация признаков
        
        Args:
            data: Входные данные
            data_type: Тип данных
            apply_scaling: Применять масштабирование
            apply_normalization: Применять нормализацию
            
        Returns:
            pd.DataFrame: DataFrame с признаками
        """
        try:
            start_time = asyncio.get_event_loop().time()
            
            self._metrics['total_requests'] += 1
            
            # Проверка состояния
            if not self._running:
                self.logger.warning("Feature Manager not running")
                return pd.DataFrame()
            
            # Генерация признаков через пайплайн
            features = await self.pipeline.process_data(data, data_type)
            
            if features.empty:
                self.logger.warning("No features generated")
                return pd.DataFrame()
            
            # Применение масштабирования
            if apply_scaling and self.scaler:
                features = await self.scaler.transform(features)
            
            # Применение нормализации
            if apply_normalization and self.normalizer:
                features = await self.normalizer.transform(features)
            
            # Обновление метрик
            processing_time = asyncio.get_event_loop().time() - start_time
            self._update_metrics(len(features.columns), processing_time, success=True)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error generating features: {e}")
            self._update_metrics(0, 0, success=False)
            return pd.DataFrame()
    
    async def generate_features_batch(self, data_batch: List[Dict[str, Any]], 
                                   data_type: str, apply_scaling: bool = True,
                                   apply_normalization: bool = False) -> pd.DataFrame:
        """
        Пакетная генерация признаков
        
        Args:
            data_batch: Пакет данных
            data_type: Тип данных
            apply_scaling: Применять масштабирование
            apply_normalization: Применять нормализацию
            
        Returns:
            pd.DataFrame: DataFrame с признаками
        """
        try:
            if not data_batch:
                return pd.DataFrame()
            
            # Обработка через пайплайн
            features = await self.pipeline.process_batch(data_batch, data_type)
            
            if features.empty:
                return pd.DataFrame()
            
            # Применение масштабирования
            if apply_scaling and self.scaler:
                features = await self.scaler.transform(features)
            
            # Применение нормализации
            if apply_normalization and self.normalizer:
                features = await self.normalizer.transform(features)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error in batch feature generation: {e}")
            return pd.DataFrame()
    
    async def fit_scalers(self, data: Union[pd.DataFrame, List[Dict[str, Any]]]) -> None:
        """
        Обучение масштабировщиков на данных
        
        Args:
            data: Данные для обучения
        """
        try:
            if not self.scaler:
                self.logger.warning("No scaler configured")
                return
            
            # Конвертация в DataFrame если необходимо
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = data
            
            await self.scaler.fit(df)
            self.logger.info("Scalers fitted successfully")
            
        except Exception as e:
            self.logger.error(f"Error fitting scalers: {e}")
    
    async def fit_normalizer(self, data: Union[pd.DataFrame, List[Dict[str, Any]]]) -> None:
        """
        Обучение нормализатора на данных
        
        Args:
            data: Данные для обучения
        """
        try:
            if not self.normalizer:
                self.logger.warning("No normalizer configured")
                return
            
            # Конвертация в DataFrame если необходимо
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = data
            
            await self.normalizer.fit(df)
            self.logger.info("Normalizer fitted successfully")
            
        except Exception as e:
            self.logger.error(f"Error fitting normalizer: {e}")
    
    def get_feature_names(self, data_type: str) -> List[str]:
        """
        Получение имен признаков для типа данных
        
        Args:
            data_type: Тип данных
            
        Returns:
            List[str]: Имена признаков
        """
        if self.pipeline:
            return self.pipeline.get_feature_names(data_type)
        return []
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Получение метрик производительности
        
        Returns:
            Dict: Метрики
        """
        metrics = self._metrics.copy()
        
        # Добавление uptime
        if self._start_time:
            metrics['uptime'] = (datetime.utcnow() - self._start_time).total_seconds()
        else:
            metrics['uptime'] = 0.0
        
        # Метрики компонентов
        if self.pipeline:
            metrics['pipeline'] = self.pipeline.get_metrics()
        
        if self.scaler:
            metrics['scaler'] = self.scaler.get_metrics()
        
        if self.normalizer:
            metrics['normalizer'] = self.normalizer.get_metrics()
        
        # Кэш метрики
        metrics['cache_size'] = len(self._feature_cache)
        
        return metrics
    
    def _update_metrics(self, features_count: int, processing_time: float, success: bool) -> None:
        """
        Обновление метрик
        
        Args:
            features_count: Количество признаков
            processing_time: Время обработки
            success: Успешность операции
        """
        if success:
            self._metrics['successful_requests'] += 1
            self._metrics['features_generated'] += features_count
        else:
            self._metrics['errors_count'] += 1
        
        self._metrics['processing_time'] += processing_time
    
    def reset_metrics(self) -> None:
        """Сброс метрик"""
        self._metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'errors_count': 0,
            'features_generated': 0,
            'processing_time': 0.0
        }
        
        # Сброс метрик компонентов
        if self.pipeline:
            self.pipeline.reset_metrics()
        
        if self.scaler:
            self.scaler.reset_metrics()
        
        if self.normalizer:
            self.normalizer.reset_metrics()
    
    def clear_caches(self) -> None:
        """Очистка всех кэшей"""
        self._feature_cache.clear()
        
        if self.pipeline:
            self.pipeline.clear_caches()
        
        self.logger.info("All caches cleared")
    
    def is_running(self) -> bool:
        """Проверка состояния работы"""
        return self._running
    
    def get_status(self) -> Dict[str, Any]:
        """
        Получение статуса менеджера
        
        Returns:
            Dict: Статус
        """
        status = {
            'running': self._running,
            'uptime': (datetime.utcnow() - self._start_time).total_seconds() if self._start_time else 0,
            'components': {
                'pipeline': self.pipeline is not None,
                'scaler': self.scaler is not None,
                'normalizer': self.normalizer is not None
            },
            'metrics': self.get_metrics()
        }
        
        return status
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.stop()
