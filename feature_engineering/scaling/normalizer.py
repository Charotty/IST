"""
Feature Normalizer

Нормализация признаков.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
import os
from pathlib import Path


class FeatureNormalizer:
    """Нормализатор признаков"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Инициализация нормализатора
        
        Args:
            config: Конфигурация
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Параметры
        self.method = config.get('method', 'l2')  # l1, l2, max
        self.apply_pca = config.get('apply_pca', False)
        self.pca_components = config.get('pca_components', None)
        self.variance_threshold = config.get('variance_threshold', 0.95)
        
        # Нормализатор и PCA
        self._normalizer = None
        self._pca = None
        
        # Метрики
        self._metrics = {
            'features_normalized': 0,
            'pca_applied': 0,
            'errors_count': 0
        }
        
        # Инициализация
        self._initialize_components()
    
    def _initialize_components(self) -> None:
        """Инициализация компонентов"""
        try:
            # Создание нормализатора
            self._normalizer = Normalizer(norm=self.method)
            
            # Создание PCA если нужно
            if self.apply_pca:
                self._pca = PCA(
                    n_components=self.pca_components,
                    svd_solver='auto' if self.pca_components is not None else 'full'
                )
            
            self.logger.info(f"Initialized normalizer with method: {self.method}")
            
        except Exception as e:
            self.logger.error(f"Error initializing components: {e}")
    
    async def fit(self, data: pd.DataFrame) -> None:
        """
        Обучение нормализатора и PCA
        
        Args:
            data: Данные для обучения
        """
        try:
            if data.empty:
                self.logger.warning("Empty data for fitting normalizer")
                return
            
            # Фильтрация числовых признаков
            numeric_data = data.select_dtypes(include=[np.number])
            
            if numeric_data.empty:
                self.logger.warning("No numeric features for normalization")
                return
            
            # Обучение нормализатора
            self._normalizer.fit(numeric_data)
            
            # Обучение PCA если нужно
            if self.apply_pca and self._pca:
                # Сначала нормализуем данные
                normalized_data = self._normalizer.transform(numeric_data)
                
                # Обучаем PCA
                self._pca.fit(normalized_data)
                
                # Автоматический выбор числа компонентов по дисперсии
                if self.pca_components is None:
                    cumulative_variance = np.cumsum(self._pca.explained_variance_ratio_)
                    n_components = np.argmax(cumulative_variance >= self.variance_threshold) + 1
                    self._pca.n_components = n_components
                    
                    # Переобучение с новым числом компонентов
                    self._pca.fit(normalized_data)
                
                self._metrics['pca_applied'] += 1
                self.logger.info(f"Fitted PCA with {self._pca.n_components_} components")
            
            self.logger.info(f"Fitted normalizer on {numeric_data.shape[1]} features")
            
        except Exception as e:
            self.logger.error(f"Error fitting normalizer: {e}")
            self._metrics['errors_count'] += 1
    
    async def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Нормализация данных
        
        Args:
            data: Данные для нормализации
            
        Returns:
            pd.DataFrame: Нормализованные данные
        """
        try:
            if data.empty:
                return data
            
            # Проверка наличия обученного нормализатора
            if self._normalizer is None:
                self.logger.warning("Normalizer not fitted, fitting on data")
                await self.fit(data)
            
            # Фильтрация числовых признаков
            numeric_data = data.select_dtypes(include=[np.number])
            
            if numeric_data.empty:
                self.logger.warning("No numeric features for normalization")
                return data
            
            # Нормализация
            normalized_values = self._normalizer.transform(numeric_data)
            
            # Создание результата
            result = data.copy()
            result[numeric_data.columns] = normalized_values
            
            # Применение PCA если нужно
            if self.apply_pca and self._pca:
                pca_values = self._pca.transform(normalized_values)
                
                # Создание PCA признаков
                pca_columns = [f'pca_{i}' for i in range(pca_values.shape[1])]
                pca_df = pd.DataFrame(pca_values, columns=pca_columns, index=data.index)
                
                # Объединение с исходными данными (заменяем числовые признаки)
                result = pd.concat([result.drop(columns=numeric_data.columns), pca_df], axis=1)
            
            self._metrics['features_normalized'] += numeric_data.shape[1]
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error normalizing data: {e}")
            self._metrics['errors_count'] += 1
            return data
    
    async def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Обучение и нормализация данных
        
        Args:
            data: Данные
            
        Returns:
            pd.DataFrame: Нормализованные данные
        """
        await self.fit(data)
        return await self.transform(data)
    
    def get_pca_info(self) -> Dict[str, Any]:
        """
        Получение информации о PCA
        
        Returns:
            Dict: Информация о PCA
        """
        if not self.apply_pca or self._pca is None:
            return {'pca_applied': False}
        
        info = {
            'pca_applied': True,
            'n_components': self._pca.n_components_,
            'explained_variance_ratio': self._pca.explained_variance_ratio_.tolist(),
            'cumulative_variance_ratio': np.cumsum(self._pca.explained_variance_ratio_).tolist(),
            'singular_values': self._pca.singular_values_.tolist()
        }
        
        return info
    
    def get_normalization_info(self) -> Dict[str, Any]:
        """
        Получение информации о нормализации
        
        Returns:
            Dict: Информация о нормализации
        """
        info = {
            'method': self.method,
            'pca_applied': self.apply_pca,
            'pca_components': self.pca_components if self.apply_pca else None
        }
        
        if self.apply_pca and self._pca:
            pca_info = self.get_pca_info()
            info.update(pca_info)
        
        return info
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Получение метрик
        
        Returns:
            Dict: Метрики
        """
        return self._metrics.copy()
    
    def reset_metrics(self) -> None:
        """Сброс метрик"""
        self._metrics = {
            'features_normalized': 0,
            'pca_applied': 0,
            'errors_count': 0
        }
    
    def clear_components(self) -> None:
        """Очистка компонентов"""
        self._normalizer = None
        self._pca = None
        self._initialize_components()
        self.logger.info("Cleared normalization components")
