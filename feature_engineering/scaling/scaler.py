"""
Feature Scaler

Масштабирование признаков.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from joblib import dump, load
import os
from pathlib import Path


class FeatureScaler:
    """Масштабирование признаков"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Инициализация масштабатора
        
        Args:
            config: Конфигурация
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Параметры
        self.method = config.get('method', 'standard')  # standard, minmax, robust
        self.feature_range = config.get('feature_range', [0, 1])
        self.save_scalers = config.get('save_scalers', True)
        self.scalers_path = config.get('scalers_path', './scalers')
        
        # Словарь обученных масштабаторов
        self._scalers = {}
        
        # Метрики
        self._metrics = {
            'features_scaled': 0,
            'scalers_trained': 0,
            'errors_count': 0
        }
        
        # Создание директории для сохранения масштабаторов
        if self.save_scalers:
            Path(self.scalers_path).mkdir(parents=True, exist_ok=True)
    
    def _create_scaler(self) -> Any:
        """
        Создание масштабатора в зависимости от метода
        
        Returns:
            Scaler: Объект масштабатора
        """
        if self.method == 'standard':
            return StandardScaler()
        elif self.method == 'minmax':
            return MinMaxScaler(feature_range=self.feature_range)
        elif self.method == 'robust':
            return RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {self.method}")
    
    async def fit(self, data: pd.DataFrame, feature_names: Optional[List[str]] = None) -> None:
        """
        Обучение масштабатора на данных
        
        Args:
            data: Данные для обучения
            feature_names: Имена признаков для масштабирования
        """
        try:
            if data.empty:
                self.logger.warning("Empty data for fitting scaler")
                return
            
            # Определение признаков для масштабирования
            if feature_names is None:
                # Исключаем нечисловые признаки
                numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
                feature_names = numeric_features
            
            # Фильтрация существующих признаков
            valid_features = [f for f in feature_names if f in data.columns]
            
            if not valid_features:
                self.logger.warning("No valid features for scaling")
                return
            
            # Создание и обучение масштабатора
            scaler = self._create_scaler()
            scaler.fit(data[valid_features])
            
            # Сохранение масштабатора
            self._scalers['default'] = scaler
            
            # Сохранение на диск
            if self.save_scalers:
                self._save_scaler(scaler, 'default')
            
            self._metrics['scalers_trained'] += 1
            self.logger.info(f"Fitted scaler on {len(valid_features)} features")
            
        except Exception as e:
            self.logger.error(f"Error fitting scaler: {e}")
            self._metrics['errors_count'] += 1
    
    async def transform(self, data: pd.DataFrame, 
                       feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Масштабирование данных
        
        Args:
            data: Данные для масштабирования
            feature_names: Имена признаков для масштабирования
            
        Returns:
            pd.DataFrame: Масштабированные данные
        """
        try:
            if data.empty:
                self.logger.warning("Empty data for scaling")
                return data
            
            # Проверка наличия обученного масштабатора
            if 'default' not in self._scalers:
                self.logger.warning("No fitted scaler found, fitting on data")
                await self.fit(data, feature_names)
            
            scaler = self._scalers['default']
            
            # Определение признаков для масштабирования
            if feature_names is None:
                numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
                feature_names = numeric_features
            
            # Фильтрация существующих признаков
            valid_features = [f for f in feature_names if f in data.columns]
            
            if not valid_features:
                self.logger.warning("No valid features for scaling")
                return data
            
            # Создание копии данных
            scaled_data = data.copy()
            
            # Масштабирование
            scaled_values = scaler.transform(data[valid_features])
            scaled_data[valid_features] = scaled_values
            
            self._metrics['features_scaled'] += len(valid_features)
            
            return scaled_data
            
        except Exception as e:
            self.logger.error(f"Error scaling data: {e}")
            self._metrics['errors_count'] += 1
            return data
    
    async def fit_transform(self, data: pd.DataFrame, 
                          feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Обучение и масштабирование данных
        
        Args:
            data: Данные
            feature_names: Имена признаков
            
        Returns:
            pd.DataFrame: Масштабированные данные
        """
        await self.fit(data, feature_names)
        return await self.transform(data, feature_names)
    
    async def inverse_transform(self, data: pd.DataFrame, 
                             feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Обратное преобразование
        
        Args:
            data: Масштабированные данные
            feature_names: Имена признаков
            
        Returns:
            pd.DataFrame: Восстановленные данные
        """
        try:
            if data.empty:
                return data
            
            if 'default' not in self._scalers:
                self.logger.warning("No fitted scaler found")
                return data
            
            scaler = self._scalers['default']
            
            # Определение признаков
            if feature_names is None:
                feature_names = data.columns.tolist()
            
            # Фильтрация существующих признаков
            valid_features = [f for f in feature_names if f in data.columns]
            
            if not valid_features:
                return data
            
            # Обратное преобразование
            restored_data = data.copy()
            restored_values = scaler.inverse_transform(data[valid_features])
            restored_data[valid_features] = restored_values
            
            return restored_data
            
        except Exception as e:
            self.logger.error(f"Error inverse transforming data: {e}")
            return data
    
    def _save_scaler(self, scaler: Any, name: str) -> None:
        """
        Сохранение масштабатора на диск
        
        Args:
            scaler: Масштабатор
            name: Имя масштабатора
        """
        try:
            scaler_path = os.path.join(self.scalers_path, f"{name}_{self.method}.joblib")
            dump(scaler, scaler_path)
            self.logger.info(f"Saved scaler to {scaler_path}")
        except Exception as e:
            self.logger.error(f"Error saving scaler: {e}")
    
    def _load_scaler(self, name: str) -> Optional[Any]:
        """
        Загрузка масштабатора с диска
        
        Args:
            name: Имя масштабатора
            
        Returns:
            Scaler: Загруженный масштабатор или None
        """
        try:
            scaler_path = os.path.join(self.scalers_path, f"{name}_{self.method}.joblib")
            
            if os.path.exists(scaler_path):
                scaler = load(scaler_path)
                self.logger.info(f"Loaded scaler from {scaler_path}")
                return scaler
            else:
                self.logger.warning(f"Scaler file not found: {scaler_path}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error loading scaler: {e}")
            return None
    
    async def load_scalers(self) -> None:
        """Загрузка всех сохраненных масштабаторов"""
        try:
            default_scaler = self._load_scaler('default')
            if default_scaler:
                self._scalers['default'] = default_scaler
                self.logger.info("Loaded default scaler")
            
        except Exception as e:
            self.logger.error(f"Error loading scalers: {e}")
    
    def get_scaler_info(self) -> Dict[str, Any]:
        """
        Получение информации о масштабаторах
        
        Returns:
            Dict: Информация о масштабаторах
        """
        info = {
            'method': self.method,
            'feature_range': self.feature_range,
            'trained_scalers': len(self._scalers),
            'scalers': list(self._scalers.keys())
        }
        
        # Добавление информации о каждом масштабаторе
        for name, scaler in self._scalers.items():
            if hasattr(scaler, 'mean_'):
                info[f'{name}_mean'] = scaler.mean_.tolist() if hasattr(scaler.mean_, 'tolist') else scaler.mean_
            if hasattr(scaler, 'scale_'):
                info[f'{name}_scale'] = scaler.scale_.tolist() if hasattr(scaler.scale_, 'tolist') else scaler.scale_
        
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
            'features_scaled': 0,
            'scalers_trained': 0,
            'errors_count': 0
        }
    
    def clear_scalers(self) -> None:
        """Очистка всех масштабаторов"""
        self._scalers.clear()
        
        # Удаление файлов масштабаторов
        if self.save_scalers and os.path.exists(self.scalers_path):
            for file in os.listdir(self.scalers_path):
                if file.endswith('.joblib'):
                    os.remove(os.path.join(self.scalers_path, file))
        
        self.logger.info("Cleared all scalers")
