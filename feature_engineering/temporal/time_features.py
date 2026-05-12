"""
Time-based Features

Признаки на основе времени.
"""

import asyncio
import logging
from typing import Dict, Any, List, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from ..base_feature_generator import BaseFeatureGenerator


class TimeBasedFeatures(BaseFeatureGenerator):
    """Генератор временных признаков"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Инициализация генератора временных признаков
        
        Args:
            config: Конфигурация
        """
        super().__init__(config)
        
        # Параметры
        self.seasonal_features = config.get('seasonal_features', True)
        self.timezone = config.get('timezone', 'UTC')
        
        # Имена признаков
        self._feature_names = self._generate_feature_names()
    
    def _generate_feature_names(self) -> List[str]:
        """Генерация имен признаков"""
        names = [
            "hour_of_day",
            "day_of_week",
            "day_of_month",
            "month_of_year",
            "quarter_of_year",
            "is_weekend",
            "is_trading_hours"
        ]
        
        if self.seasonal_features:
            names.extend([
                "sin_hour",
                "cos_hour",
                "sin_day",
                "cos_day",
                "sin_month",
                "cos_month"
            ])
        
        # Сессионные признаки
        names.extend([
            "is_asian_session",
            "is_european_session", 
            "is_american_session",
            "is_overlap_session"
        ])
        
        return names
    
    def _validate_config(self) -> None:
        """Валидация конфигурации"""
        if not isinstance(self.seasonal_features, bool):
            raise ValueError("seasonal_features must be boolean")
        
        if not isinstance(self.timezone, str):
            raise ValueError("timezone must be string")
    
    async def generate_features(self, data: Union[pd.DataFrame, Dict[str, Any]]) -> pd.DataFrame:
        """
        Генерация временных признаков
        
        Args:
            data: Данные с временными метками
            
        Returns:
            pd.DataFrame: DataFrame с временными признаками
        """
        try:
            # Конвертация в DataFrame если необходимо
            if isinstance(data, dict):
                df = pd.DataFrame([data])
            else:
                df = data.copy()
            
            # Проверка наличия временных меток
            timestamp_field = 'timestamp'
            if timestamp_field not in df.columns:
                self.logger.warning(f"Timestamp field '{timestamp_field}' not found in data")
                return pd.DataFrame()
            
            # Конвертация временных меток
            if not pd.api.types.is_datetime64_any_dtype(df[timestamp_field]):
                df[timestamp_field] = pd.to_datetime(df[timestamp_field])
            
            features = pd.DataFrame(index=df.index)
            
            # Расчет базовых временных признаков
            for idx, row in df.iterrows():
                timestamp = row[timestamp_field]
                row_features = self._calculate_time_features(timestamp)
                
                for feature_name, value in row_features.items():
                    if feature_name not in features.columns:
                        features[feature_name] = np.nan
                    features.at[idx, feature_name] = value
            
            return features
            
        except Exception as e:
            self._handle_error(e, "generate_features")
            return pd.DataFrame()
    
    def _calculate_time_features(self, timestamp: datetime) -> Dict[str, float]:
        """
        Расчет временных признаков для одной временной метки
        
        Args:
            timestamp: Временная метка
            
        Returns:
            Dict: Временные признаки
        """
        features = {}
        
        try:
            # Базовые временные признаки
            features["hour_of_day"] = float(timestamp.hour)
            features["day_of_week"] = float(timestamp.dayofweek)
            features["day_of_month"] = float(timestamp.day)
            features["month_of_year"] = float(timestamp.month)
            features["quarter_of_year"] = float((timestamp.month - 1) // 3 + 1)
            
            # Признаки выходного дня
            features["is_weekend"] = 1.0 if timestamp.dayofweek >= 5 else 0.0
            
            # Признаки торговых часов (9:00 - 16:00)
            features["is_trading_hours"] = 1.0 if 9 <= timestamp.hour < 16 else 0.0
            
            # Сессионные признаки (UTC время)
            features["is_asian_session"] = 1.0 if 0 <= timestamp.hour < 9 else 0.0
            features["is_european_session"] = 1.0 if 7 <= timestamp.hour < 16 else 0.0
            features["is_american_session"] = 1.0 if 13 <= timestamp.hour < 22 else 0.0
            
            # Пересечение сессий
            overlap_conditions = [
                (7 <= timestamp.hour < 9),   # Asia-Europe overlap
                (13 <= timestamp.hour < 16)  # Europe-America overlap
            ]
            features["is_overlap_session"] = 1.0 if any(overlap_conditions) else 0.0
            
            # Сезонные признаки (циклические кодировки)
            if self.seasonal_features:
                # Часовые циклические признаки
                hour_rad = 2 * np.pi * timestamp.hour / 24
                features["sin_hour"] = np.sin(hour_rad)
                features["cos_hour"] = np.cos(hour_rad)
                
                # Дневные циклические признаки
                day_rad = 2 * np.pi * timestamp.dayofweek / 7
                features["sin_day"] = np.sin(day_rad)
                features["cos_day"] = np.cos(day_rad)
                
                # Месячные циклические признаки
                month_rad = 2 * np.pi * timestamp.month / 12
                features["sin_month"] = np.sin(month_rad)
                features["cos_month"] = np.cos(month_rad)
            
        except Exception as e:
            self.logger.error(f"Error calculating time features: {e}")
            # Возвращаем нулевые значения при ошибке
            for name in self._feature_names:
                features[name] = 0.0
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Получение имен признаков"""
        return self._feature_names.copy()
    
    def get_required_data_types(self) -> List[str]:
        """Получение требуемых типов данных"""
        return ['ohlcv', 'trades', 'orderbook']
