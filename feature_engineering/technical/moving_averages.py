"""
Moving Averages

Расчет скользящих средних.
"""

import asyncio
import logging
from typing import Dict, Any, List, Union
import pandas as pd
import numpy as np
from ..base_feature_generator import BaseFeatureGenerator


class MovingAverages(BaseFeatureGenerator):
    """Генератор скользящих средних"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Инициализация генератора скользящих средних
        
        Args:
            config: Конфигурация
        """
        super().__init__(config)
        
        # Параметры
        self.periods = config.get('periods', [5, 10, 20, 50, 200])
        self.types = config.get('types', ['SMA', 'EMA'])
        self.price_field = config.get('price_field', 'close')
        
        # Имена признаков
        self._feature_names = []
        for period in self.periods:
            for ma_type in self.types:
                self._feature_names.append(f"{ma_type}_{period}")
    
    def _validate_config(self) -> None:
        """Валидация конфигурации"""
        if not self.periods or not isinstance(self.periods, list):
            raise ValueError("periods must be a non-empty list")
        
        if not all(isinstance(p, int) and p > 0 for p in self.periods):
            raise ValueError("All periods must be positive integers")
        
        if not self.types or not isinstance(self.types, list):
            raise ValueError("types must be a non-empty list")
        
        valid_types = ['SMA', 'EMA', 'WMA']
        if not all(t in valid_types for t in self.types):
            raise ValueError(f"types must be one of {valid_types}")
    
    async def generate_features(self, data: Union[pd.DataFrame, Dict[str, Any]]) -> pd.DataFrame:
        """
        Генерация скользящих средних
        
        Args:
            data: OHLCV данные
            
        Returns:
            pd.DataFrame: DataFrame со скользящими средними
        """
        try:
            # Конвертация в DataFrame если необходимо
            if isinstance(data, dict):
                df = pd.DataFrame([data])
            else:
                df = data.copy()
            
            # Проверка наличия нужного поля
            if self.price_field not in df.columns:
                self.logger.warning(f"Price field '{self.price_field}' not found in data")
                return pd.DataFrame()
            
            # Проверка достаточного количества данных
            min_period = min(self.periods)
            if len(df) < min_period:
                self.logger.warning(f"Insufficient data: need {min_period}, got {len(df)}")
                return pd.DataFrame()
            
            features = pd.DataFrame(index=df.index)
            
            # Расчет скользящих средних
            for period in self.periods:
                if len(df) < period:
                    continue
                
                for ma_type in self.types:
                    feature_name = f"{ma_type}_{period}"
                    
                    try:
                        if ma_type == 'SMA':
                            ma_values = self._calculate_sma(df[self.price_field], period)
                        elif ma_type == 'EMA':
                            ma_values = self._calculate_ema(df[self.price_field], period)
                        elif ma_type == 'WMA':
                            ma_values = self._calculate_wma(df[self.price_field], period)
                        else:
                            continue
                        
                        features[feature_name] = ma_values
                        
                    except Exception as e:
                        self.logger.error(f"Error calculating {feature_name}: {e}")
                        continue
            
            return features
            
        except Exception as e:
            self._handle_error(e, "generate_features")
            return pd.DataFrame()
    
    def _calculate_sma(self, prices: pd.Series, period: int) -> pd.Series:
        """
        Расчет Simple Moving Average
        
        Args:
            prices: Цены
            period: Период
            
        Returns:
            pd.Series: SMA значения
        """
        return prices.rolling(window=period, min_periods=1).mean()
    
    def _calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """
        Расчет Exponential Moving Average
        
        Args:
            prices: Цены
            period: Период
            
        Returns:
            pd.Series: EMA значения
        """
        alpha = 2 / (period + 1)
        return prices.ewm(alpha=alpha, adjust=False).mean()
    
    def _calculate_wma(self, prices: pd.Series, period: int) -> pd.Series:
        """
        Расчет Weighted Moving Average
        
        Args:
            prices: Цены
            period: Период
            
        Returns:
            pd.Series: WMA значения
        """
        weights = np.arange(1, period + 1)
        weights = weights / weights.sum()
        
        return prices.rolling(window=period, min_periods=1).apply(
            lambda x: np.sum(x * weights), raw=True
        )
    
    def get_feature_names(self) -> List[str]:
        """Получение имен признаков"""
        return self._feature_names.copy()
    
    def get_required_data_types(self) -> List[str]:
        """Получение требуемых типов данных"""
        return ['ohlcv']
