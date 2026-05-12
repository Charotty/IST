"""
Volatility Indicators

Индикаторы волатильности: Bollinger Bands, ATR.
"""

import asyncio
import logging
from typing import Dict, Any, List, Union, Tuple
import pandas as pd
import numpy as np
from ..base_feature_generator import BaseFeatureGenerator


class VolatilityIndicators(BaseFeatureGenerator):
    """Генератор индикаторов волатильности"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Инициализация генератора индикаторов волатильности
        
        Args:
            config: Конфигурация
        """
        super().__init__(config)
        
        # Параметры Bollinger Bands
        self.bb_periods = config.get('bollinger_periods', [20])
        self.bb_std = config.get('bollinger_std', [2.0])
        
        # Параметры ATR
        self.atr_periods = config.get('atr_periods', [14])
        
        # Поля цен
        self.close_field = config.get('close_field', 'close')
        self.high_field = config.get('high_field', 'high')
        self.low_field = config.get('low_field', 'low')
        
        # Имена признаков
        self._feature_names = self._generate_feature_names()
    
    def _generate_feature_names(self) -> List[str]:
        """Генерация имен признаков"""
        names = []
        
        # Bollinger Bands
        for period in self.bb_periods:
            for std in self.bb_std:
                names.extend([
                    f"BB_upper_{period}_{std}",
                    f"BB_middle_{period}_{std}",
                    f"BB_lower_{period}_{std}",
                    f"BB_width_{period}_{std}",
                    f"BB_position_{period}_{std}"
                ])
        
        # ATR
        for period in self.atr_periods:
            names.append(f"ATR_{period}")
        
        return names
    
    def _validate_config(self) -> None:
        """Валидация конфигурации"""
        # Bollinger Bands periods
        if not self.bb_periods or not isinstance(self.bb_periods, list):
            raise ValueError("bollinger_periods must be a non-empty list")
        
        if not all(isinstance(p, int) and p > 0 for p in self.bb_periods):
            raise ValueError("All bollinger_periods must be positive integers")
        
        # Bollinger Bands std
        if not self.bb_std or not isinstance(self.bb_std, list):
            raise ValueError("bollinger_std must be a non-empty list")
        
        if not all(isinstance(s, (int, float)) and s > 0 for s in self.bb_std):
            raise ValueError("All bollinger_std must be positive numbers")
        
        # ATR periods
        if not self.atr_periods or not isinstance(self.atr_periods, list):
            raise ValueError("atr_periods must be a non-empty list")
        
        if not all(isinstance(p, int) and p > 0 for p in self.atr_periods):
            raise ValueError("All atr_periods must be positive integers")
    
    async def generate_features(self, data: Union[pd.DataFrame, Dict[str, Any]]) -> pd.DataFrame:
        """
        Генерация индикаторов волатильности
        
        Args:
            data: OHLCV данные
            
        Returns:
            pd.DataFrame: DataFrame с индикаторами
        """
        try:
            # Конвертация в DataFrame если необходимо
            if isinstance(data, dict):
                df = pd.DataFrame([data])
            else:
                df = data.copy()
            
            # Проверка наличия нужных полей
            required_fields = [self.close_field, self.high_field, self.low_field]
            missing_fields = [f for f in required_fields if f not in df.columns]
            if missing_fields:
                self.logger.warning(f"Missing fields: {missing_fields}")
                return pd.DataFrame()
            
            # Проверка достаточного количества данных
            min_required = max(
                max(self.bb_periods),
                max(self.atr_periods)
            )
            
            if len(df) < min_required:
                self.logger.warning(f"Insufficient data: need {min_required}, got {len(df)}")
                return pd.DataFrame()
            
            features = pd.DataFrame(index=df.index)
            
            # Расчет Bollinger Bands
            for period in self.bb_periods:
                if len(df) >= period:
                    for std in self.bb_std:
                        bb_features = self._calculate_bollinger_bands(
                            df[self.close_field], period, std
                        )
                        for name, values in bb_features.items():
                            features[name] = values
            
            # Расчет ATR
            for period in self.atr_periods:
                if len(df) >= period:
                    atr_values = self._calculate_atr(
                        df[self.high_field],
                        df[self.low_field],
                        df[self.close_field],
                        period
                    )
                    features[f"ATR_{period}"] = atr_values
            
            return features
            
        except Exception as e:
            self._handle_error(e, "generate_features")
            return pd.DataFrame()
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int, std: float) -> Dict[str, pd.Series]:
        """
        Расчет Bollinger Bands
        
        Args:
            prices: Цены закрытия
            period: Период
            std: Количество стандартных отклонений
            
        Returns:
            Dict: Словарь с линиями Bollinger Bands
        """
        # Средняя линия (SMA)
        middle = prices.rolling(window=period, min_periods=1).mean()
        
        # Стандартное отклонение
        rolling_std = prices.rolling(window=period, min_periods=1).std()
        
        # Верхняя и нижняя линии
        upper = middle + (rolling_std * std)
        lower = middle - (rolling_std * std)
        
        # Ширина полос
        width = (upper - lower) / middle
        
        # Позиция цены относительно полос
        position = (prices - lower) / (upper - lower)
        
        # Замена NaN значений
        middle = middle.fillna(prices.iloc[0] if len(prices) > 0 else 0)
        upper = upper.fillna(prices.iloc[0] if len(prices) > 0 else 0)
        lower = lower.fillna(prices.iloc[0] if len(prices) > 0 else 0)
        width = width.fillna(0)
        position = position.fillna(0.5)
        
        return {
            f"BB_upper_{period}_{std}": upper,
            f"BB_middle_{period}_{std}": middle,
            f"BB_lower_{period}_{std}": lower,
            f"BB_width_{period}_{std}": width,
            f"BB_position_{period}_{std}": position
        }
    
    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        """
        Расчет Average True Range
        
        Args:
            high: Максимальные цены
            low: Минимальные цены
            close: Цены закрытия
            period: Период
            
        Returns:
            pd.Series: ATR значения
        """
        # Расчет True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Расчет ATR как EMA от True Range
        atr = true_range.ewm(span=period, adjust=False).mean()
        
        # Замена NaN значений
        atr = atr.fillna(0)
        
        return atr
    
    def get_feature_names(self) -> List[str]:
        """Получение имен признаков"""
        return self._feature_names.copy()
    
    def get_required_data_types(self) -> List[str]:
        """Получение требуемых типов данных"""
        return ['ohlcv']
