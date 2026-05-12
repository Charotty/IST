"""
Return Features

Признаки на основе доходностей.
"""

import asyncio
import logging
from typing import Dict, Any, List, Union
import pandas as pd
import numpy as np
from ..base_feature_generator import BaseFeatureGenerator


class ReturnFeatures(BaseFeatureGenerator):
    """Генератор признаков доходностей"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Инициализация генератора доходностей
        
        Args:
            config: Конфигурация
        """
        super().__init__(config)
        
        # Параметры
        self.return_periods = config.get('return_periods', [1, 5, 15, 60])
        self.price_field = config.get('price_field', 'close')
        self.volume_field = config.get('volume_field', 'volume')
        
        # Имена признаков
        self._feature_names = self._generate_feature_names()
    
    def _generate_feature_names(self) -> List[str]:
        """Генерация имен признаков"""
        names = []
        
        for period in self.return_periods:
            names.extend([
                f"return_{period}",
                f"log_return_{period}",
                f"abs_return_{period}",
                f"return_sign_{period}",
                f"volume_weighted_return_{period}"
            ])
        
        # Дополнительные признаки
        names.extend([
            "return_volatility",
            "return_skewness",
            "return_kurtosis",
            "return_momentum",
            "return_acceleration"
        ])
        
        return names
    
    def _validate_config(self) -> None:
        """Валидация конфигурации"""
        if not self.return_periods or not isinstance(self.return_periods, list):
            raise ValueError("return_periods must be a non-empty list")
        
        if not all(isinstance(p, int) and p > 0 for p in self.return_periods):
            raise ValueError("All return_periods must be positive integers")
    
    async def generate_features(self, data: Union[pd.DataFrame, Dict[str, Any]]) -> pd.DataFrame:
        """
        Генерация признаков доходностей
        
        Args:
            data: OHLCV данные
            
        Returns:
            pd.DataFrame: DataFrame с признаками
        """
        try:
            # Конвертация в DataFrame если необходимо
            if isinstance(data, dict):
                df = pd.DataFrame([data])
            else:
                df = data.copy()
            
            # Проверка наличия нужных полей
            if self.price_field not in df.columns:
                self.logger.warning(f"Price field '{self.price_field}' not found in data")
                return pd.DataFrame()
            
            # Проверка достаточного количества данных
            max_period = max(self.return_periods)
            if len(df) < max_period + 1:
                self.logger.warning(f"Insufficient data: need {max_period + 1}, got {len(df)}")
                return pd.DataFrame()
            
            features = pd.DataFrame(index=df.index)
            
            # Расчет базовых доходностей
            for period in self.return_periods:
                if len(df) >= period + 1:
                    period_features = self._calculate_period_returns(df, period)
                    for name, values in period_features.items():
                        features[name] = values
            
            # Расчет статистических признаков доходностей
            if len(df) >= 20:  # Минимальный период для статистики
                self._calculate_return_statistics(features, df)
            
            return features
            
        except Exception as e:
            self._handle_error(e, "generate_features")
            return pd.DataFrame()
    
    def _calculate_period_returns(self, df: pd.DataFrame, period: int) -> Dict[str, pd.Series]:
        """
        Расчет доходностей для конкретного периода
        
        Args:
            df: DataFrame с данными
            period: Период
            
        Returns:
            Dict: Признаки доходностей
        """
        features = {}
        
        try:
            prices = df[self.price_field]
            
            # Простая доходность
            simple_return = prices.pct_change(period)
            features[f"return_{period}"] = simple_return
            
            # Логарифмическая доходность
            log_return = np.log(prices / prices.shift(period))
            features[f"log_return_{period}"] = log_return
            
            # Абсолютная доходность
            abs_return = simple_return.abs()
            features[f"abs_return_{period}"] = abs_return
            
            # Знак доходности
            return_sign = np.sign(simple_return)
            features[f"return_sign_{period}"] = return_sign
            
            # Volume weighted return (если есть объем)
            if self.volume_field in df.columns:
                volumes = df[self.volume_field]
                
                # Volume weighted price
                vwp = (prices * volumes).rolling(window=period, min_periods=1).sum() / \
                      volumes.rolling(window=period, min_periods=1).sum()
                
                vwp_return = vwp.pct_change(period)
                features[f"volume_weighted_return_{period}"] = vwp_return
            else:
                features[f"volume_weighted_return_{period}"] = simple_return
            
        except Exception as e:
            self.logger.error(f"Error calculating period {period} returns: {e}")
            # Возвращаем нулевые значения
            for suffix in ["return", "log_return", "abs_return", "return_sign", "volume_weighted_return"]:
                features[f"{suffix}_{period}"] = pd.Series([0.0] * len(df), index=df.index)
        
        return features
    
    def _calculate_return_statistics(self, features: pd.DataFrame, df: pd.DataFrame) -> None:
        """
        Расчет статистических признаков доходностей
        
        Args:
            features: DataFrame с признаками
            df: DataFrame с данными
        """
        try:
            # Используем доходность за 1 период для статистики
            if "return_1" in features.columns:
                returns = features["return_1"].dropna()
                
                if len(returns) >= 10:
                    # Волатильность доходностей
                    features["return_volatility"] = returns.rolling(window=10, min_periods=1).std()
                    
                    # Асимметрия доходностей
                    features["return_skewness"] = returns.rolling(window=20, min_periods=1).skew()
                    
                    # Эксцесс доходностей
                    features["return_kurtosis"] = returns.rolling(window=20, min_periods=1).kurt()
                    
                    # Моментум доходностей
                    features["return_momentum"] = returns.rolling(window=5, min_periods=1).mean()
                    
                    # Ускорение доходностей (вторая разность)
                    features["return_acceleration"] = returns.diff().diff()
            
        except Exception as e:
            self.logger.error(f"Error calculating return statistics: {e}")
    
    def get_feature_names(self) -> List[str]:
        """Получение имен признаков"""
        return self._feature_names.copy()
    
    def get_required_data_types(self) -> List[str]:
        """Получение требуемых типов данных"""
        return ['ohlcv', 'trades']
