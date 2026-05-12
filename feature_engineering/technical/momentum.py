"""
Momentum Indicators

Индикаторы момента: RSI, MACD, Stochastic.
"""

import asyncio
import logging
from typing import Dict, Any, List, Union
import pandas as pd
import numpy as np
from ..base_feature_generator import BaseFeatureGenerator


class MomentumIndicators(BaseFeatureGenerator):
    """Генератор индикаторов момента"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Инициализация генератора индикаторов момента
        
        Args:
            config: Конфигурация
        """
        super().__init__(config)
        
        # Параметры RSI
        self.rsi_periods = config.get('rsi_periods', [14, 21])
        
        # Параметры MACD
        self.macd_params = config.get('macd_params', [12, 26, 9])
        
        # Параметры Stochastic
        self.stoch_k_period = config.get('stoch_k_period', 14)
        self.stoch_d_period = config.get('stoch_d_period', 3)
        self.stoch_smooth_k = config.get('stoch_smooth_k', 3)
        
        # Поля цен
        self.price_field = config.get('price_field', 'close')
        self.high_field = config.get('high_field', 'high')
        self.low_field = config.get('low_field', 'low')
        
        # Имена признаков
        self._feature_names = self._generate_feature_names()
    
    def _generate_feature_names(self) -> List[str]:
        """Генерация имен признаков"""
        names = []
        
        # RSI
        for period in self.rsi_periods:
            names.append(f"RSI_{period}")
        
        # MACD
        names.extend(["MACD", "MACD_signal", "MACD_histogram"])
        
        # Stochastic
        names.extend(["Stoch_K", "Stoch_D"])
        
        return names
    
    def _validate_config(self) -> None:
        """Валидация конфигурации"""
        # RSI periods
        if not self.rsi_periods or not isinstance(self.rsi_periods, list):
            raise ValueError("rsi_periods must be a non-empty list")
        
        if not all(isinstance(p, int) and p > 0 for p in self.rsi_periods):
            raise ValueError("All rsi_periods must be positive integers")
        
        # MACD params
        if not isinstance(self.macd_params, list) or len(self.macd_params) != 3:
            raise ValueError("macd_params must be a list of 3 integers")
        
        if not all(isinstance(p, int) and p > 0 for p in self.macd_params):
            raise ValueError("All macd_params must be positive integers")
        
        # Stochastic params
        if not isinstance(self.stoch_k_period, int) or self.stoch_k_period <= 0:
            raise ValueError("stoch_k_period must be a positive integer")
        
        if not isinstance(self.stoch_d_period, int) or self.stoch_d_period <= 0:
            raise ValueError("stoch_d_period must be a positive integer")
    
    async def generate_features(self, data: Union[pd.DataFrame, Dict[str, Any]]) -> pd.DataFrame:
        """
        Генерация индикаторов момента
        
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
            required_fields = [self.price_field, self.high_field, self.low_field]
            missing_fields = [f for f in required_fields if f not in df.columns]
            if missing_fields:
                self.logger.warning(f"Missing fields: {missing_fields}")
                return pd.DataFrame()
            
            # Проверка достаточного количества данных
            min_required = max(
                max(self.rsi_periods),
                max(self.macd_params),
                self.stoch_k_period + self.stoch_d_period
            )
            
            if len(df) < min_required:
                self.logger.warning(f"Insufficient data: need {min_required}, got {len(df)}")
                return pd.DataFrame()
            
            features = pd.DataFrame(index=df.index)
            
            # Расчет RSI
            for period in self.rsi_periods:
                if len(df) >= period:
                    rsi_values = self._calculate_rsi(df[self.price_field], period)
                    features[f"RSI_{period}"] = rsi_values
            
            # Расчет MACD
            if len(df) >= max(self.macd_params):
                macd_line, signal_line, histogram = self._calculate_macd(df[self.price_field])
                features["MACD"] = macd_line
                features["MACD_signal"] = signal_line
                features["MACD_histogram"] = histogram
            
            # Расчет Stochastic
            if len(df) >= (self.stoch_k_period + self.stoch_d_period):
                stoch_k, stoch_d = self._calculate_stochastic(
                    df[self.high_field], 
                    df[self.low_field], 
                    df[self.price_field]
                )
                features["Stoch_K"] = stoch_k
                features["Stoch_D"] = stoch_d
            
            return features
            
        except Exception as e:
            self._handle_error(e, "generate_features")
            return pd.DataFrame()
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """
        Расчет Relative Strength Index
        
        Args:
            prices: Цены закрытия
            period: Период
            
        Returns:
            pd.Series: RSI значения
        """
        # Расчет изменений
        delta = prices.diff()
        
        # Разделение на gains и losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        # Расчет средних gains и losses
        avg_gains = gains.rolling(window=period, min_periods=1).mean()
        avg_losses = losses.rolling(window=period, min_periods=1).mean()
        
        # Расчет RS и RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        # Замена NaN на 50 (нейтральное значение)
        rsi = rsi.fillna(50)
        
        return rsi
    
    def _calculate_macd(self, prices: pd.Series) -> tuple:
        """
        Расчет MACD
        
        Args:
            prices: Цены закрытия
            
        Returns:
            tuple: (MACD line, Signal line, Histogram)
        """
        fast_period, slow_period, signal_period = self.macd_params
        
        # Расчет EMA
        fast_ema = prices.ewm(span=fast_period, adjust=False).mean()
        slow_ema = prices.ewm(span=slow_period, adjust=False).mean()
        
        # MACD line
        macd_line = fast_ema - slow_ema
        
        # Signal line
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        
        # Histogram
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def _calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series) -> tuple:
        """
        Расчет Stochastic Oscillator
        
        Args:
            high: Максимальные цены
            low: Минимальные цены
            close: Цены закрытия
            
        Returns:
            tuple: (Stoch_K, Stoch_D)
        """
        # Расчет %K
        lowest_low = low.rolling(window=self.stoch_k_period, min_periods=1).min()
        highest_high = high.rolling(window=self.stoch_k_period, min_periods=1).max()
        
        k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
        
        # Сглаживание %K
        if self.stoch_smooth_k > 1:
            k_percent = k_percent.rolling(window=self.stoch_smooth_k, min_periods=1).mean()
        
        # Расчет %D как сглаженная %K
        d_percent = k_percent.rolling(window=self.stoch_d_period, min_periods=1).mean()
        
        # Замена NaN на 50
        k_percent = k_percent.fillna(50)
        d_percent = d_percent.fillna(50)
        
        return k_percent, d_percent
    
    def get_feature_names(self) -> List[str]:
        """Получение имен признаков"""
        return self._feature_names.copy()
    
    def get_required_data_types(self) -> List[str]:
        """Получение требуемых типов данных"""
        return ['ohlcv']
