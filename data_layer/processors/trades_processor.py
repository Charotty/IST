"""
Trades Processor

Обработка данных сделок (trades).
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging


class TradesProcessor:
    """Процессор данных сделок"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Инициализация процессора
        
        Args:
            config: Конфигурация процессора
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Параметры обработки
        self.min_trade_size = self.config.get('min_trade_size', 0.001)
        self.max_trade_size = self.config.get('max_trade_size', 1000)
        self.outlier_threshold = self.config.get('outlier_threshold', 5.0)  # std deviations
        
    def process_trades_data(self, raw_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Обработка сырых данных сделок
        
        Args:
            raw_data: Список сырых данных
            
        Returns:
            pd.DataFrame: Обработанные данные сделок
        """
        try:
            if not raw_data:
                return pd.DataFrame()
            
            # Создание DataFrame
            df = pd.DataFrame(raw_data)
            
            # Конвертация типов
            df = self._convert_types(df)
            
            # Валидация данных
            df = self._validate_data(df)
            
            # Очистка данных
            df = self._clean_data(df)
            
            # Расчет производных метрик
            df = self._calculate_metrics(df)
            
            # Сортировка по времени
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            self.logger.debug(f"Processed {len(df)} trade records")
            return df
            
        except Exception as e:
            self.logger.error(f"Error processing trades data: {e}")
            raise
    
    def detect_trade_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Детекция аномалий в сделках
        
        Args:
            df: Данные сделок
            
        Returns:
            pd.DataFrame: Данные с метками аномалий
        """
        try:
            df_copy = df.copy()
            
            if df_copy.empty:
                return df_copy
            
            # Детекция по размеру сделки
            size_mean = df_copy['volume'].mean()
            size_std = df_copy['volume'].std()
            
            df_copy['size_anomaly'] = (
                (df_copy['volume'] < size_mean - self.outlier_threshold * size_std) |
                (df_copy['volume'] > size_mean + self.outlier_threshold * size_std)
            )
            
            # Детекция по цене
            price_mean = df_copy['price'].mean()
            price_std = df_copy['price'].std()
            
            df_copy['price_anomaly'] = (
                (df_copy['price'] < price_mean - self.outlier_threshold * price_std) |
                (df_copy['price'] > price_mean + self.outlier_threshold * price_std)
            )
            
            # Детекция по времени (слишком быстрые сделки)
            if len(df_copy) > 1:
                time_diffs = df_copy['timestamp'].diff().dt.total_seconds()
                df_copy['rapid_trades'] = time_diffs < 0.001  # меньше 1ms
            else:
                df_copy['rapid_trades'] = False
            
            # Общая метка аномалии
            df_copy['is_anomaly'] = (
                df_copy['size_anomaly'] | 
                df_copy['price_anomaly'] | 
                df_copy['rapid_trades']
            )
            
            anomaly_count = df_copy['is_anomaly'].sum()
            self.logger.info(f"Detected {anomaly_count} anomalous trades out of {len(df_copy)}")
            
            return df_copy
            
        except Exception as e:
            self.logger.error(f"Error detecting trade anomalies: {e}")
            return df
    
    def calculate_trade_statistics(
        self, 
        df: pd.DataFrame, 
        window_minutes: int = 5
    ) -> Dict[str, Any]:
        """
        Расчет статистики сделок
        
        Args:
            df: Данные сделок
            window_minutes: Временное окно в минутах
            
        Returns:
            Dict: Статистика сделок
        """
        try:
            if df.empty:
                return {}
            
            # Фильтрация по временному окну
            cutoff_time = df['timestamp'].max() - pd.Timedelta(minutes=window_minutes)
            df_window = df[df['timestamp'] >= cutoff_time]
            
            if df_window.empty:
                return {}
            
            # Базовая статистика
            total_trades = len(df_window)
            total_volume = df_window['volume'].sum()
            avg_trade_size = df_window['volume'].mean()
            
            # Статистика по сторонам
            buy_trades = df_window[df_window['side'] == 'buy']
            sell_trades = df_window[df_window['side'] == 'sell']
            
            buy_volume = buy_trades['volume'].sum() if not buy_trades.empty else 0
            sell_volume = sell_trades['volume'].sum() if not sell_trades.empty else 0
            
            # Цена и волатильность
            prices = df_window['price']
            min_price = prices.min()
            max_price = prices.max()
            price_range = max_price - min_price
            
            # Расчет взвешенной средней цены (VWAP)
            vwap = (df_window['price'] * df_window['volume']).sum() / total_volume
            
            # Расчет волатильности
            returns = prices.pct_change().dropna()
            volatility = returns.std() if len(returns) > 0 else 0
            
            # Частота сделок
            time_span = (df_window['timestamp'].max() - df_window['timestamp'].min()).total_seconds()
            trade_frequency = total_trades / time_span if time_span > 0 else 0
            
            return {
                'time_window_minutes': window_minutes,
                'total_trades': total_trades,
                'total_volume': total_volume,
                'avg_trade_size': avg_trade_size,
                'buy_volume': buy_volume,
                'sell_volume': sell_volume,
                'buy_sell_ratio': buy_volume / sell_volume if sell_volume > 0 else float('inf'),
                'min_price': min_price,
                'max_price': max_price,
                'price_range': price_range,
                'vwap': vwap,
                'volatility': volatility,
                'trade_frequency': trade_frequency,
                'time_span_seconds': time_span
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating trade statistics: {e}")
            return {}
    
    def aggregate_trades_by_interval(
        self, 
        df: pd.DataFrame, 
        interval: str = '1min'
    ) -> pd.DataFrame:
        """
        Агрегация сделок по интервалу
        
        Args:
            df: Данные сделок
            interval: Интервал агрегации
            
        Returns:
            pd.DataFrame: Агрегированные данные
        """
        try:
            if df.empty:
                return pd.DataFrame()
            
            df_copy = df.copy()
            df_copy = df_copy.set_index('timestamp')
            
            # Агрегационные правила
            agg_rules = {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum',
                'trade_count': 'count',
                'buy_volume': lambda x: x[x['side'] == 'buy']['volume'].sum(),
                'sell_volume': lambda x: x[x['side'] == 'sell']['volume'].sum(),
                'avg_trade_size': 'mean',
                'vwap': lambda x: (x['price'] * x['volume']).sum() / x['volume'].sum()
            }
            
            # Агрегация
            aggregated = df_copy.resample(interval).agg(agg_rules)
            
            # Расчет дополнительных метрик
            aggregated['buy_sell_ratio'] = aggregated['buy_volume'] / aggregated['sell_volume']
            aggregated['price_change'] = aggregated['close'].pct_change()
            
            # Удаление пустых записей
            aggregated = aggregated.dropna()
            
            self.logger.debug(f"Aggregated {len(df)} trades to {len(aggregated)} {interval} bars")
            return aggregated.reset_index()
            
        except Exception as e:
            self.logger.error(f"Error aggregating trades: {e}")
            raise
    
    def detect_trading_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Детекция торговых паттернов
        
        Args:
            df: Данные сделок
            
        Returns:
            Dict: Найденные паттерны
        """
        try:
            if df.empty:
                return {}
            
            patterns = {
                'accumulation': self._detect_accumulation(df),
                'distribution': self._detect_distribution(df),
                'momentum': self._detect_momentum(df),
                'reversal': self._detect_reversal(df)
            }
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error detecting trading patterns: {e}")
            return {}
    
    def _convert_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Конвертация типов данных"""
        # Конвертация timestamp
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Конвертация числовых полей
        numeric_fields = ['price', 'volume']
        for field in numeric_fields:
            if field in df.columns:
                df[field] = pd.to_numeric(df[field], errors='coerce')
        
        # Конвертация side в lowercase
        if 'side' in df.columns:
            df['side'] = df['side'].str.lower()
        
        return df
    
    def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Валидация данных"""
        # Проверка обязательных полей
        required_fields = ['timestamp', 'price', 'volume', 'side']
        missing_fields = [field for field in required_fields if field not in df.columns]
        
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
        
        # Валидация значений
        valid_rows = (
            (df['price'] > 0) &
            (df['volume'] >= self.min_trade_size) &
            (df['volume'] <= self.max_trade_size) &
            (df['side'].isin(['buy', 'sell']))
        )
        
        return df[valid_rows]
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Очистка данных"""
        # Удаление дубликатов
        if 'timestamp' in df.columns and 'trade_id' in df.columns:
            df = df.drop_duplicates(subset=['trade_id'], keep='last')
        elif 'timestamp' in df.columns:
            df = df.drop_duplicates(subset=['timestamp'], keep='last')
        
        # Удаление записей с NaN значениями
        key_fields = ['price', 'volume', 'side']
        df = df.dropna(subset=key_fields)
        
        # Сортировка по времени
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp')
        
        return df
    
    def _calculate_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Расчет производных метрик"""
        df_copy = df.copy()
        
        # Расчет цены в USD (если нужно)
        if 'price' in df.columns and 'volume' in df.columns:
            df_copy['notional'] = df_copy['price'] * df_copy['volume']
        
        # Расчет кумулятивного объема
        if 'volume' in df.columns:
            df_copy['cumulative_volume'] = df_copy['volume'].cumsum()
        
        # Расчет скользящей средней цены
        if 'price' in df.columns:
            df_copy['sma_5'] = df_copy['price'].rolling(window=5, min_periods=1).mean()
            df_copy['sma_20'] = df_copy['price'].rolling(window=20, min_periods=1).mean()
        
        # Расчет изменения цены
        if 'price' in df.columns:
            df_copy['price_change'] = df_copy['price'].pct_change()
            df_copy['price_change_abs'] = df_copy['price_change'].abs()
        
        return df_copy
    
    def _detect_accumulation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Детекция паттерна аккумуляции"""
        try:
            if df.empty or 'price' not in df.columns or 'volume' not in df.columns:
                return {}
            
            # Анализ объема и цены
            recent_data = df.tail(20)  # последние 20 сделок
            
            if len(recent_data) < 10:
                return {}
            
            # Проверка на аккумуляцию (большой объем при стабильной цене)
            price_std = recent_data['price'].std()
            volume_mean = recent_data['volume'].mean()
            volume_std = recent_data['volume'].std()
            
            # Паттерн аккумуляции: низкая волатильность цены, высокий объем
            accumulation_score = 0
            if price_std < (df['price'].std() * 0.5):  # низкая волатильность
                accumulation_score += 0.5
            if volume_mean > (df['volume'].mean() * 1.5):  # высокий объем
                accumulation_score += 0.5
            
            return {
                'detected': accumulation_score > 0.7,
                'score': accumulation_score,
                'price_stability': price_std / df['price'].std(),
                'volume_intensity': volume_mean / df['volume'].mean()
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting accumulation: {e}")
            return {}
    
    def _detect_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Детекция паттерна дистрибуции"""
        try:
            if df.empty or 'price' not in df.columns or 'volume' not in df.columns:
                return {}
            
            recent_data = df.tail(20)
            
            if len(recent_data) < 10:
                return {}
            
            # Анализ тренда цены
            price_trend = self._analyze_price_trend(recent_data['price'])
            
            # Паттерн дистрибуции: падение цены при увеличении объема
            distribution_score = 0
            if price_trend['direction'] == 'down':
                distribution_score += 0.4
            
            # Проверка на увеличение объема
            volume_trend = self._analyze_volume_trend(recent_data['volume'])
            if volume_trend['direction'] == 'up':
                distribution_score += 0.4
            
            return {
                'detected': distribution_score > 0.6,
                'score': distribution_score,
                'price_trend': price_trend,
                'volume_trend': volume_trend
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting distribution: {e}")
            return {}
    
    def _detect_momentum(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Детекция паттерна момента"""
        try:
            if df.empty or 'price' not in df.columns:
                return {}
            
            recent_data = df.tail(10)
            
            if len(recent_data) < 5:
                return {}
            
            # Расчет момента (скорости изменения цены)
            price_changes = recent_data['price'].pct_change().dropna()
            
            if len(price_changes) < 3:
                return {}
            
            momentum_score = abs(price_changes.mean())
            momentum_direction = 'up' if price_changes.mean() > 0 else 'down'
            
            return {
                'detected': momentum_score > 0.01,  # 1% изменение
                'score': momentum_score,
                'direction': momentum_direction,
                'avg_price_change': price_changes.mean(),
                'price_volatility': price_changes.std()
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting momentum: {e}")
            return {}
    
    def _detect_reversal(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Детекция паттерна разворота"""
        try:
            if df.empty or 'price' not in df.columns:
                return {}
            
            recent_data = df.tail(15)
            
            if len(recent_data) < 8:
                return {}
            
            # Анализ изменения направления тренда
            first_half = recent_data.head(len(recent_data)//2)['price']
            second_half = recent_data.tail(len(recent_data)//2)['price']
            
            first_trend = 'up' if first_half.iloc[-1] > first_half.iloc[0] else 'down'
            second_trend = 'up' if second_half.iloc[-1] > second_half.iloc[0] else 'down'
            
            # Паттерн разворота: изменение направления тренда
            reversal_score = 0
            if first_trend != second_trend:
                reversal_score += 0.7
            
            # Дополнительная проверка по объему
            if 'volume' in recent_data.columns:
                volume_change = (second_half['volume'].mean() / first_half['volume'].mean() - 1)
                if abs(volume_change) > 0.3:  # 30% изменение объема
                    reversal_score += 0.3
            
            return {
                'detected': reversal_score > 0.5,
                'score': reversal_score,
                'previous_trend': first_trend,
                'current_trend': second_trend,
                'volume_change': volume_change if 'volume' in recent_data.columns else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting reversal: {e}")
            return {}
    
    def _analyze_price_trend(self, prices: pd.Series) -> Dict[str, Any]:
        """Анализ тренда цены"""
        if len(prices) < 2:
            return {}
        
        start_price = prices.iloc[0]
        end_price = prices.iloc[-1]
        change_pct = (end_price - start_price) / start_price
        
        direction = 'up' if change_pct > 0 else 'down' if change_pct < 0 else 'flat'
        
        return {
            'direction': direction,
            'change_percentage': change_pct * 100,
            'start_price': start_price,
            'end_price': end_price
        }
    
    def _analyze_volume_trend(self, volumes: pd.Series) -> Dict[str, Any]:
        """Анализ тренда объема"""
        if len(volumes) < 2:
            return {}
        
        start_volume = volumes.iloc[0]
        end_volume = volumes.iloc[-1]
        change_pct = (end_volume - start_volume) / start_volume if start_volume > 0 else 0
        
        direction = 'up' if change_pct > 0 else 'down' if change_pct < 0 else 'flat'
        
        return {
            'direction': direction,
            'change_percentage': change_pct * 100,
            'start_volume': start_volume,
            'end_volume': end_volume
        }
