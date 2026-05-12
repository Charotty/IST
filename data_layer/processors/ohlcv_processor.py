"""
OHLCV Processor

Обработка OHLCV (Open, High, Low, Close, Volume) данных.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging


class OHLCVProcessor:
    """Процессор OHLCV данных"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Инициализация процессора
        
        Args:
            config: Конфигурация процессора
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Параметры валидации
        self.min_volume = self.config.get('min_volume', 0.001)
        self.max_price_change = self.config.get('max_price_change', 0.5)  # 50%
        
    def process_raw_data(self, raw_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Обработка сырых OHLCV данных
        
        Args:
            raw_data: Список сырых данных
            
        Returns:
            pd.DataFrame: Обработанные OHLCV данные
        """
        if not raw_data:
            return pd.DataFrame()
        
        try:
            # Создание DataFrame
            df = pd.DataFrame(raw_data)
            
            # Конвертация типов
            df = self._convert_types(df)
            
            # Валидация данных
            df = self._validate_data(df)
            
            # Очистка данных
            df = self._clean_data(df)
            
            # Сортировка по времени
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            self.logger.debug(f"Processed {len(df)} OHLCV records")
            return df
            
        except Exception as e:
            self.logger.error(f"Error processing OHLCV data: {e}")
            raise
    
    def resample_data(
        self, 
        df: pd.DataFrame, 
        target_timeframe: str
    ) -> pd.DataFrame:
        """
        Ресемплинг данных в другой таймфрейм
        
        Args:
            df: Исходные данные
            target_timeframe: Целевой таймфрейм
            
        Returns:
            pd.DataFrame: Ресемплированные данные
        """
        try:
            # Установка timestamp как индекса
            df_copy = df.copy()
            df_copy = df_copy.set_index('timestamp')
            
            # Определение правил агрегации
            agg_rules = {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }
            
            # Ресемплинг
            resampled = df_copy.resample(target_timeframe).agg(agg_rules)
            
            # Удаление пустых значений
            resampled = resampled.dropna()
            
            # Восстановление timestamp
            resampled = resampled.reset_index()
            
            self.logger.debug(f"Resampled to {target_timeframe}: {len(resampled)} records")
            return resampled
            
        except Exception as e:
            self.logger.error(f"Error resampling data: {e}")
            raise
    
    def detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Детекция аномалий в данных
        
        Args:
            df: OHLCV данные
            
        Returns:
            pd.DataFrame: Данные с метками аномалий
        """
        try:
            df_copy = df.copy()
            
            # Расчет доходностей
            df_copy['returns'] = df_copy['close'].pct_change()
            
            # Детекция выбросов по доходностям
            return_threshold = df_copy['returns'].quantile(0.99)
            df_copy['return_anomaly'] = abs(df_copy['returns']) > return_threshold
            
            # Детекция аномалий по объему
            volume_threshold = df_copy['volume'].quantile(0.99)
            df_copy['volume_anomaly'] = df_copy['volume'] > volume_threshold
            
            # Детекция ценовых разрывов
            df_copy['price_gap'] = (
                (df_copy['open'] - df_copy['close'].shift(1)) / 
                df_copy['close'].shift(1)
            ).abs()
            df_copy['gap_anomaly'] = df_copy['price_gap'] > self.max_price_change
            
            # Общая метка аномалии
            df_copy['is_anomaly'] = (
                df_copy['return_anomaly'] | 
                df_copy['volume_anomaly'] | 
                df_copy['gap_anomaly']
            )
            
            anomaly_count = df_copy['is_anomaly'].sum()
            self.logger.info(f"Detected {anomaly_count} anomalies in {len(df_copy)} records")
            
            return df_copy
            
        except Exception as e:
            self.logger.error(f"Error detecting anomalies: {e}")
            raise
    
    def fill_missing_data(
        self, 
        df: pd.DataFrame, 
        method: str = 'forward_fill'
    ) -> pd.DataFrame:
        """
        Заполнение пропущенных данных
        
        Args:
            df: OHLCV данные
            method: Метод заполнения (forward_fill, linear, drop)
            
        Returns:
            pd.DataFrame: Данные с заполненными пропусками
        """
        try:
            df_copy = df.copy()
            
            # Создание полного временного ряда
            df_copy = df_copy.set_index('timestamp')
            full_range = pd.date_range(
                start=df_copy.index.min(),
                end=df_copy.index.max(),
                freq=df_copy.index.freq or '1min'
            )
            
            # Reindex с полным диапазоном
            df_copy = df_copy.reindex(full_range)
            
            if method == 'forward_fill':
                # Forward fill
                df_copy = df_copy.fillna(method='ffill')
            elif method == 'linear':
                # Линейная интерполяция
                numeric_columns = ['open', 'high', 'low', 'close', 'volume']
                df_copy[numeric_columns] = df_copy[numeric_columns].interpolate(method='linear')
            elif method == 'drop':
                # Удаление пропусков
                df_copy = df_copy.dropna()
            else:
                raise ValueError(f"Unknown fill method: {method}")
            
            # Восстановление timestamp
            df_copy = df_copy.reset_index()
            df_copy = df_copy.rename(columns={'index': 'timestamp'})
            
            return df_copy
            
        except Exception as e:
            self.logger.error(f"Error filling missing data: {e}")
            raise
    
    def _convert_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Конвертация типов данных"""
        # Конвертация timestamp
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Конвертация числовых полей
        numeric_fields = ['open', 'high', 'low', 'close', 'volume']
        for field in numeric_fields:
            if field in df.columns:
                df[field] = pd.to_numeric(df[field], errors='coerce')
        
        return df
    
    def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Валидация данных"""
        # Проверка обязательных полей
        required_fields = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_fields = [field for field in required_fields if field not in df.columns]
        
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
        
        # Проверка логических соотношений
        invalid_rows = (
            (df['high'] < df['low']) |
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close']) |
            (df['open'] < 0) |
            (df['high'] < 0) |
            (df['low'] < 0) |
            (df['close'] < 0) |
            (df['volume'] < 0)
        )
        
        if invalid_rows.any():
            invalid_count = invalid_rows.sum()
            self.logger.warning(f"Found {invalid_count} invalid OHLCV records")
            
            # Удаление невалидных записей
            df = df[~invalid_rows]
        
        # Фильтрация по минимальному объему
        if 'volume' in df.columns:
            df = df[df['volume'] >= self.min_volume]
        
        return df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Очистка данных"""
        # Удаление дубликатов
        if 'timestamp' in df.columns:
            df = df.drop_duplicates(subset=['timestamp'], keep='last')
        
        # Удаление записей с NaN значениями в ключевых полях
        key_fields = ['open', 'high', 'low', 'close']
        df = df.dropna(subset=key_fields)
        
        # Сортировка по времени
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp')
        
        return df
    
    def get_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Расчет статистики OHLCV данных
        
        Args:
            df: OHLCV данные
            
        Returns:
            Dict: Статистика
        """
        if df.empty:
            return {}
        
        try:
            stats = {
                'total_records': len(df),
                'date_range': {
                    'start': df['timestamp'].min().isoformat(),
                    'end': df['timestamp'].max().isoformat()
                },
                'price_stats': {
                    'min': df['close'].min(),
                    'max': df['close'].max(),
                    'mean': df['close'].mean(),
                    'std': df['close'].std()
                },
                'volume_stats': {
                    'total': df['volume'].sum(),
                    'mean': df['volume'].mean(),
                    'max': df['volume'].max()
                }
            }
            
            # Расчет волатильности
            returns = df['close'].pct_change().dropna()
            if len(returns) > 0:
                stats['volatility'] = {
                    'daily': returns.std(),
                    'annualized': returns.std() * np.sqrt(252)
                }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error calculating statistics: {e}")
            return {}
    
    def export_to_format(
        self, 
        df: pd.DataFrame, 
        format: str = 'csv',
        file_path: Optional[str] = None
    ) -> str:
        """
        Экспорт данных в various форматы
        
        Args:
            df: OHLCV данные
            format: Формат экспорта (csv, json, parquet)
            file_path: Путь для сохранения
            
        Returns:
            str: Путь к сохраненному файлу
        """
        try:
            if file_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                file_path = f"ohlcv_data_{timestamp}.{format}"
            
            if format == 'csv':
                df.to_csv(file_path, index=False)
            elif format == 'json':
                df.to_json(file_path, orient='records', date_format='iso')
            elif format == 'parquet':
                df.to_parquet(file_path, index=False)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self.logger.info(f"Exported {len(df)} records to {file_path}")
            return file_path
            
        except Exception as e:
            self.logger.error(f"Error exporting data: {e}")
            raise
