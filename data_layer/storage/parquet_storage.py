"""
Parquet Storage

Хранение данных в формате Parquet с использованием PyArrow.
"""

import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import logging
from pathlib import Path


class ParquetStorage:
    """Хранение данных в Parquet формате"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Инициализация хранилища
        
        Args:
            config: Конфигурация хранилища
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Базовый путь для хранения
        self.base_path = Path(config.get('path', './data/parquet'))
        self.compression = config.get('compression', 'snappy')
        
        # Создание директорий
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Поддиректории для разных типов данных
        self.ohlcv_path = self.base_path / 'ohlcv'
        self.trades_path = self.base_path / 'trades'
        self.orderbook_path = self.base_path / 'orderbook'
        self.onchain_path = self.base_path / 'onchain'
        
        for path in [self.ohlcv_path, self.trades_path, self.orderbook_path, self.onchain_path]:
            path.mkdir(exist_ok=True)
    
    async def save_ohlcv_data(
        self, 
        data: pd.DataFrame, 
        symbol: str, 
        timeframe: str
    ) -> str:
        """
        Сохранение OHLCV данных
        
        Args:
            data: OHLCV DataFrame
            symbol: Торговая пара
            timeframe: Таймфрейм
            
        Returns:
            str: Путь к сохраненному файлу
        """
        try:
            # Формирование имени файла
            filename = f"{symbol}_{timeframe}.parquet"
            file_path = self.ohlcv_path / filename
            
            # Проверка данных
            if data.empty:
                self.logger.warning(f"Empty OHLCV data for {symbol} {timeframe}")
                return str(file_path)
            
            # Подготовка данных
            data_copy = data.copy()
            
            # Конвертация timestamp для совместимости
            if 'timestamp' in data_copy.columns:
                data_copy['timestamp'] = pd.to_datetime(data_copy['timestamp'])
            
            # Сохранение в Parquet
            table = pa.Table.from_pandas(data_copy)
            
            # Используем write_table вместо write_dataset для совместимости
            write_table_kwargs = {
                'compression': self.compression
            }
            
            pq.write_table(
                table,
                str(file_path),
                **write_table_kwargs
            )
            
            self.logger.info(f"Saved {len(data_copy)} OHLCV records to {file_path}")
            return str(file_path)
            
        except Exception as e:
            self.logger.error(f"Error saving OHLCV data: {e}")
            raise
    
    async def load_ohlcv_data(
        self, 
        symbol: str, 
        timeframe: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Загрузка OHLCV данных
        
        Args:
            symbol: Торговая пара
            timeframe: Таймфрейм
            start_time: Время начала
            end_time: Время окончания
            
        Returns:
            pd.DataFrame: OHLCV данные
        """
        try:
            filename = f"{symbol}_{timeframe}.parquet"
            file_path = self.ohlcv_path / filename
            
            if not file_path.exists():
                self.logger.warning(f"OHLCV file not found: {file_path}")
                return pd.DataFrame()
            
            # Загрузка данных
            dataset = pq.ParquetDataset(file_path)
            table = dataset.read()
            df = table.to_pandas()
            
            # Фильтрация по времени
            if start_time and 'timestamp' in df.columns:
                df = df[df['timestamp'] >= start_time]
            
            if end_time and 'timestamp' in df.columns:
                df = df[df['timestamp'] <= end_time]
            
            # Сортировка
            if 'timestamp' in df.columns:
                df = df.sort_values('timestamp')
            
            self.logger.info(f"Loaded {len(df)} OHLCV records from {file_path}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading OHLCV data: {e}")
            raise
    
    async def save_trades_data(self, data: pd.DataFrame, symbol: str) -> str:
        """
        Сохранение данных сделок
        
        Args:
            data: Trades DataFrame
            symbol: Торговая пара
            
        Returns:
            str: Путь к сохраненному файлу
        """
        try:
            filename = f"{symbol}_trades.parquet"
            file_path = self.trades_path / filename
            
            if data.empty:
                self.logger.warning(f"Empty trades data for {symbol}")
                return str(file_path)
            
            # Подготовка данных
            data_copy = data.copy()
            
            # Конвертация timestamp
            if 'timestamp' in data_copy.columns:
                data_copy['timestamp'] = pd.to_datetime(data_copy['timestamp'])
            
            # Сохранение
            table = pa.Table.from_pandas(data_copy)
            
            pq.write_dataset(
                table,
                root_path=str(file_path),
                compression=self.compression,
                existing_data_behavior='overwrite_or_ignore'
            )
            
            self.logger.info(f"Saved {len(data_copy)} trades records to {file_path}")
            return str(file_path)
            
        except Exception as e:
            self.logger.error(f"Error saving trades data: {e}")
            raise
    
    async def load_trades_data(
        self, 
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Загрузка данных сделок
        
        Args:
            symbol: Торговая пара
            start_time: Время начала
            end_time: Время окончания
            
        Returns:
            pd.DataFrame: Trades данные
        """
        try:
            filename = f"{symbol}_trades.parquet"
            file_path = self.trades_path / filename
            
            if not file_path.exists():
                self.logger.warning(f"Trades file not found: {file_path}")
                return pd.DataFrame()
            
            # Загрузка
            dataset = pq.ParquetDataset(file_path)
            table = dataset.read()
            df = table.to_pandas()
            
            # Фильтрация по времени
            if start_time and 'timestamp' in df.columns:
                df = df[df['timestamp'] >= start_time]
            
            if end_time and 'timestamp' in df.columns:
                df = df[df['timestamp'] <= end_time]
            
            # Сортировка
            if 'timestamp' in df.columns:
                df = df.sort_values('timestamp')
            
            self.logger.info(f"Loaded {len(df)} trades records from {file_path}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading trades data: {e}")
            raise
    
    async def save_onchain_data(
        self, 
        data: pd.DataFrame, 
        symbol: str, 
        metric: str
    ) -> str:
        """
        Сохранение ончейн-данных
        
        Args:
            data: On-chain DataFrame
            symbol: Актив
            metric: Метрика
            
        Returns:
            str: Путь к сохраненному файлу
        """
        try:
            filename = f"{symbol}_{metric}.parquet"
            file_path = self.onchain_path / filename
            
            if data.empty:
                self.logger.warning(f"Empty on-chain data for {symbol} {metric}")
                return str(file_path)
            
            # Подготовка данных
            data_copy = data.copy()
            
            # Конвертация timestamp
            if 'timestamp' in data_copy.columns:
                data_copy['timestamp'] = pd.to_datetime(data_copy['timestamp'])
            
            # Сохранение
            table = pa.Table.from_pandas(data_copy)
            
            pq.write_dataset(
                table,
                root_path=str(file_path),
                compression=self.compression,
                existing_data_behavior='overwrite_or_ignore'
            )
            
            self.logger.info(f"Saved {len(data_copy)} on-chain records to {file_path}")
            return str(file_path)
            
        except Exception as e:
            self.logger.error(f"Error saving on-chain data: {e}")
            raise
    
    async def get_available_symbols(self, data_type: str = 'ohlcv') -> List[str]:
        """
        Получение списка доступных символов
        
        Args:
            data_type: Тип данных (ohlcv, trades, onchain)
            
        Returns:
            List[str]: Список символов
        """
        try:
            if data_type == 'ohlcv':
                path = self.ohlcv_path
                pattern = "*_*.parquet"
            elif data_type == 'trades':
                path = self.trades_path
                pattern = "*_trades.parquet"
            elif data_type == 'onchain':
                path = self.onchain_path
                pattern = "*_*.parquet"
            else:
                raise ValueError(f"Unknown data type: {data_type}")
            
            # Получение файлов
            files = list(path.glob(pattern))
            
            # Извлечение символов
            symbols = set()
            for file in files:
                parts = file.stem.split('_')
                if len(parts) >= 2:
                    if data_type == 'trades':
                        symbol = parts[0]
                    else:
                        symbol = parts[0]
                    symbols.add(symbol)
            
            return sorted(list(symbols))
            
        except Exception as e:
            self.logger.error(f"Error getting available symbols: {e}")
            return []
    
    async def get_data_info(self, symbol: str, data_type: str = 'ohlcv') -> Dict[str, Any]:
        """
        Получение информации о данных
        
        Args:
            symbol: Торговая пара
            data_type: Тип данных
            
        Returns:
            Dict: Информация о данных
        """
        try:
            if data_type == 'ohlcv':
                file_path = self.ohlcv_path / f"{symbol}_1m.parquet"
            elif data_type == 'trades':
                file_path = self.trades_path / f"{symbol}_trades.parquet"
            else:
                raise ValueError(f"Unknown data type: {data_type}")
            
            if not file_path.exists():
                return {'exists': False}
            
            # Получение метаданных
            dataset = pq.ParquetDataset(file_path)
            schema = dataset.schema
            fragments = dataset.get_fragments()
            
            # Расчет размера
            total_size = sum(frag.compute_computed_metadata().total_byte_size for frag in fragments)
            
            info = {
                'exists': True,
                'file_path': str(file_path),
                'schema': schema.names,
                'total_size_bytes': total_size,
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'num_fragments': len(fragments),
                'created_time': datetime.fromtimestamp(file_path.stat().st_ctime),
                'modified_time': datetime.fromtimestamp(file_path.stat().st_mtime)
            }
            
            return info
            
        except Exception as e:
            self.logger.error(f"Error getting data info: {e}")
            return {'exists': False, 'error': str(e)}
    
    async def cleanup_old_data(
        self, 
        days_to_keep: int = 30,
        data_type: str = 'all'
    ) -> None:
        """
        Очистка старых данных
        
        Args:
            days_to_keep: Количество дней для хранения
            data_type: Тип данных для очистки
        """
        try:
            cutoff_time = datetime.now() - pd.Timedelta(days=days_to_keep)
            
            paths_to_clean = []
            if data_type in ['all', 'ohlcv']:
                paths_to_clean.append(self.ohlcv_path)
            if data_type in ['all', 'trades']:
                paths_to_clean.append(self.trades_path)
            if data_type in ['all', 'onchain']:
                paths_to_clean.append(self.onchain_path)
            
            deleted_files = 0
            total_size_freed = 0
            
            for path in paths_to_clean:
                for file_path in path.glob("*.parquet"):
                    file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                    
                    if file_time < cutoff_time:
                        file_size = file_path.stat().st_size
                        file_path.unlink()
                        deleted_files += 1
                        total_size_freed += file_size
            
            self.logger.info(
                f"Cleaned up {deleted_files} files, freed {total_size_freed / (1024*1024):.2f} MB"
            )
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    async def backup_data(self, backup_path: str) -> None:
        """
        Создание резервной копии данных
        
        Args:
            backup_path: Путь для резервной копии
        """
        try:
            import shutil
            
            backup_dir = Path(backup_path)
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Копирование всех данных
            shutil.copytree(self.base_path, backup_dir / 'data', dirs_exist_ok=True)
            
            self.logger.info(f"Data backed up to {backup_path}")
            
        except Exception as e:
            self.logger.error(f"Error during backup: {e}")
            raise
