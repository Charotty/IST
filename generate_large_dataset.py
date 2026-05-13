#!/usr/bin/env python3
"""
Генератор большого датасета с использованием реализованных модулей

Скрипт для загрузки и сохранения больших объемов данных через API OKX
с использованием существующих модулей системы.
"""

import asyncio
import logging
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List
import pandas as pd
import yaml
from dotenv import load_dotenv

# Добавляем корневую директорию в путь
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_layer.data_manager import DataManager
from data_layer.connectors.okx_connector import OKXConnector
from data_layer.processors.ohlcv_processor import OHLCVProcessor
from data_layer.storage.parquet_storage import ParquetStorage

# Загрузка переменных окружения
load_dotenv()

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('dataset_generation.log')
    ]
)

logger = logging.getLogger(__name__)


class DatasetGenerator:
    """Генератор больших датасетов"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Инициализация генератора
        
        Args:
            config_path: Путь к конфигурационному файлу
        """
        self.config = self._load_config(config_path)
        self.data_manager = None
        self.dataset_info = {
            'start_time': None,
            'end_time': None,
            'symbols': [],
            'data_types': [],
            'total_records': 0,
            'file_sizes': {}
        }
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Загрузка конфигурации"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # Замена переменных окружения
            self._replace_env_vars(config)
            
            return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise
    
    def _replace_env_vars(self, config: Dict[str, Any]) -> None:
        """Замена переменных окружения в конфигурации"""
        def replace_recursive(obj):
            if isinstance(obj, dict):
                return {k: replace_recursive(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [replace_recursive(item) for item in obj]
            elif isinstance(obj, str) and obj.startswith('${') and obj.endswith('}'):
                env_var = obj[2:-1]
                return os.getenv(env_var, obj)
            else:
                return obj
        
        replace_recursive(config)
    
    async def initialize(self) -> None:
        """Инициализация компонентов"""
        try:
            logger.info("Initializing Dataset Generator...")
            
            # Инициализация DataManager
            self.data_manager = DataManager(self.config)
            await self.data_manager.start()
            
            logger.info("Dataset Generator initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing: {e}")
            raise
    
    async def generate_comprehensive_dataset(
        self,
        symbols: List[str] = None,
        timeframes: List[str] = None,
        days_back: int = 30,
        data_types: List[str] = None
    ) -> Dict[str, Any]:
        """
        Генерация комплексного датасета
        
        Args:
            symbols: Список символов
            timeframes: Список таймфреймов
            days_back: Количество дней для исторических данных
            data_types: Типы данных для загрузки
            
        Returns:
            Dict: Информация о сгенерированном датасете
        """
        if symbols is None:
            symbols = ['BTC-USDT', 'ETH-USDT', 'SOL-USDT', 'BNB-USDT', 'XRP-USDT']
        
        if timeframes is None:
            timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
        
        if data_types is None:
            data_types = ['ohlcv', 'trades', 'orderbook']
        
        self.dataset_info['start_time'] = datetime.utcnow()
        self.dataset_info['symbols'] = symbols
        self.dataset_info['data_types'] = data_types
        
        logger.info(f"Starting dataset generation for {len(symbols)} symbols")
        logger.info(f"Timeframes: {timeframes}")
        logger.info(f"Data types: {data_types}")
        logger.info(f"Days back: {days_back}")
        
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days_back)
        
        total_records = 0
        
        try:
            # Генерация OHLCV данных
            if 'ohlcv' in data_types:
                for symbol in symbols:
                    for timeframe in timeframes:
                        try:
                            logger.info(f"Loading OHLCV data for {symbol} {timeframe}")
                            
                            data = await self.data_manager.get_historical_data(
                                symbol=symbol,
                                data_type='ohlcv',
                                timeframe=timeframe,
                                start_time=start_time,
                                end_time=end_time
                            )
                            
                            if data and 'data' in data:
                                df = data['data']
                                if not df.empty:
                                    record_count = len(df)
                                    total_records += record_count
                                    logger.info(f"Loaded {record_count} records for {symbol} {timeframe}")
                                    
                                    # Сохранение информации о файле
                                    file_path = f"./data/parquet/ohlcv/{symbol}_{timeframe}.parquet"
                                    if os.path.exists(file_path):
                                        file_size = os.path.getsize(file_path)
                                        self.dataset_info['file_sizes'][f"{symbol}_{timeframe}"] = file_size
                            
                        except Exception as e:
                            logger.error(f"Error loading OHLCV for {symbol} {timeframe}: {e}")
                            continue
            
            # Генерация данных о сделках (ограниченный объем из-за размера)
            if 'trades' in data_types:
                for symbol in symbols[:2]:  # Ограничиваем количество символов для trades
                    try:
                        logger.info(f"Loading trades data for {symbol}")
                        
                        # Загружаем данные за короткий период для trades
                        trades_start = end_time - timedelta(days=1)
                        
                        # Здесь можно добавить логику для загрузки trades
                        # через stream_data или специальный метод
                        
                    except Exception as e:
                        logger.error(f"Error loading trades for {symbol}: {e}")
                        continue
            
            # Генерация данных orderbook (ограниченный объем)
            if 'orderbook' in data_types:
                for symbol in symbols[:2]:  # Ограничиваем количество символов
                    try:
                        logger.info(f"Loading orderbook data for {symbol}")
                        
                        # Здесь можно добавить логику для загрузки orderbook
                        
                    except Exception as e:
                        logger.error(f"Error loading orderbook for {symbol}: {e}")
                        continue
            
            self.dataset_info['total_records'] = total_records
            self.dataset_info['end_time'] = datetime.utcnow()
            
            logger.info(f"Dataset generation completed. Total records: {total_records}")
            
            return self.dataset_info
            
        except Exception as e:
            logger.error(f"Error in dataset generation: {e}")
            raise
    
    async def generate_streaming_sample(
        self,
        symbol: str = 'BTC-USDT',
        duration_minutes: int = 10
    ) -> Dict[str, Any]:
        """
        Генерация образца потоковых данных
        
        Args:
            symbol: Символ для потоковой загрузки
            duration_minutes: Длительность в минутах
            
        Returns:
            Dict: Информация о собранных данных
        """
        logger.info(f"Starting streaming sample for {symbol} - {duration_minutes} minutes")
        
        stream_data = {
            'symbol': symbol,
            'duration_minutes': duration_minutes,
            'trades': [],
            'orderbook_snapshots': [],
            'ohlcv_updates': []
        }
        
        try:
            end_time = datetime.utcnow() + timedelta(minutes=duration_minutes)
            
            async for data in self.data_manager.stream_data(symbol, 'trades'):
                if datetime.utcnow() > end_time:
                    break
                
                stream_data['trades'].append(data)
                
                if len(stream_data['trades']) % 100 == 0:
                    logger.info(f"Collected {len(stream_data['trades'])} trades")
            
            logger.info(f"Streaming sample completed. Collected {len(stream_data['trades'])} trades")
            
            # Сохранение потоковых данных
            await self._save_streaming_sample(stream_data)
            
            return stream_data
            
        except Exception as e:
            logger.error(f"Error in streaming sample: {e}")
            raise
    
    async def _save_streaming_sample(self, stream_data: Dict[str, Any]) -> None:
        """Сохранение образца потоковых данных"""
        try:
            # Создаем DataFrame для trades
            if stream_data['trades']:
                trades_df = pd.DataFrame(stream_data['trades'])
                trades_path = f"./data/parquet/trades/{stream_data['symbol']}_stream_sample.parquet"
                
                os.makedirs(os.path.dirname(trades_path), exist_ok=True)
                trades_df.to_parquet(trades_path, compression='snappy')
                
                logger.info(f"Streaming sample saved to {trades_path}")
            
        except Exception as e:
            logger.error(f"Error saving streaming sample: {e}")
    
    async def get_dataset_summary(self) -> Dict[str, Any]:
        """Получение сводной информации о датасете"""
        try:
            summary = {
                'dataset_info': self.dataset_info,
                'data_directory': './data/parquet',
                'directory_structure': {},
                'total_size_mb': 0
            }
            
            # Анализ структуры директорий
            if os.path.exists('./data/parquet'):
                for root, dirs, files in os.walk('./data/parquet'):
                    level = root.replace('./data/parquet', '').count(os.sep)
                    indent = ' ' * 2 * level
                    summary['directory_structure'][root] = {
                        'files': files,
                        'file_count': len(files)
                    }
                    
                    # Подсчет общего размера
                    for file in files:
                        if file.endswith('.parquet'):
                            file_path = os.path.join(root, file)
                            if os.path.exists(file_path):
                                summary['total_size_mb'] += os.path.getsize(file_path) / (1024 * 1024)
            
            # Получение статуса data manager
            if self.data_manager:
                summary['data_manager_status'] = await self.data_manager.get_data_status()
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting dataset summary: {e}")
            return {'error': str(e)}
    
    async def cleanup(self) -> None:
        """Очистка ресурсов"""
        try:
            if self.data_manager:
                await self.data_manager.stop()
                logger.info("Data manager stopped")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


async def main():
    """Главная функция"""
    generator = None
    
    try:
        # Создание генератора
        generator = DatasetGenerator()
        
        # Инициализация
        await generator.initialize()
        
        print("=" * 80)
        print("INTELLIGENT TRADING SYSTEM - DATASET GENERATOR")
        print("=" * 80)
        
        # Генерация большого датасета
        print("\n🔄 Generating comprehensive dataset...")
        dataset_info = await generator.generate_comprehensive_dataset(
            symbols=['BTC-USDT', 'ETH-USDT', 'SOL-USDT', 'BNB-USDT', 'XRP-USDT', 'ADA-USDT', 'DOT-USDT', 'LINK-USDT'],
            timeframes=['1m', '5m', '15m', '1h', '4h'],
            days_back=30,
            data_types=['ohlcv']  # Начинаем с OHLCV для экономии времени
        )
        
        print(f"✅ Dataset generation completed!")
        print(f"📊 Total records: {dataset_info['total_records']:,}")
        print(f"📈 Symbols: {len(dataset_info['symbols'])}")
        print(f"⏱️ Duration: {dataset_info['end_time'] - dataset_info['start_time']}")
        
        # Генерация потокового образца
        print("\n🌊 Generating streaming data sample...")
        stream_data = await generator.generate_streaming_sample(
            symbol='BTC-USDT',
            duration_minutes=5
        )
        
        print(f"✅ Streaming sample completed!")
        print(f"📊 Trades collected: {len(stream_data['trades'])}")
        
        # Получение сводной информации
        print("\n📋 Generating dataset summary...")
        summary = await generator.get_dataset_summary()
        
        print("\n" + "=" * 80)
        print("DATASET SUMMARY")
        print("=" * 80)
        
        print(f"📁 Data directory: {summary['data_directory']}")
        print(f"💾 Total size: {summary['total_size_mb']:.2f} MB")
        
        if 'directory_structure' in summary:
            print("\n📂 Directory structure:")
            for path, info in summary['directory_structure'].items():
                if info['file_count'] > 0:
                    print(f"  {path}: {info['file_count']} files")
        
        if 'file_sizes' in dataset_info:
            print("\n📄 Generated files:")
            for file_name, size in dataset_info['file_sizes'].items():
                size_mb = size / (1024 * 1024)
                print(f"  {file_name}.parquet: {size_mb:.2f} MB")
        
        print("\n🎯 Dataset generation completed successfully!")
        print("💡 You can now use the generated data for testing and analysis")
        
    except KeyboardInterrupt:
        print("\n⚠️ Dataset generation interrupted by user")
        
    except Exception as e:
        print(f"\n❌ Error during dataset generation: {e}")
        logger.error(f"Main error: {e}")
        
    finally:
        if generator:
            await generator.cleanup()
        
        print("\n👋 Dataset generator finished")


if __name__ == "__main__":
    # Запуск генератора
    asyncio.run(main())
