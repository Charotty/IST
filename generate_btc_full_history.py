#!/usr/bin/env python3
"""
Генератор полного исторического датасета BTC-USDT с 2020 по 2026 год

Скрипт для загрузки 6-летней истории BTC-USDT по всем таймфреймам
с учетом ограничений API и разбивкой на периоды
"""

import asyncio
import logging
import sys
import os
import argparse
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import yaml
from dotenv import load_dotenv
import time

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
        logging.FileHandler('btc_history_generation.log')
    ]
)

logger = logging.getLogger(__name__)


class BTCHistoryGenerator:
    """Генератор полной истории BTC-USDT"""
    
    def __init__(self, config_path: str = "config.yaml", timeframes: Optional[List[str]] = None):
        """
        Инициализация генератора
        
        Args:
            config_path: Путь к конфигурационному файлу
            timeframes: Список таймфреймов для загрузки (если None, загружаются все)
        """
        self.config = self._load_config(config_path)
        self.data_manager = None
        self.symbol = "BTC-USDT"
        
        # Все доступные таймфреймы
        all_timeframes = ['1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d', '1w', '1M']
        
        # Используем указанные таймфреймы или все
        self.timeframes = timeframes if timeframes else all_timeframes
        
        # Период с 2020 по 2026
        self.start_date = datetime(2020, 1, 1)
        self.end_date = datetime(2026, 5, 12)  # Текущая дата
        
        # Статистика
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_records': 0,
            'start_time': None,
            'end_time': None,
            'timeframes_completed': [],
            'errors': []
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
            logger.info("Initializing BTC History Generator...")
            
            # Инициализация DataManager
            self.data_manager = DataManager(self.config)
            await self.data_manager.start()
            
            logger.info("BTC History Generator initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing: {e}")
            raise
    
    def _split_date_range(self, start_date: datetime, end_date: datetime, timeframe: str) -> List[Tuple[datetime, datetime]]:
        """
        Разбивка диапазона дат на периоды с учетом ограничений API
        
        Args:
            start_date: Начальная дата
            end_date: Конечная дата
            timeframe: Таймфрейм
            
        Returns:
            List[Tuple]: Список периодов (start, end)
        """
        periods = []
        
        # Определяем максимальный период для каждого таймфрейма
        # OKX ограничивает количество свечей за один запрос
        max_periods = {
            '1m': 1440,    # 1 день
            '5m': 288,     # 1 день
            '15m': 96,     # 1 день
            '30m': 48,     # 1 день
            '1h': 720,     # 30 дней
            '2h': 360,     # 30 дней
            '4h': 180,     # 30 дней
            '6h': 120,     # 30 дней
            '12h': 60,     # 30 дней
            '1d': 365,     # 1 год
            '1w': 52,      # 1 год
            '1M': 12       # 1 год
        }
        
        # Определяем период в днях для каждого таймфрейма
        period_days = {
            '1m': 1, '5m': 1, '15m': 1, '30m': 1,
            '1h': 30, '2h': 30, '4h': 30, '6h': 30, '12h': 30,
            '1d': 365, '1w': 365, '1M': 365
        }
        
        days_per_period = period_days.get(timeframe, 30)
        
        current_start = start_date
        while current_start < end_date:
            current_end = min(current_start + timedelta(days=days_per_period), end_date)
            periods.append((current_start, current_end))
            current_start = current_end
        
        logger.info(f"Split {timeframe} into {len(periods)} periods")
        return periods
    
    async def download_timeframe_data(self, timeframe: str) -> bool:
        """
        Загрузка данных для одного таймфрейма через CCXT с сохранением в один файл
        
        Args:
            timeframe: Таймфрейм
            
        Returns:
            bool: Успешность загрузки
        """
        try:
            logger.info(f"Starting download for {self.symbol} {timeframe}")
            logger.info(f"Period: {self.start_date.date()} to {self.end_date.date()}")
            
            # Используем CCXT для получения полной истории
            self.stats['total_requests'] += 1
            
            # Получаем данные через CCXT коннектор напрямую
            from data_layer.connectors.ccxt_connector import CCXTConnector
            
            ccxt_connector = CCXTConnector(exchange_name='binance')
            await ccxt_connector.connect()
            
            try:
                # Получаем полную историю с автоматической пагинацией
                data = await ccxt_connector.get_full_historical_data(
                    symbol=self.symbol,
                    timeframe=timeframe,
                    start_time=self.start_date,
                    end_time=self.end_date,
                    limit=1000
                )
                
                if data and 'data' in data:
                    df = pd.DataFrame(data['data'])
                    if not df.empty:
                        # Сохраняем все данные в один файл
                        await self._save_timeframe_data(df, timeframe)
                        
                        self.stats['successful_requests'] += 1
                        self.stats['total_records'] += len(df)
                        logger.info(f"Successfully loaded and saved {len(df)} records for {timeframe}")
                        return True
                    else:
                        logger.error(f"No data received for {timeframe}")
                        self.stats['failed_requests'] += 1
                        return False
                else:
                    logger.error(f"Failed to load data for {timeframe}")
                    self.stats['failed_requests'] += 1
                    return False
            finally:
                await ccxt_connector.disconnect()
                
        except Exception as e:
            logger.error(f"Error in download_timeframe_data for {timeframe}: {e}")
            self.stats['errors'].append(f"{timeframe}: {str(e)}")
            return False
    
    async def _save_timeframe_data(self, df: pd.DataFrame, timeframe: str) -> None:
        """Сохранение данных таймфрейма в один файл"""
        try:
            # Создаем директорию если нужно
            save_dir = f"./data/parquet/ohlcv"
            os.makedirs(save_dir, exist_ok=True)
            
            # Путь к файлу
            file_path = f"{save_dir}/{self.symbol}_{timeframe}_full_history.parquet"
            
            # Сохраняем в один файл (перезаписываем если существует)
            df.to_parquet(file_path, compression='snappy', index=False)
            
            logger.info(f"Saved {len(df)} records to {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving data for {timeframe}: {e}")
            raise
    
    async def generate_full_history(self) -> Dict[str, Any]:
        """
        Генерация полной истории для всех таймфреймов
        
        Returns:
            Dict: Статистика выполнения
        """
        self.stats['start_time'] = datetime.utcnow()
        
        logger.info("=" * 80)
        logger.info("STARTING BTC-USDT FULL HISTORY GENERATION")
        logger.info("=" * 80)
        logger.info(f"Symbol: {self.symbol}")
        logger.info(f"Period: {self.start_date.date()} to {self.end_date.date()}")
        logger.info(f"Timeframes: {len(self.timeframes)}")
        logger.info(f"Total duration: {(self.end_date - self.start_date).days} days")
        
        try:
            # Загружаем данные для каждого таймфрейма
            for timeframe in self.timeframes:
                try:
                    success = await self.download_timeframe_data(timeframe)
                    
                    if success:
                        self.stats['timeframes_completed'].append(timeframe)
                        logger.info(f"Completed {timeframe}")
                    else:
                        logger.error(f"Failed {timeframe}")
                    
                    # Задержка между таймфреймами
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Critical error processing {timeframe}: {e}")
                    self.stats['errors'].append(f"Critical {timeframe}: {str(e)}")
                    continue
            
            self.stats['end_time'] = datetime.utcnow()
            
            # Вывод статистики
            self._print_final_stats()
            
            return self.stats
            
        except Exception as e:
            logger.error(f"Critical error in generate_full_history: {e}")
            self.stats['errors'].append(f"Critical: {str(e)}")
            raise
    
    def _print_final_stats(self) -> None:
        """Вывод финальной статистики"""
        duration = self.stats['end_time'] - self.stats['start_time']
        
        print("\n" + "=" * 80)
        print("BTC-USDT FULL HISTORY GENERATION COMPLETED")
        print("=" * 80)
        
        print(f"Duration: {duration}")
        print(f"Total requests: {self.stats['total_requests']}")
        print(f"Successful: {self.stats['successful_requests']}")
        print(f"Failed: {self.stats['failed_requests']}")
        print(f"Total records: {self.stats['total_records']:,}")
        print(f"Timeframes completed: {len(self.stats['timeframes_completed'])}/{len(self.timeframes)}")
        
        print(f"\nCompleted timeframes:")
        for tf in sorted(self.stats['timeframes_completed']):
            print(f"   {tf}")
        
        if self.stats['timeframes_completed'] != self.timeframes:
            print(f"\nFailed timeframes:")
            for tf in self.timeframes:
                if tf not in self.stats['timeframes_completed']:
                    print(f"   {tf}")
        
        if self.stats['errors']:
            print(f"\nErrors encountered: {len(self.stats['errors'])}")
            for error in self.stats['errors'][:5]:  # Показываем первые 5 ошибок
                print(f"   {error}")
            if len(self.stats['errors']) > 5:
                print(f"   ... and {len(self.stats['errors']) - 5} more errors")
        
        print(f"\nData saved to: ./data/parquet/ohlcv/")
        print(f"You can now use this comprehensive dataset for analysis!")
    
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
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(
        description='Генератор исторических данных BTC-USDT',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  # Загрузить все таймфреймы
  python generate_btc_full_history.py
  
  # Загрузить только 1h таймфрейм
  python generate_btc_full_history.py --timeframe 1h
  
  # Загрузить несколько таймфреймов
  python generate_btc_full_history.py --timeframes 1h 4h 1d
  
  # Загрузить с自定义 датами
  python generate_btc_full_history.py --start 2023-01-01 --end 2024-01-01
        """
    )
    
    parser.add_argument(
        '--timeframe', '-t',
        type=str,
        help='Один таймфрейм для загрузки (например: 1h, 4h, 1d)'
    )
    
    parser.add_argument(
        '--timeframes',
        nargs='+',
        type=str,
        help='Список таймфреймов для загрузки (например: 1h 4h 1d)'
    )
    
    parser.add_argument(
        '--start',
        type=str,
        help='Начальная дата в формате YYYY-MM-DD (по умолчанию: 2020-01-01)'
    )
    
    parser.add_argument(
        '--end',
        type=str,
        help='Конечная дата в формате YYYY-MM-DD (по умолчанию: текущая дата)'
    )
    
    args = parser.parse_args()
    
    # Определение таймфреймов
    timeframes = None
    if args.timeframe:
        timeframes = [args.timeframe]
    elif args.timeframes:
        timeframes = args.timeframes
    
    # Определение дат
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2026, 5, 12)
    
    if args.start:
        start_date = datetime.strptime(args.start, '%Y-%m-%d')
    
    if args.end:
        end_date = datetime.strptime(args.end, '%Y-%m-%d')
    
    generator = None
    
    try:
        # Создание генератора
        generator = BTCHistoryGenerator(timeframes=timeframes)
        
        # Обновление дат если указаны
        if args.start or args.end:
            generator.start_date = start_date
            generator.end_date = end_date
        
        # Инициализация
        await generator.initialize()
        
        # Генерация полной истории
        await generator.generate_full_history()
        
    except KeyboardInterrupt:
        print("\n⚠️ Generation interrupted by user")
        
    except Exception as e:
        print(f"\n❌ Error during generation: {e}")
        logger.error(f"Main error: {e}")
        
    finally:
        if generator:
            await generator.cleanup()
        
        print("\n👋 BTC History Generator finished")


if __name__ == "__main__":
    # Запуск генератора
    asyncio.run(main())
