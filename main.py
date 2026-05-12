#!/usr/bin/env python3
"""
Main entry point for Intelligent Trading System Data Layer
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path
from typing import Dict, Any
import os

import yaml
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from data_layer.data_manager import DataManager
from utils.logger import LoggerConfig


class DataLayerApp:
    """Основное приложение Data Layer"""
    
    def __init__(self):
        self.config = {}
        self.data_manager = None
        self.logger = None
        self._running = False
        
    async def initialize(self) -> None:
        """Инициализация приложения"""
        try:
            # Загрузка переменных окружения
            load_dotenv()
            
            # Загрузка конфигурации
            self.config = self._load_config()
            
            # Настройка логирования
            self._setup_logging()
            
            # Инициализация менеджера данных
            self.data_manager = DataManager(self.config)
            
            self.logger.info("Data Layer initialized successfully")
            
        except Exception as e:
            print(f"Failed to initialize Data Layer: {e}")
            sys.exit(1)
    
    async def start(self) -> None:
        """Запуск Data Layer"""
        try:
            self.logger.info("Starting Data Layer...")
            
            # Запуск менеджера данных
            await self.data_manager.start()
            
            self._running = True
            self.logger.info("Data Layer started successfully")
            
            # Запуск демонстрации работы
            await self._run_demo()
            
        except Exception as e:
            self.logger.error(f"Failed to start Data Layer: {e}")
            raise
    
    async def stop(self) -> None:
        """Остановка Data Layer"""
        try:
            self.logger.info("Stopping Data Layer...")
            
            self._running = False
            
            # Остановка менеджера данных
            if self.data_manager:
                await self.data_manager.stop()
            
            self.logger.info("Data Layer stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping Data Layer: {e}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Загрузка конфигурации"""
        config_path = project_root / 'config.yaml'
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Подстановка переменных окружения
        return self._substitute_env_vars(config)
    
    def _substitute_env_vars(self, obj: Any) -> Any:
        """Рекурсивная подстановка переменных окружения"""
        if isinstance(obj, dict):
            return {k: self._substitute_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._substitute_env_vars(item) for item in obj]
        elif isinstance(obj, str) and obj.startswith('${') and obj.endswith('}'):
            env_var = obj[2:-1]
            return os.getenv(env_var, obj)
        else:
            return obj
    
    def _setup_logging(self) -> None:
        """Настройка логирования"""
        logging_config = self.config.get('logging', {})
        
        # Базовая конфигурация если не указана
        if not logging_config:
            logging_config = LoggerConfig.default_config()
        
        LoggerConfig.setup_logging(logging_config)
        self.logger = logging.getLogger('DataLayerApp')
    
    async def _run_demo(self) -> None:
        """Демонстрация работы Data Layer"""
        try:
            # Ожидание инициализации
            await asyncio.sleep(2)
            
            # Демонстрация получения исторических данных
            await self._demo_historical_data()
            
            # Демонстрация потоковых данных
            await self._demo_streaming_data()
            
        except Exception as e:
            self.logger.error(f"Error in demo: {e}")
    
    async def _demo_historical_data(self) -> None:
        """Демонстрация получения исторических данных"""
        try:
            self.logger.info("=== Historical Data Demo ===")
            
            # Демонстрация получения исторических данных
            try:
                end_time = datetime.utcnow()
                start_time = end_time - timedelta(hours=1)
                
                # Используем BTC-USDT с реальными данными
                data = await self.data_manager.get_historical_data(
                    data_type='ohlcv',
                    symbol='BTC-USDT',
                    timeframe='1m',
                    start_time=start_time,
                    end_time=end_time
                )
                
                self.logger.info(f"Получено исторических данных: {len(data.get('data', []))} записей")
                if data.get('data'):
                    self.logger.info(f"Последняя запись: {data['data'][-1]}")
                
            except Exception as e:
                self.logger.error(f"Error in historical data demo: {e}")
            
            await asyncio.sleep(1)
            
        except Exception as e:
            self.logger.error(f"Error in historical data demo: {e}")
    
    async def _demo_streaming_data(self) -> None:
        """Демонстрация потоковых данных"""
        try:
            self.logger.info("=== Streaming Data Demo ===")
            
            # Потоковые данные (короткая демонстрация)
            message_count = 0
            start_time = datetime.utcnow()
            
            async for data in self.data_manager.stream_data(
                symbol='BTC-USDT',
                data_type='trades'
            ):
                message_count += 1
                
                # Логирование каждые 10 сообщений
                if message_count % 10 == 0:
                    elapsed = (datetime.utcnow() - start_time).total_seconds()
                    rate = message_count / elapsed if elapsed > 0 else 0
                    
                    self.logger.info(
                        f"Streamed {message_count} messages "
                        f"({rate:.1f} msg/sec) - Latest: {data.get('price', 'N/A')}"
                    )
                
                # Остановка через 30 секунд
                if (datetime.utcnow() - start_time).total_seconds() > 30:
                    break
            
            self.logger.info(f"Streaming demo completed. Total messages: {message_count}")
            
        except Exception as e:
            self.logger.error(f"Error in streaming demo: {e}")
    
    def _signal_handler(self, signum, frame) -> None:
        """Обработчик сигналов"""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self._running = False


async def main():
    """Главная функция"""
    app = DataLayerApp()
    
    # Настройка обработчиков сигналов
    signal.signal(signal.SIGINT, app._signal_handler)
    signal.signal(signal.SIGTERM, app._signal_handler)
    
    try:
        # Инициализация и запуск
        await app.initialize()
        await app.start()
        
    except KeyboardInterrupt:
        print("\nReceived keyboard interrupt, shutting down...")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)
    finally:
        # Очистка
        await app.stop()


if __name__ == "__main__":
    # Запуск приложения
    asyncio.run(main())
