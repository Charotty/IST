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
from feature_engineering.feature_manager import FeatureManager
from utils.logger import LoggerConfig


class DataLayerApp:
    """Основное приложение Data Layer"""
    
    def __init__(self):
        self.config = {}
        self.data_manager = None
        self.feature_manager = None
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
            
            # Инициализация менеджера признаков
            self.feature_manager = FeatureManager(self.config)
            
            self.logger.info("Data Layer initialized successfully")
            self.logger.info("Feature Engineering Layer initialized successfully")
            
        except Exception as e:
            print(f"Failed to initialize Data Layer: {e}")
            sys.exit(1)
    
    async def start(self) -> None:
        """Запуск Data Layer"""
        try:
            self.logger.info("Starting Data Layer...")
            
            # Запуск менеджера данных
            await self.data_manager.start()
            
            # Запуск менеджера признаков
            await self.feature_manager.start()
            
            self._running = True
            self.logger.info("Data Layer started successfully")
            self.logger.info("Feature Engineering Layer started successfully")
            
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
                # Остановка менеджера признаков
                if self.feature_manager:
                    await self.feature_manager.stop()
            
            # Остановка менеджера данных
            await self.data_manager.stop()
            
            self.logger.info("Data Layer stopped")
            self.logger.info("Feature Engineering Layer stopped")
            
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
        """Демонстрация работы Data Layer и Feature Engineering"""
        try:
            # Ожидание инициализации
            await asyncio.sleep(2)
            
            # Демонстрация получения исторических данных
            await self._demo_historical_data()
            
            # Демонстрация Feature Engineering
            await self._demo_feature_engineering()
            
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
                if data and data.get('data'):
                    self.logger.info(f"Последняя запись: {data['data'][-1]}")
                    
                    # Демонстрация Feature Engineering на исторических данных
                    await self._demo_feature_engineering_historical(data['data'])
                    
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
    
    async def _demo_feature_engineering_historical(self, historical_data) -> None:
        """Демонстрация Feature Engineering на исторических данных"""
        try:
            self.logger.info("--- Feature Engineering on Historical Data ---")
            
            if not historical_data:
                self.logger.warning("No historical data for feature engineering")
                return
            
            # Генерация признаков
            features = await self.feature_manager.generate_features(
                historical_data,
                'ohlcv',
                apply_scaling=True
            )
            
            if not features.empty:
                self.logger.info(f"Generated {len(features.columns)} features from {len(historical_data)} OHLCV records")
                self.logger.info(f"Feature names: {list(features.columns[:10])}")
                self.logger.info(f"Sample features values:\n{features.head(3).to_string()}")
            else:
                self.logger.warning("No features generated from historical data")
                
        except Exception as e:
            self.logger.error(f"Error in historical feature engineering demo: {e}")
    
    async def _demo_feature_engineering(self) -> None:
        """Демонстрация работы Feature Engineering Layer"""
        try:
            self.logger.info("=== Feature Engineering Demo ===")
            
            # Получение метрик Feature Manager
            metrics = self.feature_manager.get_metrics()
            self.logger.info(f"Feature Manager metrics: {metrics}")
            
            # Получение имен признаков для разных типов данных
            for data_type in ['ohlcv', 'trades', 'orderbook']:
                feature_names = self.feature_manager.get_feature_names(data_type)
                self.logger.info(f"{data_type} features: {len(feature_names)} available")
                if feature_names:
                    self.logger.info(f"Sample features: {feature_names[:5]}")
            
            # Демонстрация на потоковых данных
            await self._demo_feature_engineering_streaming()
            
        except Exception as e:
            self.logger.error(f"Error in feature engineering demo: {e}")
    
    async def _demo_feature_engineering_streaming(self) -> None:
        """Демонстрация Feature Engineering на потоковых данных"""
        try:
            self.logger.info("--- Feature Engineering on Streaming Data ---")
            
            # Сбор небольшого количества данных для демонстрации
            trades_data = []
            message_count = 0
            max_messages = 20
            
            self.logger.info(f"Collecting {max_messages} trades messages for feature engineering...")
            
            async for data in self.data_manager.stream_data('BTC-USDT', 'trades'):
                trades_data.append(data)
                message_count += 1
                
                if message_count >= max_messages:
                    break
                
                if message_count % 5 == 0:
                    self.logger.info(f"Collected {message_count}/{max_messages} messages")
            
            if trades_data:
                self.logger.info(f"Processing feature engineering on {len(trades_data)} trades...")
                
                # Генерация признаков
                features = await self.feature_manager.generate_features(
                    trades_data,
                    'trades',
                    apply_scaling=True
                )
                
                if not features.empty:
                    self.logger.info(f"✅ Generated {len(features.columns)} features from trades")
                    self.logger.info(f"📋 Feature names: {list(features.columns[:8])}")
                    self.logger.info(f"📊 Sample features:\n{features.head(2).to_string()}")
                else:
                    self.logger.warning("❌ No features generated from trades")
            else:
                self.logger.warning("No trades data collected")
                
        except Exception as e:
            self.logger.error(f"Error in streaming feature engineering demo: {e}")
    
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
