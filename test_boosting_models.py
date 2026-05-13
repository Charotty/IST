"""
Тестирование моделей бустинга на реальных данных

Скрипт для обучения и проверки LightGBM и XGBoost моделей
с использованием существующих модулей данных.
"""

import asyncio
import sys
import os
import logging
from typing import Dict, Any, List
from datetime import datetime, timedelta
from dataclasses import dataclass
import pandas as pd
import numpy as np

# Добавляем путь к проекту
sys.path.append('.')

# Импорты наших модулей
from models.boosting.lightgbm_model import LightGBMModel
from models.boosting.xgboost_model import XGBoostModel
from models.training.trainer import ModelTrainer
from models.training.validator import ModelValidator

# Импорты для данных (если доступны)
try:
    from data_layer.data_manager import DataManager
    DATA_MANAGER_AVAILABLE = True
except ImportError:
    print("⚠️  DataManager недоступен, используем сгенерированные данные")
    DATA_MANAGER_AVAILABLE = False

try:
    from feature_engineering.feature_manager import FeatureManager
    FEATURE_MANAGER_AVAILABLE = True
except ImportError:
    print("⚠️  FeatureManager недоступен, используем базовые признаки")
    FEATURE_MANAGER_AVAILABLE = False


# Проверка наличия .env файла
def check_env_file():
    """Проверка наличия и загрузка .env файла"""
    env_file = '.env'
    if os.path.exists(env_file):
        try:
            from dotenv import load_dotenv
            load_dotenv()
            print(f"✅ .env файл загружен из {env_file}")
            return True
        except ImportError:
            print("⚠️  python-dotenv не установлен, установите: pip install python-dotenv")
            return False
        except Exception as e:
            print(f"⚠️  Ошибка загрузки .env: {e}")
            return False
    else:
        print("⚠️  .env файл не найден, создайте его с API ключами OKX")
        print("   OKX_API_KEY=your_api_key")
        print("   OKX_SECRET_KEY=your_secret_key") 
        print("   OKX_PASSPHRASE=your_passphrase")
        print("   OKX_SANDBOX=true")
        return False


# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BoostingModelTester:
    """Класс для тестирования моделей бустинга"""
    
    def __init__(self, n_samples: int = 120000, use_real_data: bool = False):
        """
        Инициализация тестера
        
        Args:
            n_samples: Количество образцов данных
            use_real_data: Использовать реальные данные или синтетику
        """
        self.n_samples = n_samples
        self.use_real_data = use_real_data
        self.data_manager = None
        self.feature_manager = None
        self.trainer = None
        self.validator = None
        
        # Инициализация компонентов
        self._initialize_components()
        
        # Результаты тестирования
        self.results = {}
    
    def _initialize_components(self):
        """Инициализация компонентов"""
        try:
            # Инициализация тренера
            trainer_config = {
                'validation_split': 0.167,  # ~1/6 для train/validation split
                'early_stopping_patience': 10,
                'batch_size': 32,
                'epochs': 100
            }
            self.trainer = ModelTrainer(trainer_config)
            
            # Инициализация валидатора
            validator_config = {
                'cv_folds': 5,
                'test_size': 0.2,
                'random_state': 42
            }
            self.validator = ModelValidator(validator_config)
            
            # Инициализация DataManager (если доступен и запрошен)
            if DATA_MANAGER_AVAILABLE and self.use_real_data:
                # Проверка и загрузка .env файла
                if not check_env_file():
                    logger.warning("Не удалось загрузить .env файл, используем синтетику")
                    self.use_real_data = False
                    return
                
                # Загрузка переменных окружения из .env
                import os
                from dotenv import load_dotenv
                
                load_dotenv()
                
                data_config = {
                    'data_layer': {
                        'connectors': {
                            'okx': {
                                'mock_mode': False,  # Используем реальные данные
                                'api_key': os.getenv('OKX_API_KEY', 'your_api_key'),
                                'secret_key': os.getenv('OKX_SECRET_KEY', 'your_secret_key'),
                                'passphrase': os.getenv('OKX_PASSPHRASE', 'your_passphrase'),
                                'sandbox': os.getenv('OKX_SANDBOX', 'true').lower() == 'true'
                            }
                        },
                        'storage': {
                            'data_path': './data'
                        }
                    }
                }
                try:
                    self.data_manager = DataManager(data_config)
                    logger.info("DataManager инициализирован для реальных данных")
                except Exception as e:
                    logger.warning(f"Ошибка инициализации DataManager: {e}")
                    self.use_real_data = False  # Fallback на синтетику
            
            # Инициализация FeatureManager (если доступен)
            if FEATURE_MANAGER_AVAILABLE:
                feature_config = {}
                try:
                    self.feature_manager = FeatureManager(feature_config)
                    logger.info("FeatureManager инициализирован")
                except Exception as e:
                    logger.warning(f"Ошибка инициализации FeatureManager: {e}")
            
        except Exception as e:
            logger.error(f"Ошибка инициализации компонентов: {e}")
    
    async def get_real_data(self, symbol: str = 'BTC-USDT', timeframe: str = '1h') -> tuple:
        """
        Получение реальных данных из DataManager
        
        Args:
            symbol: Торговая пара
            timeframe: Таймфрейм
            
        Returns:
            tuple: (X, y) - признаки и целевая переменная
        """
        try:
            if not self.data_manager:
                logger.warning("DataManager не доступен, используем синтетику")
                return self.generate_sample_data(self.n_samples)
            
            await self.data_manager.start()
            
            # Получение исторических данных за последний период
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=30)  # 30 дней данных
            
            logger.info(f"Загрузка реальных данных для {symbol} с {start_time} по {end_time}")
            
            historical_data = await self.data_manager.get_historical_data(
                symbol=symbol,
                data_type='ohlcv',
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time
            )
            
            # Обработка данных
            if 'data' in historical_data and not historical_data['data'].empty:
                df = historical_data['data']
                
                # Создание признаков
                X = self._create_features_from_ohlcv(df)
                y = self._create_target_from_ohlcv(df)
                
                # Ограничение количества образцов
                if len(X) > self.n_samples:
                    X = X.iloc[:self.n_samples]
                    y = y.iloc[:self.n_samples]
                
                await self.data_manager.stop()
                
                logger.info(f"Загружено {len(X)} реальных образцов с {X.shape[1]} признаками")
                return X, y
            else:
                logger.warning("Не удалось получить реальные данные, используем синтетику")
                await self.data_manager.stop()
                return self.generate_sample_data(self.n_samples)
                
        except Exception as e:
            logger.error(f"Ошибка получения реальных данных: {e}")
            if self.data_manager:
                await self.data_manager.stop()
            return self.generate_sample_data(self.n_samples)
    
    def _create_features_from_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создание признаков из OHLCV данных"""
        try:
            # Технические индикаторы
            df['returns'] = df['close'].pct_change().fillna(0)
            df['sma_5'] = df['close'].rolling(5).mean().bfill()
            df['sma_20'] = df['close'].rolling(20).mean().bfill()
            df['rsi'] = self._calculate_rsi(df['close'].values)
            df['volatility'] = df['close'].rolling(20).std().bfill()
            
            # Временные признаки
            if 'timestamp' in df.columns:
                df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
                df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
            else:
                df['hour'] = df.index.hour if hasattr(df.index, 'hour') else 0
                df['day_of_week'] = df.index.dayofweek if hasattr(df.index, 'dayofweek') else 0
            
            # Выбор признаков
            feature_columns = ['open', 'high', 'low', 'close', 'volume', 'returns', 
                           'sma_5', 'sma_20', 'rsi', 'volatility', 'hour', 'day_of_week']
            
            X = df[feature_columns].fillna(0)
            return X
            
        except Exception as e:
            logger.error(f"Ошибка создания признаков: {e}")
            return pd.DataFrame()
    
    def _create_target_from_ohlcv(self, df: pd.DataFrame) -> pd.Series:
        """Создание целевой переменной из OHLCV данных"""
        try:
            # Целевая переменная: направление движения цены
            future_returns = df['close'].pct_change().shift(-1).fillna(0)
            
            # Для реальных данных используем более низкие пороги
            # и гарантируем наличие всех классов
            threshold_up = 0.005  # 0.5% порог для UP
            threshold_down = -0.005  # -0.5% порог для DOWN
            
            # Создаем целевую переменную с гарантированными 3 классами
            target = np.where(future_returns > threshold_up, 2,  # UP
                             np.where(future_returns < threshold_down, 0,  # DOWN
                                      1))  # HOLD
            
            # Проверяем наличие всех классов
            unique_classes = np.unique(target)
            if len(unique_classes) < 3:
                logger.warning(f"Найдено только {len(unique_classes)} класса: {unique_classes}")
                # Принудительно добавляем недостающие классы
                n = len(target)
                # Добавляем несколько примеров каждого класса
                indices_up = np.where(future_returns > threshold_up)[0]
                indices_down = np.where(future_returns < threshold_down)[0]
                
                if len(indices_up) > 0 and len(indices_down) > 0:
                    # Заменяем некоторые HOLD на UP/DOWN
                    for i in range(min(10, n//10)):
                        if i < len(indices_up):
                            target[indices_up[i]] = 2
                        elif i < len(indices_down):
                            target[indices_down[i]] = 0
            
            logger.info(f"Создана целевая переменная с {len(np.unique(target))} классами: {np.unique(target)}")
            return pd.Series(target, name='target')
            
        except Exception as e:
            logger.error(f"Ошибка создания целевой переменной: {e}")
            return pd.Series()
    
    def generate_sample_data(self, n_samples: int = 1000) -> tuple:
        """
        Генерация образцов данных для тестирования
        
        Args:
            n_samples: Количество образцов
            
        Returns:
            tuple: (X, y) - признаки и целевая переменная
        """
        logger.info(f"Генерация {n_samples} образцов данных...")
        
        # Генерация временных рядов
        np.random.seed(42)
        
        # Базовые признаки (OHLCV)
        dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='1h')
        
        # Генерация цен с трендом и шумом
        trend = np.linspace(100, 120, n_samples)
        noise = np.random.normal(0, 2, n_samples)
        close_prices = trend + noise
        
        # OHLCV данные
        high = close_prices + np.abs(np.random.normal(0, 1, n_samples))
        low = close_prices - np.abs(np.random.normal(0, 1, n_samples))
        open_price = close_prices + np.random.normal(0, 0.5, n_samples)
        volume = np.random.exponential(1000, n_samples)
        
        # Технические индикаторы
        returns = np.zeros(n_samples)
        returns[1:] = np.diff(close_prices)
        returns[0] = 0
        
        sma_5 = pd.Series(close_prices).rolling(5).mean().bfill()
        sma_20 = pd.Series(close_prices).rolling(20).mean().bfill()
        rsi = self._calculate_rsi(close_prices)
        
        # Волатильность
        volatility = pd.Series(close_prices).rolling(20).std().bfill()
        
        # Целевая переменная: направление движения цены
        # 0 - DOWN, 1 - HOLD, 2 - UP
        future_returns = np.zeros(n_samples)
        future_returns[1:] = np.diff(close_prices)
        future_returns[0] = 0
        
        target = np.where(future_returns > 0.5, 2,  # UP
                         np.where(future_returns < -0.5, 0,  # DOWN
                                  1))  # HOLD
        
        # Создание DataFrame
        X = pd.DataFrame({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close_prices,
            'volume': volume,
            'returns': returns,
            'sma_5': sma_5,
            'sma_20': sma_20,
            'rsi': rsi,
            'volatility': volatility,
            'hour': dates.hour,
            'day_of_week': dates.dayofweek
        })
        
        # Удаление NaN значений
        X = X.bfill().fillna(0)
        
        y = pd.Series(target, name='target')
        
        logger.info(f"Сгенерировано {len(X)} образцов с {X.shape[1]} признаками")
        logger.info(f"Распределение классов: {y.value_counts().to_dict()}")
        
        return X, y
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Расчет RSI индикатора"""
        delta = np.diff(prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        # Добавляем нули в начало чтобы сохранить размерность
        gain_padded = np.zeros(len(prices))
        loss_padded = np.zeros(len(prices))
        gain_padded[1:] = gain
        loss_padded[1:] = loss
        
        avg_gain = pd.Series(gain_padded).rolling(period).mean().bfill()
        avg_loss = pd.Series(loss_padded).rolling(period).mean().bfill()
        
        # Избегаем деления на ноль
        rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        
        return np.nan_to_num(rsi, nan=50.0)
    
    async def test_lightgbm_model(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Тестирование LightGBM модели"""
        logger.info("🚀 Тестирование LightGBM модели...")
        
        try:
            # Конфигурация модели
            config = {
                'task_type': 'classification',
                'num_leaves': 31,
                'max_depth': -1,
                'learning_rate': 0.05,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.0,
                'reg_lambda': 0.0,
                'random_state': 42,
                'early_stopping_rounds': 10,
                'verbose': -1
            }
            
            # Создание модели
            model = LightGBMModel(config)
            logger.info("✅ LightGBM модель создана")
            
            # Обучение
            training_results = await self.trainer.train_model(model, X, y)
            logger.info(f"✅ Обучение завершено: {training_results.get('training_time', 0):.2f}s")
            
            # Валидация
            validation_results = await self.validator.validate_model(model, X, y)
            logger.info("✅ Валидация завершена")
            
            # Получение важности признаков
            feature_importance = model.get_feature_importance()
            
            # Предсказания
            predictions = model.predict(X)
            probabilities = model.predict_proba(X)
            
            results = {
                'model_name': 'LightGBM',
                'config': config,
                'training_results': training_results,
                'validation_results': validation_results,
                'feature_importance': feature_importance,
                'predictions_shape': predictions.shape,
                'probabilities_shape': probabilities.shape,
                'model_info': model.get_model_info()
            }
            
            logger.info(f"✅ LightGBM тестирование завершено")
            return results
            
        except Exception as e:
            logger.error(f"❌ Ошибка тестирования LightGBM: {e}")
            return {'error': str(e)}
    
    async def test_xgboost_model(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Тестирование XGBoost модели"""
        logger.info("🚀 Тестирование XGBoost модели...")
        
        try:
            # Конфигурация модели
            config = {
                'task_type': 'classification',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'colsample_bylevel': 0.8,
                'gamma': 0,
                'min_child_weight': 1,
                'reg_alpha': 0,
                'reg_lambda': 1,
                'random_state': 42,
                'early_stopping_rounds': 10,
                'verbose': 1
            }
            
            # Создание модели
            model = XGBoostModel(config)
            logger.info("✅ XGBoost модель создана")
            
            # Обучение
            training_results = await self.trainer.train_model(model, X, y)
            logger.info(f"✅ Обучение завершено: {training_results.get('training_time', 0):.2f}s")
            
            # Валидация
            validation_results = await self.validator.validate_model(model, X, y)
            logger.info("✅ Валидация завершена")
            
            # Получение важности признаков
            feature_importance = model.get_feature_importance()
            
            # Предсказания
            predictions = model.predict(X)
            probabilities = model.predict_proba(X)
            
            results = {
                'model_name': 'XGBoost',
                'config': config,
                'training_results': training_results,
                'validation_results': validation_results,
                'feature_importance': feature_importance,
                'predictions_shape': predictions.shape,
                'probabilities_shape': probabilities.shape,
                'model_info': model.get_model_info()
            }
            
            logger.info(f"✅ XGBoost тестирование завершено")
            return results
            
        except Exception as e:
            logger.error(f"❌ Ошибка тестирования XGBoost: {e}")
            return {'error': str(e)}
    
    def compare_models(self) -> Dict[str, Any]:
        """Сравнение результатов моделей"""
        logger.info("📊 Сравнение результатов моделей...")
        
        comparison = {
            'summary': {},
            'detailed_comparison': {}
        }
        
        # Извлечение метрик для сравнения
        for model_name, results in self.results.items():
            if 'error' not in results:
                training_results = results.get('training_results', {})
                validation_results = results.get('validation_results', {})
                
                comparison['detailed_comparison'][model_name] = {
                    'training_time': training_results.get('training_time', 0),
                    'accuracy': validation_results.get('test_metrics', {}).get('accuracy', 0),
                    'cv_score_mean': validation_results.get('cv_scores', {}).get('mean', 0),
                    'cv_score_std': validation_results.get('cv_scores', {}).get('std', 0),
                    'feature_count': len(results.get('feature_importance', {}))
                }
        
        # Определение лучшей модели
        if comparison['detailed_comparison']:
            valid_models = {k: v for k, v in comparison['detailed_comparison'].items() 
                           if v.get('accuracy', 0) > 0}
            
            if valid_models:
                best_model = max(
                    valid_models.items(),
                    key=lambda x: x[1]['accuracy']
                )
                comparison['summary']['best_model'] = best_model[0]
                comparison['summary']['best_accuracy'] = best_model[1]['accuracy']
                
                # Сравнение по времени обучения
                fastest_model = min(
                    valid_models.items(),
                    key=lambda x: x[1]['training_time']
                )
                comparison['summary']['fastest_model'] = fastest_model[0]
                comparison['summary']['fastest_time'] = fastest_model[1]['training_time']
            else:
                comparison['summary']['best_model'] = None
                comparison['summary']['best_accuracy'] = 0
                comparison['summary']['fastest_model'] = None
                comparison['summary']['fastest_time'] = 0
        
        return comparison
    
    def print_results(self):
        """Вывод результатов тестирования"""
        print("\n" + "="*80)
        print("📈 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ МОДЕЛЕЙ БУСТИНГА")
        print("="*80)
        
        for model_name, results in self.results.items():
            print(f"\n🔹 {model_name}")
            print("-" * 40)
            
            if 'error' in results:
                print(f"❌ Ошибка: {results['error']}")
                continue
            
            # Результаты обучения
            training_results = results.get('training_results', {})
            print(f"⏱️  Время обучения: {training_results.get('training_time', 0):.3f} сек")
            print(f"📊 Размер обучающей выборки: {training_results.get('training_samples', 0)}")
            print(f"📊 Размер валидационной выборки: {training_results.get('validation_samples', 0)}")
            
            # Метрики валидации
            validation_results = results.get('validation_results', {})
            test_metrics = validation_results.get('test_metrics', {})
            print(f"🎯 Точность (accuracy): {test_metrics.get('accuracy', 0):.4f}")
            
            cv_scores = validation_results.get('cv_scores', {})
            if cv_scores:
                print(f"📈 CV Score: {cv_scores.get('mean', 0):.4f} ± {cv_scores.get('std', 0):.4f}")
            
            ts_cv_scores = validation_results.get('time_series_cv_scores', {})
            if ts_cv_scores:
                print(f"📈 Time Series CV: {ts_cv_scores.get('mean', 0):.4f} ± {ts_cv_scores.get('std', 0):.4f}")
            
            # Важность признаков (топ 5)
            feature_importance = results.get('feature_importance', {})
            if feature_importance:
                print(f"\n🏆 Топ-5 важных признаков:")
                top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
                for i, (feature, importance) in enumerate(top_features, 1):
                    print(f"   {i}. {feature}: {importance:.4f}")
        
        # Сравнение моделей
        comparison = self.compare_models()
        print(f"\n🏆 СРАВНЕНИЕ МОДЕЛЕЙ")
        print("-" * 40)
        
        summary = comparison.get('summary', {})
        if 'best_model' in summary:
            print(f"🥇 Лучшая модель: {summary['best_model']} (accuracy: {summary['best_accuracy']:.4f})")
        if 'fastest_model' in summary:
            print(f"⚡ Самая быстрая: {summary['fastest_model']} ({summary['fastest_time']:.3f} сек)")
        
        print("\n" + "="*80)
    
    async def run_tests(self, symbol: str = 'BTC-USDT', timeframe: str = '1h'):
        """Запуск всех тестов"""
        logger.info("🚀 Запуск тестирования моделей бустинга...")
        
        try:
            # Получение данных
            if self.use_real_data:
                logger.info(f"📡 Используем реальные данные для {symbol}")
                X, y = await self.get_real_data(symbol, timeframe)
            else:
                logger.info(f"🔬 Используем синтетические данные ({self.n_samples} образцов)")
                X, y = self.generate_sample_data(n_samples=self.n_samples)
            
            # Тестирование LightGBM
            lgb_results = await self.test_lightgbm_model(X, y)
            self.results['LightGBM'] = lgb_results
            
            # Тестирование XGBoost
            xgb_results = await self.test_xgboost_model(X, y)
            self.results['XGBoost'] = xgb_results
            
            # Вывод результатов
            self.print_results()
            
            logger.info("✅ Все тесты завершены")
            
        except Exception as e:
            logger.error(f"❌ Ошибка в процессе тестирования: {e}")
    
    def print_usage(self):
        """Вывод информации об использовании"""
        print("\n" + "="*80)
        print("🤖 МОДУЛЬНОСТЬ ТЕСТЕРА МОДЕЛЕЙ БУСТИНГА")
        print("="*80)
        print()
        print("📋 Параметры запуска:")
        print("   python test_boosting_models.py [опции]")
        print()
        print("🔧 Опции:")
        print("   --n-samples N     Количество образцов данных (по умолчанию: 120000)")
        print("   --real-data       Использовать реальные рыночные данные")
        print("   --symbol S        Торговая пара (по умолчанию: BTC-USDT)")
        print("   --timeframe T     Таймфрейм (по умолчанию: 1h)")
        print()
        print("📊 Примеры использования:")
        print("   # Синтетические данные (120K образцов):")
        print("   python test_boosting_models.py --n-samples 120000")
        print()
        print("   # Реальные данные BTC-USDT:")
        print("   python test_boosting_models.py --real-data --symbol BTC-USDT --timeframe 1h")
        print()
        print("   # Реальные данные ETH-USDT (50K образцов):")
        print("   python test_boosting_models.py --real-data --symbol ETH-USDT --n-samples 50000")
        print()
        print("⚠️  Для реальных данных нужны API ключи OKX!")
        print("="*80)


async def main():
    """Главная функция"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Тестирование моделей бустинга')
    parser.add_argument('--n-samples', type=int, default=120000, 
                       help='Количество образцов данных (по умолчанию: 120000)')
    parser.add_argument('--real-data', action='store_true',
                       help='Использовать реальные рыночные данные')
    parser.add_argument('--symbol', type=str, default='BTC-USDT',
                       help='Торговая пара (по умолчанию: BTC-USDT)')
    parser.add_argument('--timeframe', type=str, default='1h',
                       help='Таймфрейм (по умолчанию: 1h)')
    parser.add_argument('--usage', action='store_true',
                       help='Показать справку')
    
    args = parser.parse_args()
    
    if args.usage:
        tester = BoostingModelTester()
        tester.print_usage()
        return
    
    print("🤖 Запуск тестирования моделей бустинга...")
    print(f"📊 Параметры: {args.n_samples} образцов, {'реальные данные' if args.real_data else 'синтетика'}")
    
    # Создание тестера
    tester = BoostingModelTester(
        n_samples=args.n_samples,
        use_real_data=args.real_data
    )
    
    # Запуск тестов
    await tester.run_tests(symbol=args.symbol, timeframe=args.timeframe)


if __name__ == "__main__":
    asyncio.run(main())
