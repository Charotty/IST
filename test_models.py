"""
Models Layer Test

Тестирование Models Layer с реальными данными BTC-USDT.
"""

import asyncio
import logging
import sys
from datetime import datetime, timedelta
from typing import Dict, Any
import pandas as pd
import numpy as np

# Добавление путей
sys.path.append('.')

from models.model_manager import ModelManager


async def test_models_layer():
    """Тестирование Models Layer"""
    
    print("🚀 Testing Models Layer with real BTC-USDT data...")
    
    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )
    
    try:
        # Конфигурация
        config = {
            'models': {
                'registry': {
                    'registry_path': './models_registry',
                    'max_versions': 5,
                    'auto_cleanup': True
                },
                'training': {
                    'validation_split': 0.2,
                    'early_stopping_patience': 5,
                    'batch_size': 16,
                    'epochs': 10  # Уменьшено для теста
                }
            }
        }
        
        print("\n📦 Initializing Models Layer...")
        
        # Инициализация Model Manager
        model_manager = ModelManager(config)
        
        print("\n🔄 Starting Model Manager...")
        await model_manager.start()
        
        print("\n📊 Testing Model Registration...")
        
        # Тестирование регистрации моделей
        await test_model_registration(model_manager)
        
        print("\n📈 Testing Model Training...")
        
        # Тестирование обучения моделей
        await test_model_training(model_manager)
        
        print("\n🔮 Testing Model Prediction...")
        
        # Тестирование предсказаний
        await test_model_prediction(model_manager)
        
        print("\n📋 Testing Model Management...")
        
        # Тестирование управления моделями
        await test_model_management(model_manager)
        
        print("\n📈 Getting Model Manager Metrics...")
        
        # Метрики Model Manager
        metrics = model_manager.get_metrics()
        print(f"  ✅ Total models: {metrics.get('total_models', 0)}")
        print(f"  ✅ Active models: {metrics.get('active_models', 0)}")
        print(f"  ✅ Predictions made: {metrics.get('predictions_made', 0)}")
        print(f"  ✅ Training jobs: {metrics.get('training_jobs', 0)}")
        print(f"  ✅ Errors count: {metrics.get('errors_count', 0)}")
        
        print("\n🧹 Cleaning up...")
        
        # Очистка
        await model_manager.stop()
        
        print("\n🎉 Models Layer test completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Models Layer test failed: {e}")
        return False


async def test_model_registration(model_manager: ModelManager) -> None:
    """Тестирование регистрации моделей"""
    try:
        print("  📝 Testing model registration...")
        
        # Тестовые конфигурации моделей
        lstm_config = {
            'type': 'lstm',
            'task_type': 'classification',
            'input_dim': 10,
            'hidden_dim': 32,
            'num_layers': 2,
            'output_dim': 3,
            'dropout': 0.2,
            'learning_rate': 0.001,
            'batch_size': 16,
            'epochs': 5,
            'sequence_length': 30
        }
        
        transformer_config = {
            'type': 'transformer',
            'task_type': 'classification',
            'input_dim': 10,
            'd_model': 64,
            'nhead': 4,
            'num_layers': 2,
            'output_dim': 3,
            'dropout': 0.1,
            'learning_rate': 0.0001,
            'batch_size': 16,
            'epochs': 5,
            'sequence_length': 30
        }
        
        # Создание тестовых моделей
        from models.deep_learning.lstm_model import LSTMModel
        from models.deep_learning.transformer_model import TransformerModel
        
        lstm_model = LSTMModel(lstm_config)
        transformer_model = TransformerModel(transformer_config)
        
        # Регистрация моделей
        lstm_id = await model_manager.register_model(
            'lstm_test', lstm_model, lstm_config,
            {'description': 'Test LSTM model for BTC-USDT'}
        )
        
        transformer_id = await model_manager.register_model(
            'transformer_test', transformer_model, transformer_config,
            {'description': 'Test Transformer model for BTC-USDT'}
        )
        
        print(f"    ✅ Registered LSTM model: {lstm_id}")
        print(f"    ✅ Registered Transformer model: {transformer_id}")
        
        # Тестирование списка моделей
        models = await model_manager.list_models()
        print(f"    📋 Total registered models: {len(models)}")
        
        for model in models:
            print(f"      - {model.get('model_id', 'unknown')}: {model.get('model_name', 'unknown')}")
        
    except Exception as e:
        print(f"    ❌ Model registration test error: {e}")


async def test_model_training(model_manager: ModelManager) -> None:
    """Тестирование обучения моделей"""
    try:
        print("  🏋 Testing model training...")
        
        # Генерация тестовых данных
        X_train, y_train = generate_test_data(1000, 10)
        
        # Получение списка моделей
        models = await model_manager.list_models()
        
        if not models:
            print("    ⚠️  No models available for training")
            return
        
        # Обучение первой модели
        model_id = models[0]['model_id']
        
        print(f"    🎯 Training model: {model_id}")
        
        # Обучение модели
        training_results = await model_manager.train_model(
            model_id, X_train, y_train
        )
        
        if training_results:
            print(f"    ✅ Model trained successfully")
            print(f"    📊 Training time: {training_results.get('training_time', 0):.2f}s")
            print(f"    📈 Version: {training_results.get('version', 'unknown')}")
        else:
            print("    ❌ Model training failed")
        
    except Exception as e:
        print(f"    ❌ Model training test error: {e}")


async def test_model_prediction(model_manager: ModelManager) -> None:
    """Тестирование предсказаний"""
    try:
        print("  🔮 Testing model prediction...")
        
        # Генерация тестовых данных
        X_test, _ = generate_test_data(100, 10)
        
        # Получение списка моделей
        models = await model_manager.list_models()
        
        if not models:
            print("    ⚠️  No models available for prediction")
            return
        
        # Предсказание с первой модели
        model_id = models[0]['model_id']
        
        print(f"    🔮 Making predictions with model: {model_id}")
        
        # Предсказание
        predictions = await model_manager.predict(model_id, X_test)
        
        if len(predictions) > 0:
            print(f"    ✅ Made {len(predictions)} predictions")
            print(f"    📊 Prediction classes: {np.unique(predictions)}")
        else:
            print("    ❌ No predictions made")
        
        # Предсказание вероятностей
        probabilities = await model_manager.predict_proba(model_id, X_test)
        
        if len(probabilities) > 0:
            print(f"    ✅ Generated probabilities with shape: {probabilities.shape}")
            print(f"    📊 Sample probabilities: {probabilities[0]}")
        else:
            print("    ❌ No probabilities generated")
        
    except Exception as e:
        print(f"    ❌ Model prediction test error: {e}")


async def test_model_management(model_manager: ModelManager) -> None:
    """Тестирование управления моделями"""
    try:
        print("  📋 Testing model management...")
        
        # Получение информации о моделях
        models = await model_manager.list_models()
        
        for model in models[:2]:  # Тестирование первых двух моделей
            model_id = model['model_id']
            
            # Информация о модели
            info = await model_manager.get_model_info(model_id)
            if info:
                print(f"    📊 Model {model_id} info:")
                print(f"      - Type: {info.get('model_name', 'unknown')}")
                print(f"      - Status: {info.get('status', 'unknown')}")
                print(f"      - Created: {info.get('created_at', 'unknown')}")
            
            # Версии модели
            versions = await model_manager.list_versions(model_id)
            if versions:
                print(f"    📈 Model {model_id} has {len(versions)} versions")
                for version in versions[-2:]:  # Последние 2 версии
                    print(f"      - {version.get('version_id', 'unknown')}: {version.get('created_at', 'unknown')}")
            else:
                print(f"    ⚠️  Model {model_id} has no versions")
        
    except Exception as e:
        print(f"    ❌ Model management test error: {e}")


def generate_test_data(n_samples: int, n_features: int) -> tuple:
    """
    Генерация тестовых данных для моделей
    
    Args:
        n_samples: Количество образцов
        n_features: Количество признаков
        
    Returns:
        tuple: (X, y) данные
    """
    np.random.seed(42)
    
    # Генерация признаков
    X = np.random.randn(n_samples, n_features)
    
    # Генерация целей (3 класса: BUY=0, SELL=1, HOLD=2)
    y = np.random.randint(0, 3, n_samples)
    
    # Конвертация в DataFrame
    feature_names = [f'feature_{i}' for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name='target')
    
    return X_df, y_series


if __name__ == "__main__":
    asyncio.run(test_models_layer())
