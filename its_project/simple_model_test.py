#!/usr/bin/env python3
"""
Простой тест моделей для проверки работоспособности.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from datetime import datetime

def test_imports():
    """Тест импортов."""
    print("Тест импортов...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch не импортируется: {e}")
        return False
    
    try:
        import sklearn
        print(f"✓ Scikit-learn {sklearn.__version__}")
    except ImportError as e:
        print(f"✗ Scikit-learn не импортируется: {e}")
        return False
    
    try:
        from models.boosting_model import BoostingModel
        print("✓ BoostingModel импортируется")
    except ImportError as e:
        print(f"✗ BoostingModel не импортируется: {e}")
        return False
    
    try:
        from models.gru_model import GRUModel
        print("✓ GRUModel импортируется")
    except ImportError as e:
        print(f"✗ GRUModel не импортируется: {e}")
        return False
    
    return True

def create_sample_data(n_samples=500):
    """Создание простых тестовых данных."""
    print(f"\nСоздание {n_samples} тестовых образцов...")
    
    np.random.seed(42)
    
    # Простые признаки
    X = np.random.randn(n_samples, 5)
    
    # Простые правила для меток
    y = np.zeros(n_samples)
    y[X[:, 0] > 0.5] = 2  # BUY
    y[X[:, 0] < -0.5] = 0  # SELL
    y[(X[:, 0] >= -0.5) & (X[:, 0] <= 0.5)] = 1  # HOLD
    
    print(f"Распределение: SELL={np.sum(y==0)}, HOLD={np.sum(y==1)}, BUY={np.sum(y==2)}")
    return X, y

def test_boosting_model(X, y):
    """Тест Gradient Boosting модели."""
    print("\nТест Gradient Boosting модели...")
    
    try:
        from sklearn.model_selection import train_test_split
        
        # Разделение данных
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Создание и обучение модели
        config = {
            "n_estimators": 10,  # Меньше для быстрого теста
            "learning_rate": 0.1,
            "max_depth": 2,
            "random_state": 42
        }
        
        model = BoostingModel(config)
        model.fit(X_train, y_train)
        
        # Предсказания
        y_pred = model.predict(X_test)
        
        # Метрики
        accuracy = np.mean(y_pred == y_test)
        print(f"✓ Accuracy: {accuracy:.4f}")
        
        return True, accuracy
        
    except Exception as e:
        print(f"✗ Ошибка: {e}")
        return False, 0

def test_gru_model(X, y):
    """Тест GRU модели."""
    print("\nТест GRU модели...")
    
    try:
        from sklearn.model_selection import train_test_split
        
        # Создание последовательностей
        def create_sequences(data, targets, seq_len=10):
            sequences = []
            seq_targets = []
            for i in range(len(data) - seq_len + 1):
                sequences.append(data[i:i + seq_len])
                seq_targets.append(targets[i + seq_len - 1])
            return np.array(sequences), np.array(seq_targets)
        
        X_seq, y_seq = create_sequences(X, y)
        
        # Разделение данных
        X_train, X_test, y_train, y_test = train_test_split(
            X_seq, y_seq, test_size=0.2, random_state=42
        )
        
        # Конфигурация GRU
        config = {
            "input_size": X.shape[1],
            "hidden_size": 32,  # Меньше для быстрого теста
            "num_layers": 1,
            "dropout": 0.1,
            "epochs": 5,  # Меньше эпох
            "batch_size": 16,
            "learning_rate": 0.01
        }
        
        model = GRUModel(config)
        
        # Обучение
        model.fit(X_train, y_train)
        
        # Предсказания
        y_pred = model.predict(X_test)
        
        # Метрики
        accuracy = np.mean(y_pred == y_test)
        print(f"✓ Accuracy: {accuracy:.4f}")
        
        return True, accuracy
        
    except Exception as e:
        print(f"✗ Ошибка: {e}")
        return False, 0

def main():
    """Главная функция."""
    print("=" * 50)
    print("ПРОСТОЙ ТЕСТ МОДЕЛЕЙ ITS")
    print("=" * 50)
    
    # Тест импортов
    if not test_imports():
        print("\n❌ Тест импортов не пройден")
        return
    
    # Создание данных
    X, y = create_sample_data(500)
    
    # Тест моделей
    boosting_success, boosting_acc = test_boosting_model(X, y)
    gru_success, gru_acc = test_gru_model(X, y)
    
    # Результаты
    print("\n" + "=" * 50)
    print("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
    print("=" * 50)
    
    print(f"Gradient Boosting: {'✓ УСПЕХ' if boosting_success else '✗ ОШИБКА'}")
    if boosting_success:
        print(f"  Accuracy: {boosting_acc:.4f}")
    
    print(f"GRU: {'✓ УСПЕХ' if gru_success else '✗ ОШИБКА'}")
    if gru_success:
        print(f"  Accuracy: {gru_acc:.4f}")
    
    if boosting_success and gru_success:
        print(f"\n✅ Обе модели работают успешно!")
        print(f"   Средняя точность: {(boosting_acc + gru_acc) / 2:.4f}")
    else:
        print(f"\n❌ Некоторые модели не работают")
    
    # Сохранение результатов
    results = {
        'timestamp': datetime.now().isoformat(),
        'boosting': {
            'success': boosting_success,
            'accuracy': boosting_acc
        },
        'gru': {
            'success': gru_success,
            'accuracy': gru_acc
        }
    }
    
    try:
        import json
        with open('simple_test_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n📄 Результаты сохранены в simple_test_results.json")
    except Exception as e:
        print(f"⚠️ Не удалось сохранить результаты: {e}")

if __name__ == "__main__":
    main()
