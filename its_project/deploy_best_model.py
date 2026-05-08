#!/usr/bin/env python3
"""
Скрипт для развертывания лучшей модели (Gradient Boosting) в торговую систему.
"""

import numpy as np
import pandas as pd
import joblib
import json
from datetime import datetime
import sys
import os

# Добавляем путь для импортов
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def train_production_model():
    """Обучение продакшн-модели на полном датасете."""
    print("🚀 Обучение продакшн модели Gradient Boosting...")
    
    # Создание большего датасета для продакшена
    X, y = create_production_data(5000)
    
    # Разделение данных
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Создание и обучение модели
    try:
        from models.boosting_model import BoostingModel
        
        config = {
            "n_estimators": 100,  # Увеличим для продакшена
            "learning_rate": 0.1,
            "max_depth": 3,
            "random_state": 42,
            "class_weight": "balanced"  # Для баланса классов
        }
        
        model = BoostingModel(config)
        
    except ImportError:
        print("⚠️ Используем sklearn GradientBoosting")
        from sklearn.ensemble import GradientBoostingClassifier
        
        model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
    
    # Обучение
    print("📚 Обучение модели...")
    model.fit(X_train, y_train)
    
    # Оценка
    from sklearn.metrics import accuracy_score, classification_report
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"✅ Accuracy: {accuracy:.4f}")
    
    # Сохранение модели
    model_path = "production_gradient_boosting_model.pkl"
    joblib.dump(model, model_path)
    print(f"💾 Модель сохранена в {model_path}")
    
    return model, accuracy

def create_production_data(n_samples=5000):
    """Создание продакшн-датасета."""
    print(f"📊 Создание {n_samples} продакшн образцов...")
    
    np.random.seed(42)
    
    # Более реалистичные признаки
    rsi = np.random.beta(2, 2, n_samples) * 100  # RSI с бета-распределением
    macd = np.random.normal(0, 0.5, n_samples)  # Меньший разброс MACD
    bb_upper = np.random.normal(105, 8, n_samples)  # BB bands
    bb_lower = np.random.normal(95, 8, n_samples)
    volume = np.random.lognormal(10, 1, n_samples)  # Логнормальное распределение объема
    price_change = np.random.normal(0, 0.015, n_samples)  # Меньший разброс цены
    
    # Order book признаки
    bid_ask_spread = np.random.exponential(0.0005, n_samples)
    order_imbalance = np.random.normal(0, 0.3, n_samples)  # Центрированное распределение
    
    # Sentiment признаки
    sentiment_score = np.random.normal(0, 0.4, n_samples)  # Более реалистичный sentiment
    
    # Матрица признаков
    X = np.column_stack([
        rsi, macd, bb_upper, bb_lower, volume, price_change,
        bid_ask_spread, order_imbalance, sentiment_score
    ])
    
    # Более сбалансированные метки
    y = np.zeros(n_samples, dtype=int)
    
    # Улучшенные правила для классификации
    buy_signal = (
        (rsi < 35) & 
        (sentiment_score > 0.3) & 
        (price_change > 0.005) &
        (order_imbalance > 0.1)
    )
    
    sell_signal = (
        (rsi > 65) & 
        (sentiment_score < -0.3) & 
        (price_change < -0.005) &
        (order_imbalance < -0.1)
    )
    
    y[buy_signal] = 2  # BUY
    y[sell_signal] = 0  # SELL
    y[~(buy_signal | sell_signal)] = 1  # HOLD
    
    print(f"   Распределение: SELL={np.sum(y==0)}, HOLD={np.sum(y==1)}, BUY={np.sum(y==2)}")
    
    return X, y

def create_trading_interface():
    """Создание интерфейса для торговли."""
    
    class TradingModel:
        """Класс-обертка для торговой модели."""
        
        def __init__(self, model_path="production_gradient_boosting_model.pkl"):
            self.model = joblib.load(model_path)
            self.feature_names = [
                'RSI', 'MACD', 'BB_Upper', 'BB_Lower', 'Volume',
                'Price_Change', 'Bid_Ask_Spread', 'Order_Imbalance', 'Sentiment'
            ]
        
        def predict_signal(self, features):
            """Предсказание торгового сигнала."""
            if isinstance(features, dict):
                # Преобразование словаря в массив
                feature_array = np.array([features.get(name, 0) for name in self.feature_names])
            else:
                feature_array = np.array(features)
            
            # Предсказание
            prediction = self.model.predict(feature_array.reshape(1, -1))[0]
            probabilities = self.model.predict_proba(feature_array.reshape(1, -1))[0]
            
            # Конвертация в торговый сигнал
            signal_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
            confidence = max(probabilities)
            
            return {
                'signal': signal_map[prediction],
                'confidence': confidence,
                'probabilities': {
                    'SELL': probabilities[0],
                    'HOLD': probabilities[1],
                    'BUY': probabilities[2]
                }
            }
        
        def get_feature_importance(self):
            """Получение важности признаков."""
            if hasattr(self.model, 'feature_importances_'):
                return dict(zip(self.feature_names, self.model.feature_importances_))
            return None
    
    return TradingModel

def create_model_monitoring():
    """Создание системы мониторинга модели."""
    
    class ModelMonitor:
        """Мониторинг производительности модели."""
        
        def __init__(self):
            self.predictions = []
            self.accuracies = []
            self.confidences = []
        
        def log_prediction(self, prediction, actual=None):
            """Логирование предсказания."""
            self.predictions.append({
                'timestamp': datetime.now(),
                'prediction': prediction,
                'actual': actual
            })
            
            if 'confidence' in prediction:
                self.confidences.append(prediction['confidence'])
        
        def calculate_accuracy(self, window_size=100):
            """Расчет точности за окно."""
            if len(self.predictions) < window_size:
                return None
            
            recent = self.predictions[-window_size:]
            correct = 0
            total = 0
            
            for pred in recent:
                if pred['actual'] is not None:
                    if pred['prediction']['signal'] == pred['actual']:
                        correct += 1
                    total += 1
            
            return correct / total if total > 0 else None
        
        def get_stats(self):
            """Получение статистики."""
            return {
                'total_predictions': len(self.predictions),
                'avg_confidence': np.mean(self.confidences) if self.confidences else 0,
                'recent_accuracy': self.calculate_accuracy()
            }
    
    return ModelMonitor

def main():
    """Главная функция."""
    print("🚀 РАЗВЕРТЫВАНИЕ ПРОДАКШН МОДЕЛИ")
    print("="*50)
    
    # Обучение продакшн модели
    model, accuracy = train_production_model()
    
    # Создание интерфейсов
    TradingModel = create_trading_interface()
    ModelMonitor = create_model_monitoring()
    
    # Демонстрация работы
    print("\n🧪 Демонстрация работы модели:")
    
    # Создание экземпляров
    trading_model = TradingModel()
    monitor = ModelMonitor()
    
    # Тестовые признаки
    test_features = {
        'RSI': 25.0,  # Oversold
        'MACD': -0.2,
        'BB_Upper': 102.0,
        'BB_Lower': 98.0,
        'Volume': 1500000,
        'Price_Change': 0.02,
        'Bid_Ask_Spread': 0.0003,
        'Order_Imbalance': 0.2,
        'Sentiment': 0.6
    }
    
    # Предсказание
    prediction = trading_model.predict_signal(test_features)
    print(f"📊 Сигнал: {prediction['signal']}")
    print(f"🎯 Уверенность: {prediction['confidence']:.3f}")
    print(f"📈 Вероятности: {prediction['probabilities']}")
    
    # Важность признаков
    importance = trading_model.get_feature_importance()
    if importance:
        print("\n🎯 Топ-5 важных признаков:")
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
        for name, score in sorted_features:
            print(f"   {name}: {score:.3f}")
    
    # Сохранение конфигурации
    config = {
        'model_type': 'GradientBoosting',
        'accuracy': accuracy,
        'features': trading_model.feature_names,
        'deployment_time': datetime.now().isoformat(),
        'test_prediction': prediction
    }
    
    with open('production_model_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✅ Продакшн модель развернута успешно!")
    print(f"📄 Конфигурация сохранена в production_model_config.json")
    
    return trading_model, monitor

if __name__ == "__main__":
    main()
