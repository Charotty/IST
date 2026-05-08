#!/usr/bin/env python3
"""
Рабочий скрипт для тестирования моделей ITS.
"""

import numpy as np
import pandas as pd
from datetime import datetime
import json
import sys
import os

# Добавляем путь для импортов
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_dependencies():
    """Проверка зависимостей."""
    print("🔍 Проверка зависимостей...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   Device: {device}")
    except ImportError as e:
        print(f"❌ PyTorch: {e}")
        return False
    
    try:
        import sklearn
        print(f"✅ Scikit-learn {sklearn.__version__}")
    except ImportError as e:
        print(f"❌ Scikit-learn: {e}")
        return False
    
    return True

def create_test_data(n_samples=1000):
    """Создание тестовых данных."""
    print(f"📊 Создание {n_samples} тестовых образцов...")
    
    np.random.seed(42)
    
    # Технические индикаторы
    rsi = np.random.uniform(0, 100, n_samples)
    macd = np.random.normal(0, 1, n_samples)
    bb_upper = np.random.normal(100, 10, n_samples)
    bb_lower = np.random.normal(90, 10, n_samples)
    volume = np.random.exponential(1000000, n_samples)
    price_change = np.random.normal(0, 0.02, n_samples)
    
    # Order book признаки
    bid_ask_spread = np.random.exponential(0.001, n_samples)
    order_imbalance = np.random.uniform(-1, 1, n_samples)
    
    # Sentiment признаки
    sentiment_score = np.random.uniform(-1, 1, n_samples)
    
    # Матрица признаков
    X = np.column_stack([
        rsi, macd, bb_upper, bb_lower, volume, price_change,
        bid_ask_spread, order_imbalance, sentiment_score
    ])
    
    # Метки классов
    y = np.zeros(n_samples, dtype=int)
    
    # Правила для классификации
    buy_signal = (rsi < 30) & (sentiment_score > 0.5) & (price_change > 0.01)
    sell_signal = (rsi > 70) & (sentiment_score < -0.5) & (price_change < -0.01)
    
    y[buy_signal] = 2  # BUY
    y[sell_signal] = 0  # SELL
    y[~(buy_signal | sell_signal)] = 1  # HOLD
    
    print(f"   Распределение: SELL={np.sum(y==0)}, HOLD={np.sum(y==1)}, BUY={np.sum(y==2)}")
    
    return X, y

def test_gradient_boosting(X, y):
    """Тест Gradient Boosting модели."""
    print("\n🚀 Тест Gradient Boosting модели...")
    
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        
        # Разделение данных
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Импорт и создание модели
        try:
            from models.boosting_model import BoostingModel
            
            config = {
                "n_estimators": 50,
                "learning_rate": 0.1,
                "max_depth": 3,
                "random_state": 42
            }
            
            model = BoostingModel(config)
            
        except ImportError:
            print("   ⚠️ Используем sklearn GradientBoosting вместо кастомной модели")
            from sklearn.ensemble import GradientBoostingClassifier
            
            model = GradientBoostingClassifier(
                n_estimators=50,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )
        
        # Обучение
        print("   📚 Обучение модели...")
        model.fit(X_train, y_train)
        
        # Предсказания
        y_pred = model.predict(X_test)
        
        # Метрики
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"   ✅ Accuracy: {accuracy:.4f}")
        print(f"   📊 F1-score (macro): {report['macro avg']['f1-score']:.4f}")
        
        # Важность признаков (если доступна)
        if hasattr(model, 'feature_importances_'):
            feature_names = ['RSI', 'MACD', 'BB_Upper', 'BB_Lower', 'Volume', 
                           'Price_Change', 'Bid_Ask_Spread', 'Order_Imbalance', 'Sentiment']
            importances = model.feature_importances_
            top_idx = np.argsort(importances)[-3:]
            print(f"   🎯 Топ-3 важных признака:")
            for idx in top_idx:
                print(f"      {feature_names[idx]}: {importances[idx]:.3f}")
        
        return {
            'success': True,
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'feature_importance': model.feature_importances_.tolist() if hasattr(model, 'feature_importances_') else None
        }
        
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")
        return {'success': False, 'error': str(e)}

def test_gru_model(X, y):
    """Тест GRU модели."""
    print("\n🧠 Тест GRU модели...")
    
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        
        # Создание последовательностей
        def create_sequences(data, targets, seq_len=10):
            sequences = []
            seq_targets = []
            for i in range(len(data) - seq_len + 1):
                sequences.append(data[i:i + seq_len])
                seq_targets.append(targets[i + seq_len - 1])
            return np.array(sequences), np.array(seq_targets)
        
        X_seq, y_seq = create_sequences(X, y, seq_len=10)
        
        # Разделение данных
        X_train, X_test, y_train, y_test = train_test_split(
            X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq
        )
        
        # Импорт модели
        try:
            from models.gru_model import GRUModel
            
            config = {
                "input_size": X.shape[1],
                "hidden_size": 32,
                "num_layers": 1,
                "dropout": 0.1,
                "epochs": 10,
                "batch_size": 16,
                "learning_rate": 0.01,
                "patience": 5
            }
            
            model = GRUModel(config)
            
            print("   📚 Обучение GRU модели...")
            
            # Callback для прогресса
            def progress_callback(epoch, total_epochs, loss):
                if epoch % 5 == 0:
                    print(f"      Epoch {epoch}/{total_epochs}, Loss: {loss:.4f}")
            
            model.fit(X_train, y_train, progress_callback=progress_callback)
            
        except ImportError as e:
            print(f"   ⚠️ Не удалось импортировать GRU модель: {e}")
            return {'success': False, 'error': str(e)}
        
        # Предсказания
        y_pred = model.predict(X_test)
        
        # Метрики
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"   ✅ Accuracy: {accuracy:.4f}")
        print(f"   📊 F1-score (macro): {report['macro avg']['f1-score']:.4f}")
        
        return {
            'success': True,
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm.tolist()
        }
        
    except Exception as e:
        print(f"   ❌ Ошибка: {e}")
        return {'success': False, 'error': str(e)}

def create_simple_report(results):
    """Создание простого отчета."""
    print("\n" + "="*60)
    print("📋 ОТЧЕТ О ТЕСТИРОВАНИИ МОДЕЛЕЙ")
    print("="*60)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"⏰ Время: {timestamp}")
    
    # Результаты по моделям
    for model_name, result in results.items():
        print(f"\n🤖 {model_name.upper()}:")
        
        if result['success']:
            print(f"   ✅ Статус: Успешно")
            print(f"   🎯 Accuracy: {result['accuracy']:.4f}")
            
            if 'classification_report' in result:
                macro_avg = result['classification_report']['macro avg']
                print(f"   📊 Precision: {macro_avg['precision']:.4f}")
                print(f"   📊 Recall: {macro_avg['recall']:.4f}")
                print(f"   📊 F1-score: {macro_avg['f1-score']:.4f}")
            
            if 'feature_importance' in result and result['feature_importance']:
                print(f"   🎯 Важность признаков: доступна")
        else:
            print(f"   ❌ Статус: Ошибка")
            print(f"   📝 Ошибка: {result['error']}")
    
    # Сводка
    successful_models = [name for name, result in results.items() if result['success']]
    accuracies = [result['accuracy'] for result in results.values() if result['success']]
    
    print(f"\n📈 СВОДКА:")
    print(f"   Всего моделей: {len(results)}")
    print(f"   Успешных: {len(successful_models)}")
    
    if accuracies:
        print(f"   Средняя accuracy: {np.mean(accuracies):.4f}")
        print(f"   Лучшая accuracy: {np.max(accuracies):.4f}")
        print(f"   Худшая accuracy: {np.min(accuracies):.4f}")
        
        best_model = successful_models[np.argmax(accuracies)]
        print(f"   Лучшая модель: {best_model}")
    
    print("="*60)
    
    # Сохранение отчета
    report_data = {
        'timestamp': timestamp,
        'results': results,
        'summary': {
            'total_models': len(results),
            'successful_models': len(successful_models),
            'mean_accuracy': float(np.mean(accuracies)) if accuracies else None,
            'best_accuracy': float(np.max(accuracies)) if accuracies else None,
            'best_model': best_model if accuracies else None
        }
    }
    
    try:
        with open('model_test_report.json', 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        print(f"📄 Отчет сохранен в model_test_report.json")
    except Exception as e:
        print(f"⚠️ Не удалось сохранить отчет: {e}")

def main():
    """Главная функция."""
    print("🚀 ЗАПУСК ТЕСТИРОВАНИЯ МОДЕЛЕЙ ITS")
    print("="*60)
    
    # Проверка зависимостей
    if not test_dependencies():
        print("\n❌ Не все зависимости установлены")
        return
    
    # Создание данных
    X, y = create_test_data(1000)
    
    # Тестирование моделей
    results = {}
    
    # Gradient Boosting
    results['gradient_boosting'] = test_gradient_boosting(X, y)
    
    # GRU
    results['gru'] = test_gru_model(X, y)
    
    # Создание отчета
    create_simple_report(results)
    
    print(f"\n✅ Тестирование завершено!")

if __name__ == "__main__":
    main()
