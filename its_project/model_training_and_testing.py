#!/usr/bin/env python3
"""
Модуль для обучения и тестирования моделей ITS.
Создает синтетические данные, обучает модели и собирает метрики.
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelTrainerTester:
    """Класс для обучения и тестирования моделей."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Инициализация с конфигурацией."""
        self.config_path = config_path
        self.results = {}
        self.models = {}
        
    def load_test_data(self) -> Dict[str, pd.DataFrame]:
        """Загрузка тестовых данных."""
        data_path = Path("tests/data")
        data = {}
        
        try:
            # Загрузка OHLCV данных
            ohlcv_path = data_path / "ohlcv_test.parquet"
            if ohlcv_path.exists():
                data['ohlcv'] = pd.read_parquet(ohlcv_path)
                logger.info(f"Загружено OHLCV данных: {data['ohlcv'].shape}")
            
            # Загрузка данных orderbook
            orderbook_path = data_path / "orderbook_test.parquet"
            if orderbook_path.exists():
                data['orderbook'] = pd.read_parquet(orderbook_path)
                logger.info(f"Загружено OrderBook данных: {data['orderbook'].shape}")
            
            # Загрузка sentiment данных
            sentiment_path = data_path / "sentiment_test.parquet"
            if sentiment_path.exists():
                data['sentiment'] = pd.read_parquet(sentiment_path)
                logger.info(f"Загружено Sentiment данных: {data['sentiment'].shape}")
                
        except Exception as e:
            logger.error(f"Ошибка загрузки данных: {e}")
            
        return data
    
    def create_synthetic_data(self, n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Создание синтетических данных для обучения."""
        logger.info(f"Создание {n_samples} синтетических образцов")
        
        # Генерация признаков
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
        
        # Создание матрицы признаков
        X = np.column_stack([
            rsi, macd, bb_upper, bb_lower, volume, price_change,
            bid_ask_spread, order_imbalance, sentiment_score
        ])
        
        # Создание меток на основе правил
        y = np.zeros(n_samples, dtype=int)
        
        # Правила для классификации
        buy_signal = (rsi < 30) & (sentiment_score > 0.5) & (price_change > 0.01)
        sell_signal = (rsi > 70) & (sentiment_score < -0.5) & (price_change < -0.01)
        
        y[buy_signal] = 2  # BUY
        y[sell_signal] = 0  # SELL
        y[~(buy_signal | sell_signal)] = 1  # HOLD
        
        logger.info(f"Распределение классов: SELL={np.sum(y==0)}, HOLD={np.sum(y==1)}, BUY={np.sum(y==2)}")
        
        return X, y
    
    def create_sequences(self, X: np.ndarray, y: np.ndarray, seq_length: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Создание последовательностей для GRU модели."""
        sequences = []
        targets = []
        
        for i in range(len(X) - seq_length + 1):
            sequences.append(X[i:i + seq_length])
            targets.append(y[i + seq_length - 1])
        
        return np.array(sequences), np.array(targets)
    
    def train_and_test_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Обучение и тестирование всех моделей."""
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
        
        # Разделение данных
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        results = {}
        
        # 1. Gradient Boosting Model
        logger.info("Обучение Gradient Boosting модели...")
        try:
            from models.boosting_model import BoostingModel
            
            boosting_config = {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 3,
                "random_state": 42
            }
            
            boosting_model = BoostingModel(boosting_config)
            boosting_model.fit(X_train, y_train)
            
            # Предсказания
            y_pred_boost = boosting_model.predict(X_test)
            y_proba_boost = boosting_model.predict_proba(X_test)
            
            # Метрики
            results['boosting'] = {
                'model': boosting_model,
                'predictions': y_pred_boost,
                'probabilities': y_proba_boost,
                'accuracy': accuracy_score(y_test, y_pred_boost),
                'classification_report': classification_report(y_test, y_pred_boost, output_dict=True),
                'confusion_matrix': confusion_matrix(y_test, y_pred_boost).tolist(),
                'feature_importance': boosting_model.get_feature_importance().tolist() if boosting_model.get_feature_importance() is not None else None
            }
            
            logger.info(f"Gradient Boosting accuracy: {results['boosting']['accuracy']:.4f}")
            
        except Exception as e:
            logger.error(f"Ошибка в Gradient Boosting: {e}")
            results['boosting'] = {'error': str(e)}
        
        # 2. GRU Model
        logger.info("Обучение GRU модели...")
        try:
            from models.gru_model import GRUModel
            
            # Создание последовательностей для GRU
            X_seq_train, y_seq_train = self.create_sequences(X_train, y_train, seq_length=10)
            X_seq_test, y_seq_test = self.create_sequences(X_test, y_test, seq_length=10)
            
            gru_config = {
                "input_size": X.shape[1],
                "hidden_size": 64,
                "num_layers": 2,
                "dropout": 0.2,
                "epochs": 50,
                "batch_size": 32,
                "learning_rate": 0.001,
                "patience": 10
            }
            
            gru_model = GRUModel(gru_config)
            
            # Callback для прогресса
            def progress_callback(epoch, total_epochs, loss):
                if epoch % 10 == 0:
                    logger.info(f"GRU Epoch {epoch}/{total_epochs}, Loss: {loss:.4f}")
            
            gru_model.fit(X_seq_train, y_seq_train, progress_callback=progress_callback)
            
            # Предсказания
            y_pred_gru = gru_model.predict(X_seq_test)
            y_proba_gru = gru_model.predict_proba(X_seq_test)
            
            # Метрики
            results['gru'] = {
                'model': gru_model,
                'predictions': y_pred_gru,
                'probabilities': y_proba_gru,
                'accuracy': accuracy_score(y_seq_test, y_pred_gru),
                'classification_report': classification_report(y_seq_test, y_pred_gru, output_dict=True),
                'confusion_matrix': confusion_matrix(y_seq_test, y_pred_gru).tolist()
            }
            
            logger.info(f"GRU accuracy: {results['gru']['accuracy']:.4f}")
            
        except Exception as e:
            logger.error(f"Ошибка в GRU: {e}")
            results['gru'] = {'error': str(e)}
        
        return results
    
    def evaluate_models(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Оценка и сравнение моделей."""
        evaluation = {
            'model_comparison': {},
            'best_model': None,
            'summary': {}
        }
        
        valid_models = {k: v for k, v in results.items() if 'error' not in v}
        
        if not valid_models:
            logger.error("Нет успешно обученных моделей")
            return evaluation
        
        # Сравнение метрик
        for model_name, model_results in valid_models.items():
            evaluation['model_comparison'][model_name] = {
                'accuracy': model_results['accuracy'],
                'precision_macro': model_results['classification_report']['macro avg']['precision'],
                'recall_macro': model_results['classification_report']['macro avg']['recall'],
                'f1_macro': model_results['classification_report']['macro avg']['f1-score']
            }
        
        # Определение лучшей модели
        best_model = max(evaluation['model_comparison'].items(), 
                        key=lambda x: x[1]['accuracy'])
        evaluation['best_model'] = {
            'name': best_model[0],
            'accuracy': best_model[1]['accuracy']
        }
        
        # Сводная статистика
        accuracies = [v['accuracy'] for v in evaluation['model_comparison'].values()]
        evaluation['summary'] = {
            'total_models': len(valid_models),
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'best_accuracy': np.max(accuracies),
            'worst_accuracy': np.min(accuracies)
        }
        
        return evaluation
    
    def save_results(self, results: Dict[str, Any], evaluation: Dict[str, Any]) -> None:
        """Сохранение результатов."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Создание директории для результатов
        results_dir = Path("model_test_results")
        results_dir.mkdir(exist_ok=True)
        
        # Сохранение метрик
        metrics_file = results_dir / f"metrics_{timestamp}.json"
        
        # Подготовка данных для JSON (исключая объекты моделей)
        json_results = {}
        for model_name, model_data in results.items():
            if 'error' not in model_data:
                json_results[model_name] = {
                    'accuracy': model_data['accuracy'],
                    'classification_report': model_data['classification_report'],
                    'confusion_matrix': model_data['confusion_matrix']
                }
                if 'feature_importance' in model_data and model_data['feature_importance'] is not None:
                    json_results[model_name]['feature_importance'] = model_data['feature_importance']
            else:
                json_results[model_name] = model_data
        
        with open(metrics_file, 'w') as f:
            json.dump({
                'results': json_results,
                'evaluation': evaluation,
                'timestamp': timestamp
            }, f, indent=2)
        
        logger.info(f"Результаты сохранены в {metrics_file}")
        
        # Сохранение моделей
        for model_name, model_data in results.items():
            if 'error' not in model_data:
                model_file = results_dir / f"{model_name}_model_{timestamp}.pkl"
                model_data['model'].save(model_file)
                logger.info(f"Модель {model_name} сохранена в {model_file}")
    
    def create_visualizations(self, results: Dict[str, Any], evaluation: Dict[str, Any]) -> None:
        """Создание визуализаций."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        viz_dir = Path("model_test_results/visualizations")
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Сравнение точности моделей
        valid_models = {k: v for k, v in results.items() if 'error' not in v}
        
        if valid_models:
            accuracies = [v['accuracy'] for v in valid_models.values()]
            model_names = list(valid_models.keys())
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(model_names, accuracies, color=['skyblue', 'lightcoral', 'lightgreen'])
            plt.title('Сравнение точности моделей')
            plt.ylabel('Accuracy')
            plt.ylim(0, 1)
            
            # Добавление значений на бары
            for bar, acc in zip(bars, accuracies):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{acc:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(viz_dir / f"accuracy_comparison_{timestamp}.png", dpi=300)
            plt.close()
            
            # 2. Матрицы ошибок
            fig, axes = plt.subplots(1, len(valid_models), figsize=(15, 5))
            if len(valid_models) == 1:
                axes = [axes]
            
            for idx, (model_name, model_data) in enumerate(valid_models.items()):
                cm = np.array(model_data['confusion_matrix'])
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=['SELL', 'HOLD', 'BUY'],
                           yticklabels=['SELL', 'HOLD', 'BUY'],
                           ax=axes[idx])
                axes[idx].set_title(f'{model_name} Confusion Matrix')
                axes[idx].set_xlabel('Predicted')
                axes[idx].set_ylabel('Actual')
            
            plt.tight_layout()
            plt.savefig(viz_dir / f"confusion_matrices_{timestamp}.png", dpi=300)
            plt.close()
            
            # 3. Важность признаков (для Gradient Boosting)
            if 'boosting' in valid_models and valid_models['boosting'].get('feature_importance') is not None:
                feature_importance = np.array(valid_models['boosting']['feature_importance'])
                feature_names = ['RSI', 'MACD', 'BB_Upper', 'BB_Lower', 'Volume', 
                               'Price_Change', 'Bid_Ask_Spread', 'Order_Imbalance', 'Sentiment']
                
                plt.figure(figsize=(12, 6))
                indices = np.argsort(feature_importance)[::-1]
                plt.bar(range(len(feature_importance)), feature_importance[indices])
                plt.xticks(range(len(feature_importance)), [feature_names[i] for i in indices], rotation=45)
                plt.title('Feature Importance (Gradient Boosting)')
                plt.ylabel('Importance')
                plt.tight_layout()
                plt.savefig(viz_dir / f"feature_importance_{timestamp}.png", dpi=300)
                plt.close()
        
        logger.info(f"Визуализации сохранены в {viz_dir}")
    
    def run_complete_testing(self) -> Dict[str, Any]:
        """Запуск полного цикла тестирования."""
        logger.info("Начало полного цикла тестирования моделей")
        
        # 1. Создание данных
        X, y = self.create_synthetic_data(n_samples=2000)
        
        # 2. Обучение и тестирование моделей
        results = self.train_and_test_models(X, y)
        
        # 3. Оценка моделей
        evaluation = self.evaluate_models(results)
        
        # 4. Сохранение результатов
        self.save_results(results, evaluation)
        
        # 5. Создание визуализаций
        self.create_visualizations(results, evaluation)
        
        # 6. Вывод сводки
        self.print_summary(evaluation)
        
        return {
            'results': results,
            'evaluation': evaluation
        }
    
    def print_summary(self, evaluation: Dict[str, Any]) -> None:
        """Вывод сводки результатов."""
        print("\n" + "="*60)
        print("СВОДКА РЕЗУЛЬТАТОВ ТЕСТИРОВАНИЯ МОДЕЛЕЙ")
        print("="*60)
        
        print(f"\nВсего моделей обучено: {evaluation['summary']['total_models']}")
        print(f"Средняя точность: {evaluation['summary']['mean_accuracy']:.4f}")
        print(f"Лучший результат: {evaluation['summary']['best_accuracy']:.4f}")
        
        print(f"\nЛучшая модель: {evaluation['best_model']['name']}")
        print(f"Точность: {evaluation['best_model']['accuracy']:.4f}")
        
        print("\nДетальные результаты:")
        for model_name, metrics in evaluation['model_comparison'].items():
            print(f"\n{model_name.upper()}:")
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision_macro']:.4f}")
            print(f"  Recall:    {metrics['recall_macro']:.4f}")
            print(f"  F1-Score:  {metrics['f1_macro']:.4f}")
        
        print("\n" + "="*60)


def main():
    """Главная функция."""
    trainer = ModelTrainerTester()
    results = trainer.run_complete_testing()
    return results


if __name__ == "__main__":
    main()
