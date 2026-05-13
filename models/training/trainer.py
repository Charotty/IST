"""
Model Trainer

Универсальный тренер для обучения моделей.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score


class ModelTrainer:
    """Универсальный тренер моделей"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Инициализация тренера
        
        Args:
            config: Конфигурация тренера
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Параметры
        self.validation_split = config.get('validation_split', 0.2)
        self.early_stopping_patience = config.get('early_stopping_patience', 10)
        self.batch_size = config.get('batch_size', 32)
        self.epochs = config.get('epochs', 100)
        
        # Метрики
        self._metrics = {
            'models_trained': 0,
            'training_time': 0.0,
            'best_scores': {}
        }
    
    async def train_model(self, model: Any, X: pd.DataFrame, y: pd.Series,
                       config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Обучение модели
        
        Args:
            model: Модель для обучения
            X: Признаки
            y: Целевая переменная
            config: Дополнительная конфигурация
            
        Returns:
            Dict: Результаты обучения
        """
        try:
            start_time = time.time()
            
            # Использование конфигурации тренера если не передана
            training_config = config or self.config
            
            # Валидация входных данных
            if not self._validate_data(X, y):
                raise ValueError("Invalid input data")
            
            # Разделение на train/validation
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=training_config.get('validation_split', self.validation_split),
                random_state=42, stratify=y if len(y.unique()) > 2 else None
            )
            
            self.logger.info(f"Training {model.__class__.__name__} on {len(X_train)} samples")
            self.logger.info(f"Validation on {len(X_val)} samples")
            
            # Обучение модели
            if hasattr(model, 'fit'):
                model.fit(X_train, y_train)
            else:
                raise ValueError("Model must have fit method")
            
            # Предсказания на валидационных данных
            y_pred_val = model.predict(X_val)
            
            # Расчет метрик
            metrics = self._calculate_metrics(y_val, y_pred_val)
            
            training_time = time.time() - start_time
            
            # Сохранение лучших метрик
            model_type = model.get_prediction_type()
            if model_type not in self._metrics['best_scores']:
                self._metrics['best_scores'][model_type] = {}
            
            self._metrics['best_scores'][model_type][model.__class__.__name__] = {
                'score': metrics.get('accuracy', metrics.get('r2', 0)),
                'training_time': training_time,
                'timestamp': time.time()
            }
            
            self._metrics['models_trained'] += 1
            self._metrics['training_time'] += training_time
            
            results = {
                'model_name': model.__class__.__name__,
                'training_samples': len(X_train),
                'validation_samples': len(X_val),
                'training_time': training_time,
                'metrics': metrics,
                'model_info': model.get_model_info() if hasattr(model, 'get_model_info') else {}
            }
            
            self.logger.info(f"Model trained successfully in {training_time:.2f}s")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error training model: {e}")
            return {'error': str(e)}
    
    def _validate_data(self, X: pd.DataFrame, y: pd.Series) -> bool:
        """
        Валидация входных данных
        
        Args:
            X: Признаки
            y: Целевая переменная
            
        Returns:
            bool: True если данные валидны
        """
        if X is None or y is None:
            return False
        
        if X.empty or y.empty:
            return False
        
        if len(X) != len(y):
            return False
        
        return True
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Расчет метрик
        
        Args:
            y_true: Истинные значения
            y_pred: Предсказанные значения
            
        Returns:
            Dict: Метрики
        """
        try:
            # Определение типа задачи
            if len(y_true.shape) == 1:  # 1D массив - классификация
                unique_classes = len(np.unique(y_true))
                
                if unique_classes == 2:
                    # Бинарная классификация
                    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
                    
                    return {
                        'accuracy': accuracy_score(y_true, y_pred),
                        'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
                        'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
                        'f1': f1_score(y_true, y_pred, average='binary', zero_division=0),
                        'roc_auc': roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) == 2 else 0.0
                    }
                else:
                    # Мультиклассовая классификация
                    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                    
                    return {
                        'accuracy': accuracy_score(y_true, y_pred),
                        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
                        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
                        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
                    }
            else:
                # Регрессия
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                
                mse = mean_squared_error(y_true, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_true, y_pred)
                r2 = r2_score(y_true, y_pred)
                
                return {
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2,
                    'accuracy': max(0, r2)  # Для совместимости
                }
                
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}")
            return {}
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Получение метрик тренера
        
        Returns:
            Dict: Метрики
        """
        return self._metrics.copy()
    
    def reset_metrics(self) -> None:
        """Сброс метрик"""
        self._metrics = {
            'models_trained': 0,
            'training_time': 0.0,
            'best_scores': {}
        }
    
    def get_best_models(self) -> Dict[str, Any]:
        """
        Получение лучших моделей по типам
        
        Returns:
            Dict: Лучшие модели
        """
        return self._metrics.get('best_scores', {})
