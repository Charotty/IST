"""
Base Model

Базовый класс для всех моделей машинного обучения.
"""

import asyncio
import logging
import json
import pickle
import os
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from .model_interface import ModelInterface


class BaseModel(ModelInterface):
    """Базовый класс для всех моделей"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Инициализация базовой модели
        
        Args:
            config: Конфигурация модели
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Атрибуты модели
        self.model = None
        self.is_trained = False
        self.model_info = {}
        self.training_history = []
        
        # Метаданные
        self.created_at = datetime.utcnow()
        self.last_trained = None
        self.version = config.get('version', '1.0.0')
        
        # Параметры
        self.task_type = config.get('task_type', 'classification')  # classification, regression
        self.feature_names = []
        self.target_name = None
        
        # Метрики
        self.metrics = {
            'training_time': 0.0,
            'prediction_time': 0.0,
            'predictions_count': 0,
            'accuracy': 0.0,
            'error_count': 0
        }
        
        # Валидация конфигурации
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Валидация конфигурации модели"""
        if not isinstance(self.config, dict):
            raise ValueError("Config must be a dictionary")
        
        if 'task_type' not in self.config:
            raise ValueError("task_type must be specified in config")
        
        valid_task_types = ['classification', 'regression', 'probability']
        if self.task_type not in valid_task_types:
            raise ValueError(f"task_type must be one of {valid_task_types}")
    
    def _update_metrics(self, metric_name: str, value: float) -> None:
        """
        Обновление метрик
        
        Args:
            metric_name: Имя метрики
            value: Значение
        """
        self.metrics[metric_name] = value
    
    def _log_training_start(self, X_shape: Tuple[int, int]) -> None:
        """
        Логирование начала обучения
        
        Args:
            X_shape: Размерность данных
        """
        self.logger.info(f"Training {self.__class__.__name__} on {X_shape[0]} samples, {X_shape[1]} features")
        self.logger.info(f"Task type: {self.task_type}")
        self.logger.info(f"Config: {self.config}")
    
    def _log_training_end(self, training_time: float) -> None:
        """
        Логирование окончания обучения
        
        Args:
            training_time: Время обучения
        """
        self.logger.info(f"Training completed in {training_time:.2f} seconds")
        self._update_metrics('training_time', training_time)
        self.last_trained = datetime.utcnow()
    
    def _log_prediction(self, n_predictions: int, prediction_time: float) -> None:
        """
        Логирование предсказания
        
        Args:
            n_predictions: Количество предсказаний
            prediction_time: Время предсказания
        """
        self.logger.debug(f"Made {n_predictions} predictions in {prediction_time:.4f} seconds")
        self._update_metrics('predictions_count', 
                          self.metrics['predictions_count'] + n_predictions)
        self._update_metrics('prediction_time', prediction_time)
    
    def _handle_error(self, error: Exception, context: str) -> None:
        """
        Обработка ошибок
        
        Args:
            error: Исключение
            context: Контекст ошибки
        """
        self._update_metrics('error_count', self.metrics['error_count'] + 1)
        self.logger.error(f"Error in {context}: {error}")
    
    def _save_metadata(self, path: str) -> None:
        """
        Сохранение метаданных модели
        
        Args:
            path: Путь для сохранения
        """
        try:
            metadata = {
                'model_class': self.__class__.__name__,
                'version': self.version,
                'task_type': self.task_type,
                'config': self.config,
                'created_at': self.created_at.isoformat(),
                'last_trained': self.last_trained.isoformat() if self.last_trained else None,
                'is_trained': self.is_trained,
                'feature_names': self.feature_names,
                'target_name': self.target_name,
                'metrics': self.metrics,
                'model_info': self.model_info
            }
            
            metadata_path = f"{path}.metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            self.logger.info(f"Model metadata saved to {metadata_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving metadata: {e}")
    
    def _load_metadata(self, path: str) -> None:
        """
        Загрузка метаданных модели
        
        Args:
            path: Путь к модели
        """
        try:
            metadata_path = f"{path}.metadata.json"
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                self.version = metadata.get('version', '1.0.0')
                self.task_type = metadata.get('task_type', 'classification')
                self.config = metadata.get('config', {})
                self.created_at = datetime.fromisoformat(metadata.get('created_at', datetime.utcnow().isoformat()))
                self.last_trained = datetime.fromisoformat(metadata['last_trained']) if metadata.get('last_trained') else None
                self.is_trained = metadata.get('is_trained', False)
                self.feature_names = metadata.get('feature_names', [])
                self.target_name = metadata.get('target_name')
                self.metrics = metadata.get('metrics', {})
                self.model_info = metadata.get('model_info', {})
                
                self.logger.info(f"Model metadata loaded from {metadata_path}")
            else:
                self.logger.warning(f"No metadata file found: {metadata_path}")
                
        except Exception as e:
            self.logger.error(f"Error loading metadata: {e}")
    
    def prepare_data(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Подготовка данных для модели
        
        Args:
            X: Признаки
            y: Целевая переменная
            
        Returns:
            Tuple: Подготовленные данные
        """
        try:
            # Валидация
            if not self.validate_input(X):
                raise ValueError("Invalid input data")
            
            # Сохранение имен признаков
            if self.feature_names == []:
                self.feature_names = X.columns.tolist()
            
            # Сохранение имени цели
            if y is not None:
                self.target_name = y.name if y.name else 'target'
            
            # Базовая подготовка (можно переопределить в дочерних классах)
            X_clean = X.copy()
            
            # Удаление NaN значений
            X_clean = X_clean.fillna(0)
            
            if y is not None:
                y_clean = y.copy()
                y_clean = y_clean.fillna(0)
                return X_clean, y_clean
            else:
                return X_clean, None
                
        except Exception as e:
            self._handle_error(e, "prepare_data")
            raise
    
    def evaluate_predictions(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Оценка предсказаний
        
        Args:
            y_true: Истинные значения
            y_pred: Предсказанные значения
            
        Returns:
            Dict: Метрики оценки
        """
        try:
            if self.task_type == 'classification':
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                
                return {
                    'accuracy': accuracy_score(y_true, y_pred),
                    'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
                    'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
                    'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
                }
            elif self.task_type == 'regression':
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                
                return {
                    'mse': mean_squared_error(y_true, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                    'mae': mean_absolute_error(y_true, y_pred),
                    'r2': r2_score(y_true, y_pred)
                }
            else:
                return {}
                
        except Exception as e:
            self._handle_error(e, "evaluate_predictions")
            return {}
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Получение информации о модели
        
        Returns:
            Dict: Информация о модели
        """
        return {
            'class_name': self.__class__.__name__,
            'version': self.version,
            'task_type': self.task_type,
            'is_trained': self.is_trained,
            'created_at': self.created_at.isoformat(),
            'last_trained': self.last_trained.isoformat() if self.last_trained else None,
            'feature_count': len(self.feature_names),
            'feature_names': self.feature_names[:10],  # Первые 10 признаков
            'target_name': self.target_name,
            'config': self.config,
            'metrics': self.metrics,
            'model_info': self.model_info
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Получение метрик
        
        Returns:
            Dict: Метрики
        """
        return self.metrics.copy()
    
    def reset_metrics(self) -> None:
        """Сброс метрик"""
        self.metrics = {
            'training_time': 0.0,
            'prediction_time': 0.0,
            'predictions_count': 0,
            'accuracy': 0.0,
            'error_count': 0
        }
    
    def get_prediction_type(self) -> str:
        """
        Получение типа предсказания
        
        Returns:
            str: Тип предсказания
        """
        return self.task_type
    
    def get_supported_tasks(self) -> List[str]:
        """
        Получение поддерживаемых задач
        
        Returns:
            List[str]: Поддерживаемые задачи
        """
        return ['classification', 'regression', 'probability']
    
    def __str__(self) -> str:
        """Строковое представление"""
        return f"{self.__class__.__name__}(task={self.task_type}, trained={self.is_trained})"
    
    def __repr__(self) -> str:
        """Подробное строковое представление"""
        return (f"{self.__class__.__name__}("
                f"task_type='{self.task_type}', "
                f"version='{self.version}', "
                f"is_trained={self.is_trained}, "
                f"feature_count={len(self.feature_names)})")
