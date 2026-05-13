"""
XGBoost Model

Модель градиентного бустинга XGBoost.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
try:
    import xgboost as xgb
except ImportError:
    xgb = None
import joblib
import os

from ..base.base_model import BaseModel
try:
    from sklearn.base import BaseEstimator
except ImportError:
    BaseEstimator = object


class XGBoostModel(BaseModel, BaseEstimator):
    """XGBoost модель градиентного бустинга"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Инициализация XGBoost модели
        
        Args:
            config: Конфигурация модели
        """
        super().__init__(config)
        
        if xgb is None:
            raise ImportError("xgboost is not installed. Please install it with: pip install xgboost")
        
        # Параметры XGBoost
        self.max_depth = config.get('max_depth', 6)
        self.learning_rate = config.get('learning_rate', 0.1)
        self.n_estimators = config.get('n_estimators', 100)
        self.subsample = config.get('subsample', 0.8)
        self.colsample_bytree = config.get('colsample_bytree', 0.8)
        self.colsample_bylevel = config.get('colsample_bylevel', 0.8)
        self.gamma = config.get('gamma', 0)
        self.min_child_weight = config.get('min_child_weight', 1)
        self.reg_alpha = config.get('reg_alpha', 0)
        self.reg_lambda = config.get('reg_lambda', 1)
        self.random_state = config.get('random_state', 42)
        self.n_jobs = config.get('n_jobs', -1)
        
        # Параметры обучения
        self.early_stopping_rounds = config.get('early_stopping_rounds', 10)
        self.verbose = config.get('verbose', 1)
        
        # Инициализация модели
        self._build_model()
        
        # История обучения
        self.training_history = []
        self.best_iteration = None
    
    def _build_model(self) -> None:
        """Построение XGBoost модели"""
        try:
            # Определение параметров в зависимости от типа задачи
            if self.task_type == 'classification':
                if len(set(self.target_name or [])) <= 2:
                    objective = 'binary:logistic'
                    eval_metric = 'logloss'
                else:
                    objective = 'multi:softprob'
                    eval_metric = 'mlogloss'
            else:
                objective = 'reg:squarederror'
                eval_metric = 'rmse'
            
            # Параметры модели
            params = {
                'max_depth': self.max_depth,
                'learning_rate': self.learning_rate,
                'n_estimators': self.n_estimators,
                'subsample': self.subsample,
                'colsample_bytree': self.colsample_bytree,
                'colsample_bylevel': self.colsample_bylevel,
                'gamma': self.gamma,
                'min_child_weight': self.min_child_weight,
                'reg_alpha': self.reg_alpha,
                'reg_lambda': self.reg_lambda,
                'random_state': self.random_state,
                'n_jobs': self.n_jobs,
                'verbose': self.verbose
            }
            
            # Создание модели
            if self.task_type == 'classification':
                self.model = xgb.XGBClassifier(**params)
            else:
                self.model = xgb.XGBRegressor(**params)
            
            self.model_info = {
                'architecture': 'XGBoost',
                'objective': objective,
                'eval_metric': eval_metric,
                'max_depth': self.max_depth,
                'learning_rate': self.learning_rate,
                'n_estimators': self.n_estimators,
                'subsample': self.subsample,
                'colsample_bytree': self.colsample_bytree,
                'colsample_bylevel': self.colsample_bylevel,
                'gamma': self.gamma,
                'min_child_weight': self.min_child_weight,
                'reg_alpha': self.reg_alpha,
                'reg_lambda': self.reg_lambda
            }
            
            self.logger.info(f"XGBoost model built with objective: {objective}")
            
        except Exception as e:
            self._handle_error(e, "_build_model")
            raise
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> None:
        """
        Обучение XGBoost модели
        
        Args:
            X: Признаки
            y: Целевая переменная
            **kwargs: Дополнительные параметры
        """
        try:
            start_time = time.time()
            
            # Подготовка данных
            X_clean, y_clean = self.prepare_data(X, y)
            
            self._log_training_start(X_clean.shape)
            
            # Разделение на train/validation для early stopping
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                X_clean, y_clean, 
                test_size=0.2, 
                random_state=self.random_state,
                stratify=y_clean if self.task_type == 'classification' else None
            )
            
            # Обучение с early stopping
            eval_set = [(X_val, y_val)]
            
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=self.verbose
            )
            
            # Сохранение лучшей итерации
            if hasattr(self.model, 'best_iteration'):
                self.best_iteration = self.model.best_iteration
            
            # Сохранение истории обучения
            if hasattr(self.model, 'evals_result_'):
                evals_result = self.model.evals_result_
                for i, metric_name in enumerate(evals_result['validation_0'].keys()):
                    for epoch, value in enumerate(evals_result['validation_0'][metric_name]):
                        self.training_history.append({
                            'epoch': epoch + 1,
                            'metric': metric_name,
                            'value': value
                        })
            
            self.is_trained = True
            training_time = time.time() - start_time
            self._log_training_end(training_time)
            
        except Exception as e:
            self._handle_error(e, "fit")
            raise
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Предсказание
        
        Args:
            X: Признаки
            
        Returns:
            np.ndarray: Предсказания
        """
        try:
            start_time = time.time()
            
            if not self.is_trained:
                raise ValueError("Model must be trained before prediction")
            
            # Подготовка данных
            X_clean, _ = self.prepare_data(X)
            
            # Предсказание
            predictions = self.model.predict(X_clean)
            
            prediction_time = time.time() - start_time
            self._log_prediction(len(predictions), prediction_time)
            
            return predictions
            
        except Exception as e:
            self._handle_error(e, "predict")
            return np.array([])
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Предсказание вероятностей
        
        Args:
            X: Признаки
            
        Returns:
            np.ndarray: Вероятности
        """
        try:
            start_time = time.time()
            
            if not self.is_trained:
                raise ValueError("Model must be trained before prediction")
            
            if self.task_type != 'classification':
                raise ValueError("predict_proba is only available for classification models")
            
            # Подготовка данных
            X_clean, _ = self.prepare_data(X)
            
            # Предсказание вероятностей
            probabilities = self.model.predict_proba(X_clean)
            
            prediction_time = time.time() - start_time
            self._log_prediction(len(probabilities), prediction_time)
            
            return probabilities
            
        except Exception as e:
            self._handle_error(e, "predict_proba")
            return np.array([])
    
    def save_model(self, path: str) -> None:
        """
        Сохранение модели
        
        Args:
            path: Путь для сохранения
        """
        try:
            # Создание директории если необходимо
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Сохранение модели
            model_data = {
                'model': self.model,
                'config': self.config,
                'training_history': self.training_history,
                'best_iteration': self.best_iteration,
                'model_info': self.model_info
            }
            
            joblib.dump(model_data, path)
            
            # Сохранение метаданных
            self._save_metadata(path)
            
            self.logger.info(f"XGBoost model saved to {path}")
            
        except Exception as e:
            self._handle_error(e, "save_model")
            raise
    
    def load_model(self, path: str) -> None:
        """
        Загрузка модели
        
        Args:
            path: Путь к модели
        """
        try:
            # Загрузка модели
            model_data = joblib.load(path)
            
            self.model = model_data['model']
            self.config = model_data['config']
            self.training_history = model_data.get('training_history', [])
            self.best_iteration = model_data.get('best_iteration')
            self.model_info = model_data.get('model_info', {})
            
            # Обновление параметров
            self.max_depth = self.config.get('max_depth', 6)
            self.learning_rate = self.config.get('learning_rate', 0.1)
            self.n_estimators = self.config.get('n_estimators', 100)
            self.subsample = self.config.get('subsample', 0.8)
            self.colsample_bytree = self.config.get('colsample_bytree', 0.8)
            self.colsample_bylevel = self.config.get('colsample_bylevel', 0.8)
            self.gamma = self.config.get('gamma', 0)
            self.min_child_weight = self.config.get('min_child_weight', 1)
            self.reg_alpha = self.config.get('reg_alpha', 0)
            self.reg_lambda = self.config.get('reg_lambda', 1)
            self.random_state = self.config.get('random_state', 42)
            self.n_jobs = self.config.get('n_jobs', -1)
            self.early_stopping_rounds = self.config.get('early_stopping_rounds', 10)
            self.verbose = self.config.get('verbose', 1)
            
            self.is_trained = True
            
            # Загрузка метаданных
            self._load_metadata(path)
            
            self.logger.info(f"XGBoost model loaded from {path}")
            
        except Exception as e:
            self._handle_error(e, "load_model")
            raise
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Получение важности признаков
        
        Returns:
            Dict: Важность признаков
        """
        try:
            if not self.is_trained:
                return None
            
            # Получение важности признаков
            importance = self.model.feature_importances_
            feature_names = self.feature_names
            
            if len(importance) != len(feature_names):
                self.logger.warning("Feature importance length mismatch")
                return None
            
            # Создание словаря важности
            feature_importance = {}
            for i, feature_name in enumerate(feature_names):
                feature_importance[feature_name] = float(importance[i])
            
            # Нормализация
            total_importance = sum(feature_importance.values())
            if total_importance > 0:
                for feature_name in feature_importance:
                    feature_importance[feature_name] /= total_importance
            
            return feature_importance
            
        except Exception as e:
            self._handle_error(e, "get_feature_importance")
            return None
    
    def get_training_history(self) -> List[Dict[str, Any]]:
        """
        Получение истории обучения
        
        Returns:
            List: История обучения
        """
        return self.training_history.copy()
    
    def get_best_iteration(self) -> Optional[int]:
        """
        Получение лучшей итерации
        
        Returns:
            int: Лучшая итерация
        """
        return self.best_iteration
