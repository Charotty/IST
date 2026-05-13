"""
Ensemble Model

Ансамбль моделей для улучшения предсказаний.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
import joblib

from ..base.base_model import BaseModel


class EnsembleModel(BaseModel):
    """Ансамбль моделей"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Инициализация ансамбля
        
        Args:
            config: Конфигурация ансамбля
        """
        super().__init__(config)
        
        # Параметры ансамбля
        self.ensemble_type = config.get('ensemble_type', 'voting')  # voting, stacking, blending
        self.models = []
        self.model_configs = config.get('models', [])
        self.voting_strategy = config.get('voting_strategy', 'soft')  # hard, soft
        self.weight_strategy = config.get('weight_strategy', 'equal')  # equal, performance, custom
        
        # Параметры stacking
        self.meta_model_type = config.get('meta_model_type', 'logistic')  # logistic, linear
        self.meta_model = None
        
        # Веса моделей
        self.model_weights = []
        self.model_performance = {}
        
        # Инициализация моделей
        self._initialize_models()
        
        # Построение ансамбля
        self._build_ensemble()
    
    def _initialize_models(self) -> None:
        """Инициализация моделей ансамбля"""
        try:
            from ..deep_learning.lstm_model import LSTMModel
            from ..deep_learning.transformer_model import TransformerModel
            from ..boosting.lightgbm_model import LightGBMModel
            from ..boosting.xgboost_model import XGBoostModel
            
            model_classes = {
                'lstm': LSTMModel,
                'transformer': TransformerModel,
                'lightgbm': LightGBMModel,
                'xgboost': XGBoostModel
            }
            
            for model_config in self.model_configs:
                model_type = model_config.get('type')
                model_class = model_classes.get(model_type)
                
                if model_class:
                    model = model_class(model_config)
                    self.models.append(model)
                    self.logger.info(f"Initialized {model_type} model for ensemble")
                else:
                    self.logger.warning(f"Unknown model type: {model_type}")
            
            # Инициализация весов
            if self.weight_strategy == 'equal':
                self.model_weights = [1.0 / len(self.models)] * len(self.models)
            else:
                self.model_weights = [1.0] * len(self.models)
            
        except Exception as e:
            self._handle_error(e, "_initialize_models")
            raise
    
    def _build_ensemble(self) -> None:
        """Построение ансамбля"""
        try:
            if self.ensemble_type == 'voting':
                # Voting ансамбль
                if self.task_type == 'classification':
                    self.ensemble = VotingClassifier(
                        estimators=[(f"model_{i}", model) for i, model in enumerate(self.models)],
                        voting=self.voting_strategy,
                        weights=self.model_weights
                    )
                else:
                    self.ensemble = VotingRegressor(
                        estimators=[(f"model_{i}", model) for i, model in enumerate(self.models)],
                        weights=self.model_weights
                    )
            
            elif self.ensemble_type == 'stacking':
                # Stacking ансамбль
                self._build_stacking_ensemble()
            
            elif self.ensemble_type == 'blending':
                # Blending ансамбль
                self._build_blending_ensemble()
            
            else:
                raise ValueError(f"Unknown ensemble type: {self.ensemble_type}")
            
            self.model_info = {
                'ensemble_type': self.ensemble_type,
                'num_models': len(self.models),
                'voting_strategy': self.voting_strategy,
                'weight_strategy': self.weight_strategy,
                'models': [model.get_model_info() for model in self.models]
            }
            
            self.logger.info(f"Built {self.ensemble_type} ensemble with {len(self.models)} models")
            
        except Exception as e:
            self._handle_error(e, "_build_ensemble")
            raise
    
    def _build_stacking_ensemble(self) -> None:
        """Построение stacking ансамбля"""
        try:
            # Meta модель для stacking
            if self.meta_model_type == 'logistic':
                self.meta_model = LogisticRegression(random_state=42)
            elif self.meta_model_type == 'linear':
                self.meta_model = LinearRegression()
            else:
                raise ValueError(f"Unknown meta model type: {self.meta_model_type}")
            
            self.ensemble = None  # Будет определен в predict
            
        except Exception as e:
            self._handle_error(e, "_build_stacking_ensemble")
            raise
    
    def _build_blending_ensemble(self) -> None:
        """Построение blending ансамбля"""
        try:
            # Blending использует взвешенное среднее предсказаний
            self.ensemble = None  # Будет определен в predict
            
        except Exception as e:
            self._handle_error(e, "_build_blending_ensemble")
            raise
    
    def _get_model_predictions(self, X: pd.DataFrame) -> List[np.ndarray]:
        """
        Получение предсказаний от всех моделей
        
        Args:
            X: Входные данные
            
        Returns:
            List: Предсказания моделей
        """
        predictions = []
        
        for i, model in enumerate(self.models):
            try:
                if model.is_trained:
                    pred = model.predict(X)
                    predictions.append(pred)
                else:
                    self.logger.warning(f"Model {i} is not trained")
                    predictions.append(np.zeros(len(X)))
            except Exception as e:
                self.logger.error(f"Error getting predictions from model {i}: {e}")
                predictions.append(np.zeros(len(X)))
        
        return predictions
    
    def _get_model_probabilities(self, X: pd.DataFrame) -> List[np.ndarray]:
        """
        Получение вероятностей от всех моделей
        
        Args:
            X: Входные данные
            
        Returns:
            List: Вероятности моделей
        """
        probabilities = []
        
        for i, model in enumerate(self.models):
            try:
                if model.is_trained:
                    prob = model.predict_proba(X)
                    probabilities.append(prob)
                else:
                    self.logger.warning(f"Model {i} is not trained")
                    # Равномерные вероятности для необученной модели
                    uniform_prob = np.full((len(X), 3), 1.0/3.0)
                    probabilities.append(uniform_prob)
            except Exception as e:
                self.logger.error(f"Error getting probabilities from model {i}: {e}")
                uniform_prob = np.full((len(X), 3), 1.0/3.0)
                probabilities.append(uniform_prob)
        
        return probabilities
    
    def _voting_predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Предсказание через voting
        
        Args:
            X: Входные данные
            
        Returns:
            np.ndarray: Предсказания
        """
        try:
            if self.task_type == 'classification':
                return self.ensemble.predict(X)
            else:
                return self.ensemble.predict(X)
        except Exception as e:
            self._handle_error(e, "_voting_predict")
            return np.array([])
    
    def _stacking_predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Предсказание через stacking
        
        Args:
            X: Входные данные
            
        Returns:
            np.ndarray: Предсказания
        """
        try:
            # Получение предсказаний от базовых моделей
            base_predictions = self._get_model_predictions(X)
            
            if not base_predictions:
                return np.array([])
            
            # Создание признаков для meta модели
            meta_features = np.column_stack(base_predictions)
            
            # Предсказание meta моделью
            meta_pred = self.meta_model.predict(meta_features)
            
            return meta_pred
            
        except Exception as e:
            self._handle_error(e, "_stacking_predict")
            return np.array([])
    
    def _blending_predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Предсказание через blending
        
        Args:
            X: Входные данные
            
        Returns:
            np.ndarray: Предсказания
        """
        try:
            # Получение предсказаний от всех моделей
            predictions = self._get_model_predictions(X)
            
            if not predictions:
                return np.array([])
            
            # Взвешенное среднее
            weighted_pred = np.zeros_like(predictions[0])
            
            for i, pred in enumerate(predictions):
                weight = self.model_weights[i] if i < len(self.model_weights) else 1.0
                weighted_pred += weight * pred
            
            # Нормализация весов
            total_weight = sum(self.model_weights[:len(predictions)])
            if total_weight > 0:
                weighted_pred /= total_weight
            
            return weighted_pred
            
        except Exception as e:
            self._handle_error(e, "_blending_predict")
            return np.array([])
    
    def _voting_predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Предсказание вероятностей через voting
        
        Args:
            X: Входные данные
            
        Returns:
            np.ndarray: Вероятности
        """
        try:
            if self.task_type == 'classification':
                return self.ensemble.predict_proba(X)
            else:
                # Для регрессии возвращаем предсказания как "вероятности"
                pred = self.ensemble.predict(X)
                # Конвертируем в формат вероятностей
                prob_array = np.zeros((len(pred), 3))
                prob_array[:, 0] = pred  # Предсказанное значение как "вероятность"
                return prob_array
        except Exception as e:
            self._handle_error(e, "_voting_predict_proba")
            return np.array([])
    
    def _stacking_predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Предсказание вероятностей через stacking
        
        Args:
            X: Входные данные
            
        Returns:
            np.ndarray: Вероятности
        """
        try:
            # Получение вероятностей от базовых моделей
            base_probabilities = self._get_model_probabilities(X)
            
            if not base_probabilities:
                return np.array([])
            
            # Создание признаков для meta модели
            meta_features = np.column_stack(base_probabilities)
            
            # Предсказание meta моделью
            if hasattr(self.meta_model, 'predict_proba'):
                meta_prob = self.meta_model.predict_proba(meta_features)
            else:
                # Если meta модель не поддерживает predict_proba
                meta_pred = self.meta_model.predict(meta_features)
                # Конвертируем в вероятности
                meta_prob = np.zeros((len(meta_pred), 3))
                for i, pred_class in enumerate(meta_pred):
                    if pred_class < 3:
                        meta_prob[i, pred_class] = 1.0
                    else:
                        meta_prob[i, 0] = 1.0
            
            return meta_prob
            
        except Exception as e:
            self._handle_error(e, "_stacking_predict_proba")
            return np.array([])
    
    def _blending_predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Предсказание вероятностей через blending
        
        Args:
            X: Входные данные
            
        Returns:
            np.ndarray: Вероятности
        """
        try:
            # Получение вероятностей от всех моделей
            probabilities = self._get_model_probabilities(X)
            
            if not probabilities:
                return np.array([])
            
            # Взвешенное среднее вероятностей
            weighted_prob = np.zeros_like(probabilities[0])
            
            for i, prob in enumerate(probabilities):
                weight = self.model_weights[i] if i < len(self.model_weights) else 1.0
                weighted_prob += weight * prob
            
            # Нормализация весов
            total_weight = sum(self.model_weights[:len(probabilities)])
            if total_weight > 0:
                weighted_prob /= total_weight
            
            return weighted_prob
            
        except Exception as e:
            self._handle_error(e, "_blending_predict_proba")
            return np.array([])
    
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> None:
        """
        Обучение ансамбля
        
        Args:
            X: Признаки
            y: Целевая переменная
            **kwargs: Дополнительные параметры
        """
        try:
            start_time = time.time()
            
            self._log_training_start(X.shape)
            
            # Обучение всех моделей
            for i, model in enumerate(self.models):
                try:
                    self.logger.info(f"Training model {i+1}/{len(self.models)}: {model.__class__.__name__}")
                    model.fit(X, y, **kwargs)
                    self.model_performance[i] = model.get_metrics()
                except Exception as e:
                    self.logger.error(f"Error training model {i}: {e}")
                    self.model_performance[i] = {'error': str(e)}
            
            # Обучение ансамбля
            if self.ensemble_type == 'voting':
                self.ensemble.fit(X, y)
            elif self.ensemble_type == 'stacking':
                self._fit_stacking_ensemble(X, y)
            # Blending не требует обучения
            
            self.is_trained = True
            training_time = time.time() - start_time
            self._log_training_end(training_time)
            
        except Exception as e:
            self._handle_error(e, "fit")
            raise
    
    def _fit_stacking_ensemble(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Обучение stacking ансамбля"""
        try:
            # Получение предсказаний от базовых моделей (cross-validation)
            from sklearn.model_selection import cross_val_predict
            
            base_predictions = []
            for model in self.models:
                if model.is_trained:
                    pred = cross_val_predict(model, X, y, cv=3)
                    base_predictions.append(pred)
                else:
                    base_predictions.append(np.zeros(len(X)))
            
            if not base_predictions:
                return
            
            # Создание признаков для meta модели
            meta_features = np.column_stack(base_predictions)
            
            # Обучение meta модели
            self.meta_model.fit(meta_features, y)
            
        except Exception as e:
            self._handle_error(e, "_fit_stacking_ensemble")
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
                raise ValueError("Ensemble must be trained before prediction")
            
            if not self.validate_input(X):
                return np.array([])
            
            # Предсказание в зависимости от типа ансамбля
            if self.ensemble_type == 'voting':
                predictions = self._voting_predict(X)
            elif self.ensemble_type == 'stacking':
                predictions = self._stacking_predict(X)
            elif self.ensemble_type == 'blending':
                predictions = self._blending_predict(X)
            else:
                predictions = np.array([])
            
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
                raise ValueError("Ensemble must be trained before prediction")
            
            if not self.validate_input(X):
                return np.array([])
            
            # Предсказание в зависимости от типа ансамбля
            if self.ensemble_type == 'voting':
                probabilities = self._voting_predict_proba(X)
            elif self.ensemble_type == 'stacking':
                probabilities = self._stacking_predict_proba(X)
            elif self.ensemble_type == 'blending':
                probabilities = self._blending_predict_proba(X)
            else:
                probabilities = np.array([])
            
            prediction_time = time.time() - start_time
            self._log_prediction(len(probabilities), prediction_time)
            
            return probabilities
            
        except Exception as e:
            self._handle_error(e, "predict_proba")
            return np.array([])
    
    def save_model(self, path: str) -> None:
        """
        Сохранение ансамбля
        
        Args:
            path: Путь для сохранения
        """
        try:
            import os
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Сохранение всех моделей
            for i, model in enumerate(self.models):
                model_path = f"{path}_model_{i}"
                model.save_model(model_path)
            
            # Сохранение ансамбля
            ensemble_data = {
                'ensemble_type': self.ensemble_type,
                'voting_strategy': self.voting_strategy,
                'weight_strategy': self.weight_strategy,
                'model_weights': self.model_weights,
                'model_performance': self.model_performance,
                'meta_model': self.meta_model,
                'config': self.config
            }
            
            if self.ensemble:
                joblib.dump(self.ensemble, f"{path}_ensemble.pkl")
            
            joblib.dump(ensemble_data, f"{path}_ensemble_data.pkl")
            
            # Сохранение метаданных
            self._save_metadata(path)
            
            self.logger.info(f"Ensemble model saved to {path}")
            
        except Exception as e:
            self._handle_error(e, "save_model")
            raise
    
    def load_model(self, path: str) -> None:
        """
        Загрузка ансамбля
        
        Args:
            path: Путь к модели
        """
        try:
            # Загрузка ансамбля
            import joblib
            ensemble_data = joblib.load(f"{path}_ensemble_data.pkl")
            
            self.ensemble_type = ensemble_data['ensemble_type']
            self.voting_strategy = ensemble_data['voting_strategy']
            self.weight_strategy = ensemble_data['weight_strategy']
            self.model_weights = ensemble_data['model_weights']
            self.model_performance = ensemble_data['model_performance']
            self.config = ensemble_data['config']
            
            # Загрузка ансамбля
            if os.path.exists(f"{path}_ensemble.pkl"):
                self.ensemble = joblib.load(f"{path}_ensemble.pkl")
            
            # Загрузка моделей
            self.models = []
            for i in range(len(ensemble_data['config']['models'])):
                model_path = f"{path}_model_{i}"
                model_type = ensemble_data['config']['models'][i]['type']
                
                from ..deep_learning.lstm_model import LSTMModel
                from ..deep_learning.transformer_model import TransformerModel
                from ..boosting.lightgbm_model import LightGBMModel
                from ..boosting.xgboost_model import XGBoostModel
                
                model_classes = {
                    'lstm': LSTMModel,
                    'transformer': TransformerModel,
                    'lightgbm': LightGBMModel,
                    'xgboost': XGBoostModel
                }
                
                model_class = model_classes.get(model_type)
                if model_class:
                    model = model_class(ensemble_data['config']['models'][i])
                    model.load_model(model_path)
                    self.models.append(model)
            
            self.is_trained = True
            
            # Загрузка метаданных
            self._load_metadata(path)
            
            self.logger.info(f"Ensemble model loaded from {path}")
            
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
            # Агрегация важности от всех моделей
            all_importance = {}
            
            for i, model in enumerate(self.models):
                importance = model.get_feature_importance()
                if importance:
                    for feature, imp in importance.items():
                        if feature in all_importance:
                            all_importance[feature] += imp
                        else:
                            all_importance[feature] = imp
            
            # Нормализация важности
            total_importance = sum(all_importance.values())
            if total_importance > 0:
                for feature in all_importance:
                    all_importance[feature] /= total_importance
            
            return all_importance
            
        except Exception as e:
            self._handle_error(e, "get_feature_importance")
            return None
    
    def get_ensemble_info(self) -> Dict[str, Any]:
        """
        Получение информации об ансамбле
        
        Returns:
            Dict: Информация об ансамбле
        """
        return {
            'ensemble_type': self.ensemble_type,
            'num_models': len(self.models),
            'voting_strategy': self.voting_strategy,
            'weight_strategy': self.weight_strategy,
            'model_weights': self.model_weights,
            'model_performance': self.model_performance,
            'models': [model.get_model_info() for model in self.models]
        }
