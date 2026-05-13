"""
Model Validator

Валидатор моделей для оценки производительности.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import joblib


class ModelValidator:
    """Валидатор моделей"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Инициализация валидатора
        
        Args:
            config: Конфигурация валидатора
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Параметры
        self.cv_folds = config.get('cv_folds', 5)
        self.test_size = config.get('test_size', 0.2)
        self.random_state = config.get('random_state', 42)
        
        # Метрики
        self._metrics = {
            'models_validated': 0,
            'validation_time': 0.0,
            'best_scores': {}
        }
    
    async def validate_model(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Валидация модели
        
        Args:
            model: Модель для валидации
            X: Признаки
            y: Целевая переменная
            
        Returns:
            Dict: Результаты валидации
        """
        try:
            start_time = asyncio.get_event_loop().time()
            
            # Проверка наличия метода predict
            if not hasattr(model, 'predict'):
                raise ValueError("Model must have predict method")
            
            # Определение типа задачи
            task_type = model.get_prediction_type()
            
            if task_type == 'classification':
                results = await self._validate_classification_model(model, X, y)
            elif task_type == 'regression':
                results = await self._validate_regression_model(model, X, y)
            else:
                results = await self._validate_generic_model(model, X, y)
            
            validation_time = asyncio.get_event_loop().time() - start_time
            results['validation_time'] = validation_time
            
            # Сохранение лучших результатов
            model_name = model.__class__.__name__
            if model_name not in self._metrics['best_scores']:
                self._metrics['best_scores'][model_name] = {}
            
            current_best = self._metrics['best_scores'][model_name].get('score', -float('inf'))
            new_score = results.get('metrics', {}).get('accuracy', results.get('metrics', {}).get('r2', -float('inf')))
            
            if new_score > current_best:
                self._metrics['best_scores'][model_name] = {
                    'score': new_score,
                    'metrics': results.get('metrics', {}),
                    'timestamp': asyncio.get_event_loop().time()
                }
            
            self._metrics['models_validated'] += 1
            self._metrics['validation_time'] += validation_time
            
            self.logger.info(f"Model {model_name} validated successfully")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error validating model: {e}")
            return {'error': str(e)}
    
    async def _validate_classification_model(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Валидация классификационной модели"""
        try:
            # Cross-validation
            cv_scores = cross_val_score(
                model, X, y,
                cv=self.cv_folds,
                scoring='accuracy',
                n_jobs=-1
            )
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=self.cv_folds)
            ts_scores = cross_val_score(
                model, X, y,
                cv=tscv,
                scoring='accuracy',
                n_jobs=-1
            )
            
            # Train-test split
            X_train, X_test, y_train, y_test = self._train_test_split(X, y)
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Метрики на тестовых данных
            test_accuracy = accuracy_score(y_test, y_pred)
            
            # Feature importance если доступно
            feature_importance = model.get_feature_importance()
            
            return {
                'model_type': 'classification',
                'cv_scores': {
                    'mean': cv_scores.mean(),
                    'std': cv_scores.std(),
                    'min': cv_scores.min(),
                    'max': cv_scores.max()
                },
                'time_series_cv_scores': {
                    'mean': ts_scores.mean(),
                    'std': ts_scores.std(),
                    'min': ts_scores.min(),
                    'max': ts_scores.max()
                },
                'test_metrics': {
                    'accuracy': test_accuracy,
                    'samples': len(y_test)
                },
                'feature_importance': feature_importance,
                'model_info': model.get_model_info() if hasattr(model, 'get_model_info') else {}
            }
            
        except Exception as e:
            self.logger.error(f"Error in classification validation: {e}")
            return {'error': str(e)}
    
    async def _validate_regression_model(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Валидация регрессионной модели"""
        try:
            # Cross-validation
            cv_scores = cross_val_score(
                model, X, y,
                cv=self.cv_folds,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=self.cv_folds)
            ts_scores = cross_val_score(
                model, X, y,
                cv=tscv,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            
            # Train-test split
            X_train, X_test, y_train, y_test = self._train_test_split(X, y)
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Метрики на тестовых данных
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            # Feature importance если доступно
            feature_importance = model.get_feature_importance()
            
            return {
                'model_type': 'regression',
                'cv_scores': {
                    'mse_mean': -cv_scores.mean(),  # Отрицаем так как cross_val_score возвращает -MSE
                    'mse_std': cv_scores.std(),
                    'mse_min': -cv_scores.min(),
                    'mse_max': -cv_scores.max()
                },
                'time_series_cv_scores': {
                    'mse_mean': -ts_scores.mean(),
                    'mse_std': ts_scores.std(),
                    'mse_min': -ts_scores.min(),
                    'mse_max': -ts_scores.max()
                },
                'test_metrics': {
                    'mse': mse,
                    'rmse': rmse,
                    'r2': r2,
                    'samples': len(y_test)
                },
                'feature_importance': feature_importance,
                'model_info': model.get_model_info() if hasattr(model, 'get_model_info') else {}
            }
            
        except Exception as e:
            self.logger.error(f"Error in regression validation: {e}")
            return {'error': str(e)}
    
    async def _validate_generic_model(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Валидация универсальной модели"""
        try:
            # Train-test split
            X_train, X_test, y_train, y_test = self._train_test_split(X, y)
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Базовые метрики
            if hasattr(model, 'predict_proba'):
                # Классификация
                accuracy = accuracy_score(y_test, y_pred)
                return {
                    'model_type': 'classification',
                    'test_metrics': {
                        'accuracy': accuracy,
                        'samples': len(y_test)
                    },
                    'model_info': model.get_model_info() if hasattr(model, 'get_model_info') else {}
                }
            else:
                # Регрессия
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                return {
                    'model_type': 'regression',
                    'test_metrics': {
                        'mse': mse,
                        'r2': r2,
                        'samples': len(y_test)
                    },
                    'model_info': model.get_model_info() if hasattr(model, 'get_model_info') else {}
                }
                
        except Exception as e:
            self.logger.error(f"Error in generic validation: {e}")
            return {'error': str(e)}
    
    def _train_test_split(self, X: pd.DataFrame, y: pd.Series) -> Tuple:
        """Разделение на train/test"""
        from sklearn.model_selection import train_test_split
        
        return train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y if len(y.unique()) > 2 else None
        )
    
    async def evaluate_model(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Оценка модели
        
        Args:
            model: Модель для оценки
            X: Признаки
            y: Целевая переменная
            
        Returns:
            Dict: Результаты оценки
        """
        try:
            # Разделение данных
            X_train, X_test, y_train, y_test = self._train_test_split(X, y)
            
            # Обучение
            model.fit(X_train, y_train)
            
            # Предсказание
            y_pred = model.predict(X_test)
            
            # Расчет метрик
            task_type = model.get_prediction_type()
            
            if task_type == 'classification':
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
                
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                    'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                    'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
                    'classification_report': classification_report(y_test, y_pred, output_dict=True)
                }
            else:
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                metrics = {
                    'mse': mse,
                    'rmse': np.sqrt(mse),
                    'mae': mae,
                    'r2': r2
                }
            
            return {
                'model_type': task_type,
                'test_samples': len(y_test),
                'train_samples': len(y_train),
                'metrics': metrics,
                'model_info': model.get_model_info() if hasattr(model, 'get_model_info') else {}
            }
            
        except Exception as e:
            self.logger.error(f"Error evaluating model: {e}")
            return {'error': str(e)}
    
    async def compare_models(self, models: List[Any], X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Сравнение моделей
        
        Args:
            models: Список моделей
            X: Признаки
            y: Целевая переменная
            
        Returns:
            Dict: Результаты сравнения
        """
        try:
            comparison_results = {}
            
            for model in models:
                model_name = model.__class__.__name__
                
                # Валидация модели
                results = await self.validate_model(model, X, y)
                
                if 'error' not in results:
                    comparison_results[model_name] = results
                else:
                    comparison_results[model_name] = {'error': results['error']}
            
            # Ранжирование моделей
            valid_results = {k: v for k, v in comparison_results.items() if 'error' not in v}
            
            if valid_results:
                # Определение метрики для ранжирования
                first_result = list(valid_results.values())[0]
                if 'test_metrics' in first_result:
                    if 'accuracy' in first_result['test_metrics']:
                        ranking_metric = 'accuracy'
                        higher_is_better = True
                    elif 'r2' in first_result['test_metrics']:
                        ranking_metric = 'r2'
                        higher_is_better = True
                    else:
                        ranking_metric = 'mse'
                        higher_is_better = False  # для MSE меньше лучше
                
                # Сортировка моделей
                sorted_models = sorted(
                    valid_results.items(),
                    key=lambda x: x[1]['test_metrics'].get(ranking_metric, 0),
                    reverse=higher_is_better
                )
                
                comparison_results['ranking'] = {
                    'metric': ranking_metric,
                    'higher_is_better': higher_is_better,
                    'sorted_models': [model[0] for model in sorted_models]
                }
            
            comparison_results['summary'] = {
                'total_models': len(models),
                'successful_validations': len(valid_results),
                'failed_validations': len(models) - len(valid_results)
            }
            
            return comparison_results
            
        except Exception as e:
            self.logger.error(f"Error comparing models: {e}")
            return {'error': str(e)}
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Получение метрик валидатора
        
        Returns:
            Dict: Метрики
        """
        return self._metrics.copy()
    
    def reset_metrics(self) -> None:
        """Сброс метрик"""
        self._metrics = {
            'models_validated': 0,
            'validation_time': 0.0,
            'best_scores': {}
        }
