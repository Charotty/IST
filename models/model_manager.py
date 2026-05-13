"""
Model Manager

Главный менеджер моделей для координации всех компонентов.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import pandas as pd
import numpy as np

from .registry.model_registry import ModelRegistry
from .training.trainer import ModelTrainer
from .training.validator import ModelValidator


class ModelManager:
    """Главный менеджер моделей"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Инициализация менеджера моделей
        
        Args:
            config: Конфигурация менеджера
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Инициализация компонентов
        self.registry = None
        self.trainer = None
        self.validator = None
        
        # Состояние системы
        self._running = False
        self._start_time = None
        
        # Активные модели
        self._active_models = {}
        self._model_cache = {}
        
        # Метрики
        self._metrics = {
            'total_models': 0,
            'active_models': 0,
            'predictions_made': 0,
            'training_jobs': 0,
            'errors_count': 0
        }
        
        # Инициализация
        self._initialize_components()
    
    def _initialize_components(self) -> None:
        """Инициализация компонентов"""
        try:
            # Инициализация реестра
            registry_config = self.config.get('registry', {})
            self.registry = ModelRegistry(registry_config)
            
            # Инициализация тренера
            training_config = self.config.get('training', {})
            self.trainer = ModelTrainer(training_config)
            
            # Инициализация валидатора
            validation_config = self.config.get('validation', {})
            self.validator = ModelValidator(validation_config)
            
            self.logger.info("Model Manager components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing components: {e}")
            raise
    
    async def start(self) -> None:
        """Запуск менеджера моделей"""
        try:
            if self._running:
                self.logger.warning("Model Manager is already running")
                return
            
            self._running = True
            self._start_time = datetime.utcnow()
            
            # Запуск компонентов
            # Реестр не требует запуска
            
            self.logger.info("Model Manager started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start Model Manager: {e}")
            raise
    
    async def stop(self) -> None:
        """Остановка менеджера моделей"""
        try:
            if not self._running:
                return
            
            self._running = False
            
            # Остановка всех активных моделей
            for model_id, model in self._active_models.items():
                try:
                    if hasattr(model, 'cleanup'):
                        await model.cleanup()
                    self.logger.info(f"Cleaned up model {model_id}")
                except Exception as e:
                    self.logger.error(f"Error cleaning up model {model_id}: {e}")
            
            # Очистка кэша
            self._active_models.clear()
            self._model_cache.clear()
            
            self.logger.info("Model Manager stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Error stopping Model Manager: {e}")
    
    async def register_model(self, model_name: str, model: Any, 
                          config: Dict[str, Any], 
                          metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Регистрация модели
        
        Args:
            model_name: Имя модели
            model: Объект модели
            config: Конфигурация модели
            metadata: Дополнительные метаданные
            
        Returns:
            str: ID модели
        """
        try:
            # Регистрация в реестре
            model_id = await self.registry.register_model(
                model_name, model, config, metadata
            )
            
            # Добавление в активные модели
            self._active_models[model_id] = model
            
            # Обновление метрик
            self._metrics['total_models'] += 1
            self._metrics['active_models'] = len(self._active_models)
            
            self.logger.info(f"Model {model_id} registered successfully")
            
            return model_id
            
        except Exception as e:
            self._metrics['errors_count'] += 1
            self.logger.error(f"Error registering model: {e}")
            raise
    
    async def load_model(self, model_id: str, version: Optional[str] = None) -> Any:
        """
        Загрузка модели
        
        Args:
            model_id: ID модели
            version: Версия модели
            
        Returns:
            Any: Загруженная модель
        """
        try:
            # Проверка кэша
            cache_key = f"{model_id}_{version or 'latest'}"
            if cache_key in self._model_cache:
                self.logger.info(f"Model {model_id} loaded from cache")
                return self._model_cache[cache_key]
            
            # Загрузка из реестра
            model = await self.registry.load_model(model_id, version)
            
            # Сохранение в кэш
            self._model_cache[cache_key] = model
            
            # Добавление в активные модели
            self._active_models[model_id] = model
            
            self._metrics['active_models'] = len(self._active_models)
            
            self.logger.info(f"Model {model_id} loaded successfully")
            
            return model
            
        except Exception as e:
            self._metrics['errors_count'] += 1
            self.logger.error(f"Error loading model: {e}")
            raise
    
    async def train_model(self, model_id: str, X: pd.DataFrame, y: pd.Series,
                        training_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Обучение модели
        
        Args:
            model_id: ID модели
            X: Признаки
            y: Целевая переменная
            training_config: Конфигурация обучения
            
        Returns:
            Dict: Результаты обучения
        """
        try:
            # Проверка наличия модели
            if model_id not in self._active_models:
                raise ValueError(f"Model {model_id} not found")
            
            model = self._active_models[model_id]
            
            # Обучение модели
            self._metrics['training_jobs'] += 1
            
            training_results = await self.trainer.train_model(
                model, X, y, training_config
            )
            
            # Валидация модели
            validation_results = await self.validator.validate_model(
                model, X, y
            )
            
            # Сохранение обученной модели
            new_version = await self.registry.save_model(
                model_id, model, training_config
            )
            
            # Обновление кэша
            cache_key = f"{model_id}_{new_version}"
            self._model_cache[cache_key] = model
            
            results = {
                'model_id': model_id,
                'version': new_version,
                'training_results': training_results,
                'validation_results': validation_results,
                'training_time': training_results.get('training_time', 0),
                'performance_metrics': validation_results.get('metrics', {})
            }
            
            self.logger.info(f"Model {model_id} trained successfully")
            
            return results
            
        except Exception as e:
            self._metrics['errors_count'] += 1
            self.logger.error(f"Error training model: {e}")
            raise
    
    async def predict(self, model_id: str, X: pd.DataFrame, 
                   version: Optional[str] = None) -> np.ndarray:
        """
        Предсказание
        
        Args:
            model_id: ID модели
            X: Признаки
            version: Версия модели
            
        Returns:
            np.ndarray: Предсказания
        """
        try:
            # Загрузка модели если необходимо
            if model_id not in self._active_models:
                model = await self.load_model(model_id, version)
            else:
                model = self._active_models[model_id]
            
            # Предсказание
            predictions = model.predict(X)
            
            # Обновление метрик
            self._metrics['predictions_made'] += len(predictions)
            
            return predictions
            
        except Exception as e:
            self._metrics['errors_count'] += 1
            self.logger.error(f"Error making prediction: {e}")
            raise
    
    async def predict_proba(self, model_id: str, X: pd.DataFrame,
                          version: Optional[str] = None) -> np.ndarray:
        """
        Предсказание вероятностей
        
        Args:
            model_id: ID модели
            X: Признаки
            version: Версия модели
            
        Returns:
            np.ndarray: Вероятности
        """
        try:
            # Загрузка модели если необходимо
            if model_id not in self._active_models:
                model = await self.load_model(model_id, version)
            else:
                model = self._active_models[model_id]
            
            # Предсказание вероятностей
            probabilities = model.predict_proba(X)
            
            # Обновление метрик
            self._metrics['predictions_made'] += len(probabilities)
            
            return probabilities
            
        except Exception as e:
            self._metrics['errors_count'] += 1
            self.logger.error(f"Error making probability prediction: {e}")
            raise
    
    async def evaluate_model(self, model_id: str, X: pd.DataFrame, y: pd.Series,
                           version: Optional[str] = None) -> Dict[str, Any]:
        """
        Оценка модели
        
        Args:
            model_id: ID модели
            X: Признаки
            y: Целевая переменная
            version: Версия модели
            
        Returns:
            Dict: Результаты оценки
        """
        try:
            # Загрузка модели если необходимо
            if model_id not in self._active_models:
                model = await self.load_model(model_id, version)
            else:
                model = self._active_models[model_id]
            
            # Оценка модели
            evaluation_results = await self.validator.evaluate_model(
                model, X, y
            )
            
            return evaluation_results
            
        except Exception as e:
            self._metrics['errors_count'] += 1
            self.logger.error(f"Error evaluating model: {e}")
            raise
    
    async def list_models(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Список моделей
        
        Args:
            status: Фильтр по статусу
            
        Returns:
            List: Список моделей
        """
        try:
            return await self.registry.list_models(status)
        except Exception as e:
            self._metrics['errors_count'] += 1
            self.logger.error(f"Error listing models: {e}")
            return []
    
    async def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Получение информации о модели
        
        Args:
            model_id: ID модели
            
        Returns:
            Dict: Информация о модели
        """
        try:
            return await self.registry.get_model_info(model_id)
        except Exception as e:
            self._metrics['errors_count'] += 1
            self.logger.error(f"Error getting model info: {e}")
            return None
    
    async def delete_model(self, model_id: str) -> None:
        """
        Удаление модели
        
        Args:
            model_id: ID модели
        """
        try:
            # Удаление из активных моделей
            if model_id in self._active_models:
                del self._active_models[model_id]
            
            # Удаление из кэша
            keys_to_remove = [k for k in self._model_cache.keys() if k.startswith(model_id)]
            for key in keys_to_remove:
                del self._model_cache[key]
            
            # Удаление из реестра
            await self.registry.delete_model(model_id)
            
            # Обновление метрик
            self._metrics['total_models'] -= 1
            self._metrics['active_models'] = len(self._active_models)
            
            self.logger.info(f"Model {model_id} deleted successfully")
            
        except Exception as e:
            self._metrics['errors_count'] += 1
            self.logger.error(f"Error deleting model: {e}")
            raise
    
    async def compare_models(self, model_ids: List[str], X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Сравнение моделей
        
        Args:
            model_ids: Список ID моделей
            X: Признаки
            y: Целевая переменная
            
        Returns:
            Dict: Результаты сравнения
        """
        try:
            comparison_results = {}
            
            for model_id in model_ids:
                try:
                    evaluation = await self.evaluate_model(model_id, X, y)
                    comparison_results[model_id] = evaluation
                except Exception as e:
                    self.logger.error(f"Error evaluating model {model_id}: {e}")
                    comparison_results[model_id] = {'error': str(e)}
            
            # Добавление сводной информации
            comparison_results['summary'] = {
                'total_models': len(model_ids),
                'successful_evaluations': len([r for r in comparison_results.values() if 'error' not in r]),
                'best_model': None,
                'ranking': []
            }
            
            # Ранжирование моделей
            successful_results = {k: v for k, v in comparison_results.items() if 'error' not in v}
            if successful_results:
                # Сортировка по метрике (например, accuracy или R²)
                sorted_models = sorted(
                    successful_results.items(),
                    key=lambda x: x[1].get('metrics', {}).get('accuracy', 0) or 
                                   x[1].get('metrics', {}).get('r2', -float('inf')),
                    reverse=True
                )
                
                comparison_results['summary']['ranking'] = sorted_models
                comparison_results['summary']['best_model'] = sorted_models[0][0] if sorted_models else None
            
            return comparison_results
            
        except Exception as e:
            self._metrics['errors_count'] += 1
            self.logger.error(f"Error comparing models: {e}")
            raise
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Получение метрик
        
        Returns:
            Dict: Метрики менеджера
        """
        metrics = self._metrics.copy()
        
        # Добавление uptime
        if self._start_time:
            metrics['uptime'] = (datetime.utcnow() - self._start_time).total_seconds()
        else:
            metrics['uptime'] = 0.0
        
        # Метрики компонентов
        if self.registry:
            metrics['registry'] = self.registry.get_metrics()
        
        if self.trainer:
            metrics['trainer'] = self.trainer.get_metrics()
        
        if self.validator:
            metrics['validator'] = self.validator.get_metrics()
        
        # Дополнительные метрики
        metrics['cache_size'] = len(self._model_cache)
        metrics['active_model_ids'] = list(self._active_models.keys())
        
        return metrics
    
    def reset_metrics(self) -> None:
        """Сброс метрик"""
        self._metrics = {
            'total_models': 0,
            'active_models': 0,
            'predictions_made': 0,
            'training_jobs': 0,
            'errors_count': 0
        }
        
        # Сброс метрик компонентов
        if self.registry:
            self.registry.reset_metrics()
        
        if self.trainer:
            self.trainer.reset_metrics()
        
        if self.validator:
            self.validator.reset_metrics()
    
    def clear_cache(self) -> None:
        """Очистка кэша"""
        self._model_cache.clear()
        self.logger.info("Model cache cleared")
    
    def is_running(self) -> bool:
        """Проверка состояния работы"""
        return self._running
    
    def get_status(self) -> Dict[str, Any]:
        """
        Получение статуса менеджера
        
        Returns:
            Dict: Статус
        """
        return {
            'running': self._running,
            'uptime': (datetime.utcnow() - self._start_time).total_seconds() if self._start_time else 0,
            'active_models': len(self._active_models),
            'total_models': self._metrics['total_models'],
            'cache_size': len(self._model_cache),
            'components': {
                'registry': self.registry is not None,
                'trainer': self.trainer is not None,
                'validator': self.validator is not None
            },
            'metrics': self.get_metrics()
        }
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.stop()
