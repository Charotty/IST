"""
Model Registry

Реестр моделей для управления версиями и метаданными.
"""

import asyncio
import logging
import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import hashlib
import joblib
import pickle


class ModelRegistry:
    """Реестр моделей"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Инициализация реестра моделей
        
        Args:
            config: Конфигурация реестра
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Параметры
        self.registry_path = config.get('registry_path', './models_registry')
        self.max_versions = config.get('max_versions', 10)
        self.auto_cleanup = config.get('auto_cleanup', True)
        
        # Структура реестра
        self._registry = {}
        self._models = {}
        self._versions = {}
        
        # Метрики
        self._metrics = {
            'models_registered': 0,
            'versions_created': 0,
            'models_loaded': 0,
            'models_saved': 0,
            'cleanup_count': 0
        }
        
        # Инициализация
        self._initialize_registry()
    
    def _initialize_registry(self) -> None:
        """Инициализация реестра"""
        try:
            # Создание директории реестра
            os.makedirs(self.registry_path, exist_ok=True)
            
            # Загрузка существующего реестра
            registry_file = os.path.join(self.registry_path, 'registry.json')
            if os.path.exists(registry_file):
                with open(registry_file, 'r') as f:
                    self._registry = json.load(f)
                
                self.logger.info(f"Registry loaded from {registry_file}")
            else:
                self._registry = {
                    'models': {},
                    'versions': {},
                    'created_at': datetime.utcnow().isoformat(),
                    'updated_at': datetime.utcnow().isoformat()
                }
                self._save_registry()
            
            self._models = self._registry.get('models', {})
            self._versions = self._registry.get('versions', {})
            
        except Exception as e:
            self.logger.error(f"Error initializing registry: {e}")
            raise
    
    def _save_registry(self) -> None:
        """Сохранение реестра"""
        try:
            self._registry['updated_at'] = datetime.utcnow().isoformat()
            
            registry_file = os.path.join(self.registry_path, 'registry.json')
            with open(registry_file, 'w') as f:
                json.dump(self._registry, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving registry: {e}")
    
    def _generate_model_id(self, model_name: str, config: Dict[str, Any]) -> str:
        """
        Генерация ID модели
        
        Args:
            model_name: Имя модели
            config: Конфигурация модели
            
        Returns:
            str: ID модели
        """
        # Создание хэша из имени и конфигурации
        config_str = json.dumps(config, sort_keys=True)
        hash_input = f"{model_name}_{config_str}"
        model_id = hashlib.md5(hash_input.encode()).hexdigest()[:12]
        
        return f"{model_name}_{model_id}"
    
    def _generate_version_id(self, model_id: str, version: str) -> str:
        """
        Генерация ID версии
        
        Args:
            model_id: ID модели
            version: Версия
            
        Returns:
            str: ID версии
        """
        return f"{model_id}_v{version}"
    
    async def register_model(self, model_name: str, model: Any, 
                          config: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> str:
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
            # Генерация ID модели
            model_id = self._generate_model_id(model_name, config)
            
            # Проверка существования модели
            if model_id in self._models:
                self.logger.warning(f"Model {model_id} already exists, updating...")
            
            # Метаданные модели
            model_metadata = {
                'model_id': model_id,
                'model_name': model_name,
                'config': config,
                'metadata': metadata or {},
                'created_at': datetime.utcnow().isoformat(),
                'updated_at': datetime.utcnow().isoformat(),
                'status': 'active',
                'versions': []
            }
            
            # Сохранение модели
            model_path = os.path.join(self.registry_path, f"{model_id}.pkl")
            joblib.dump(model, model_path)
            
            # Обновление реестра
            self._models[model_id] = model_metadata
            self._registry['models'][model_id] = model_metadata
            
            # Сохранение реестра
            self._save_registry()
            
            self._update_metrics('models_registered', 1)
            self.logger.info(f"Model {model_id} registered successfully")
            
            return model_id
            
        except Exception as e:
            self.logger.error(f"Error registering model: {e}")
            raise
    
    async def create_version(self, model_id: str, model: Any, 
                           version: str, changelog: Optional[str] = None,
                           performance_metrics: Optional[Dict[str, float]] = None) -> str:
        """
        Создание версии модели
        
        Args:
            model_id: ID модели
            model: Объект модели
            version: Версия
            changelog: Описание изменений
            performance_metrics: Метрики производительности
            
        Returns:
            str: ID версии
        """
        try:
            # Проверка существования модели
            if model_id not in self._models:
                raise ValueError(f"Model {model_id} not found")
            
            # Генерация ID версии
            version_id = self._generate_version_id(model_id, version)
            
            # Метаданные версии
            version_metadata = {
                'version_id': version_id,
                'model_id': model_id,
                'version': version,
                'changelog': changelog or f"Version {version}",
                'performance_metrics': performance_metrics or {},
                'created_at': datetime.utcnow().isoformat(),
                'status': 'active'
            }
            
            # Сохранение версии
            version_path = os.path.join(self.registry_path, f"{version_id}.pkl")
            joblib.dump(model, version_path)
            
            # Обновление реестра
            if model_id not in self._versions:
                self._versions[model_id] = []
            
            self._versions[model_id].append(version_metadata)
            self._registry['versions'][model_id] = self._versions[model_id]
            
            # Обновление метаданных модели
            self._models[model_id]['versions'].append(version_id)
            self._models[model_id]['updated_at'] = datetime.utcnow().isoformat()
            self._registry['models'][model_id] = self._models[model_id]
            
            # Сохранение реестра
            self._save_registry()
            
            # Cleanup старых версий
            if self.auto_cleanup:
                await self._cleanup_old_versions(model_id)
            
            self._update_metrics('versions_created', 1)
            self.logger.info(f"Version {version_id} created successfully")
            
            return version_id
            
        except Exception as e:
            self.logger.error(f"Error creating version: {e}")
            raise
    
    async def _cleanup_old_versions(self, model_id: str) -> None:
        """
        Очистка старых версий модели
        
        Args:
            model_id: ID модели
        """
        try:
            if model_id not in self._versions:
                return
            
            versions = self._versions[model_id]
            
            if len(versions) <= self.max_versions:
                return
            
            # Сортировка версий по времени создания
            versions.sort(key=lambda x: x['created_at'], reverse=True)
            
            # Удаление самых старых версий
            versions_to_remove = versions[self.max_versions:]
            
            for version in versions_to_remove:
                version_id = version['version_id']
                
                # Удаление файла версии
                version_path = os.path.join(self.registry_path, f"{version_id}.pkl")
                if os.path.exists(version_path):
                    os.remove(version_path)
                    self.logger.info(f"Removed old version {version_id}")
                
                # Удаление из реестра
                self._versions[model_id] = [v for v in self._versions[model_id] 
                                        if v['version_id'] != version_id]
            
            # Обновление реестра
            self._registry['versions'][model_id] = self._versions[model_id]
            self._models[model_id]['versions'] = [v['version_id'] for v in self._versions[model_id]]
            
            self._save_registry()
            self._update_metrics('cleanup_count', len(versions_to_remove))
            
        except Exception as e:
            self.logger.error(f"Error cleaning up old versions: {e}")
    
    async def load_model(self, model_id: str, version: Optional[str] = None) -> Any:
        """
        Загрузка модели
        
        Args:
            model_id: ID модели
            version: Версия (если None, последняя)
            
        Returns:
            Any: Загруженная модель
        """
        try:
            # Поиск модели в реестре
            if model_id not in self._models:
                raise ValueError(f"Model {model_id} not found")
            
            # Определение версии для загрузки
            if version:
                version_id = self._generate_version_id(model_id, version)
                version_path = os.path.join(self.registry_path, f"{version_id}.pkl")
            else:
                # Загрузка последней версии
                model_path = os.path.join(self.registry_path, f"{model_id}.pkl")
                version_path = model_path
            
            # Проверка существования файла
            if not os.path.exists(version_path):
                raise FileNotFoundError(f"Model file not found: {version_path}")
            
            # Загрузка модели
            model = joblib.load(version_path)
            
            self._update_metrics('models_loaded', 1)
            self.logger.info(f"Model {model_id} loaded successfully")
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
    
    async def save_model(self, model_id: str, model: Any, version: Optional[str] = None,
                       changelog: Optional[str] = None,
                       performance_metrics: Optional[Dict[str, float]] = None) -> str:
        """
        Сохранение модели
        
        Args:
            model_id: ID модели
            model: Объект модели
            version: Версия
            changelog: Описание изменений
            performance_metrics: Метрики производительности
            
        Returns:
            str: ID версии
        """
        try:
            # Проверка существования модели
            if model_id not in self._models:
                raise ValueError(f"Model {model_id} not found")
            
            # Генерация версии если не указана
            if not version:
                # Автоинкремент версии
                existing_versions = self._models[model_id].get('versions', [])
                if existing_versions:
                    last_version = existing_versions[-1]
                    version_num = int(last_version.split('_v')[1]) + 1
                    version = str(version_num)
                else:
                    version = "1"
            
            # Создание новой версии
            version_id = await self.create_version(
                model_id, model, version, changelog, performance_metrics
            )
            
            self._update_metrics('models_saved', 1)
            
            return version_id
            
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            raise
    
    async def list_models(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Список моделей
        
        Args:
            status: Фильтр по статусу
            
        Returns:
            List: Список моделей
        """
        models = []
        
        for model_id, metadata in self._models.items():
            if status is None or metadata.get('status') == status:
                models.append(metadata)
        
        return models
    
    async def list_versions(self, model_id: str) -> List[Dict[str, Any]]:
        """
        Список версий модели
        
        Args:
            model_id: ID модели
            
        Returns:
            List: Список версий
        """
        if model_id not in self._versions:
            return []
        
        return self._versions[model_id].copy()
    
    async def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Получение информации о модели
        
        Args:
            model_id: ID модели
            
        Returns:
            Dict: Информация о модели
        """
        return self._models.get(model_id)
    
    async def get_version_info(self, version_id: str) -> Optional[Dict[str, Any]]:
        """
        Получение информации о версии
        
        Args:
            version_id: ID версии
            
        Returns:
            Dict: Информация о версии
        """
        for model_versions in self._versions.values():
            for version in model_versions:
                if version['version_id'] == version_id:
                    return version
        
        return None
    
    async def delete_model(self, model_id: str) -> None:
        """
        Удаление модели
        
        Args:
            model_id: ID модели
        """
        try:
            # Удаление файла модели
            model_path = os.path.join(self.registry_path, f"{model_id}.pkl")
            if os.path.exists(model_path):
                os.remove(model_path)
            
            # Удаление версий
            if model_id in self._versions:
                for version in self._versions[model_id]:
                    version_path = os.path.join(self.registry_path, f"{version['version_id']}.pkl")
                    if os.path.exists(version_path):
                        os.remove(version_path)
                
                del self._versions[model_id]
            
            # Удаление из реестра
            if model_id in self._models:
                del self._models[model_id]
            
            # Сохранение реестра
            self._registry['models'] = self._models
            self._registry['versions'] = self._versions
            self._save_registry()
            
            self.logger.info(f"Model {model_id} deleted successfully")
            
        except Exception as e:
            self.logger.error(f"Error deleting model: {e}")
            raise
    
    async def update_model_status(self, model_id: str, status: str) -> None:
        """
        Обновление статуса модели
        
        Args:
            model_id: ID модели
            status: Новый статус
        """
        try:
            if model_id not in self._models:
                raise ValueError(f"Model {model_id} not found")
            
            self._models[model_id]['status'] = status
            self._models[model_id]['updated_at'] = datetime.utcnow().isoformat()
            
            # Сохранение реестра
            self._registry['models'][model_id] = self._models[model_id]
            self._save_registry()
            
            self.logger.info(f"Model {model_id} status updated to {status}")
            
        except Exception as e:
            self.logger.error(f"Error updating model status: {e}")
            raise
    
    def _update_metrics(self, metric_name: str, value: int) -> None:
        """
        Обновление метрик
        
        Args:
            metric_name: Имя метрики
            value: Значение
        """
        self._metrics[metric_name] += value
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Получение метрик
        
        Returns:
            Dict: Метрики реестра
        """
        metrics = self._metrics.copy()
        
        # Дополнительные метрики
        metrics['total_models'] = len(self._models)
        metrics['total_versions'] = sum(len(versions) for versions in self._versions.values())
        metrics['registry_size'] = self._get_registry_size()
        
        return metrics
    
    def _get_registry_size(self) -> int:
        """Получение размера реестра в байтах"""
        try:
            total_size = 0
            
            for root, dirs, files in os.walk(self.registry_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    if os.path.exists(file_path):
                        total_size += os.path.getsize(file_path)
            
            return total_size
            
        except Exception:
            return 0
    
    def reset_metrics(self) -> None:
        """Сброс метрик"""
        self._metrics = {
            'models_registered': 0,
            'versions_created': 0,
            'models_loaded': 0,
            'models_saved': 0,
            'cleanup_count': 0
        }
