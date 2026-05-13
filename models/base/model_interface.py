"""
Model Interface

Интерфейс для всех моделей машинного обучения.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Tuple
import pandas as pd
import numpy as np


class ModelInterface(ABC):
    """Интерфейс для всех моделей"""
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> None:
        """
        Обучение модели
        
        Args:
            X: Признаки
            y: Целевая переменная
            **kwargs: Дополнительные параметры
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Предсказание
        
        Args:
            X: Признаки
            
        Returns:
            np.ndarray: Предсказания
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Предсказание вероятностей
        
        Args:
            X: Признаки
            
        Returns:
            np.ndarray: Вероятности
        """
        pass
    
    @abstractmethod
    def save_model(self, path: str) -> None:
        """
        Сохранение модели
        
        Args:
            path: Путь для сохранения
        """
        pass
    
    @abstractmethod
    def load_model(self, path: str) -> None:
        """
        Загрузка модели
        
        Args:
            path: Путь к модели
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Получение информации о модели
        
        Returns:
            Dict: Информация о модели
        """
        pass
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Получение важности признаков
        
        Returns:
            Dict: Важность признаков или None
        """
        return None
    
    def validate_input(self, X: pd.DataFrame) -> bool:
        """
        Валидация входных данных
        
        Args:
            X: Входные данные
            
        Returns:
            bool: True если данные валидны
        """
        if X is None or X.empty:
            return False
        return True
    
    def get_prediction_type(self) -> str:
        """
        Получение типа предсказания
        
        Returns:
            str: 'classification', 'regression', 'probability'
        """
        return 'classification'
    
    def get_supported_tasks(self) -> List[str]:
        """
        Получение поддерживаемых задач
        
        Returns:
            List[str]: Список поддерживаемых задач
        """
        return ['classification', 'regression']
