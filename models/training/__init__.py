"""
Training Module

Утилиты для обучения и валидации моделей.
"""

from .trainer import ModelTrainer
from .validator import ModelValidator

__all__ = ['ModelTrainer', 'ModelValidator']
