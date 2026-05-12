"""
Feature Scaling Module

Нормализация и масштабирование признаков.
"""

from .normalizer import FeatureNormalizer
from .scaler import FeatureScaler

__all__ = ['FeatureNormalizer', 'FeatureScaler']
