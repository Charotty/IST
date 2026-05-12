"""
Feature Selection Module

Отбор наиболее информативных признаков.
"""

from .importance import FeatureImportance
from .correlation_filter import CorrelationFilter

__all__ = ['FeatureImportance', 'CorrelationFilter']
