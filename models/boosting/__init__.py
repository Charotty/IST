"""
Boosting Models Module

Модели градиентного бустинга.
"""

from .lightgbm_model import LightGBMModel
from .xgboost_model import XGBoostModel

__all__ = ['LightGBMModel', 'XGBoostModel']
