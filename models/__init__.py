"""
Models Layer — Adaptive Multi-Model Prediction Layer

The models module contains all predictive architectures used by the adaptive trading system.
Unlike traditional trading systems based on a single predictive model, this architecture uses
dynamic model selection based on market regime.

**Production / multi-model path:** prefer ``orchestration.InferenceOrchestrator`` (все модели →
meta-weighting → decision → risk). ``InferenceEngine`` + ``ModelRouter`` — legacy single-router
контур; не смешивайте оба входа в одном деплое без явного адаптера.

Экспорты верхнего уровня загружаются **лениво** (``__getattr__``), чтобы ``import models.tabular...``
не подтягивал TensorFlow / sklearn / все модели сразу — только при обращении к атрибуту.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

__all__ = [
    "RegimeDetector",
    "ModelRouter",
    "GRUTrendModel",
    "XGBoostMeanReversionModel",
    "CNNVolatilityModel",
    "LogisticMetaFilter",
    "PositionSizer",
    "ProbabilityCalibrator",
    "ModelEvaluator",
    "ModelRegistry",
    "ModelTrainer",
    "InferenceEngine",
    "save_model",
    "load_model",
    "calculate_feature_importance",
    "normalize_features",
    "calculate_rolling_metrics",
    "detect_drift",
    "create_feature_combinations",
    "validate_data",
]

_CLASS_EXPORTS = {
    "RegimeDetector": ("models.regime.regime_detector", "RegimeDetector"),
    "ModelRouter": ("models.router.model_router", "ModelRouter"),
    "GRUTrendModel": ("models.trend.gru_model", "GRUTrendModel"),
    "XGBoostMeanReversionModel": ("models.mean_reversion.xgboost_model", "XGBoostMeanReversionModel"),
    "CNNVolatilityModel": ("models.volatility.cnn_model", "CNNVolatilityModel"),
    "LogisticMetaFilter": ("models.meta.logistic_filter", "LogisticMetaFilter"),
    "ProbabilityCalibrator": ("models.calibration.probability_calibrator", "ProbabilityCalibrator"),
    "ModelEvaluator": ("models.evaluation.model_evaluator", "ModelEvaluator"),
    "ModelRegistry": ("models.registry.model_registry", "ModelRegistry"),
    "ModelTrainer": ("models.training.trainer", "ModelTrainer"),
    "InferenceEngine": ("models.inference.inference_engine", "InferenceEngine"),
}

_UTILS = frozenset(
    {
        "save_model",
        "load_model",
        "calculate_feature_importance",
        "normalize_features",
        "calculate_rolling_metrics",
        "detect_drift",
        "create_feature_combinations",
        "validate_data",
    }
)


def __getattr__(name: str):
    if name == "PositionSizer":
        from models.sizing.position_sizer import ConfidencePositionSizer

        return ConfidencePositionSizer
    if name in _CLASS_EXPORTS:
        mod_name, attr = _CLASS_EXPORTS[name]
        mod = importlib.import_module(mod_name)
        return getattr(mod, attr)
    if name in _UTILS:
        helpers = importlib.import_module("models.utils.helpers")
        return getattr(helpers, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:
    from models.calibration.probability_calibrator import ProbabilityCalibrator
    from models.evaluation.model_evaluator import ModelEvaluator
    from models.inference.inference_engine import InferenceEngine
    from models.meta.logistic_filter import LogisticMetaFilter
    from models.mean_reversion.xgboost_model import XGBoostMeanReversionModel
    from models.registry.model_registry import ModelRegistry
    from models.regime.regime_detector import RegimeDetector
    from models.router.model_router import ModelRouter
    from models.training.trainer import ModelTrainer
    from models.trend.gru_model import GRUTrendModel
    from models.volatility.cnn_model import CNNVolatilityModel
