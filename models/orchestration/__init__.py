"""Backward-compat: фабрика перенесена в ``orchestration.model_factory``."""

from orchestration.model_factory import (
    build_orchestration_models,
    infer_training_feature_columns,
    meta_weighting_from_config,
)

__all__ = [
    "build_orchestration_models",
    "infer_training_feature_columns",
    "meta_weighting_from_config",
]
