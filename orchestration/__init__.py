"""
Orchestration Layer

Unified orchestration for training and inference.
Provides explicit contract: features → regime → all model predictions → meta weighting → decision → risk
"""

from .training_orchestrator import TrainingOrchestrator
from .inference_orchestrator import InferenceOrchestrator
from .orchestrator_config import OrchestratorConfig
from .config_validate import validate_pipeline_config, raise_if_invalid
from .artifact_bundle import (
    save_orchestrator_bundle,
    load_orchestrator_bundle,
    validate_bundle_feature_schema,
)

__all__ = [
    "TrainingOrchestrator",
    "InferenceOrchestrator",
    "OrchestratorConfig",
    "validate_pipeline_config",
    "raise_if_invalid",
    "save_orchestrator_bundle",
    "load_orchestrator_bundle",
    "validate_bundle_feature_schema",
]
