"""
Models Layer — Adaptive Multi-Model Prediction Layer

The models module contains all predictive architectures used by the adaptive trading system.
Unlike traditional trading systems based on a single predictive model, this architecture uses
dynamic model selection based on market regime.
"""

# Regime Detection
from .regime import RegimeDetector

# Model Router
from .router import ModelRouter

# Specialized Models
from .trend import GRUTrendModel
from .mean_reversion import XGBoostMeanReversionModel
from .volatility import CNNVolatilityModel

# Meta Filter
from .meta import LogisticMetaFilter

# Position Sizing
from .sizing import PositionSizer

# Calibration
from .calibration import ProbabilityCalibrator

# Evaluation
from .evaluation import ModelEvaluator

# Registry
from .registry import ModelRegistry

# Training
from .training import ModelTrainer

# Inference
from .inference import InferenceEngine

# Utilities
from .utils import (
    save_model,
    load_model,
    calculate_feature_importance,
    normalize_features,
    calculate_rolling_metrics,
    detect_drift,
    create_feature_combinations,
    validate_data
)

__all__ = [
    # Regime Detection
    'RegimeDetector',
    
    # Model Router
    'ModelRouter',
    
    # Specialized Models
    'GRUTrendModel',
    'XGBoostMeanReversionModel',
    'CNNVolatilityModel',
    
    # Meta Filter
    'LogisticMetaFilter',
    
    # Position Sizing
    'PositionSizer',
    
    # Calibration
    'ProbabilityCalibrator',
    
    # Evaluation
    'ModelEvaluator',
    
    # Registry
    'ModelRegistry',
    
    # Training
    'ModelTrainer',
    
    # Inference
    'InferenceEngine',
    
    # Utilities
    'save_model',
    'load_model',
    'calculate_feature_importance',
    'normalize_features',
    'calculate_rolling_metrics',
    'detect_drift',
    'create_feature_combinations',
    'validate_data',
]
