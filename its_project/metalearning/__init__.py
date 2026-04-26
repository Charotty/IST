from .metrics import MLMetrics, TradingMetrics
from .cv import TimeSeriesSplitter, WalkForwardValidator
from .selector import ModelSelector
from .stacking import StackingEnsemble

try:
    from .hyperopt import HyperparameterOptimizer
    _has_hyperopt = True
except ImportError:
    _has_hyperopt = False
    HyperparameterOptimizer = None

__all__ = [
    "MLMetrics",
    "TradingMetrics",
    "TimeSeriesSplitter",
    "WalkForwardValidator",
    "ModelSelector",
    "StackingEnsemble",
]

if _has_hyperopt:
    __all__.append("HyperparameterOptimizer")