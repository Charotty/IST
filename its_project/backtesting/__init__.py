from .base import BaseBacktester, Trade, BacktestResult
from .simple import SimpleBacktester
try:
    from .performance import PerformanceAnalyzer
except ImportError:
    PerformanceAnalyzer = None
try:
    from .walkforward import WalkForwardValidator, MultiAssetWalkForward
except ImportError:
    WalkForwardValidator = None
    MultiAssetWalkForward = None

__all__ = [
    "BaseBacktester",
    "Trade",
    "BacktestResult",
    "SimpleBacktester",
]

if PerformanceAnalyzer is not None:
    __all__.append("PerformanceAnalyzer")
if WalkForwardValidator is not None:
    __all__.extend(["WalkForwardValidator", "MultiAssetWalkForward"])