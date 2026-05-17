"""GUI API layer — typed access to orchestration without Qt dependencies."""

from gui.api.client import IstGuiClient
from gui.api.types import (
    BacktestRunSummary,
    BundleInfo,
    ChartBar,
    ChartPayload,
    DataHealth,
    ExecutionAccountSnapshot,
    ExecutionStepResult,
    ExplainSnapshot,
    FoldEquityPoint,
    JobEvent,
    OrderRecord,
    PipelineStepStatus,
    PositionRecord,
    RegimeBar,
    SymbolEntry,
)

__all__ = [
    "IstGuiClient",
    "SymbolEntry",
    "DataHealth",
    "BundleInfo",
    "ExplainSnapshot",
    "RegimeBar",
    "ChartBar",
    "ChartPayload",
    "FoldEquityPoint",
    "BacktestRunSummary",
    "JobEvent",
    "PipelineStepStatus",
    "ExecutionAccountSnapshot",
    "ExecutionStepResult",
    "OrderRecord",
    "PositionRecord",
]
