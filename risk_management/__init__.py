"""
Risk Management Module

Управление размером позиции и выходами: ATR-based sizing, ATR trailing stop,
координация с RL risk multiplier.
"""

from .position_sizer import PositionSizer
from .atr_trailing_stop import apply_atr_trailing_stop
from .risk_pipeline import RiskPipeline
from .orchestrator_risk_bridge import OrchestratorRiskBridge

__all__ = [
    'PositionSizer',
    'apply_atr_trailing_stop',
    'RiskPipeline',
    'OrchestratorRiskBridge',
]
