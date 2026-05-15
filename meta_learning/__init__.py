"""
Meta-Learning Layer

Adaptive orchestration and signal aggregation layer of the trading system.

This module is responsible for:
- ensemble aggregation
- adaptive model weighting
- regime-aware signal routing
- confidence filtering
- final trade approval
"""

from .signal_assembler import SignalAssembler
from .ensemble import EnsembleAggregator
from .dynamic_meta import DynamicMetaWeighting
from .thresholds import ThresholdManager

__all__ = [
    'SignalAssembler',
    'EnsembleAggregator',
    'DynamicMetaWeighting',
    'ThresholdManager',
]
