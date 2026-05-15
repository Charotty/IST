"""
Decision Layer for ITS

This layer provides thin rule-based assembly of trading decisions from outputs of models + meta_learning.
It serves as the final rules and entry point for risk management and backtesting.

Components:
- Signal rules: final_signal (MetaFilter + MTF) and integrated_signal (Regime-Adaptive Ensemble)
- Decision pipeline: Selection of signal variant based on configuration
"""

from .signal_rules import compute_final_signal, compute_integrated_signal
from .decision_pipeline import DecisionPipeline

__all__ = [
    'compute_final_signal',
    'compute_integrated_signal',
    'DecisionPipeline',
]
