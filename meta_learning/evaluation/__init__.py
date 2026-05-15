"""
Evaluation Module

Evaluation metrics and analysis for meta-learning layer.

Focus areas:
- Financial metrics (Profit Factor, Sharpe Ratio, Max Drawdown, Trade Quality, Turnover Reduction)
- ML metrics (calibration quality, confidence stability, regime specialization, ensemble consistency)
"""

from .meta_evaluator import MetaEvaluator
from .regime_analyzer import RegimeAnalyzer

__all__ = ['MetaEvaluator', 'RegimeAnalyzer']
