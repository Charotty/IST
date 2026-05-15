"""
Meta Evaluator

Evaluation metrics for meta-learning layer.

Financial Metrics:
- Profit Factor
- Sharpe Ratio
- Max Drawdown
- Trade Quality
- Turnover Reduction

ML Metrics:
- Calibration quality
- Confidence stability
- Regime specialization
- Ensemble consistency
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    financial_metrics: Dict[str, float]
    ml_metrics: Dict[str, float]
    regime_metrics: Dict[str, float]


class MetaEvaluator:
    """
    Evaluator for meta-learning layer performance.
    
    Evaluates both financial and ML metrics to assess meta-learning effectiveness.
    """
    
    def __init__(
        self,
        risk_free_rate: float = 0.0,
    ):
        """
        Initialize MetaEvaluator.
        
        Args:
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
        """
        self.risk_free_rate = risk_free_rate
    
    def evaluate(
        self,
        signals: np.ndarray,
        returns: np.ndarray,
        meta_prob: np.ndarray,
        regime_pred: Optional[np.ndarray] = None,
    ) -> EvaluationResult:
        """
        Evaluate meta-learning performance.
        
        Args:
            signals: Trading signals (-1, 0, 1)
            returns: Asset returns
            meta_prob: Meta probabilities
            regime_pred: Regime predictions (optional)
            
        Returns:
            EvaluationResult with all metrics
        """
        financial_metrics = self._calculate_financial_metrics(signals, returns)
        ml_metrics = self._calculate_ml_metrics(signals, meta_prob)
        
        if regime_pred is not None:
            regime_metrics = self._calculate_regime_metrics(signals, returns, regime_pred)
        else:
            regime_metrics = {}
        
        return EvaluationResult(
            financial_metrics=financial_metrics,
            ml_metrics=ml_metrics,
            regime_metrics=regime_metrics,
        )
    
    def _calculate_financial_metrics(
        self,
        signals: np.ndarray,
        returns: np.ndarray,
    ) -> Dict[str, float]:
        """
        Calculate financial metrics.
        
        Args:
            signals: Trading signals
            returns: Asset returns
            
        Returns:
            Dictionary of financial metrics
        """
        # Calculate strategy returns
        strategy_returns = signals * returns
        
        # Total return
        total_return = np.sum(strategy_returns)
        
        # Sharpe ratio
        if np.std(strategy_returns) > 0:
            sharpe_ratio = (np.mean(strategy_returns) - self.risk_free_rate) / np.std(strategy_returns)
        else:
            sharpe_ratio = 0.0
        
        # Max drawdown
        cumulative = np.cumsum(strategy_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        max_drawdown = np.min(drawdown)
        
        # Profit factor
        positive_returns = strategy_returns[strategy_returns > 0]
        negative_returns = strategy_returns[strategy_returns < 0]
        
        if len(negative_returns) > 0 and np.sum(np.abs(negative_returns)) > 0:
            profit_factor = np.sum(positive_returns) / np.sum(np.abs(negative_returns))
        else:
            profit_factor = float('inf') if len(positive_returns) > 0 else 0.0
        
        # Trade quality (average return per trade)
        trade_mask = signals != 0
        if np.sum(trade_mask) > 0:
            trade_quality = np.mean(strategy_returns[trade_mask])
        else:
            trade_quality = 0.0
        
        # Win rate
        if len(strategy_returns[trade_mask]) > 0:
            win_rate = np.sum(strategy_returns[trade_mask] > 0) / len(strategy_returns[trade_mask])
        else:
            win_rate = 0.0
        
        # Turnover (signal changes)
        signal_changes = np.sum(np.diff(signals) != 0)
        turnover = signal_changes / len(signals) if len(signals) > 0 else 0.0
        
        return {
            "total_return": float(total_return),
            "sharpe_ratio": float(sharpe_ratio),
            "max_drawdown": float(max_drawdown),
            "profit_factor": float(profit_factor),
            "trade_quality": float(trade_quality),
            "win_rate": float(win_rate),
            "turnover": float(turnover),
        }
    
    def _calculate_ml_metrics(
        self,
        signals: np.ndarray,
        meta_prob: np.ndarray,
    ) -> Dict[str, float]:
        """
        Calculate ML metrics.
        
        Args:
            signals: Trading signals
            meta_prob: Meta probabilities
            
        Returns:
            Dictionary of ML metrics
        """
        # Confidence stability (std of meta prob for trades)
        trade_mask = signals != 0
        if np.sum(trade_mask) > 0:
            confidence_stability = np.std(meta_prob[trade_mask])
        else:
            confidence_stability = 0.0
        
        # Average confidence for trades
        if np.sum(trade_mask) > 0:
            avg_confidence = np.mean(meta_prob[trade_mask])
        else:
            avg_confidence = 0.0
        
        # Signal rate (proportion of non-zero signals)
        signal_rate = np.sum(trade_mask) / len(signals) if len(signals) > 0 else 0.0
        
        # Calibration (simplified: compare avg confidence with win rate)
        if np.sum(trade_mask) > 0:
            # This is a simplified calibration metric
            # In practice, you'd compare predicted vs actual probabilities
            calibration_score = avg_confidence  # Placeholder
        else:
            calibration_score = 0.0
        
        return {
            "confidence_stability": float(confidence_stability),
            "avg_confidence": float(avg_confidence),
            "signal_rate": float(signal_rate),
            "calibration_score": float(calibration_score),
        }
    
    def _calculate_regime_metrics(
        self,
        signals: np.ndarray,
        returns: np.ndarray,
        regime_pred: np.ndarray,
    ) -> Dict[str, float]:
        """
        Calculate regime-specific metrics.
        
        Args:
            signals: Trading signals
            returns: Asset returns
            regime_pred: Regime predictions
            
        Returns:
            Dictionary of regime metrics
        """
        strategy_returns = signals * returns
        
        # Trend regime metrics
        trend_mask = regime_pred == 1
        if np.sum(trend_mask) > 0:
            trend_return = np.sum(strategy_returns[trend_mask])
            trend_signal_rate = np.sum(signals[trend_mask] != 0) / np.sum(trend_mask)
        else:
            trend_return = 0.0
            trend_signal_rate = 0.0
        
        # Range regime metrics
        range_mask = regime_pred == 0
        if np.sum(range_mask) > 0:
            range_return = np.sum(strategy_returns[range_mask])
            range_signal_rate = np.sum(signals[range_mask] != 0) / np.sum(range_mask)
        else:
            range_return = 0.0
            range_signal_rate = 0.0
        
        return {
            "trend_return": float(trend_return),
            "range_return": float(range_return),
            "trend_signal_rate": float(trend_signal_rate),
            "range_signal_rate": float(range_signal_rate),
        }
    
    def compare_ensembles(
        self,
        signals_dict: Dict[str, np.ndarray],
        returns: np.ndarray,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple ensemble methods.
        
        Args:
            signals_dict: Dictionary of signals from different methods
            returns: Asset returns
            
        Returns:
            Dictionary of metrics for each method
        """
        results = {}
        
        for method_name, signals in signals_dict.items():
            results[method_name] = self._calculate_financial_metrics(signals, returns)
        
        return results
