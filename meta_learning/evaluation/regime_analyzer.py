"""
Regime Analyzer

Analysis of regime-specific performance and model specialization.
"""

import numpy as np
from typing import Dict, Optional, List, Tuple


class RegimeAnalyzer:
    """
    Analyzer for regime-specific performance.
    
    Analyzes how models and signals perform under different market regimes.
    """
    
    def __init__(self):
        """Initialize RegimeAnalyzer."""
        pass
    
    def analyze_regime_performance(
        self,
        signals: np.ndarray,
        returns: np.ndarray,
        regime_pred: np.ndarray,
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze performance by regime.
        
        Args:
            signals: Trading signals
            returns: Asset returns
            regime_pred: Regime predictions (1=trend, 0=range)
            
        Returns:
            Dictionary of metrics per regime
        """
        strategy_returns = signals * returns
        
        results = {}
        
        # Trend regime
        trend_mask = regime_pred == 1
        if np.sum(trend_mask) > 0:
            results['trend'] = self._calculate_regime_metrics(
                strategy_returns[trend_mask],
                signals[trend_mask],
            )
        else:
            results['trend'] = {}
        
        # Range regime
        range_mask = regime_pred == 0
        if np.sum(range_mask) > 0:
            results['range'] = self._calculate_regime_metrics(
                strategy_returns[range_mask],
                signals[range_mask],
            )
        else:
            results['range'] = {}
        
        return results
    
    def _calculate_regime_metrics(
        self,
        returns: np.ndarray,
        signals: np.ndarray,
    ) -> Dict[str, float]:
        """
        Calculate metrics for a specific regime.
        
        Args:
            returns: Strategy returns for this regime
            signals: Trading signals for this regime
            
        Returns:
            Dictionary of metrics
        """
        total_return = np.sum(returns)
        
        if len(returns) > 0:
            sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0.0
        else:
            sharpe_ratio = 0.0
        
        signal_count = np.sum(signals != 0)
        signal_rate = signal_count / len(signals) if len(signals) > 0 else 0.0
        
        win_rate = np.sum(returns > 0) / len(returns) if len(returns) > 0 else 0.0
        
        return {
            "total_return": float(total_return),
            "sharpe_ratio": float(sharpe_ratio),
            "signal_count": int(signal_count),
            "signal_rate": float(signal_rate),
            "win_rate": float(win_rate),
        }
    
    def analyze_model_specialization(
        self,
        model_predictions: Dict[str, np.ndarray],
        regime_pred: np.ndarray,
        true_returns: np.ndarray,
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze model specialization by regime.
        
        Args:
            model_predictions: Dictionary of model predictions
            regime_pred: Regime predictions
            true_returns: True returns
            
        Returns:
            Dictionary of specialization metrics per model
        """
        results = {}
        
        for model_name, predictions in model_predictions.items():
            # Convert predictions to signals (simple threshold)
            signals = np.where(predictions > 0.5, 1, np.where(predictions < 0.5, -1, 0))
            
            # Calculate regime-specific performance
            regime_performance = self.analyze_regime_performance(
                signals,
                true_returns,
                regime_pred,
            )
            
            results[model_name] = regime_performance
        
        return results
    
    def get_regime_transitions(
        self,
        regime_pred: np.ndarray,
    ) -> Dict[str, float]:
        """
        Analyze regime transitions.
        
        Args:
            regime_pred: Regime predictions
            
        Returns:
            Dictionary of transition metrics
        """
        transitions = np.sum(regime_pred[1:] != regime_pred[:-1])
        total_transitions = transitions / len(regime_pred) if len(regime_pred) > 0 else 0.0
        
        # Count regime durations
        regime_changes = np.where(regime_pred[1:] != regime_pred[:-1])[0]
        if len(regime_changes) > 0:
            durations = np.diff(np.concatenate([[-1], regime_changes, [len(regime_pred) - 1]]))
            avg_duration = np.mean(durations)
        else:
            avg_duration = len(regime_pred)
        
        return {
            "transition_rate": float(total_transitions),
            "avg_regime_duration": float(avg_duration),
            "total_transitions": int(transitions),
        }
