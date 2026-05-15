"""
Dynamic Meta Weighting

Dynamic regime-aware ensemble weighting.

This is the CORE adaptive orchestration layer of the system.
The module dynamically changes model importance depending on current market regime.

Regime Routing:
- regime_pred = 1: Trend / strong directional market
- regime_pred = 0: Range / weak directional market

Current Adaptive Weights:

Trend Regime:
{
    'lgb': 0.10,
    'lstm': 0.45,
    'cnn': 0.10,
    'trans': 0.35
}
Priority: LSTM, Transformer (sequential models perform better during momentum persistence)

Range Regime:
{
    'lgb': 0.55,
    'lstm': 0.10,
    'cnn': 0.25,
    'trans': 0.10
}
Priority: LightGBM, CNN (tabular and local-pattern models perform better during ranging)
"""

import numpy as np
from typing import Dict, Optional, List


class DynamicMetaWeighting:
    """
    Dynamic regime-aware ensemble weighting.
    
    Dynamically changes model importance depending on current market regime.
    """
    
    def __init__(
        self,
        trend_weights: Optional[Dict[str, float]] = None,
        range_weights: Optional[Dict[str, float]] = None,
        default_mode: str = "regime_adaptive",
    ):
        """
        Initialize DynamicMetaWeighting.
        
        Args:
            trend_weights: Weights for trend regime (default: LSTM/Transformer dominant)
            range_weights: Weights for range regime (default: LightGBM/CNN dominant)
            default_mode: Default ensemble mode ('regime_adaptive', 'fixed_trend', 'fixed_range')
        """
        # Default trend weights: LSTM + Transformer dominant
        self.trend_weights = trend_weights or {
            'lgb': 0.10,
            'lstm': 0.45,
            'cnn': 0.10,
            'trans': 0.35
        }
        
        # Default range weights: LightGBM + CNN dominant
        self.range_weights = range_weights or {
            'lgb': 0.55,
            'lstm': 0.10,
            'cnn': 0.25,
            'trans': 0.10
        }
        
        self.default_mode = default_mode
        self.models = list(self.trend_weights.keys())
    
    def get_weights(
        self,
        regime_pred: np.ndarray,
        mode: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Get dynamic weights based on regime predictions.
        
        Args:
            regime_pred: Regime predictions (1=trend, 0=range)
            mode: Ensemble mode (uses default_mode if None)
            
        Returns:
            Dictionary of weight arrays for each model
        """
        if mode is None:
            mode = self.default_mode
        
        if mode == "regime_adaptive":
            return self._regime_adaptive_weights(regime_pred)
        elif mode == "fixed_trend":
            return self._fixed_weights(self.trend_weights, regime_pred.shape)
        elif mode == "fixed_range":
            return self._fixed_weights(self.range_weights, regime_pred.shape)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def _regime_adaptive_weights(
        self,
        regime_pred: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Get regime-adaptive weights.
        
        Args:
            regime_pred: Regime predictions (1=trend, 0=range)
            
        Returns:
            Dictionary of weight arrays for each model
        """
        weights_dict = {}
        
        for model in self.models:
            # Initialize weight array
            weights = np.zeros_like(regime_pred, dtype=float)
            
            # Apply trend weights where regime_pred == 1
            weights[regime_pred == 1] = self.trend_weights[model]
            
            # Apply range weights where regime_pred == 0
            weights[regime_pred == 0] = self.range_weights[model]
            
            weights_dict[model] = weights
        
        return weights_dict
    
    def _fixed_weights(
        self,
        base_weights: Dict[str, float],
        shape: tuple,
    ) -> Dict[str, np.ndarray]:
        """
        Get fixed weights (same for all samples).
        
        Args:
            base_weights: Base weight dictionary
            shape: Shape of output arrays
            
        Returns:
            Dictionary of weight arrays for each model
        """
        weights_dict = {}
        
        for model in self.models:
            weights_dict[model] = np.full(shape, base_weights[model])
        
        return weights_dict
    
    def apply_dynamic_weighting(
        self,
        predictions: Dict[str, np.ndarray],
        regime_pred: np.ndarray,
        mode: Optional[str] = None,
    ) -> np.ndarray:
        """
        Apply dynamic weighting to predictions.
        
        Args:
            predictions: Dictionary of model predictions {model_name: probability_array}
            regime_pred: Regime predictions (1=trend, 0=range)
            mode: Ensemble mode (uses default_mode if None)
            
        Returns:
            Weighted ensemble probability
        """
        # Get dynamic weights
        weights_dict = self.get_weights(regime_pred, mode)
        
        # Apply weights
        ensemble_prob = np.zeros_like(next(iter(predictions.values())))
        
        for model in self.models:
            if model in predictions:
                ensemble_prob += weights_dict[model] * predictions[model]
        
        return ensemble_prob
    
    def get_integrated_signal(
        self,
        predictions: Dict[str, np.ndarray],
        regime_pred: np.ndarray,
        direction_soft_signal: np.ndarray,
        meta_threshold: float,
        mode: Optional[str] = None,
    ) -> np.ndarray:
        """
        Get integrated signal with meta filtering.
        
        Current logic:
        integrated_signal = np.where(
            (meta_mgmt_prob > integrated_threshold)
            & (direction_soft_signal != 0),
            direction_soft_signal,
            0
        )
        
        Args:
            predictions: Dictionary of model predictions
            regime_pred: Regime predictions
            direction_soft_signal: Directional soft signal
            meta_threshold: Meta probability threshold
            mode: Ensemble mode
            
        Returns:
            Integrated trading signal
        """
        # Apply dynamic weighting
        meta_mgmt_prob = self.apply_dynamic_weighting(predictions, regime_pred, mode)
        
        # Apply threshold filtering
        integrated_signal = np.where(
            (meta_mgmt_prob > meta_threshold)
            & (direction_soft_signal != 0),
            direction_soft_signal,
            0
        )
        
        return integrated_signal
    
    def get_regime_statistics(
        self,
        regime_pred: np.ndarray,
    ) -> Dict[str, float]:
        """
        Get statistics about regime predictions.
        
        Args:
            regime_pred: Regime predictions
            
        Returns:
            Dictionary of statistics
        """
        total = len(regime_pred)
        trend_count = np.sum(regime_pred == 1)
        range_count = np.sum(regime_pred == 0)
        
        return {
            "total_samples": total,
            "trend_count": int(trend_count),
            "range_count": int(range_count),
            "trend_percentage": float(trend_count / total) if total > 0 else 0.0,
            "range_percentage": float(range_count / total) if total > 0 else 0.0,
        }
    
    def update_weights(
        self,
        trend_weights: Optional[Dict[str, float]] = None,
        range_weights: Optional[Dict[str, float]] = None,
    ):
        """
        Update regime weights.
        
        Args:
            trend_weights: New trend weights
            range_weights: New range weights
        """
        if trend_weights is not None:
            # Normalize weights
            total = sum(trend_weights.values())
            self.trend_weights = {k: v / total for k, v in trend_weights.items()}
        
        if range_weights is not None:
            # Normalize weights
            total = sum(range_weights.values())
            self.range_weights = {k: v / total for k, v in range_weights.items()}
