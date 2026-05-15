"""
Ensemble Aggregator

Model probability aggregation.

Provides baseline ensemble logic.

Current ensemble:
ensemble_prob = (lgb_p + lstm_p + cnn_p + trans_p) / 4

Models used:
- LightGBM: Tabular / range logic
- LSTM: Sequential trend modeling
- CNN: Volatility structure
- Transformer: Long-range temporal context

Purpose of baseline ensemble:
- benchmarking
- ensemble comparison
- fallback aggregation
"""

import numpy as np
from typing import Dict, Optional, List


class EnsembleAggregator:
    """
    Baseline ensemble aggregator for model probability aggregation.
    
    Provides simple averaging and weighted averaging methods for combining
    predictions from multiple models.
    """
    
    def __init__(
        self,
        models: Optional[List[str]] = None,
        weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize EnsembleAggregator.
        
        Args:
            models: List of model names (default: ['lgb', 'lstm', 'cnn', 'trans'])
            weights: Optional weights for each model (default: equal weights)
        """
        self.models = models or ['lgb', 'lstm', 'cnn', 'trans']
        
        if weights is None:
            # Equal weights by default
            self.weights = {model: 1.0 / len(self.models) for model in self.models}
        else:
            # Normalize weights to sum to 1
            total = sum(weights.values())
            self.weights = {k: v / total for k, v in weights.items()}
    
    def simple_average(
        self,
        predictions: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        Simple average ensemble.
        
        ensemble_prob = (lgb_p + lstm_p + cnn_p + trans_p) / 4
        
        Args:
            predictions: Dictionary of model predictions {model_name: probability_array}
            
        Returns:
            Ensemble probability array
        """
        ensemble_prob = np.zeros_like(next(iter(predictions.values())))
        
        for model in self.models:
            if model in predictions:
                ensemble_prob += predictions[model]
        
        ensemble_prob /= len(self.models)
        
        return ensemble_prob
    
    def weighted_average(
        self,
        predictions: Dict[str, np.ndarray],
        weights: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """
        Weighted average ensemble.
        
        Args:
            predictions: Dictionary of model predictions {model_name: probability_array}
            weights: Optional weights for each model (uses instance weights if None)
            
        Returns:
            Ensemble probability array
        """
        if weights is None:
            weights = self.weights
        
        ensemble_prob = np.zeros_like(next(iter(predictions.values())))
        
        for model in self.models:
            if model in predictions and model in weights:
                ensemble_prob += weights[model] * predictions[model]
        
        return ensemble_prob
    
    def median_ensemble(
        self,
        predictions: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        Median ensemble (robust to outliers).
        
        Args:
            predictions: Dictionary of model predictions {model_name: probability_array}
            
        Returns:
            Ensemble probability array (median across models)
        """
        # Stack predictions along new axis
        stacked = np.stack([predictions[model] for model in self.models if model in predictions], axis=0)
        
        # Compute median along model axis
        ensemble_prob = np.median(stacked, axis=0)
        
        return ensemble_prob
    
    def rank_ensemble(
        self,
        predictions: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        Rank-based ensemble (converts predictions to ranks, then averages).
        
        Args:
            predictions: Dictionary of model predictions {model_name: probability_array}
            
        Returns:
            Ensemble probability array
        """
        # Convert each prediction to ranks
        ranked_predictions = {}
        for model in self.models:
            if model in predictions:
                ranked_predictions[model] = self._to_ranks(predictions[model])
        
        # Average ranks
        ensemble_prob = np.zeros_like(next(iter(ranked_predictions.values())))
        for model in self.models:
            if model in ranked_predictions:
                ensemble_prob += ranked_predictions[model]
        
        ensemble_prob /= len(ranked_predictions)
        
        # Convert back to probability scale
        ensemble_prob = self._from_ranks(ensemble_prob)
        
        return ensemble_prob
    
    def _to_ranks(self, arr: np.ndarray) -> np.ndarray:
        """Convert array to ranks (0 to 1)."""
        ranks = np.zeros_like(arr)
        for i in range(arr.shape[0]):
            ranks[i] = (np.argsort(np.argsort(arr[i])) + 1) / len(arr[i])
        return ranks
    
    def _from_ranks(self, ranks: np.ndarray) -> np.ndarray:
        """Convert ranks back to probability scale."""
        return ranks  # Already in 0-1 range
    
    def get_ensemble_statistics(
        self,
        predictions: Dict[str, np.ndarray],
        ensemble_prob: np.ndarray,
    ) -> Dict[str, float]:
        """
        Get statistics about the ensemble.
        
        Args:
            predictions: Dictionary of model predictions
            ensemble_prob: Ensemble probability
            
        Returns:
            Dictionary of statistics
        """
        # Calculate variance across models
        stacked = np.stack([predictions[model] for model in self.models if model in predictions], axis=0)
        model_variance = np.var(stacked, axis=0).mean()
        
        # Calculate correlation with individual models
        correlations = {}
        for model in self.models:
            if model in predictions:
                corr = np.corrcoef(predictions[model].flatten(), ensemble_prob.flatten())[0, 1]
                correlations[model] = corr
        
        return {
            "model_variance": model_variance,
            "model_correlations": correlations,
            "ensemble_mean": ensemble_prob.mean(),
            "ensemble_std": ensemble_prob.std(),
        }
