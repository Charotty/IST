"""
Weight Utils

Utility functions for weight management and normalization.
"""

import numpy as np
from typing import Dict, List, Optional


class WeightUtils:
    """
    Utility functions for weight management.
    """
    
    @staticmethod
    def normalize_weights(
        weights: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Normalize weights to sum to 1.
        
        Args:
            weights: Dictionary of weights
            
        Returns:
            Normalized weights dictionary
        """
        total = sum(weights.values())
        if total == 0:
            return {k: 1.0 / len(weights) for k in weights.keys()}
        return {k: v / total for k, v in weights.items()}
    
    @staticmethod
    def equal_weights(
        models: List[str],
    ) -> Dict[str, float]:
        """
        Create equal weights for all models.
        
        Args:
            models: List of model names
            
        Returns:
            Dictionary of equal weights
        """
        weight = 1.0 / len(models)
        return {model: weight for model in models}
    
    @staticmethod
    def interpolate_weights(
        weights1: Dict[str, float],
        weights2: Dict[str, float],
        alpha: float = 0.5,
    ) -> Dict[str, float]:
        """
        Interpolate between two weight dictionaries.
        
        Args:
            weights1: First weight dictionary
            weights2: Second weight dictionary
            alpha: Interpolation parameter (0=weights1, 1=weights2)
            
        Returns:
            Interpolated weights dictionary
        """
        # Ensure both have same keys
        all_keys = set(weights1.keys()) | set(weights2.keys())
        
        interpolated = {}
        for key in all_keys:
            w1 = weights1.get(key, 0.0)
            w2 = weights2.get(key, 0.0)
            interpolated[key] = (1 - alpha) * w1 + alpha * w2
        
        return WeightUtils.normalize_weights(interpolated)
    
    @staticmethod
    def clip_weights(
        weights: Dict[str, float],
        min_weight: float = 0.0,
        max_weight: float = 1.0,
    ) -> Dict[str, float]:
        """
        Clip weights to a specified range.
        
        Args:
            weights: Dictionary of weights
            min_weight: Minimum weight value
            max_weight: Maximum weight value
            
        Returns:
            Clipped and normalized weights dictionary
        """
        clipped = {k: np.clip(v, min_weight, max_weight) for k, v in weights.items()}
        return WeightUtils.normalize_weights(clipped)
    
    @staticmethod
    def weight_entropy(
        weights: Dict[str, float],
    ) -> float:
        """
        Calculate entropy of weight distribution.
        
        Args:
            weights: Dictionary of weights
            
        Returns:
            Entropy value (higher = more diversified)
        """
        weights_array = np.array(list(weights.values()))
        # Avoid log(0)
        weights_array = weights_array[weights_array > 0]
        entropy = -np.sum(weights_array * np.log(weights_array))
        return float(entropy)
    
    @staticmethod
    def weight_concentration(
        weights: Dict[str, float],
    ) -> float:
        """
        Calculate concentration of weight distribution (Herfindahl index).
        
        Args:
            weights: Dictionary of weights
            
        Returns:
            Concentration value (higher = more concentrated)
        """
        weights_array = np.array(list(weights.values()))
        concentration = np.sum(weights_array ** 2)
        return float(concentration)
    
    @staticmethod
    def top_k_weights(
        weights: Dict[str, float],
        k: int = 2,
    ) -> Dict[str, float]:
        """
        Get top k weights and set others to zero.
        
        Args:
            weights: Dictionary of weights
            k: Number of top weights to keep
            
        Returns:
            Dictionary with only top k weights
        """
        sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        top_k = dict(sorted_weights[:k])
        
        # Renormalize
        return WeightUtils.normalize_weights(top_k)
    
    @staticmethod
    def adaptive_weight_update(
        current_weights: Dict[str, float],
        performance: Dict[str, float],
        learning_rate: float = 0.1,
    ) -> Dict[str, float]:
        """
        Update weights based on performance metrics.
        
        Args:
            current_weights: Current weight dictionary
            performance: Performance metrics per model
            learning_rate: Learning rate for weight updates
            
        Returns:
            Updated weights dictionary
        """
        # Convert performance to relative performance
        perf_values = np.array(list(performance.values()))
        if np.std(perf_values) > 0:
            normalized_perf = (perf_values - np.mean(perf_values)) / np.std(perf_values)
        else:
            normalized_perf = np.zeros_like(perf_values)
        
        # Update weights
        updated = {}
        for i, (model, weight) in enumerate(current_weights.items()):
            if model in performance:
                updated[model] = weight + learning_rate * normalized_perf[i] * weight
            else:
                updated[model] = weight
        
        # Ensure non-negative and normalize
        updated = {k: max(v, 0.0) for k, v in updated.items()}
        return WeightUtils.normalize_weights(updated)
