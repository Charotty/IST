"""
Meta-Learning Layer for Dynamic Model Selection
Based on VKR requirements for adaptive model selection
"""

import numpy as np
from typing import Dict, Any, List, Optional, Union
from collections import deque
import logging

logger = logging.getLogger(__name__)


class MetaLearner:
    """
    Meta-learning layer for dynamic model selection.
    
    According to VKR, this layer selects the most effective model
    or weights their predictions based on current market context.
    """
    
    def __init__(self, models: Dict[str, Any], window_size: int = 100):
        """
        Initialize meta-learner.
        
        Args:
            models: Dictionary of model instances
            window_size: Window size for performance tracking
        """
        self.models = models
        self.model_names = list(models.keys())
        self.window_size = window_size
        
        # Performance tracking
        self.model_performance = {name: deque(maxlen=window_size) for name in self.model_names}
        self.model_weights = {name: 1.0 / len(models) for name in self.model_names}
        
        # Market regime tracking
        self.current_regime = "neutral"
        self.regime_history = deque(maxlen=50)
        
    def update_performance(self, model_name: str, metric: float, metric_type: str = "accuracy"):
        """
        Update performance tracking for a model.
        
        Args:
            model_name: Name of the model
            metric: Performance metric value
            metric_type: Type of metric (accuracy, sharpe, etc.)
        """
        if model_name in self.model_performance:
            self.model_performance[model_name].append({
                "metric": metric,
                "type": metric_type,
                "timestamp": len(self.model_performance[model_name])
            })
    
    def get_model_weights(self) -> Dict[str, float]:
        """
        Get current model weights based on recent performance.
        
        Returns:
            Dictionary of model weights
        """
        # Calculate recent performance scores
        scores = {}
        for name in self.model_names:
            if len(self.model_performance[name]) > 0:
                recent_metrics = [m["metric"] for m in self.model_performance[name]]
                scores[name] = np.mean(recent_metrics[-10:]) if len(recent_metrics) >= 10 else np.mean(recent_metrics)
            else:
                scores[name] = 0.5  # Default score
        
        # Normalize to weights
        total_score = sum(scores.values())
        if total_score > 0:
            weights = {name: score / total_score for name, score in scores.items()}
        else:
            weights = {name: 1.0 / len(self.models) for name in self.model_names}
        
        self.model_weights = weights
        return weights
    
    def select_best_model(self, metric_type: str = "accuracy") -> str:
        """
        Select the best performing model.
        
        Args:
            metric_type: Type of metric to use for selection
            
        Returns:
            Name of the best model
        """
        best_model = None
        best_score = -float('inf')
        
        for name in self.model_names:
            if len(self.model_performance[name]) > 0:
                recent_metrics = [m["metric"] for m in self.model_performance[name] 
                               if m["type"] == metric_type]
                if recent_metrics:
                    avg_score = np.mean(recent_metrics[-10:])
                    if avg_score > best_score:
                        best_score = avg_score
                        best_model = name
        
        return best_model if best_model else self.model_names[0]
    
    def ensemble_predict(self, X: np.ndarray, method: str = "weighted") -> np.ndarray:
        """
        Make ensemble prediction using multiple models.
        
        Args:
            X: Input features
            method: Ensemble method (weighted, voting, stacking)
            
        Returns:
            Ensemble prediction
        """
        predictions = {}
        probabilities = {}
        
        # Get predictions from all models
        for name, model in self.models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    prob = model.predict_proba(X)
                    pred = np.argmax(prob, axis=1)
                    probabilities[name] = prob
                else:
                    pred = model.predict(X)
                predictions[name] = pred
            except Exception as e:
                logger.error(f"Error predicting with {name}: {e}")
                continue
        
        if not predictions:
            raise ValueError("No models available for prediction")
        
        if method == "weighted":
            return self._weighted_ensemble(predictions, probabilities)
        elif method == "voting":
            return self._voting_ensemble(predictions)
        else:
            return self._weighted_ensemble(predictions, probabilities)
    
    def _weighted_ensemble(self, predictions: Dict[str, np.ndarray],
                          probabilities: Dict[str, np.ndarray]) -> np.ndarray:
        """Weighted ensemble based on model performance."""
        weights = self.get_model_weights()
        
        if probabilities:
            # Weight probabilities
            weighted_probs = np.zeros_like(list(probabilities.values())[0])
            for name, prob in probabilities.items():
                weighted_probs += weights[name] * prob
            
            return np.argmax(weighted_probs, axis=1)
        else:
            # Weight predictions (for regression)
            weighted_pred = np.zeros_like(list(predictions.values())[0])
            for name, pred in predictions.items():
                weighted_pred += weights[name] * pred
            return weighted_pred
    
    def _voting_ensemble(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Majority voting ensemble."""
        pred_array = np.array(list(predictions.values()))
        
        # For classification
        if pred_array.ndim == 1:
            # Single prediction per model
            from scipy.stats import mode
            return mode(pred_array, axis=0)[0].flatten()
        else:
            # Multiple predictions
            return np.apply_along_axis(lambda x: np.bincount(x).argmax(), 0, pred_array)
    
    def detect_market_regime(self, X: np.ndarray) -> str:
        """
        Detect current market regime.
        
        Args:
            X: Recent market features
            
        Returns:
            Regime: trending, ranging, volatile
        """
        # Calculate volatility
        if X.shape[1] > 1:
            returns = X[:, 1] if X.shape[1] > 1 else X[:, 0]
            volatility = np.std(returns)
            trend = np.mean(returns)
            
            if volatility > 0.02:  # High volatility
                regime = "volatile"
            elif abs(trend) > 0.005:  # Strong trend
                regime = "trending"
            else:
                regime = "ranging"
        else:
            regime = "neutral"
        
        self.current_regime = regime
        self.regime_history.append(regime)
        return regime
    
    def get_regime_specific_weights(self) -> Dict[str, float]:
        """
        Get model weights specific to current market regime.
        
        Returns:
            Dictionary of regime-specific weights
        """
        # Different models perform better in different regimes
        regime_weights = {
            "trending": {"gru": 0.4, "transformer": 0.3, "cnn": 0.2, "siamese": 0.1},
            "ranging": {"gru": 0.2, "transformer": 0.2, "cnn": 0.3, "siamese": 0.3},
            "volatile": {"gru": 0.3, "transformer": 0.4, "cnn": 0.2, "siamese": 0.1},
            "neutral": {name: 0.25 for name in self.model_names}
        }
        
        weights = regime_weights.get(self.current_regime, regime_weights["neutral"])
        
        # Adjust for available models
        available_weights = {}
        for name in self.model_names:
            available_weights[name] = weights.get(name, 0.25)
        
        # Normalize
        total = sum(available_weights.values())
        if total > 0:
            available_weights = {k: v/total for k, v in available_weights.items()}
        
        return available_weights


class AdaptiveModelSelector:
    """
    Adaptive model selector that continuously learns which model performs best.
    """
    
    def __init__(self, models: Dict[str, Any], adaptation_interval: int = 50):
        """
        Initialize adaptive model selector.
        
        Args:
            models: Dictionary of model instances
            adaptation_interval: Interval for weight adaptation
        """
        self.models = models
        self.meta_learner = MetaLearner(models)
        self.adaptation_interval = adaptation_interval
        self.prediction_count = 0
        
    def predict(self, X: np.ndarray) -> tuple:
        """
        Make prediction with adaptive model selection.
        
        Args:
            X: Input features
            
        Returns:
            Tuple of (prediction, selected_model_name)
        """
        # Detect market regime
        regime = self.meta_learner.detect_market_regime(X)
        
        # Get regime-specific weights
        weights = self.meta_learner.get_regime_specific_weights()
        
        # Select best model based on regime
        best_model_name = max(weights, key=weights.get)
        
        # Make prediction
        prediction = self.models[best_model_name].predict(X)
        
        self.prediction_count += 1
        
        return prediction, best_model_name
    
    def update(self, model_name: str, prediction: np.ndarray, 
              true_labels: np.ndarray):
        """
        Update meta-learner with prediction results.
        
        Args:
            model_name: Name of the model used
            prediction: Model predictions
            true_labels: True labels
        """
        accuracy = np.mean(prediction == true_labels)
        self.meta_learner.update_performance(model_name, accuracy, "accuracy")
        
        # Adapt weights periodically
        if self.prediction_count % self.adaptation_interval == 0:
            self.meta_learner.get_model_weights()
