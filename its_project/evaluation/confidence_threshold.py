"""
Dynamic Confidence Thresholding for Trading Decisions
Based on VKR requirements for confidence-based execution
"""

import numpy as np
from typing import Dict, Any, List, Optional
from scipy import stats


class ConfidenceThreshold:
    """
    Dynamic confidence thresholding for trading decisions.
    
    According to VKR, this allows the system to execute trades only
    when the model has high confidence, filtering out low-probability signals.
    """
    
    def __init__(self, initial_threshold: float = 0.65, 
                 min_threshold: float = 0.5, max_threshold: float = 0.9,
                 adaptation_rate: float = 0.05):
        """
        Initialize confidence threshold.
        
        Args:
            initial_threshold: Starting confidence threshold
            min_threshold: Minimum allowed threshold
            max_threshold: Maximum allowed threshold
            adaptation_rate: Rate of threshold adaptation
        """
        self.threshold = initial_threshold
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.adaptation_rate = adaptation_rate
        
        # Performance tracking
        self.recent_predictions = []
        self.recent_accuracy = []
        self.window_size = 100
        
    def should_execute(self, probability: float, predicted_class: int) -> bool:
        """
        Determine if trade should be executed based on confidence.
        
        Args:
            probability: Model's confidence score
            predicted_class: Predicted class (0=SELL, 1=HOLD, 2=BUY)
            
        Returns:
            True if trade should be executed, False otherwise
        """
        return probability >= self.threshold
    
    def update_threshold(self, recent_accuracy: float, recent_sharpe: float):
        """
        Adapt threshold based on recent performance.
        
        Args:
            recent_accuracy: Recent prediction accuracy
            recent_sharpe: Recent Sharpe ratio
        """
        # If performance is good, lower threshold to get more trades
        if recent_accuracy > 0.7 and recent_sharpe > 1.0:
            self.threshold = max(
                self.min_threshold,
                self.threshold - self.adaptation_rate
            )
        # If performance is poor, raise threshold to be more selective
        elif recent_accuracy < 0.5 or recent_sharpe < 0.5:
            self.threshold = min(
                self.max_threshold,
                self.threshold + self.adaptation_rate
            )
    
    def get_optimal_threshold(self, probabilities: np.ndarray, 
                            true_labels: np.ndarray) -> float:
        """
        Find optimal threshold using ROC curve analysis.
        
        Args:
            probabilities: Predicted probabilities
            true_labels: True labels
            
        Returns:
            Optimal threshold value
        """
        from sklearn.metrics import roc_curve, auc
        
        # Get ROC curve for binary classification (BUY vs not BUY)
        binary_labels = (true_labels == 2).astype(int)
        binary_probs = probabilities[:, 2] if len(probabilities.shape) > 1 else probabilities
        
        fpr, tpr, thresholds = roc_curve(binary_labels, binary_probs)
        
        # Find optimal threshold (maximize TPR - FPR)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        
        # Clamp to valid range
        optimal_threshold = max(self.min_threshold, min(self.max_threshold, optimal_threshold))
        
        return optimal_threshold
    
    def evaluate_threshold(self, probabilities: np.ndarray, 
                         true_labels: np.ndarray, 
                         threshold: Optional[float] = None) -> Dict[str, float]:
        """
        Evaluate performance at a given threshold.
        
        Args:
            probabilities: Predicted probabilities
            true_labels: True labels
            threshold: Threshold to evaluate (uses current if None)
            
        Returns:
            Dictionary with performance metrics
        """
        if threshold is None:
            threshold = self.threshold
        
        # Get predictions at threshold
        if len(probabilities.shape) > 1:
            max_probs = np.max(probabilities, axis=1)
            predictions = np.argmax(probabilities, axis=1)
        else:
            max_probs = probabilities
            predictions = (probabilities > threshold).astype(int)
        
        # Filter by threshold
        mask = max_probs >= threshold
        filtered_predictions = predictions[mask]
        filtered_labels = true_labels[mask]
        
        if len(filtered_labels) == 0:
            return {
                "accuracy": 0.0,
                "coverage": 0.0,
                "threshold": threshold
            }
        
        # Calculate metrics
        accuracy = np.mean(filtered_predictions == filtered_labels)
        coverage = len(filtered_labels) / len(true_labels)
        
        return {
            "accuracy": accuracy,
            "coverage": coverage,
            "threshold": threshold,
            "num_trades": len(filtered_labels)
        }
    
    def adapt_threshold_rolling(self, predictions: List[int], 
                                true_labels: List[int],
                                probabilities: List[float]):
        """
        Adapt threshold using rolling window of recent performance.
        
        Args:
            predictions: List of predictions
            true_labels: List of true labels
            probabilities: List of confidence scores
        """
        # Add to recent history
        for pred, true, prob in zip(predictions, true_labels, probabilities):
            self.recent_predictions.append((pred, true, prob))
            if len(self.recent_predictions) > self.window_size:
                self.recent_predictions.pop(0)
        
        # Calculate recent accuracy
        if len(self.recent_predictions) >= 10:
            recent_correct = sum(1 for pred, true, _ in self.recent_predictions 
                               if pred == true)
            recent_acc = recent_correct / len(self.recent_predictions)
            self.recent_accuracy.append(recent_acc)
            
            # Adapt threshold
            if len(self.recent_accuracy) >= 5:
                avg_acc = np.mean(self.recent_accuracy[-5:])
                if avg_acc > 0.7:
                    self.threshold = max(self.min_threshold, self.threshold - 0.01)
                elif avg_acc < 0.5:
                    self.threshold = min(self.max_threshold, self.threshold + 0.01)


class EnsembleConfidenceThreshold:
    """
    Confidence thresholding for ensemble models.
    
    Combines predictions from multiple models and applies
    confidence-based filtering.
    """
    
    def __init__(self, model_thresholds: Optional[Dict[str, float]] = None):
        """
        Initialize ensemble confidence threshold.
        
        Args:
            model_thresholds: Dictionary of model-specific thresholds
        """
        self.model_thresholds = model_thresholds or {}
        self.global_threshold = 0.65
        
    def should_execute_ensemble(self, model_predictions: Dict[str, np.ndarray],
                                model_probabilities: Dict[str, np.ndarray]) -> bool:
        """
        Determine if ensemble trade should be executed.
        
        Args:
            model_predictions: Dictionary of model predictions
            model_probabilities: Dictionary of model probabilities
            
        Returns:
            True if trade should be executed
        """
        # Check if all models agree
        unique_predictions = set(model_predictions.values())
        if len(unique_predictions) != 1:
            return False
        
        # Get average confidence
        avg_confidence = np.mean([np.max(probs) for probs in model_probabilities.values()])
        
        return avg_confidence >= self.global_threshold
    
    def weighted_confidence(self, model_probabilities: Dict[str, np.ndarray],
                          model_weights: Dict[str, float]) -> float:
        """
        Calculate weighted confidence from ensemble.
        
        Args:
            model_probabilities: Dictionary of model probabilities
            model_weights: Dictionary of model weights
            
        Returns:
            Weighted confidence score
        """
        weighted_conf = 0.0
        total_weight = 0.0
        
        for model_name, probs in model_probabilities.items():
            weight = model_weights.get(model_name, 1.0)
            max_prob = np.max(probs)
            weighted_conf += weight * max_prob
            total_weight += weight
        
        return weighted_conf / total_weight if total_weight > 0 else 0.0
