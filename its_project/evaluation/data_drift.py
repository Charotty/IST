"""
MLOps with Data Drift Detection
Based on VKR requirements for monitoring data distribution changes
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from scipy import stats
from collections import deque
import logging

logger = logging.getLogger(__name__)


class DataDriftDetector:
    """
    Data drift detection for monitoring distribution changes.
    
    According to VKR, this detects when market data distribution
    deviates from training data, triggering retraining.
    """
    
    def __init__(self, threshold: float = 0.05, window_size: int = 100):
        """
        Initialize data drift detector.
        
        Args:
            threshold: Statistical significance threshold for drift detection
            window_size: Window size for rolling statistics
        """
        self.threshold = threshold
        self.window_size = window_size
        
        # Reference distribution (from training data)
        self.reference_mean = None
        self.reference_std = None
        self.reference_dist = None
        
        # Current data window
        self.current_window = deque(maxlen=window_size)
        
        # Drift history
        self.drift_history = deque(maxlen=50)
        self.drift_detected = False
        
    def fit_reference(self, X: np.ndarray):
        """
        Fit reference distribution from training data.
        
        Args:
            X: Training data
        """
        self.reference_mean = np.mean(X, axis=0)
        self.reference_std = np.std(X, axis=0)
        self.reference_dist = X
        
        logger.info(f"Reference distribution fitted: mean={self.reference_mean[:3]}, std={self.reference_std[:3]}")
    
    def detect_drift(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Detect data drift using statistical tests.
        
        Args:
            X: Current data batch
            
        Returns:
            Dictionary with drift detection results
        """
        if self.reference_mean is None:
            logger.warning("Reference distribution not fitted")
            return {"drift_detected": False, "p_value": 1.0}
        
        # Add to current window
        self.current_window.extend(X)
        
        if len(self.current_window) < self.window_size:
            return {"drift_detected": False, "p_value": 1.0, "reason": "Insufficient data"}
        
        # Get current window as array
        current_data = np.array(self.current_window)
        current_mean = np.mean(current_data, axis=0)
        current_std = np.std(current_data, axis=0)
        
        # Kolmogorov-Smirnov test for distribution shift
        p_values = []
        for feature_idx in range(min(X.shape[1], self.reference_mean.shape[0])):
            ref_feature = self.reference_dist[:, feature_idx]
            curr_feature = current_data[:, feature_idx]
            
            if len(ref_feature) > 0 and len(curr_feature) > 0:
                try:
                    ks_stat, p_value = stats.ks_2samp(ref_feature, curr_feature)
                    p_values.append(p_value)
                except:
                    p_values.append(1.0)
        
        # Average p-value across features
        avg_p_value = np.mean(p_values) if p_values else 1.0
        
        # Detect drift
        drift_detected = avg_p_value < self.threshold
        
        result = {
            "drift_detected": drift_detected,
            "p_value": avg_p_value,
            "reference_mean": self.reference_mean[:3],
            "current_mean": current_mean[:3],
            "num_features": len(p_values)
        }
        
        if drift_detected:
            result["reason"] = f"Statistical significance (p={avg_p_value:.4f}) below threshold"
            self.drift_history.append(result)
            self.drift_detected = True
            logger.warning(f"Data drift detected: p-value={avg_p_value:.4f}")
        
        return result
    
    def detect_drift_kl_divergence(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Detect drift using KL divergence.
        
        Args:
            X: Current data batch
            
        Returns:
            Dictionary with drift detection results
        """
        if self.reference_mean is None or self.reference_std is None:
            return {"drift_detected": False, "kl_divergence": 0.0}
        
        # Add to current window
        self.current_window.extend(X)
        
        if len(self.current_window) < self.window_size:
            return {"drift_detected": False, "kl_divergence": 0.0}
        
        current_data = np.array(self.current_window)
        current_mean = np.mean(current_data, axis=0)
        current_std = np.std(current_data, axis=0)
        
        # Calculate KL divergence (Gaussian approximation)
        kl_div = 0.5 * (
            np.sum((current_mean - self.reference_mean) ** 2 / self.reference_std ** 2) +
            np.sum(self.reference_std ** 2 / current_std ** 2) -
            len(self.reference_mean) +
            np.sum(current_std ** 2 / self.reference_std ** 2)
        )
        
        drift_detected = kl_div > 5.0  # Threshold for KL divergence
        
        result = {
            "drift_detected": drift_detected,
            "kl_divergence": kl_div,
            "threshold": 5.0
        }
        
        if drift_detected:
            self.drift_history.append(result)
            self.drift_detected = True
            logger.warning(f"Data drift detected (KL): {kl_div:.4f}")
        
        return result
    
    def reset_drift(self):
        """Reset drift detection state."""
        self.drift_detected = False
        self.current_window.clear()


class ModelPerformanceMonitor:
    """
    Monitor model performance and trigger retraining when needed.
    """
    
    def __init__(self, min_accuracy: float = 0.5, min_sharpe: float = 0.5,
                 window_size: int = 100):
        """
        Initialize performance monitor.
        
        Args:
            min_accuracy: Minimum acceptable accuracy
            min_sharpe: Minimum acceptable Sharpe ratio
            window_size: Window size for performance tracking
        """
        self.min_accuracy = min_accuracy
        self.min_sharpe = min_sharpe
        self.window_size = window_size
        
        self.accuracy_history = deque(maxlen=window_size)
        self.sharpe_history = deque(maxlen=window_size)
        
        self.retraining_needed = False
        
    def update_performance(self, accuracy: float, sharpe: float):
        """
        Update performance metrics.
        
        Args:
            accuracy: Current accuracy
            sharpe: Current Sharpe ratio
        """
        self.accuracy_history.append(accuracy)
        self.sharpe_history.append(sharpe)
        
        # Check if retraining is needed
        if len(self.accuracy_history) >= self.window_size:
            avg_accuracy = np.mean(self.accuracy_history)
            avg_sharpe = np.mean(self.sharpe_history)
            
            if avg_accuracy < self.min_accuracy or avg_sharpe < self.min_sharpe:
                self.retraining_needed = True
                logger.warning(f"Performance degraded: acc={avg_accuracy:.3f}, sharpe={avg_sharpe:.3f}")
    
    def should_retrain(self) -> bool:
        """
        Check if model retraining is needed.
        
        Returns:
            True if retraining is needed
        """
        return self.retraining_needed
    
    def reset(self):
        """Reset monitoring state."""
        self.retraining_needed = False
        self.accuracy_history.clear()
        self.sharpe_history.clear()


class MLOpsPipeline:
    """
    Complete MLOps pipeline with data drift detection and retraining.
    """
    
    def __init__(self, model: Any, drift_threshold: float = 0.05,
                 min_accuracy: float = 0.5, min_sharpe: float = 0.5):
        """
        Initialize MLOps pipeline.
        
        Args:
            model: Model instance
            drift_threshold: Threshold for drift detection
            min_accuracy: Minimum acceptable accuracy
            min_sharpe: Minimum acceptable Sharpe ratio
        """
        self.model = model
        self.drift_detector = DataDriftDetector(threshold=drift_threshold)
        self.performance_monitor = ModelPerformanceMonitor(
            min_accuracy=min_accuracy, min_sharpe=min_sharpe
        )
        
        self.retraining_count = 0
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit model and reference distribution.
        
        Args:
            X: Training features
            y: Training labels
        """
        # Fit model
        if hasattr(self.model, 'fit'):
            self.model.fit(X, y)
        
        # Fit reference distribution
        self.drift_detector.fit_reference(X)
        
        logger.info("MLOps pipeline fitted")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make prediction with drift detection.
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        # Check for data drift
        drift_result = self.drift_detector.detect_drift(X)
        
        if drift_result["drift_detected"]:
            logger.warning("Data drift detected, retraining recommended")
        
        # Make prediction
        prediction = self.model.predict(X)
        
        return prediction
    
    def update_performance(self, accuracy: float, sharpe: float):
        """
        Update performance monitoring.
        
        Args:
            accuracy: Current accuracy
            sharpe: Current Sharpe ratio
        """
        self.performance_monitor.update_performance(accuracy, sharpe)
    
    def should_retrain(self) -> bool:
        """
        Check if retraining is needed.
        
        Returns:
            True if retraining is needed
        """
        return (self.drift_detector.drift_detected or 
                self.performance_monitor.should_retrain())
    
    def retrain(self, X: np.ndarray, y: np.ndarray):
        """
        Retrain model with new data.
        
        Args:
            X: New training features
            y: New training labels
        """
        logger.info("Retraining model...")
        
        # Reset drift detection
        self.drift_detector.reset_drift()
        self.performance_monitor.reset()
        
        # Retrain model
        if hasattr(self.model, 'fit'):
            self.model.fit(X, y)
        
        # Update reference distribution
        self.drift_detector.fit_reference(X)
        
        self.retraining_count += 1
        logger.info(f"Model retrained (count: {self.retraining_count})")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current MLOps status.
        
        Returns:
            Dictionary with status information
        """
        return {
            "drift_detected": self.drift_detector.drift_detected,
            "retraining_needed": self.performance_monitor.should_retrain(),
            "retraining_count": self.retraining_count,
            "avg_accuracy": np.mean(self.performance_monitor.accuracy_history) if self.performance_monitor.accuracy_history else 0.0,
            "avg_sharpe": np.mean(self.performance_monitor.sharpe_history) if self.performance_monitor.sharpe_history else 0.0
        }
