"""
Probability Calibrator

Calibrates model probabilities to improve reliability and confidence estimation.

Primary model: LightGBM
- Strong tabular performance
- Stable calibration
- Low-latency inference

Planned alternatives:
- Logistic Regression
- CatBoost
- Stacking Meta-Model
"""

import numpy as np
from typing import Optional, Dict, Tuple
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV


class ProbabilityCalibrator:
    """
    Probability calibrator for meta-learning layer.
    
    Calibrates model probabilities to improve reliability and confidence estimation.
    """
    
    def __init__(
        self,
        method: str = "isotonic",
        random_state: int = 42,
    ):
        """
        Initialize ProbabilityCalibrator.
        
        Args:
            method: Calibration method ('isotonic', 'sigmoid', 'none')
            random_state: Random seed for reproducibility
        """
        self.method = method
        self.random_state = random_state
        self.calibrator = None
        self.is_fitted = False
    
    def fit(
        self,
        probabilities: np.ndarray,
        true_labels: np.ndarray,
    ) -> 'ProbabilityCalibrator':
        """
        Fit calibrator on labeled data.
        
        Args:
            probabilities: Predicted probabilities (n_samples,)
            true_labels: True binary labels (n_samples,)
            
        Returns:
            Self (fitted calibrator)
        """
        if self.method == "isotonic":
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
            self.calibrator.fit(probabilities, true_labels)
        elif self.method == "sigmoid":
            # Use Platt scaling (sigmoid)
            # For binary classification, we can use logistic regression
            from sklearn.linear_model import LogisticRegression
            self.calibrator = LogisticRegression(random_state=self.random_state)
            self.calibrator.fit(probabilities.reshape(-1, 1), true_labels)
        elif self.method == "none":
            self.calibrator = None
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")
        
        self.is_fitted = True
        return self
    
    def calibrate(
        self,
        probabilities: np.ndarray,
    ) -> np.ndarray:
        """
        Calibrate probabilities.
        
        Args:
            probabilities: Predicted probabilities to calibrate
            
        Returns:
            Calibrated probabilities
        """
        if not self.is_fitted and self.method != "none":
            raise ValueError("Calibrator must be fitted before calibration")
        
        if self.method == "none" or self.calibrator is None:
            return probabilities.copy()
        
        if self.method == "isotonic":
            return self.calibrator.predict(probabilities)
        elif self.method == "sigmoid":
            return self.calibrator.predict_proba(probabilities.reshape(-1, 1))[:, 1]
        else:
            return probabilities.copy()
    
    def fit_calibrate(
        self,
        probabilities: np.ndarray,
        true_labels: np.ndarray,
    ) -> np.ndarray:
        """
        Fit calibrator and return calibrated probabilities.
        
        Args:
            probabilities: Predicted probabilities
            true_labels: True binary labels
            
        Returns:
            Calibrated probabilities
        """
        self.fit(probabilities, true_labels)
        return self.calibrate(probabilities)
    
    def get_calibration_metrics(
        self,
        probabilities: np.ndarray,
        true_labels: np.ndarray,
    ) -> Dict[str, float]:
        """
        Get calibration metrics.
        
        Args:
            probabilities: Predicted probabilities
            true_labels: True binary labels
            
        Returns:
            Dictionary of calibration metrics
        """
        from sklearn.metrics import brier_score_loss, log_loss
        
        # Brier score (lower is better)
        brier_score = brier_score_loss(true_labels, probabilities)
        
        # Log loss (lower is better)
        log_loss_score = log_loss(true_labels, probabilities)
        
        # Expected calibration error (simplified)
        n_bins = 10
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(probabilities, bin_edges) - 1
        
        ece = 0.0
        for i in range(n_bins):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                avg_conf = np.mean(probabilities[mask])
                avg_acc = np.mean(true_labels[mask])
                ece += np.abs(avg_conf - avg_acc) * np.sum(mask) / len(probabilities)
        
        return {
            "brier_score": float(brier_score),
            "log_loss": float(log_loss_score),
            "expected_calibration_error": float(ece),
        }
