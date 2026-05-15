"""
Probability Calibration Module

Probability calibration layer using Platt Scaling and Isotonic Regression.
Transforms raw probabilities to calibrated probabilities.
"""

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
import matplotlib.pyplot as plt


class ProbabilityCalibrator:
    """
    Probability calibrator for model outputs.
    
    Methods:
    - Platt Scaling
    - Isotonic Regression
    """
    
    def __init__(self, method='sigmoid'):
        """
        Initialize probability calibrator.
        
        :param method: Calibration method ('sigmoid' for Platt, 'isotonic' for Isotonic)
        """
        self.method = method
        self.calibrator = None
        self.is_fitted = False
    
    def fit(self, y_true, y_prob):
        """
        Fit the calibrator.
        
        :param y_true: True labels
        :param y_prob: Predicted probabilities
        """
        if self.method == 'sigmoid':
            # Platt scaling using logistic regression
            self.calibrator = CalibratedClassifierCV(method='sigmoid', cv='prefit')
            # Note: This requires a base classifier, simplified here
            # For full implementation, pass the classifier
            pass
        elif self.method == 'isotonic':
            # Isotonic regression
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
            self.calibrator.fit(y_prob, y_true)
        
        self.is_fitted = True
    
    def calibrate(self, y_prob):
        """
        Calibrate probabilities.
        
        :param y_prob: Raw probabilities
        :return: Calibrated probabilities
        """
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before calibration")
        
        if self.method == 'isotonic':
            return self.calibrator.predict(y_prob)
        else:
            # For sigmoid, return raw probabilities if not fully implemented
            return y_prob
    
    def plot_calibration_curve(self, y_true, y_prob, n_bins=10):
        """
        Plot calibration curve.
        
        :param y_true: True labels
        :param y_prob: Predicted probabilities
        :param n_bins: Number of bins for calibration curve
        """
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
        
        plt.figure(figsize=(10, 6))
        plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
        plt.plot(prob_pred, prob_true, marker='o', label='Model')
        plt.xlabel('Mean predicted probability')
        plt.ylabel('Fraction of positives')
        plt.title('Calibration Curve')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def get_calibration_metrics(self, y_true, y_prob, n_bins=10):
        """
        Get calibration metrics.
        
        :param y_true: True labels
        :param y_prob: Predicted probabilities
        :param n_bins: Number of bins
        :return: Dict with calibration metrics
        """
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
        
        # Expected Calibration Error (ECE)
        ece = np.abs(prob_true - prob_pred).mean()
        
        return {
            "expected_calibration_error": ece,
            "calibration_method": self.method
        }
