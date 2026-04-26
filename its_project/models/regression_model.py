from __future__ import annotations

import logging
from typing import Dict, Any, List, Optional
import numpy as np
import joblib
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

from its_project.models.base import BaseModel

logger = logging.getLogger(__name__)


class RegressionModel(BaseModel):
    """
    Regression model for predicting price changes (ΔP_hat).
    
    Supports Linear Regression and Gradient Boosting Regressor.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.model_type = config.get("model_type", "linear")
        self._init_model()
        
    def _init_model(self) -> None:
        """Initialize the underlying regression model."""
        if self.model_type == "linear":
            self.model = LinearRegression()
        elif self.model_type == "gradient_boosting":
            self.model = GradientBoostingRegressor(
                n_estimators=self.config.get("n_estimators", 100),
                learning_rate=self.config.get("learning_rate", 0.1),
                max_depth=self.config.get("max_depth", 3),
                random_state=self.config.get("random_state", 42),
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RegressionModel":
        """
        Train the regression model.
        
        Args:
            X: features, shape (n_samples, n_features)
            y: target price changes, shape (n_samples,)
            
        Returns:
            self (for chaining)
        """
        self.validate_input(X, y)
        
        logger.info(f"Training {self.model_type} regression model on {len(X)} samples")
        
        # For 3D input (neural network style), flatten to 2D
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        
        self.model.fit(X, y)
        self._is_fitted = True
        
        # Log training metrics
        if hasattr(self.model, 'score'):
            score = self.model.score(X, y)
            logger.info(f"Training R² score: {score:.4f}")
        
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict price changes.
        
        Args:
            X: features, shape (n_samples, n_features)
            
        Returns:
            predicted price changes, shape (n_samples,)
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted")
        
        self.validate_input(X)
        
        # For 3D input, flatten to 2D
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        
        predictions = self.model.predict(X)
        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return dummy probabilities for compatibility with BaseModel interface.
        
        For regression, this returns a normalized confidence based on
        prediction magnitude.
        
        Args:
            X: features, shape (n_samples, n_features)
            
        Returns:
            dummy probabilities, shape (n_samples, 3) for [SELL, HOLD, BUY]
        """
        predictions = self.predict(X)
        
        # Convert regression predictions to probabilities
        # Negative predictions -> SELL, near zero -> HOLD, positive -> BUY
        proba = np.zeros((len(predictions), 3))
        
        for i, pred in enumerate(predictions):
            if pred < -0.001:  # Significant negative
                proba[i, 0] = 0.7  # SELL
                proba[i, 1] = 0.2  # HOLD
                proba[i, 2] = 0.1  # BUY
            elif pred > 0.001:  # Significant positive
                proba[i, 0] = 0.1  # SELL
                proba[i, 1] = 0.2  # HOLD
                proba[i, 2] = 0.7  # BUY
            else:  # Near zero
                proba[i, 0] = 0.15  # SELL
                proba[i, 1] = 0.7  # HOLD
                proba[i, 2] = 0.15  # BUY
        
        return proba

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance if available.
        
        Returns:
            feature importance array or None
        """
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            return np.abs(self.model.coef_)
        return None

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5,
        scoring: str = 'r2'
    ) -> Dict[str, float]:
        """
        Perform cross-validation.
        
        Args:
            X: features
            y: target
            cv: number of folds
            scoring: scoring metric
            
        Returns:
            dictionary with cv scores
        """
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
        
        scores = cross_val_score(self.model, X, y, cv=cv, scoring=scoring)
        
        return {
            f"cv_{scoring}_mean": float(scores.mean()),
            f"cv_{scoring}_std": float(scores.std()),
            f"cv_{scoring}_scores": scores.tolist(),
        }

    def get_metadata(self) -> Dict[str, Any]:
        """Model metadata."""
        metadata = super().get_metadata()
        metadata.update({
            "model_type": self.model_type,
            "regression_model": str(type(self.model).__name__),
        })
        
        if self._is_fitted:
            importance = self.get_feature_importance()
            if importance is not None:
                metadata["feature_importance"] = importance.tolist()
        
        return metadata
