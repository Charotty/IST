from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import numpy as np
import joblib
from pathlib import Path


class BaseModel(ABC):
    """Base class for all ML models (immutable interface)."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self._is_fitted = False
        self.feature_names: Optional[List[str]] = None

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseModel":
        """
        Train the model.

        Args:
            X: features, shape (n_samples, n_features)
            y: target, shape (n_samples,)

        Returns:
            self (for chaining)
        """
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: features, shape (n_samples, n_features)

        Returns:
            predictions, shape (n_samples,)
            Values: 0 (sell), 1 (hold), 2 (buy)
        """
        ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: features, shape (n_samples, n_features)

        Returns:
            probabilities, shape (n_samples, n_classes)
        """
        ...

    def get_confidence(self, X: np.ndarray) -> np.ndarray:
        """
        Confidence scores (max probability).

        Returns:
            confidence scores, shape (n_samples,)
        """
        proba = self.predict_proba(X)
        return np.max(proba, axis=1)

    def save(self, path: Path) -> None:
        """Serialize model to disk."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted, nothing to save")
        model_data = {
            "model": self,
            "config": self.config,
            "feature_names": self.feature_names,
            "metadata": self.get_metadata(),
        }
        joblib.dump(model_data, path)

    @classmethod
    def load(cls, path: Path) -> "BaseModel":
        """Deserialize model from disk."""
        model_data = joblib.load(path)
        model = model_data["model"]
        model.config = model_data["config"]
        model.feature_names = model_data["feature_names"]
        return model

    def get_metadata(self) -> Dict[str, Any]:
        """Model metadata."""
        return {
            "model_type": self.__class__.__name__,
            "is_fitted": self._is_fitted,
            "config": self.config,
        }

    def validate_input(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """Validate input shape and type."""
        if not isinstance(X, np.ndarray):
            raise TypeError(f"X must be np.ndarray, got {type(X)}")
        # Allow both 2D (sklearn-style) and 3D (neural network-style) inputs
        if X.ndim not in [2, 3]:
            raise ValueError(f"X must be 2D or 3D, got {X.ndim}D")
        if self._is_fitted and self.feature_names is not None:
            # For 2D: check features in last dimension
            # For 3D: check features in last dimension (sequence, features)
            expected = len(self.feature_names)
            if X.shape[-1] != expected:
                raise ValueError(f"Expected {expected} features, got {X.shape[-1]}")
        
        if y is not None:
            if not isinstance(y, np.ndarray):
                raise TypeError(f"y must be np.ndarray, got {type(y)}")
            if y.ndim != 1:
                raise ValueError(f"y must be 1D, got {y.ndim}D")
            # For 3D input, y length should match first dimension (samples)
            if len(X) != len(y):
                raise ValueError(f"X and y must have same length: {len(X)} vs {len(y)}")
