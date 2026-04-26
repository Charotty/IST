from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

try:
    from its_project.models.base import BaseModel
except ImportError:
    from .base import BaseModel

logger = logging.getLogger(__name__)


class BoostingModel(BaseModel):
    """Gradient Boosting model for trading predictions."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)

        # Model parameters
        self.n_estimators = config.get("n_estimators", 100)
        self.learning_rate = config.get("learning_rate", 0.1)
        self.max_depth = config.get("max_depth", 3)
        self.min_samples_split = config.get("min_samples_split", 2)
        self.min_samples_leaf = config.get("min_samples_leaf", 1)
        self.subsample = config.get("subsample", 1.0)
        self.max_features = config.get("max_features", None)
        self.random_state = config.get("random_state", 42)
        self.class_weight = config.get("class_weight", None)

        # Initialize model
        self.model = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            subsample=self.subsample,
            max_features=self.max_features,
            random_state=self.random_state
        )

        self._fitted = False
        self._feature_names = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BoostingModel":
        """Fit the Gradient Boosting model."""
        self.validate_input(X, y)

        # Ensure 2D input for sklearn
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)

        # Handle class imbalance with sample weights
        sample_weight = None
        if self.class_weight == "balanced":
            from sklearn.utils.class_weight import compute_sample_weight
            sample_weight = compute_sample_weight("balanced", y)
            logger.info("Using balanced sample weights for class imbalance")

        self.model.fit(X, y, sample_weight=sample_weight)
        self._fitted = True
        self.feature_names = [f"feat_{i}" for i in range(X.shape[1])]
        logger.info(f"Model fitted with {self.n_estimators} estimators")

        # Log feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            top_features = np.argsort(self.model.feature_importances_)[-5:]
            logger.info(f"Top 5 feature indices: {top_features}")

        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self._fitted:
            raise ValueError("Model must be fitted before prediction")
        
        self.validate_input(X)
        
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self._fitted:
            raise ValueError("Model must be fitted before prediction")
        
        self.validate_input(X)
        
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        
        return self.model.predict_proba(X)
    
    def get_confidence(self, X: np.ndarray) -> np.ndarray:
        """Get prediction confidence (max probability)."""
        probabilities = self.predict_proba(X)
        return np.max(probabilities, axis=1)
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance scores."""
        if not self._fitted:
            return None
        
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        return None
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict[str, float]:
        """Perform cross-validation."""
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        
        scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')
        return {
            'mean_accuracy': scores.mean(),
            'std_accuracy': scores.std(),
            'scores': scores.tolist()
        }
    
    def save(self, path: str) -> None:
        """Save model state."""
        import joblib
        state = {
            'model': self.model,
            'config': self.config,
            'fitted': self._fitted,
            'feature_names': self._feature_names
        }
        joblib.dump(state, path)
    
    @classmethod
    def load(cls, path: str) -> "BoostingModel":
        """Load model from file."""
        import joblib
        state = joblib.load(path)
        model = cls(state['config'])
        model.model = state['model']
        model._fitted = state['fitted']
        model._feature_names = state.get('feature_names')
        return model
    
    @property
    def is_fitted(self) -> bool:
        return self._fitted
