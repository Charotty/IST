from __future__ import annotations

from typing import Dict, Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression

from its_project.models.base import BaseModel


class EnsembleModel(BaseModel):
    """Ensemble of classical sklearn models."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        # Base models
        self.models = {
            "lr": LogisticRegression(max_iter=1000, random_state=42),
            "rf": RandomForestClassifier(
                n_estimators=config.get("rf_estimators", 100),
                max_depth=config.get("rf_max_depth", 10),
                random_state=42,
            ),
            "gb": GradientBoostingClassifier(
                n_estimators=config.get("gb_estimators", 100),
                max_depth=config.get("gb_max_depth", 5),
                random_state=42,
            ),
        }
        # Voting classifier (soft voting)
        self.ensemble = VotingClassifier(
            estimators=list(self.models.items()),
            voting="soft",
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "EnsembleModel":
        """Train ensemble."""
        self.validate_input(X)
        self.feature_names = [f"feat_{i}" for i in range(X.shape[1])]
        self.ensemble.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        self.validate_input(X)
        return self.ensemble.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        self.validate_input(X)
        return self.ensemble.predict_proba(X)
