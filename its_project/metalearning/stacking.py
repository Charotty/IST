from __future__ import annotations

from typing import List

import numpy as np
from sklearn.model_selection import cross_val_predict

from its_project.models.base import BaseModel


class StackingEnsemble(BaseModel):
    """Stacking ensemble of base models with meta model."""

    def __init__(
        self,
        base_models: List[BaseModel],
        meta_model: BaseModel,
        config: dict | None = None,
    ) -> None:
        super().__init__(config or {})
        self.base_models = base_models
        self.meta_model = meta_model

    def fit(self, X: np.ndarray, y: np.ndarray) -> "StackingEnsemble":
        """Fit stacking ensemble (base models + meta model)."""
        # Step 1: Get out-of-fold predictions from base models
        base_predictions = []
        for model in self.base_models:
            model.fit(X, y)
            pred_proba = cross_val_predict(
                model, X, y, cv=5, method="predict_proba"
            )
            base_predictions.append(pred_proba)

        # Step 2: Stack predictions as meta features
        meta_features = np.hstack(base_predictions)

        # Step 3: Fit meta model
        self.meta_model.fit(meta_features, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if not self.is_fitted:
            raise RuntimeError("StackingEnsemble not fitted")
        base_predictions = []
        for model in self.base_models:
            pred_proba = model.predict_proba(X)
            base_predictions.append(pred_proba)
        meta_features = np.hstack(base_predictions)
        return self.meta_model.predict(meta_features)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise RuntimeError("StackingEnsemble not fitted")
        base_predictions = []
        for model in self.base_models:
            pred_proba = model.predict_proba(X)
            base_predictions.append(pred_proba)
        meta_features = np.hstack(base_predictions)
        return self.meta_model.predict_proba(meta_features)
