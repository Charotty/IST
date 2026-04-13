from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd

from its_project.features.base import BaseFeature

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """Immutable pipeline: sequence of pure feature transformations."""

    def __init__(self, features: List[BaseFeature]) -> None:
        if not features:
            raise ValueError("FeaturePipeline requires at least one feature")
        self.features = features
        self._cached_feature_names: List[str] | None = None

    def fit(self, data: pd.DataFrame) -> "FeaturePipeline":
        """
        No-op for pure pipeline; returns self for compatibility.
        Some features may need to compute statistics (e.g., scalers).
        """
        for i, feature in enumerate(self.features):
            if hasattr(feature, "fit"):
                logger.debug("Fitting feature %d/%s", i, feature.__class__.__name__)
                feature.fit(data)
        return self

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        """
        Apply all features; returns concatenated array.
        Input data is not mutated.
        """
        all_features = []
        names = []
        for i, feature in enumerate(self.features):
            logger.debug("Transforming feature %d/%s", i, feature.__class__.__name__)
            feats = feature.calculate(data)
            if feats.ndim == 1:
                feats = feats.reshape(-1, 1)
            all_features.append(feats)
            names.extend(feature.get_feature_names())
        if not all_features:
            return np.empty((len(data), 0))
        result = np.hstack(all_features)
        self._cached_feature_names = names
        return result

    def fit_transform(self, data: pd.DataFrame) -> np.ndarray:
        """Fit + transform in one pass."""
        return self.fit(data).transform(data)

    def get_feature_names(self) -> List[str]:
        """Return cached or compute feature names."""
        if self._cached_feature_names is None:
            raise RuntimeError("Feature names not available; call transform first")
        return self._cached_feature_names.copy()
