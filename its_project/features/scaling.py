from __future__ import annotations

from typing import Dict, Any, Tuple

import numpy as np

from its_project.features.base import BaseFeature


class FeatureScaler(BaseFeature):
    """Pure feature scaling wrapper; decorates any BaseFeature with scaling."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.method = config.get("method", "zscore")  # zscore or minmax
        self.axis = config.get("axis", 0)
        self._stats: Dict[str, Any] = {}

    def fit(self, features: np.ndarray) -> "FeatureScaler":
        """Collect statistics for scaling; pure (returns new instance)."""
        if self.method == "zscore":
            mean = np.mean(features, axis=self.axis, keepdims=True)
            std = np.std(features, axis=self.axis, keepdims=True)
            self._stats = {"mean": mean.squeeze(self.axis), "std": std.squeeze(self.axis)}
        elif self.method == "minmax":
            mn = np.min(features, axis=self.axis, keepdims=True)
            mx = np.max(features, axis=self.axis, keepdims=True)
            self._stats = {"min": mn.squeeze(self.axis), "max": mx.squeeze(self.axis)}
        else:
            raise ValueError(f"Unknown method: {self.method}")
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Pure scaling; returns new array."""
        if not self._stats:
            raise RuntimeError("Scaler not fitted; call fit first")
        if self.method == "zscore":
            mean = self._stats["mean"]
            std = self._stats["std"]
            if mean.ndim == 1:
                mean = mean.reshape(1, -1)
                std = std.reshape(1, -1)
            return (features - mean) / (std + 1e-10)
        if self.method == "minmax":
            mn = self._stats["min"]
            mx = self._stats["max"]
            if mn.ndim == 1:
                mn = mn.reshape(1, -1)
                mx = mx.reshape(1, -1)
            return (features - mn) / (mx - mn + 1e-10)
        raise ValueError(f"Unknown method: {self.method}")

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """Fit + transform; pure."""
        return self.fit(features).transform(features)

    def inverse_transform(self, normalized: np.ndarray) -> np.ndarray:
        """Inverse scaling; pure."""
        if not self._stats:
            raise RuntimeError("Scaler not fitted; call fit first")
        if self.method == "zscore":
            mean = self._stats["mean"]
            std = self._stats["std"]
            if mean.ndim == 1:
                mean = mean.reshape(1, -1)
                std = std.reshape(1, -1)
            return normalized * (std + 1e-10) + mean
        if self.method == "minmax":
            mn = self._stats["min"]
            mx = self._stats["max"]
            if mn.ndim == 1:
                mn = mn.reshape(1, -1)
                mx = mx.reshape(1, -1)
            return normalized * (mx - mn + 1e-10) + mn
        raise ValueError(f"Unknown method: {self.method}")

    # BaseFeature dummy methods (not used for scaling)
    def calculate(self, data) -> np.ndarray:
        raise NotImplementedError("FeatureScaler is a wrapper; use fit_transform")

    def get_feature_names(self) -> list:
        return []
