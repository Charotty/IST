from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any, List
import pandas as pd
import numpy as np


class BaseFeature(ABC):
    """Base class for all features (pure, immutable)."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self._feature_names: List[str] = []

    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> np.ndarray:
        """Pure function: calculate features from input data."""
        ...

    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Return list of feature names (order matches columns)."""
        ...

    def validate_input(self, data: pd.DataFrame) -> None:
        """Validate input data; raise if invalid."""
        required = ["open", "high", "low", "close", "volume"]
        missing = set(required) - set(data.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        if data.isnull().any().any():
            raise ValueError("Input contains NaN values")

    def normalize(self, values: np.ndarray, method: str = "zscore") -> np.ndarray:
        """Pure normalization; returns new array."""
        if method == "zscore":
            return (values - values.mean()) / (values.std() + 1e-10)
        if method == "minmax":
            mn, mx = values.min(), values.max()
            return (values - mn) / (mx - mn + 1e-10)
        raise ValueError(f"Unknown method: {method}")
