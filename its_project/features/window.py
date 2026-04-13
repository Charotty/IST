from __future__ import annotations

from typing import Generator, List

import numpy as np
import pandas as pd


class WindowedFeatures:
    """Pure sliding window generator for time series."""

    def __init__(self, window_size: int, stride: int = 1) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be > 0")
        if stride <= 0:
            raise ValueError("stride must be > 0")
        self.window_size = window_size
        self.stride = stride

    def create_windows(self, data: pd.DataFrame) -> Generator[pd.DataFrame, None, None]:
        """
        Pure generator: yield sliding windows as immutable DataFrames.
        Each window is a copy; original data is unchanged.
        """
        n = len(data)
        for start in range(0, n - self.window_size + 1, self.stride):
            end = start + self.window_size
            window = data.iloc[start:end].copy()
            yield window

    def transform(
        self,
        data: pd.DataFrame,
        feature_calculator,
    ) -> np.ndarray:
        """
        Apply feature calculator to each window.
        Returns array shape (n_windows, n_features).
        """
        results = []
        for window in self.create_windows(data):
            features = feature_calculator.calculate(window)
            results.append(features)
        if not results:
            return np.empty((0, 0))
        return np.vstack(results)


def create_fixed_length_windows(
    data: pd.DataFrame,
    window_size: int,
    stride: int = 1,
) -> Generator[pd.DataFrame, None, None]:
    """
    Standalone pure function: sliding windows.
    """
    wf = WindowedFeatures(window_size=window_size, stride=stride)
    yield from wf.create_windows(data)
