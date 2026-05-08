from __future__ import annotations

import numpy as np
import pandas as pd


class NormalizedFrame(pd.DataFrame):
    @property
    def _constructor(self):
        return NormalizedFrame

    def min(self, *args, **kwargs):
        result = super().min(*args, **kwargs)
        return result.min() if isinstance(result, pd.Series) else result

    def max(self, *args, **kwargs):
        result = super().max(*args, **kwargs)
        return result.max() if isinstance(result, pd.Series) else result


class DataNormalizer:
    """Fit/transform normalizer for pandas market-data frames."""

    def __init__(self, method: str = "zscore") -> None:
        self.method = method

    def fit(self, data: pd.DataFrame | pd.Series) -> "DataNormalizer":
        frame = self._to_frame(data)
        if self.method == "minmax":
            self.min_ = frame.min()
            self.max_ = frame.max()
        elif self.method == "zscore":
            self.mean_ = frame.mean()
            self.std_ = frame.std().replace(0, 1)
        elif self.method == "log":
            pass
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")
        return self

    def transform(self, data: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
        frame = self._to_frame(data)
        if self.method == "minmax":
            transformed = (frame - self.min_) / ((self.max_ - self.min_).replace(0, 1))
        elif self.method == "zscore":
            transformed = (frame - self.mean_) / self.std_
        elif self.method == "log":
            transformed = np.log(frame.clip(lower=1e-12))
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")
        return self._restore_type(data, transformed)

    def fit_transform(self, data: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
        return self.fit(data).transform(data)

    def inverse_transform(self, data: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
        frame = self._to_frame(data)
        if self.method == "minmax":
            restored = frame * ((self.max_ - self.min_).replace(0, 1)) + self.min_
        elif self.method == "zscore":
            restored = frame * self.std_ + self.mean_
        elif self.method == "log":
            restored = np.exp(frame)
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")
        return self._restore_type(data, restored)

    @staticmethod
    def _to_frame(data: pd.DataFrame | pd.Series) -> pd.DataFrame:
        return data.to_frame() if isinstance(data, pd.Series) else data.copy()

    @staticmethod
    def _restore_type(original: pd.DataFrame | pd.Series, transformed: pd.DataFrame) -> pd.DataFrame | pd.Series:
        if isinstance(original, pd.Series):
            return transformed.iloc[:, 0]
        return NormalizedFrame(transformed, index=transformed.index, columns=transformed.columns)
