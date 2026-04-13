from __future__ import annotations

from typing import Iterator, Tuple
import numpy as np


class TimeSeriesSplitter:
    """Time series cross-validation (no future leakage)."""

    def __init__(
        self,
        n_splits: int = 5,
        test_size: Optional[int] = None,
        gap: int = 0,
    ) -> None:
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap

    def split(self, X: np.ndarray) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate train/test indices respecting temporal order."""
        n_samples = len(X)
        if self.test_size is None:
            test_size = n_samples // (self.n_splits + 1)
        else:
            test_size = self.test_size

        for i in range(self.n_splits):
            test_end = n_samples - i * test_size
            test_start = test_end - test_size
            train_end = test_start - self.gap
            train_start = 0
            if train_end <= train_start:
                break
            train_indices = np.arange(train_start, train_end)
            test_indices = np.arange(test_start, test_end)
            yield train_indices, test_indices


class WalkForwardValidator:
    """Walk-forward validation for trading strategies."""

    def __init__(self, train_size: int, test_size: int, step_size: int) -> None:
        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size

    def split(self, X: np.ndarray) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate walk-forward splits."""
        n_samples = len(X)
        start = 0
        while start + self.train_size + self.test_size <= n_samples:
            train_end = start + self.train_size
            test_end = train_end + self.test_size
            train_indices = np.arange(start, train_end)
            test_indices = np.arange(train_end, test_end)
            yield train_indices, test_indices
            start += self.step_size
