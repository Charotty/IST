"""Regression tests for audit items (threshold safety, meta labels)."""

import numpy as np
import pandas as pd
import pytest

from utils.data_leakage_prevention import create_safe_meta_labels, compute_safe_threshold


def test_compute_safe_threshold_requires_causal_mechanism():
    x = np.array([0.1, 0.9, 0.5, 0.4])
    with pytest.raises(ValueError, match="compute_safe_threshold requires"):
        compute_safe_threshold(x, "median", window=None, expanding=False, train_threshold=None)


def test_create_safe_meta_labels_lags_direction_prob_by_default():
    n = 20
    close = pd.Series(np.linspace(100, 110, n))
    direction_prob = pd.Series([0.3 if i % 2 == 0 else 0.7 for i in range(n)])
    df = pd.DataFrame({"close": close, "direction_prob": direction_prob})

    y_same = create_safe_meta_labels(
        df, horizon=3, direction_prob_same_bar_allowed=True
    )
    y_lag = create_safe_meta_labels(
        df, horizon=3, direction_prob_same_bar_allowed=False
    )
    assert not y_same.equals(y_lag)
