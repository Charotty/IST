"""
LightGBM tabular model for orchestration (key ``lgb``).

Row-wise ``predict_proba`` aligned with ``XGBoostMeanReversionModel`` / orchestrator
``predict(features: DataFrame) -> ndarray`` contract.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None


class LightGBMTabularModel:
    def __init__(
        self,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        max_depth: int = -1,
    ):
        if lgb is None:
            raise ImportError("lightgbm is required for LightGBMTabularModel")
        self.model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            objective="binary",
            verbosity=-1,
            class_weight="balanced",
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
        )
        self.feature_cols: Optional[list] = None
        self.is_fitted = False

    def fit(self, df: pd.DataFrame, y: pd.Series) -> None:
        if self.feature_cols is None:
            raise ValueError("feature_cols must be set before fit")
        mask = ~y.isna()
        X = df.loc[mask, self.feature_cols].astype(float)
        yy = y[mask].astype(int)
        self.model.fit(X, yy)
        self.is_fitted = True

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted or self.feature_cols is None:
            raise ValueError("Model must be trained before prediction")
        X = df[self.feature_cols].astype(float)
        return self.model.predict_proba(X)[:, 1].astype(float)
