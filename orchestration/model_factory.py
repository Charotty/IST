"""
Фабрика моделей под ``OrchestratorConfig.model_keys`` (ленивые импорты тяжёлых зависимостей).

Единый контракт: ``fit(DataFrame, Series)`` и ``predict(DataFrame) -> np.ndarray``.

Модуль расположен в ``orchestration``, чтобы импорт не выполнял ``models/__init__.py`` со всем стеком ML.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from orchestration.orchestrator_config import OrchestratorConfig
from meta_learning.dynamic_meta import DynamicMetaWeighting

_LEAKY_SUBSTR = (
    "meta_prob",
    "signal",
    "target",
    "future_",
    "order_book",
    "obi",
)
_BASE_OHLC = {"open", "high", "low", "close", "volume"}


def infer_training_feature_columns(df: pd.DataFrame) -> List[str]:
    """Numeric feature columns suitable for multi-model training (no raw OHLCV leak-set)."""
    cols: List[str] = []
    for c in df.columns:
        cl = c.lower()
        if any(s in cl for s in _LEAKY_SUBSTR):
            continue
        if c in _BASE_OHLC:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        cols.append(c)
    return cols


class _TrainCallableAdapter:
    def __init__(self, inner: Any):
        self.inner = inner

    def fit(self, df: pd.DataFrame, y: pd.Series) -> None:
        self.inner.train(df, y)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        o = self.inner.predict(df)
        return np.asarray(o, dtype=float)


class _DLTrainAdapter:
    def __init__(
        self,
        inner: Any,
        epochs: int,
        batch_size: int,
        *,
        mixed_precision: bool = False,
    ):
        self.inner = inner
        self.epochs = epochs
        self.batch_size = batch_size
        self.mixed_precision = mixed_precision

    def fit(self, df: pd.DataFrame, y: pd.Series) -> None:
        from orchestration.dl_training import setup_mixed_precision

        setup_mixed_precision(self.mixed_precision)
        try:
            self.inner.train(
                df,
                y,
                epochs=self.epochs,
                batch_size=self.batch_size,
                mixed_precision=self.mixed_precision,
            )
        finally:
            setup_mixed_precision(False)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.inner.predict(df), dtype=float)


def meta_weighting_from_config(cfg: OrchestratorConfig) -> DynamicMetaWeighting:
    mode = cfg.ensemble_mode
    if mode not in ("regime_adaptive", "fixed_trend", "fixed_range"):
        mode = "regime_adaptive"
    return DynamicMetaWeighting(
        trend_weights=cfg.trend_weights.copy(),
        range_weights=cfg.range_weights.copy(),
        default_mode=mode,
    )


def build_orchestration_models(
    config: OrchestratorConfig,
    df: pd.DataFrame,
    *,
    feature_columns: Optional[List[str]] = None,
    dl_epochs: int = 5,
    dl_batch_size: int = 64,
    mixed_precision: bool = False,
    tabular_profile: Optional[Any] = None,
    tune_level: Optional[str] = None,
) -> Dict[str, Any]:
    from orchestration.tabular_accel import TabularTrainingProfile, tabular_profile_for_level

    cols = feature_columns if feature_columns is not None else infer_training_feature_columns(df)
    if len(cols) < 1:
        raise ValueError("No feature columns inferred; check DataFrame contents")
    w = config.feature_window_size
    models: Dict[str, Any] = {}

    if tabular_profile is None and tune_level:
        tabular_profile = tabular_profile_for_level(tune_level)
    elif tabular_profile is None:
        tabular_profile = tabular_profile_for_level("confirm")
    if not isinstance(tabular_profile, TabularTrainingProfile):
        tabular_profile = tabular_profile_for_level(str(tune_level or "confirm"))

    lgb_kw = dict(tabular_profile.lgb_extra or {})
    lgb_n = int(lgb_kw.pop("n_estimators", tabular_profile.n_estimators))
    xgb_kw = dict(tabular_profile.xgb_extra or {})
    xgb_n = int(xgb_kw.pop("n_estimators", tabular_profile.n_estimators))

    for key in config.model_keys:
        if key == "lgb":
            from models.tabular.lightgbm_tabular_model import LightGBMTabularModel

            m = LightGBMTabularModel(n_estimators=lgb_n, **lgb_kw)
            m.feature_cols = list(cols)
            models[key] = m
        elif key == "xgb":
            from models.mean_reversion.xgboost_model import XGBoostMeanReversionModel

            m = XGBoostMeanReversionModel(n_estimators=xgb_n, **xgb_kw)
            m.feature_cols = list(cols)
            models[key] = _TrainCallableAdapter(m)
        elif key == "gru":
            from models.trend.gru_model import GRUTrendModel

            m = GRUTrendModel(window_size=w, n_features=len(cols))
            m.feature_cols = list(cols)
            models[key] = _DLTrainAdapter(
                m,
                epochs=dl_epochs,
                batch_size=dl_batch_size,
                mixed_precision=mixed_precision,
            )
        elif key == "cnn":
            from models.volatility.cnn_model import CNNVolatilityModel

            m = CNNVolatilityModel(window_size=w, n_features=len(cols))
            m.feature_cols = list(cols)
            # CNN Conv1D + f16: cuDNN autotune mismatch on GTX 16xx / cuDNN 9.x
            models[key] = _DLTrainAdapter(
                m,
                epochs=dl_epochs,
                batch_size=dl_batch_size,
                mixed_precision=False,
            )
        else:
            raise ValueError(
                f"Unknown model key {key!r}; extend orchestration/model_factory.py"
            )
    return models
