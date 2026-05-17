"""
Four-model orchestration: factory + WFO smoke (synthetic / small parquet).

``pytest tests/test_orchestration_four_models.py -v``
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from orchestration.glue import default_horizon_labels, run_wfo_backtest_on_feature_table
from orchestration.integration_smoke import SmokeModel, SmokeRegime, build_synthetic_features
from orchestration.orchestrator_config import OrchestratorConfig
from orchestration import TrainingOrchestrator
from meta_learning.dynamic_meta import DynamicMetaWeighting


pytest.importorskip("sklearn")
pytest.importorskip("lightgbm")
pytest.importorskip("xgboost")


def _meta_matching_config(cfg: OrchestratorConfig) -> DynamicMetaWeighting:
    w = {k: 1.0 / len(cfg.model_keys) for k in cfg.model_keys}
    return DynamicMetaWeighting(trend_weights=w.copy(), range_weights=w.copy())


@pytest.fixture
def four_model_config():
    return OrchestratorConfig(
        model_keys=["lgb", "gru", "xgb", "cnn"],
        train_window_size=400,
        test_window_size=100,
        walk_forward_step=100,
        prediction_horizon=12,
        apply_decision_pipeline=False,
        ensemble_mode="regime_adaptive",
    )


@pytest.fixture
def synthetic_features():
    return build_synthetic_features(n=900)


def test_four_model_keys_all_predict(four_model_config, synthetic_features):
    models = {k: SmokeModel(k) for k in four_model_config.model_keys}
    assert set(models.keys()) == {"lgb", "gru", "xgb", "cnn"}
    for key, model in models.items():
        preds = model.predict(synthetic_features)
        assert len(preds) == len(synthetic_features)
        assert np.all(np.isfinite(preds)), key


def test_four_model_wfo_smoke(four_model_config, synthetic_features):
    y = default_horizon_labels(synthetic_features["close"], four_model_config.prediction_horizon)
    models = {k: SmokeModel(k) for k in four_model_config.model_keys}
    orch = TrainingOrchestrator(four_model_config)
    orch.initialize(
        models=models,
        regime_detector=SmokeRegime(),
        meta_weighting=_meta_matching_config(four_model_config),
    )
    folds = orch.walk_forward_backtest(synthetic_features, y)
    assert len(folds) >= 1


@pytest.mark.integration
def test_model_factory_four_keys_predict(synthetic_features):
    tf = pytest.importorskip("tensorflow")
    del tf  # noqa: F841 — only check import

    from orchestration.model_factory import build_orchestration_models

    cfg = OrchestratorConfig(
        model_keys=["lgb", "gru", "xgb", "cnn"],
        feature_window_size=12,
        train_window_size=400,
        test_window_size=100,
        walk_forward_step=100,
        prediction_horizon=12,
        apply_decision_pipeline=False,
    )
    models = build_orchestration_models(cfg, synthetic_features, dl_epochs=1)
    assert set(models.keys()) == {"lgb", "gru", "xgb", "cnn"}
    y = default_horizon_labels(synthetic_features["close"], cfg.prediction_horizon)
    train = synthetic_features.iloc[:600]
    y_train = y.iloc[:600].dropna()
    X_train = train.loc[y_train.index]
    for key, model in models.items():
        if hasattr(model, "fit"):
            model.fit(X_train, y_train)
        preds = model.predict(synthetic_features.iloc[:200])
        assert len(preds) == 200
        assert np.all(np.isfinite(np.asarray(preds, dtype=float))), key


@pytest.mark.integration
def test_from_parquet_full_models_on_demo():
    from orchestration.real_data_benchmark import default_real_parquet
    from orchestration.glue import run_wfo_backtest_from_parquet
    from orchestration.benchmark_runner import load_canonical_config_path

    pq = default_real_parquet(ROOT)
    if not pq.is_file():
        pytest.skip(f"missing parquet: {pq}")
    tf = pytest.importorskip("tensorflow")
    del tf

    cfg_path = load_canonical_config_path(ROOT / "config.yaml")
    folds = run_wfo_backtest_from_parquet(
        pq,
        cfg_path,
        raw_ohlcv=True,
        light_only=False,
        dl_epochs=1,
        orchestrator_config_override=OrchestratorConfig(
            model_keys=["lgb", "gru", "xgb", "cnn"],
            train_window_size=500,
            test_window_size=120,
            walk_forward_step=200,
            prediction_horizon=12,
            apply_decision_pipeline=False,
        ),
    )
    assert len(folds) >= 1


@pytest.mark.integration
def test_canonical_features_preserve_mtf_columns():
    synced = ROOT / "data" / "synced" / "BTC-USDT_1h_mtf.parquet"
    profile = ROOT / "config" / "profiles" / "canonical_4model.yaml"
    if not synced.is_file() or not profile.is_file():
        pytest.skip("MTF synced parquet or canonical profile missing")

    from orchestration.canonical_pipeline import build_canonical_features

    out = build_canonical_features(synced, config_path=profile)
    for col in ("rsi_15m", "adx_4h"):
        assert col in out.columns, f"missing MTF column {col}"
