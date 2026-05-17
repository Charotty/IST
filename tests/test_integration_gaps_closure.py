"""Closing tests for integration gaps: config validation, bundle, factory, parquet glue."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from orchestration import (
    OrchestratorConfig,
    validate_pipeline_config,
    save_orchestrator_bundle,
    load_orchestrator_bundle,
    validate_bundle_feature_schema,
)
from orchestration.model_factory import build_orchestration_models, infer_training_feature_columns
from orchestration.glue import run_wfo_backtest_from_parquet, SimpleHourlyRegimeStub


def _require_tabular_ml_stack():
    """LightGBM (sklearn API), XGBoost и scikit-learn для табличного контура оркестратора."""
    pytest.importorskip("lightgbm")
    pytest.importorskip("xgboost")
    pytest.importorskip("sklearn")


def test_config_yaml_validates():
    cfg_path = ROOT / "config.yaml"
    if not cfg_path.is_file():
        pytest.skip("no repo config.yaml")
    errs, warns = validate_pipeline_config(cfg_path)
    assert not errs, errs


def test_orchestrator_from_yaml_ignores_unknown_keys():
    cfg_path = ROOT / "config.yaml"
    if not cfg_path.is_file():
        pytest.skip("no repo config.yaml")
    with pytest.warns(UserWarning, match="Ignoring unknown orchestration"):
        OrchestratorConfig.from_yaml(cfg_path)


def test_feature_columns_infer():
    df = pd.DataFrame({"close": [1.0, 2.0], "rsi": [50, 51], "meta_prob_x": [0, 0]})
    cols = infer_training_feature_columns(df)
    assert "rsi" in cols
    assert "meta_prob_x" not in cols


def test_bundle_roundtrip(tmp_path):
    _require_tabular_ml_stack()
    mk = ["lgb", "xgb"]
    w = {k: 0.5 for k in mk}
    cfg = OrchestratorConfig(
        model_keys=mk,
        trend_weights=w.copy(),
        range_weights=w.copy(),
        breakout_weights=w.copy(),
        apply_decision_pipeline=False,
    )
    n = 80
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "close": 100 + rng.standard_normal(n).cumsum() * 0.05,
            "rsi": rng.uniform(30, 70, n),
            "volatility": rng.uniform(0.01, 0.03, n),
        }
    )
    models = build_orchestration_models(cfg, df, dl_epochs=1)
    y = (df["close"].shift(-5) > df["close"]).astype(float)
    y.iloc[-5:] = np.nan
    mask = ~y.isna()
    for m in models.values():
        m.fit(df[mask], y[mask])
    feat = infer_training_feature_columns(df)
    save_orchestrator_bundle(
        tmp_path, cfg, models, SimpleHourlyRegimeStub(), feat, train_meta_threshold=0.5
    )
    assert (tmp_path / "manifest.json").is_file()
    cfg2, models2, regime2, feat2, thr, h = load_orchestrator_bundle(tmp_path)
    assert cfg2.model_keys == mk
    assert len(feat2) == len(feat)
    assert thr == 0.5
    assert validate_bundle_feature_schema(tmp_path, feat2)


def test_parquet_glue_e2e_light(tmp_path):
    _require_tabular_ml_stack()
    cfg_path = ROOT / "config.yaml"
    if not cfg_path.is_file():
        pytest.skip("no repo config.yaml")
    n = 320
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "close": 100 + rng.standard_normal(n).cumsum() * 0.08,
            "rsi": rng.uniform(25, 75, n),
            "macd_hist": rng.standard_normal(n) * 0.01,
            "ema_slope": rng.standard_normal(n) * 0.02,
            "adx": rng.uniform(15, 45, n),
            "rsi_15m": rng.uniform(25, 75, n),
            "ema_slope_15m": rng.standard_normal(n) * 0.02,
            "rsi_4h": rng.uniform(25, 75, n),
            "adx_4h": rng.uniform(15, 45, n),
            "volatility": rng.uniform(0.01, 0.04, n),
            "atr": rng.uniform(0.3, 2.0, n),
        },
        index=pd.date_range("2024-06-01", periods=n, freq="h"),
    )
    pq = tmp_path / "feat.parquet"
    df.to_parquet(pq)
    mk = ["lgb", "xgb"]
    w = {k: 0.5 for k in mk}
    small = OrchestratorConfig(
        model_keys=mk,
        trend_weights=w.copy(),
        range_weights=w.copy(),
        breakout_weights=w.copy(),
        train_window_size=160,
        test_window_size=60,
        walk_forward_step=50,
        prediction_horizon=12,
        embargo_period=3,
        apply_decision_pipeline=False,
    )
    out = run_wfo_backtest_from_parquet(
        pq,
        cfg_path,
        validate_config=True,
        dl_epochs=1,
        orchestrator_config_override=small,
    )
    assert len(out) >= 1
    assert "Sharpe Ratio" in out.columns


def test_inference_engine_warns_on_initialize():
    from models.inference.inference_engine import InferenceEngine

    eng = InferenceEngine()
    with pytest.warns(DeprecationWarning, match="legacy"):
        eng.initialize(object(), object())
