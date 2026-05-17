"""
Интеграция оркестратора без Mock-моделей: реальный OHLCV parquet → FeatureEngine → LGB+XGB → WFO.

Запуск: ``pytest tests/test_orchestration_real_models.py -m integration -v``

При отсутствии demo parquet или ML-библиотек тесты пропускаются (skip).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


pytest.importorskip("sklearn")
pytest.importorskip("lightgbm")
pytest.importorskip("xgboost")

from orchestration.real_data_benchmark import (
    default_real_parquet,
    features_from_ohlcv_parquet,
    run_lightgbm_xgboost_wfo_backtest,
    run_demo_benchmark,
    run_benchmark_report,
)


def _demo_path():
    p = default_real_parquet(ROOT)
    if not p.is_file():
        pytest.skip(f"missing OHLCV parquet: {p}")
    return p


@pytest.fixture(scope="module")
def demo_parquet():
    p = _demo_path()
    if not p.is_file():
        pytest.skip(f"missing demo OHLCV parquet: {p}")
    return p


@pytest.mark.integration
def test_demo_parquet_loadable(demo_parquet):
    df = pd.read_parquet(demo_parquet)
    for col in ("open", "high", "low", "close", "volume"):
        assert col in df.columns
    assert len(df) > 500


@pytest.mark.integration
def test_real_features_no_nan_tail(demo_parquet):
    feat = features_from_ohlcv_parquet(demo_parquet, max_rows=2200)
    assert "close" in feat.columns
    assert "rsi" in feat.columns
    assert not feat[["rsi", "atr", "volatility"]].iloc[100:].isna().any().any()


@pytest.mark.integration
def test_real_models_wfo_backtest_runs(demo_parquet):
    feat = features_from_ohlcv_parquet(demo_parquet, max_rows=2200)
    out = run_lightgbm_xgboost_wfo_backtest(feat)
    assert len(out) >= 1
    assert "Sharpe Ratio" in out.columns
    assert "Walk-Forward Efficiency" in out.columns
    assert "Calmar Ratio" in out.columns
    assert np.all(np.isfinite(out["Sharpe Ratio"].astype(float)))


@pytest.mark.integration
def test_run_demo_benchmark_entrypoint(demo_parquet):
    out = run_demo_benchmark(parquet_path=demo_parquet, max_rows=2200)
    assert len(out) >= 1
    assert "Fold" in out.columns


@pytest.mark.integration
def test_benchmark_report_criteria_on_real_btc(demo_parquet):
    report = run_benchmark_report(demo_parquet, max_rows=2200)
    assert report["feature_rows"] > 500
    assert "criteria" in report
    assert "acceptance" in report["criteria"]
    assert "summary" in report
    folds = report["fold_metrics"]
    assert len(folds) >= 1
