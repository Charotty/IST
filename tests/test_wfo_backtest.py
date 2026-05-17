"""WFO + Backtester integration (leakage-safe splits)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestration import TrainingOrchestrator, OrchestratorConfig
from meta_learning.dynamic_meta import DynamicMetaWeighting
from risk_management import OrchestratorRiskBridge


def _meta(cfg: OrchestratorConfig) -> DynamicMetaWeighting:
    w = {k: 1.0 / len(cfg.model_keys) for k in cfg.model_keys}
    return DynamicMetaWeighting(trend_weights=w.copy(), range_weights=w.copy())


class _M:
    def __init__(self, key: str):
        self.key = key

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.full(len(features), 0.53 + hash(self.key) % 5 * 0.002)

    def fit(self, features: pd.DataFrame, targets: pd.Series) -> None:
        return None


class _R:
    def get_regime_info(self, features: pd.DataFrame):
        n = len(features)
        return {
            "market_regime": "trend",
            "regime_pred": np.array([1 if i % 2 == 0 else 0 for i in range(n)]),
            "trade_allowed": True,
        }


def test_walk_forward_backtest_runs():
    n = 700
    rng = np.random.default_rng(0)
    feat = pd.DataFrame(
        {
            "close": 100 + rng.standard_normal(n).cumsum() * 0.1,
            "volatility": rng.uniform(0.01, 0.03, n),
            "atr": rng.uniform(0.5, 2.0, n),
        },
        index=pd.date_range("2024-01-01", periods=n, freq="h"),
    )
    y = (feat["close"].shift(-12) > feat["close"]).astype(float)
    y.iloc[-12:] = np.nan

    cfg = OrchestratorConfig(
        model_keys=["lgb", "gru", "xgb", "cnn"],
        train_window_size=350,
        test_window_size=100,
        walk_forward_step=80,
        prediction_horizon=12,
        apply_decision_pipeline=False,
    )
    models = {k: _M(k) for k in cfg.model_keys}
    orch = TrainingOrchestrator(cfg)
    orch.initialize(
        models=models,
        regime_detector=_R(),
        meta_weighting=_meta(cfg),
        risk_manager=OrchestratorRiskBridge(),
    )
    out = orch.walk_forward_backtest(feat, y)
    assert len(out) >= 1
    assert "Sharpe Ratio" in out.columns
    assert "Walk-Forward Efficiency" in out.columns
    assert "Sortino Ratio" in out.columns
    assert "IS_Sharpe Ratio" in out.columns
