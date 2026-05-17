"""
Smoke / integration: synthetic features → TrainingOrchestrator → WFO backtest.

Запуск из корня репозитория: ``python -m orchestration``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any

from .training_orchestrator import TrainingOrchestrator
from .orchestrator_config import OrchestratorConfig
from meta_learning.dynamic_meta import DynamicMetaWeighting
from risk_management import OrchestratorRiskBridge


def _meta_matching_config(cfg: OrchestratorConfig) -> DynamicMetaWeighting:
    w = {k: 1.0 / len(cfg.model_keys) for k in cfg.model_keys}
    return DynamicMetaWeighting(trend_weights=w.copy(), range_weights=w.copy())


class SmokeModel:
    def __init__(self, model_key: str, base: float = 0.53):
        self.model_key = model_key
        self.base = base

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        n = len(features)
        h = hash(self.model_key) % 7
        return np.full(n, self.base + h * 0.0012)

    def fit(self, features: pd.DataFrame, targets: pd.Series) -> None:
        return None


class SmokeRegime:
    def get_regime_info(self, features: pd.DataFrame) -> Dict[str, Any]:
        n = len(features)
        regime_pred = np.array([1 if i % 30 < 15 else 0 for i in range(n)])
        return {
            "market_regime": "trend",
            "regime_pred": regime_pred,
            "trade_allowed": True,
        }


def build_synthetic_features(n: int = 800, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 + rng.standard_normal(n).cumsum() * 0.15
    vol = rng.uniform(0.01, 0.04, n)
    return pd.DataFrame(
        {
            "close": close,
            "volume": rng.integers(1000, 5000, n),
            "volatility": vol,
            "atr": (vol * close).clip(0.01, None),
            "rsi": rng.uniform(25, 75, n),
        },
        index=pd.date_range("2024-01-01", periods=n, freq="h"),
    )


def build_targets(features: pd.DataFrame, horizon: int = 12) -> pd.Series:
    fwd = features["close"].shift(-horizon)
    y = (fwd > features["close"]).astype(float)
    y.iloc[-horizon:] = np.nan
    return y


def run_smoke_wfo() -> pd.DataFrame:
    feat = build_synthetic_features()
    tgt = build_targets(feat)
    cfg = OrchestratorConfig(
        model_keys=["lgb", "gru", "xgb", "cnn"],
        train_window_size=400,
        test_window_size=120,
        walk_forward_step=80,
        prediction_horizon=12,
        apply_decision_pipeline=False,
    )
    models = {k: SmokeModel(k) for k in cfg.model_keys}
    orch = TrainingOrchestrator(cfg)
    orch.initialize(
        models=models,
        regime_detector=SmokeRegime(),
        meta_weighting=_meta_matching_config(cfg),
        risk_manager=OrchestratorRiskBridge(),
    )
    return orch.walk_forward_backtest(feat, tgt)


def main() -> None:
    df = run_smoke_wfo()
    print("WFO backtest smoke OK, folds:", len(df))
    cols = ["Fold", "Sharpe Ratio", "Total Return (%)", "Max Drawdown (%)"]
    print(df[[c for c in cols if c in df.columns]].to_string(index=False))


if __name__ == "__main__":
    main()
