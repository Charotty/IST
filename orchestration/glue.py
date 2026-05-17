"""
Сквозная сборка: parquet → признаки (опционально) → оркестратор → WFO-бэктест.

Не заменяет data_layer/OKX; для live используйте ``execution.paper_loop``.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

from feature_engineering.config import FeatureEngineeringConfig
from feature_engineering.feature_manager import FeatureManager
from orchestration import TrainingOrchestrator
from orchestration.orchestrator_config import OrchestratorConfig
from orchestration.config_validate import raise_if_invalid
from orchestration.model_factory import (
    build_orchestration_models,
    infer_training_feature_columns,
    meta_weighting_from_config,
)
from risk_management import OrchestratorRiskBridge


def load_feature_table_from_parquet(
    parquet_path: str | Path,
    *,
    config_yaml: Optional[Path] = None,
    raw_ohlcv: bool = False,
) -> pd.DataFrame:
    """
    Читает parquet. Если ``raw_ohlcv=True``, прогоняет ``FeatureManager`` (конфиг из YAML при переданном пути).
    Иначе ожидаются уже готовые признаки + ``close``.
    """
    p = Path(parquet_path)
    df = pd.read_parquet(p)
    if raw_ohlcv:
        fe = FeatureEngineeringConfig.from_yaml(config_yaml) if config_yaml else FeatureEngineeringConfig()
        df = FeatureManager(fe).transform(df)
    return df


def default_horizon_labels(
    close: pd.Series,
    horizon: int,
    min_return: float = 0.0,
) -> pd.Series:
    fwd = close.shift(-horizon)
    fwd_ret = (fwd / close) - 1.0
    if min_return and min_return > 0:
        y = (fwd_ret > min_return).astype(float)
    else:
        y = (fwd > close).astype(float)
    y.iloc[-horizon:] = np.nan
    return y


class SimpleHourlyRegimeStub:
    """Лёгкий regime_pred для CLI/smoke без обучения RegimeDetector."""

    def get_regime_info(self, features: pd.DataFrame) -> Dict[str, Any]:
        n = len(features)
        r = np.array([1 if i % 20 < 10 else 0 for i in range(n)])
        return {
            "market_regime": "trend",
            "regime_pred": r,
            "trade_allowed": True,
        }


class MomentumRegimeDetector:
    """Режим по SMA(close): только прошлые данные, без ML."""

    def __init__(self, sma_period: int = 50):
        self.sma_period = sma_period

    def get_regime_info(self, features: pd.DataFrame) -> Dict[str, Any]:
        close = features["close"].astype(float)
        sma = close.rolling(self.sma_period, min_periods=1).mean()
        r = (close > sma).astype(int).to_numpy()
        return {
            "market_regime": "trend" if int(r[-1]) == 1 else "range",
            "regime_pred": r,
            "trade_allowed": True,
        }


def _orchestrator_config_for_parquet_wfo(
    config_yaml: Path,
    *,
    validate_config: bool,
    light_only: bool,
    orchestrator_config_override: Optional[OrchestratorConfig] = None,
) -> OrchestratorConfig:
    if orchestrator_config_override is not None:
        cfg = orchestrator_config_override
        if validate_config:
            raise_if_invalid(config_yaml)
        return cfg
    if validate_config:
        raise_if_invalid(config_yaml)
    cfg = OrchestratorConfig.from_yaml(config_yaml)
    if light_only:
        mk = ["lgb", "xgb"]
        w = {k: 1.0 / len(mk) for k in mk}
        cfg = replace(
            cfg,
            model_keys=mk,
            trend_weights=w.copy(),
            range_weights=w.copy(),
            breakout_weights=w.copy(),
        )
    return cfg


def run_wfo_backtest_on_feature_table(
    df: pd.DataFrame,
    cfg: OrchestratorConfig,
    *,
    dl_epochs: int = 3,
) -> pd.DataFrame:
    """WFO на готовой таблице признаков (без повторной загрузки parquet)."""
    if "close" not in df.columns:
        raise ValueError("DataFrame must contain 'close'")
    y = default_horizon_labels(df["close"], cfg.prediction_horizon)
    models = build_orchestration_models(cfg, df, dl_epochs=dl_epochs)
    orch = TrainingOrchestrator(cfg)
    orch.initialize(
        models=models,
        regime_detector=MomentumRegimeDetector(),
        meta_weighting=meta_weighting_from_config(cfg),
        risk_manager=OrchestratorRiskBridge(),
    )
    return orch.walk_forward_backtest(df, y)


def run_wfo_backtest_from_parquet(
    parquet_path: str | Path,
    config_yaml: Path,
    *,
    raw_ohlcv: bool = False,
    validate_config: bool = True,
    dl_epochs: int = 3,
    light_only: bool = False,
    orchestrator_config_override: Optional[OrchestratorConfig] = None,
) -> pd.DataFrame:
    """
    Полный контур: валидация конфига → загрузка фич → фабрика моделей → ``walk_forward_backtest``.

    ``light_only=True`` — только ``lgb`` + ``xgb`` (без TensorFlow). Иначе — полный набор из
    ``config.model_keys`` (нужен tensorflow для ``gru``/``cnn``).

    ``orchestrator_config_override`` — подмена конфига (например меньшие окна в тестах).
    """
    cfg = _orchestrator_config_for_parquet_wfo(
        config_yaml,
        validate_config=validate_config,
        light_only=light_only,
        orchestrator_config_override=orchestrator_config_override,
    )
    df = load_feature_table_from_parquet(
        parquet_path, config_yaml=config_yaml if raw_ohlcv else None, raw_ohlcv=raw_ohlcv
    )
    return run_wfo_backtest_on_feature_table(df, cfg, dl_epochs=dl_epochs)


def run_from_parquet_report(
    parquet_path: Union[str, Path],
    config_yaml: Union[str, Path],
    *,
    raw_ohlcv: bool = False,
    validate_config: bool = True,
    dl_epochs: int = 3,
    light_only: bool = False,
    orchestrator_config_override: Optional[OrchestratorConfig] = None,
) -> Dict[str, Any]:
    """Как ``run_wfo_backtest_from_parquet``, но с ``fold_metrics``, ``summary`` и ``criteria``."""
    from orchestration.benchmark_runner import build_report

    config_path = Path(config_yaml)
    p = Path(parquet_path)
    cfg = _orchestrator_config_for_parquet_wfo(
        config_path,
        validate_config=validate_config,
        light_only=light_only,
        orchestrator_config_override=orchestrator_config_override,
    )
    df = load_feature_table_from_parquet(
        p, config_yaml=config_path if raw_ohlcv else None, raw_ohlcv=raw_ohlcv
    )
    folds = run_wfo_backtest_on_feature_table(df, cfg, dl_epochs=dl_epochs)
    from orchestration.canonical_pipeline import journal_profile_tag

    label = "from-parquet-light" if light_only else "from-parquet-full"
    params = {
        "full_models": not light_only,
        "dl_epochs": dl_epochs,
        "raw_ohlcv": raw_ohlcv,
        "model_keys": list(cfg.model_keys),
        "config_path": str(config_path.resolve()),
        "profile": journal_profile_tag(config_path),
    }
    return build_report(
        folds,
        parquet=str(p.resolve()),
        feature_rows=len(df),
        max_rows=None,
        config_path=config_path,
        label=label,
        params=params,
        stage="wfo-full-models" if not light_only else "wfo-light",
    )


def inference_stack_from_bundle(bundle_dir: str | Path):
    """Загрузка bundle; верните и передайте в ``InferenceOrchestrator.initialize``."""
    from orchestration.artifact_bundle import load_orchestrator_bundle
    from orchestration import InferenceOrchestrator

    cfg, models, regime, feat_cols, train_thr, _ = load_orchestrator_bundle(bundle_dir)
    orch = InferenceOrchestrator(cfg)
    meta = meta_weighting_from_config(cfg)
    return {
        "orchestrator": orch,
        "config": cfg,
        "models": models,
        "regime_detector": regime,
        "meta_weighting": meta,
        "feature_columns": feat_cols,
        "train_meta_threshold": train_thr,
    }
