"""
Единая точка WFO-бэктеста на реальных признаках с произвольным ``OrchestratorConfig``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import yaml

from backtesting.criteria_evaluator import evaluate_backtest_levels, summarize_wfo_folds
from backtesting.metrics_config import load_backtesting_config
from orchestration import TrainingOrchestrator
from orchestration.glue import (
    MomentumRegimeDetector,
    SimpleHourlyRegimeStub,
    default_horizon_labels,
)
from orchestration.model_factory import build_orchestration_models, meta_weighting_from_config
from orchestration.orchestrator_config import OrchestratorConfig
from risk_management import OrchestratorRiskBridge


_ARCHIVE_TUNING_BEST = (
    Path(__file__).resolve().parents[1]
    / "config/archive/discussion/orchestration_tuning_best_lgb_xgb.yaml"
)


def load_tuning_best_params(
    config_path: str | Path = "config.yaml",
    *,
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
    include_archive: bool = True,
) -> Dict[str, Any]:
    """
  Archived 2-model profile (discussion only). Prefer canonical ``orchestration`` / ``config/profiles/canonical_4model.yaml``.

  Lookup order: per-symbol YAML → root ``orchestration_tuning_best`` → archive file.
    """
    if symbol and timeframe:
        from orchestration.symbols import tuning_best_for

        per_sym = tuning_best_for(symbol, timeframe)
        if per_sym:
            return per_sym
    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    block = raw.get("orchestration_tuning_best")
    if block:
        return dict(block)
    if include_archive and _ARCHIVE_TUNING_BEST.is_file():
        archived = yaml.safe_load(_ARCHIVE_TUNING_BEST.read_text(encoding="utf-8")) or {}
        return dict(archived.get("orchestration") or archived.get("orchestration_tuning_best") or {})
    return {}


def load_canonical_config_path(config_path: str | Path = "config.yaml") -> Path:
    """Prefer ``config/profiles/canonical_4model.yaml`` when present."""
    root = Path(config_path).resolve().parent
    canonical = root / "profiles" / "canonical_4model.yaml"
    if canonical.is_file():
        return canonical
    return Path(config_path)


def orchestrator_config_from_yaml(config_path: str | Path = "config.yaml") -> OrchestratorConfig:
    """Load 4-model canonical orchestrator config."""
    return OrchestratorConfig.from_yaml(load_canonical_config_path(config_path))


def regime_detector_for(name: str):
    if name == "momentum":
        return MomentumRegimeDetector()
    return SimpleHourlyRegimeStub()


def orchestrator_config_from_params(
    p: Dict[str, Any],
    *,
    config_path: str | Path = "config.yaml",
) -> OrchestratorConfig:
    from orchestration.tuning_config import orchestrator_config_from_tuning_params

    return orchestrator_config_from_tuning_params(
        p, config_path=p.get("config_path", config_path)
    )


def config_from_tuning_best(
    config_path: str | Path = "config.yaml",
    *,
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
) -> OrchestratorConfig:
    p = load_tuning_best_params(config_path, symbol=symbol, timeframe=timeframe)
    if not p:
        return lgb_xgb_config()
    required = (
        "train_window_size",
        "test_window_size",
        "walk_forward_step",
        "prediction_horizon",
        "direction_threshold",
    )
    for key in required:
        if key not in p:
            return lgb_xgb_config()
    return orchestrator_config_from_params(p, config_path=config_path)


def run_wfo_on_features(
    features: pd.DataFrame,
    cfg: OrchestratorConfig,
    *,
    use_risk_bridge: bool = True,
    regime: str = "momentum",
    dl_epochs: int = 5,
    config_path: str | Path = "config.yaml",
    tuning_params: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    from orchestration.tuning_loop import _fit_wfo_windows_to_rows

    if "close" not in features.columns:
        raise ValueError("features must include 'close'")
    cfg = _fit_wfo_windows_to_rows(cfg, len(features))
    tp = tuning_params or {}
    y = default_horizon_labels(
        features["close"],
        cfg.prediction_horizon,
        min_return=float(getattr(cfg, "label_min_return", 0.0) or 0.0),
    )
    level = str(tp.get("tune_level", "confirm"))
    models = build_orchestration_models(
        cfg,
        features,
        dl_epochs=dl_epochs,
        dl_batch_size=int(tp.get("dl_batch_size", 64)),
        mixed_precision=bool(tp.get("_mixed_precision", level != "confirm")),
        tune_level=level,
    )
    orch = TrainingOrchestrator(cfg)
    cap = float(getattr(cfg, "max_position_fraction", 1.0) or 1.0)
    risk = (
        OrchestratorRiskBridge(max_position_fraction=cap)
        if use_risk_bridge
        else None
    )
    orch.initialize(
        models=models,
        regime_detector=regime_detector_for(regime),
        meta_weighting=meta_weighting_from_config(cfg),
        risk_manager=risk,
    )
    return orch.walk_forward_backtest(features, y)


def build_report(
    folds: pd.DataFrame,
    *,
    parquet: str,
    feature_rows: int,
    max_rows: Optional[int],
    config_path: str | Path = "config.yaml",
    label: str = "",
    params: Optional[Dict[str, Any]] = None,
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
    stage: Optional[str] = None,
    train_span: Optional[Any] = None,
    holdout_span: Optional[Any] = None,
) -> Dict[str, Any]:
    bt_cfg = load_backtesting_config(config_path)
    levels = evaluate_backtest_levels(folds, bt_cfg)
    merged_params = dict(params or {})
    if "profile" not in merged_params and merged_params.get("config_path"):
        from orchestration.canonical_pipeline import journal_profile_tag

        merged_params.setdefault("profile", journal_profile_tag(merged_params["config_path"]))

    return {
        "parquet": parquet,
        "feature_rows": feature_rows,
        "max_rows": max_rows,
        "label": label,
        "profile": merged_params.get("profile"),
        "params": merged_params,
        "symbol": symbol,
        "timeframe": timeframe,
        "stage": stage,
        "train_span": train_span,
        "holdout_span": holdout_span,
        "fold_metrics": folds,
        "summary": summarize_wfo_folds(folds),
        "criteria": levels,
    }


def lgb_xgb_config(
    *,
    train_window_size: int = 900,
    test_window_size: int = 180,
    walk_forward_step: int = 360,
    prediction_horizon: int = 12,
    embargo_period: int = 3,
    direction_threshold: float = 0.52,
    apply_decision_pipeline: bool = True,
    meta_threshold_mode: str = "median",
    ensemble_mode: str = "regime_adaptive",
    min_signal_margin: float = 0.0,
    trade_mode: str = "both",
) -> OrchestratorConfig:
    mk = ["lgb", "xgb"]
    w = {k: 0.5 for k in mk}
    cfg = OrchestratorConfig(
        model_keys=mk,
        trend_weights=w.copy(),
        range_weights=w.copy(),
        breakout_weights=w.copy(),
        train_window_size=train_window_size,
        test_window_size=test_window_size,
        walk_forward_step=walk_forward_step,
        prediction_horizon=prediction_horizon,
        embargo_period=embargo_period,
        direction_threshold=direction_threshold,
        apply_decision_pipeline=apply_decision_pipeline,
        meta_threshold_mode=meta_threshold_mode,
        ensemble_mode=ensemble_mode,
    )
    cfg.trade_mode = trade_mode
    cfg.min_signal_margin = min_signal_margin
    return cfg
