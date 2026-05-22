"""
Реальные OHLCV → признаки → обучение LGB+XGB → WFO-бэктест + проверка критериев.

Используется интеграционными тестами и CLI отчётом (без Mock-моделей).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from backtesting.criteria_evaluator import evaluate_backtest_levels, summarize_wfo_folds
from backtesting.metrics_config import load_backtesting_config
from backtesting.results_journal import BacktestResultsJournal
from feature_engineering.config import FeatureEngineeringConfig
from feature_engineering.feature_engine import FeatureEngine
from orchestration.benchmark_runner import (
    build_report,
    config_from_tuning_best,
    load_tuning_best_params,
    lgb_xgb_config,
    run_wfo_on_features,
)
from orchestration.orchestrator_config import OrchestratorConfig


def default_real_parquet(repo_root: Optional[Path] = None) -> Path:
    """Предпочитает полный BTC-USDT 1h; иначе demo parquet."""
    root = repo_root or Path(__file__).resolve().parent.parent
    btc = root / "data" / "ohlcv" / "BTC-USDT_1h.parquet"
    if btc.is_file():
        return btc
    return root / "data" / "ohlcv" / "demo_BTC-USDT_1h.parquet"


def default_demo_parquet(repo_root: Optional[Path] = None) -> Path:
    return default_real_parquet(repo_root)


def features_from_ohlcv_parquet(
    parquet_path: str | Path,
    *,
    max_rows: Optional[int] = None,
    use_feature_cache: bool = False,
    symbol: Optional[str] = None,
    timeframe: str = "1h",
    config_path: str | Path = "config/profiles/canonical_4model.yaml",
) -> pd.DataFrame:
    if use_feature_cache and symbol:
        from orchestration.feature_store import load_features_from_parquet

        return load_features_from_parquet(
            parquet_path,
            symbol=symbol,
            timeframe=timeframe,
            max_rows=max_rows,
            use_feature_cache=True,
            config_path=config_path,
        )
    raw = pd.read_parquet(parquet_path)
    fe = FeatureEngine(raw, FeatureEngineeringConfig())
    fe.add_indicators()
    df = fe.get_processed_data()
    if max_rows is not None and len(df) > max_rows:
        df = df.iloc[-max_rows:].copy()
    return df


def run_lightgbm_xgboost_wfo_backtest(
    features: pd.DataFrame,
    *,
    train_window_size: int = 900,
    test_window_size: int = 180,
    walk_forward_step: int = 360,
    prediction_horizon: int = 12,
    embargo_period: int = 3,
    apply_decision_pipeline: bool = False,
    orchestrator_config: Optional[OrchestratorConfig] = None,
    use_risk_bridge: bool = True,
) -> pd.DataFrame:
    cfg = orchestrator_config or lgb_xgb_config(
        train_window_size=train_window_size,
        test_window_size=test_window_size,
        walk_forward_step=walk_forward_step,
        prediction_horizon=prediction_horizon,
        embargo_period=embargo_period,
        apply_decision_pipeline=apply_decision_pipeline,
    )
    return run_wfo_on_features(features, cfg, use_risk_bridge=use_risk_bridge)


def run_benchmark_report(
    parquet_path: str | Path,
    *,
    max_rows: Optional[int] = None,
    config_path: str | Path = "config.yaml",
    orchestrator_config: Optional[OrchestratorConfig] = None,
    use_tuning_best: bool = False,
    use_risk_bridge: Optional[bool] = None,
    dl_epochs: int = 5,
    label: str = "report-real",
    symbol: Optional[str] = None,
    timeframe: Optional[str] = None,
    stage: Optional[str] = None,
    train_span: Optional[Any] = None,
    holdout_span: Optional[Any] = None,
    use_feature_cache: bool = False,
) -> Dict[str, Any]:
    feat = features_from_ohlcv_parquet(
        parquet_path,
        max_rows=max_rows,
        use_feature_cache=use_feature_cache,
        symbol=symbol,
        timeframe=timeframe or "1h",
        config_path=config_path,
    )
    tuning = (
        load_tuning_best_params(config_path, symbol=symbol, timeframe=timeframe)
        if use_tuning_best
        else {}
    )
    eval_config_path = config_path
    if tuning.get("config_path"):
        eval_config_path = tuning["config_path"]
    from orchestration.benchmark_runner import orchestrator_config_from_yaml

    cfg = orchestrator_config or (
        config_from_tuning_best(config_path, symbol=symbol, timeframe=timeframe)
        if use_tuning_best and tuning
        else orchestrator_config_from_yaml(config_path)
    )
    bridge = (
        use_risk_bridge
        if use_risk_bridge is not None
        else bool(tuning.get("use_risk_bridge", False))
    )
    regime = str(tuning.get("regime", "momentum"))
    params = {**tuning, **(cfg.to_dict() if hasattr(cfg, "to_dict") else {})}
    from orchestration.tuning_config import dl_epochs_from_params

    epochs = dl_epochs_from_params(tuning, default=dl_epochs)
    folds = run_wfo_on_features(
        feat,
        cfg,
        use_risk_bridge=bridge,
        regime=regime,
        dl_epochs=epochs,
        config_path=eval_config_path,
        tuning_params=tuning if use_tuning_best else None,
    )
    report = build_report(
        folds,
        parquet=str(Path(parquet_path).resolve()),
        feature_rows=len(feat),
        max_rows=max_rows,
        config_path=eval_config_path,
        label=label,
        params=params,
        symbol=symbol,
        timeframe=timeframe,
        stage=stage,
        train_span=train_span,
        holdout_span=holdout_span,
    )
    report["max_rows"] = max_rows
    return report


def run_demo_benchmark(
    parquet_path: Optional[Path] = None,
    *,
    repo_root: Optional[Path] = None,
    max_rows: Optional[int] = 2200,
) -> pd.DataFrame:
    pq = parquet_path or default_real_parquet(repo_root)
    report = run_benchmark_report(pq, max_rows=max_rows)
    return report["fold_metrics"]


def write_benchmark_artifacts(
    report: Dict[str, Any],
    *,
    json_out: Optional[Path] = None,
    journal: Optional[BacktestResultsJournal] = None,
    journal_label: str = "",
) -> Optional[str]:
    """Сериализация в JSON и опционально в журнал. Возвращает run_id журнала."""
    if json_out is not None:
        import json

        json_out = Path(json_out)
        json_out.parent.mkdir(parents=True, exist_ok=True)
        folds = report["fold_metrics"]
        payload = {
            "parquet": report["parquet"],
            "feature_rows": report["feature_rows"],
            "max_rows": report.get("max_rows"),
            "summary": report["summary"],
            "acceptance_passed": report["criteria"]["acceptance"]["passed"],
            "target_passed": report["criteria"]["target"]["passed"],
            "acceptance_checks": report["criteria"]["acceptance"]["checks"],
            "target_checks": report["criteria"]["target"]["checks"],
            "folds": folds.to_dict(orient="records") if hasattr(folds, "to_dict") else folds,
            "params": report.get("params"),
        }
        json_out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    if journal is not None:
        return journal.append_run(
            report,
            label=journal_label or report.get("label", ""),
            params=report.get("params"),
        )
    return None
