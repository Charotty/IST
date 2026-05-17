"""
Подбор параметров WFO на реальных OHLCV до прохождения acceptance (и target).

Каждый прогон пишется в ``BacktestResultsJournal``.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from backtesting.results_journal import BacktestResultsJournal
from orchestration.benchmark_runner import (
    build_report,
    load_tuning_best_params,
    orchestrator_config_from_params,
    regime_detector_for,
)
from orchestration.glue import default_horizon_labels
from orchestration.real_data_benchmark import default_real_parquet, features_from_ohlcv_parquet


def _score_report(report: Dict[str, Any]) -> Tuple:
    acc = report["criteria"]["acceptance"]
    tgt = report["criteria"]["target"]
    s = report["summary"]
    n_acc = sum(1 for c in acc["checks"] if c["passed"])
    n_tgt = sum(1 for c in tgt["checks"] if c["passed"])
    return (
        1 if acc["passed"] else 0,
        1 if tgt["passed"] else 0,
        n_acc,
        n_tgt,
        float(s.get("mean_sharpe") or -1e9),
        float(s.get("mean_profit_factor") or 0),
        float(s.get("mean_wfe") or 0),
        float(s.get("mean_total_return_pct") or -1e9),
    )


def _baseline_defaults(
    config_path: str | Path = "config.yaml",
    *,
    canonical: bool = True,
) -> Dict[str, Any]:
    """Defaults for symbol tune/holdout; canonical path uses 4 models from profile YAML."""
    if canonical:
        from orchestration.benchmark_runner import orchestrator_config_from_yaml

        cfg = orchestrator_config_from_yaml(config_path)
        return {
            "regime": "momentum",
            "trade_mode": cfg.trade_mode,
            "use_risk_bridge": bool(getattr(cfg, "use_risk_bridge", True)),
            "prediction_horizon": cfg.prediction_horizon,
            "direction_threshold": cfg.direction_threshold,
            "apply_decision_pipeline": cfg.apply_decision_pipeline,
            "train_window_size": cfg.train_window_size,
            "meta_threshold_mode": cfg.meta_threshold_mode,
            "ensemble_mode": cfg.ensemble_mode,
            "test_window_size": cfg.test_window_size,
            "walk_forward_step": cfg.walk_forward_step,
            "min_signal_margin": cfg.min_signal_margin,
            "signal_strategy": cfg.signal_strategy,
            "label_min_return": float(getattr(cfg, "label_min_return", 0.0) or 0.0),
            "model_keys": list(cfg.model_keys),
            "volatility_filter_percentile": 0.0,
            "max_position_fraction": 1.0,
            "config_path": str(config_path),
            "profile": "canonical_4model",
        }
    return {
        "regime": "momentum",
        "trade_mode": "both",
        "use_risk_bridge": False,
        "prediction_horizon": 24,
        "direction_threshold": 0.6,
        "apply_decision_pipeline": True,
        "train_window_size": 1500,
        "meta_threshold_mode": "percentile",
        "ensemble_mode": "fixed_range",
        "test_window_size": 180,
        "walk_forward_step": 360,
        "min_signal_margin": 0.08,
        "signal_strategy": "ensemble",
        "label_min_return": 0.0,
        "model_keys": ["lgb", "xgb"],
        "volatility_filter_percentile": 0.0,
        "max_position_fraction": 1.0,
    }


def iter_refine_baseline_candidates(
    baseline: Optional[Dict[str, Any]] = None,
    *,
    canonical: bool = True,
) -> Iterator[Dict[str, Any]]:
    """Локальный перебор ±2–3 параметра вокруг зафиксированного baseline."""
    config_path = (baseline or {}).get("config_path", "config.yaml")
    base = {**_baseline_defaults(config_path, canonical=canonical), **(baseline or {})}
    use_rb = bool(base.get("use_risk_bridge", False))
    yield {**base, "use_risk_bridge": use_rb}

    sweeps: Dict[str, List[Any]] = {
        "min_signal_margin": [0.06, 0.07, 0.08, 0.09, 0.10, 0.11],
        "direction_threshold": [0.56, 0.58, 0.60, 0.62],
        "label_min_return": [0.0, 0.003, 0.004, 0.005],
        "prediction_horizon": [12, 24],
        "signal_strategy": ["ensemble", "momentum_confirm"],
        "trade_mode": ["long_only", "both"],
        "volatility_filter_percentile": [0.0, 90.0],
        "max_position_fraction": [1.0, 0.5, 0.35],
        "test_window_size": [160, 180, 200],
        "walk_forward_step": [300, 330, 360],
        **(
            {}
            if canonical and len(base.get("model_keys", [])) >= 4
            else {"model_keys": [["lgb", "xgb"], ["lgb"]]}
        ),
    }
    seen: set = set()

    def _key(p: Dict[str, Any]) -> str:
        return repr(sorted((k, v) for k, v in p.items() if k != "use_risk_bridge"))

    for name, values in sweeps.items():
        for val in values:
            if base.get(name) == val:
                continue
            p = {**base, name: val, "use_risk_bridge": base.get("use_risk_bridge", False)}
            k = _key(p)
            if k not in seen:
                seen.add(k)
                yield p

    combos = [
        dict(
            min_signal_margin=0.09,
            label_min_return=0.004,
            volatility_filter_percentile=90.0,
            max_position_fraction=0.5,
            signal_strategy="momentum_confirm",
        ),
        dict(
            min_signal_margin=0.10,
            label_min_return=0.003,
            volatility_filter_percentile=90.0,
            trade_mode="long_only",
        ),
        dict(
            min_signal_margin=0.08,
            label_min_return=0.005,
            max_position_fraction=0.35,
            direction_threshold=0.58,
        ),
        dict(
            min_signal_margin=0.07,
            label_min_return=0.003,
            volatility_filter_percentile=90.0,
            max_position_fraction=0.35,
            signal_strategy="momentum_confirm",
            trade_mode="long_only",
        ),
    ]
    for patch in combos:
        p = {**base, **patch, "use_risk_bridge": base.get("use_risk_bridge", False)}
        k = _key(p)
        if k not in seen:
            seen.add(k)
            yield p


def iter_tuning_candidates() -> Iterator[Dict[str, Any]]:
    """Упорядоченный перебор: сначала наиболее перспективные комбинации."""
    regimes = ("momentum", "stub")
    trade_modes = ("long_only", "both")
    use_risk = (False, True)
    horizons = (12, 24, 8)
    d_thresh = (0.55, 0.58, 0.52, 0.60)
    apply_dp = (True, False)
    train_windows = (1200, 900, 1500)
    meta_modes = ("median", "percentile")
    ensemble_modes = ("regime_adaptive", "fixed_range")
    test_windows = (180, 240)
    steps = (360, 300)

    margins = (0.0, 0.03, 0.05, 0.08)
    priority = []
    for strat in ("momentum_confirm", "ensemble"):
        for min_ret in (0.0, 0.003, 0.005):
            for mk in (["lgb", "xgb"], ["lgb"]):
                priority.append(
                    dict(
                        regime="momentum",
                        trade_mode="long_only",
                        use_risk_bridge=False,
                        prediction_horizon=24,
                        direction_threshold=0.54,
                        apply_decision_pipeline=False,
                        train_window_size=1500,
                        meta_threshold_mode="median",
                        ensemble_mode="fixed_range",
                        test_window_size=200,
                        walk_forward_step=280,
                        min_signal_margin=0.05,
                        signal_strategy=strat,
                        momentum_sma_period=80,
                        label_min_return=min_ret,
                        model_keys=mk,
                    )
                )
    for margin in margins:
        priority.append(
            dict(
                regime="momentum",
                trade_mode="long_only",
                use_risk_bridge=False,
                prediction_horizon=24,
                direction_threshold=0.56,
                apply_decision_pipeline=True,
                train_window_size=1500,
                meta_threshold_mode="percentile",
                ensemble_mode="fixed_range",
                test_window_size=240,
                walk_forward_step=300,
                min_signal_margin=margin,
            )
        )
    priority.extend(
        [
            dict(
                regime="momentum",
                trade_mode="long_only",
                use_risk_bridge=False,
                prediction_horizon=12,
                direction_threshold=0.58,
                apply_decision_pipeline=True,
                train_window_size=1200,
                meta_threshold_mode="median",
                ensemble_mode="regime_adaptive",
                test_window_size=180,
                walk_forward_step=360,
                min_signal_margin=0.05,
            ),
            dict(
                regime="momentum",
                trade_mode="both",
                use_risk_bridge=False,
                prediction_horizon=24,
                direction_threshold=0.60,
                apply_decision_pipeline=True,
                train_window_size=1500,
                meta_threshold_mode="percentile",
                ensemble_mode="fixed_range",
                test_window_size=180,
                walk_forward_step=360,
                min_signal_margin=0.08,
            ),
        ]
    )
    for p in priority:
        yield p

    for combo in itertools.product(
        regimes,
        trade_modes,
        use_risk,
        horizons,
        d_thresh,
        apply_dp,
        train_windows,
        meta_modes,
        ensemble_modes,
        test_windows,
        steps,
    ):
        yield {
            "regime": combo[0],
            "trade_mode": combo[1],
            "use_risk_bridge": combo[2],
            "prediction_horizon": combo[3],
            "direction_threshold": combo[4],
            "apply_decision_pipeline": combo[5],
            "train_window_size": combo[6],
            "meta_threshold_mode": combo[7],
            "ensemble_mode": combo[8],
            "test_window_size": combo[9],
            "walk_forward_step": combo[10],
            "min_signal_margin": 0.0,
        }


def _fit_wfo_windows_to_rows(cfg, n_rows: int):
    """Shrink train/test/step when the feature matrix is shorter than canonical WFO windows."""
    from dataclasses import replace

    need = cfg.train_window_size + cfg.test_window_size + int(getattr(cfg, "embargo_period", 0) or 0) + 20
    if n_rows >= need:
        return cfg
    train = max(200, int(n_rows * 0.55))
    test = max(40, int(n_rows * 0.18))
    step = max(40, test)
    return replace(
        cfg,
        train_window_size=train,
        test_window_size=test,
        walk_forward_step=step,
    )


def run_single_trial(
    features,
    params: Dict[str, Any],
    *,
    config_path: str | Path = "config.yaml",
    parquet_label: str = "",
) -> Dict[str, Any]:
    cfg = _fit_wfo_windows_to_rows(orchestrator_config_from_params(params), len(features))
    from orchestration import TrainingOrchestrator
    from orchestration.model_factory import build_orchestration_models, meta_weighting_from_config
    from risk_management import OrchestratorRiskBridge

    y = default_horizon_labels(
        features["close"],
        cfg.prediction_horizon,
        min_return=float(getattr(cfg, "label_min_return", 0.0) or 0.0),
    )
    models = build_orchestration_models(cfg, features, dl_epochs=5)
    orch = TrainingOrchestrator(cfg)
    cap = float(params.get("max_position_fraction", getattr(cfg, "max_position_fraction", 1.0)) or 1.0)
    risk = (
        OrchestratorRiskBridge(max_position_fraction=cap)
        if params["use_risk_bridge"]
        else None
    )
    orch.initialize(
        models=models,
        regime_detector=regime_detector_for(params["regime"]),
        meta_weighting=meta_weighting_from_config(cfg),
        risk_manager=risk,
    )
    folds = orch.walk_forward_backtest(features, y)
    from orchestration.canonical_pipeline import journal_profile_tag

    report_params = {k: v for k, v in params.items() if not k.startswith("_")}
    report_params.setdefault("profile", journal_profile_tag(config_path))
    report_params.setdefault("config_path", str(config_path))
    return build_report(
        folds,
        parquet=parquet_label,
        feature_rows=len(features),
        max_rows=None,
        config_path=config_path,
        label=params.get("_label", ""),
        params=report_params,
    )


def tune_until_criteria(
    parquet_path: Optional[Path] = None,
    *,
    max_rows: Optional[int] = 8000,
    max_trials: int = 80,
    stop_on_target: bool = True,
    mode: str = "grid",
    journal_root: str | Path = "docs/backtest_journal",
    config_path: str | Path = "config.yaml",
) -> Dict[str, Any]:
    """
    Перебор кандидатов до acceptance (обязательно) и опционально target.

    ``mode``: ``grid`` (широкий) | ``refine`` (локально вокруг orchestration_tuning_best).
    """
    pq = Path(parquet_path or default_real_parquet())
    features = features_from_ohlcv_parquet(pq, max_rows=max_rows)
    journal = BacktestResultsJournal(journal_root)

    if mode == "refine":
        baseline = load_tuning_best_params(config_path) or _baseline_defaults()
        candidates = iter_refine_baseline_candidates(baseline)
        label_prefix = "refine"
    else:
        candidates = iter_tuning_candidates()
        label_prefix = "trial"

    best_report: Optional[Dict[str, Any]] = None
    best_params: Optional[Dict[str, Any]] = None
    best_score: Tuple = (-1,)
    run_ids: List[str] = []
    acceptance_hit = False
    target_hit = False

    for trial_idx, params in enumerate(candidates):
        if trial_idx >= max_trials:
            break
        params = dict(params)
        params["_label"] = f"{label_prefix}_{trial_idx + 1}"
        try:
            report = run_single_trial(
                features,
                params,
                config_path=config_path,
                parquet_label=str(pq.resolve()),
            )
        except Exception as exc:
            journal.append_run(
                {
                    "parquet": str(pq),
                    "feature_rows": len(features),
                    "max_rows": max_rows,
                    "summary": {},
                    "criteria": {
                        "acceptance": {"passed": False, "checks": []},
                        "target": {"passed": False, "checks": []},
                    },
                    "fold_metrics": [],
                },
                label=f"{label_prefix}_{trial_idx + 1}_ERROR",
                params=params,
                notes=str(exc),
            )
            continue

        report["max_rows"] = max_rows
        rid = journal.append_run(
            report,
            label=params["_label"],
            params={k: v for k, v in params.items() if not k.startswith("_")},
        )
        run_ids.append(rid)

        sc = _score_report(report)
        if sc > best_score:
            best_score = sc
            best_report = report
            best_params = params

        if report["criteria"]["acceptance"]["passed"]:
            acceptance_hit = True
            if not stop_on_target:
                break
        if report["criteria"]["target"]["passed"]:
            target_hit = True
            if stop_on_target:
                break

    return {
        "parquet": str(pq.resolve()),
        "feature_rows": len(features),
        "max_rows": max_rows,
        "mode": mode,
        "trials_run": len(run_ids),
        "acceptance_achieved": acceptance_hit,
        "target_achieved": target_hit,
        "best_score": best_score,
        "best_params": {k: v for k, v in (best_params or {}).items() if not k.startswith("_")},
        "best_report": best_report,
        "run_ids": run_ids,
        "journal_root": str(Path(journal_root).resolve()),
    }


def apply_best_params_to_config_yaml(
    best_params: Dict[str, Any],
    yaml_path: Path = Path("config.yaml"),
) -> None:
    """Записывает лучшие orchestration-параметры в config.yaml (merge)."""
    import yaml

    path = Path(yaml_path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    orch = raw.setdefault("orchestration", {})
    for key in (
        "train_window_size",
        "test_window_size",
        "walk_forward_step",
        "prediction_horizon",
        "direction_threshold",
        "apply_decision_pipeline",
        "meta_threshold_mode",
        "ensemble_mode",
        "trade_mode",
        "embargo_period",
        "min_signal_margin",
        "signal_strategy",
        "label_min_return",
        "volatility_filter_percentile",
        "max_position_fraction",
    ):
        if key in best_params:
            orch[key] = best_params[key]
    tuned = {k: v for k, v in best_params.items() if k != "use_risk_bridge"}
    tuned["use_risk_bridge"] = best_params.get("use_risk_bridge", False)
    raw["orchestration_tuning_best"] = tuned
    path.write_text(yaml.dump(raw, default_flow_style=False, allow_unicode=True), encoding="utf-8")
