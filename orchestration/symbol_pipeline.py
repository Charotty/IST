"""
High-level пер-символьный пайплайн для GUI/CLI.

Контракт:

1. ``download_ohlcv`` — OHLCV → ``data/ohlcv/<slug>.parquet`` (через ``data_layer``).
2. ``build_features`` — parquet → DataFrame с признаками (``FeatureEngine``).
3. ``tune_for_symbol`` — refine WFO на ``train_span``, лучший конфиг → ``config/symbols/<slug>.yaml``.
4. ``holdout_for_symbol`` — один WFO-прогон на ``holdout_span`` с зафиксированными параметрами.
5. ``train_final_for_symbol`` — fit на ВСЕХ доступных данных и сохранение
   ``orchestration.artifact_bundle`` в ``artifacts/<slug>/<run_id>/``.
6. ``write_symbol_manifest`` — сводный manifest + указатель ``LATEST.txt``.

GUI/CLI-обёртка ``prepare_symbol`` вызывает эти шаги последовательно и пишет
прогресс в ``artifacts/<slug>/active/<task_id>.json``.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from backtesting.results_journal import BacktestResultsJournal
from feature_engineering.config import FeatureEngineeringConfig
from feature_engineering.feature_engine import FeatureEngine

from .benchmark_runner import (
    orchestrator_config_from_params,
    regime_detector_for,
)
from .glue import default_horizon_labels
from .model_factory import (
    build_orchestration_models,
    infer_training_feature_columns,
    meta_weighting_from_config,
)
from .symbols import SymbolPaths, paths_for, write_symbol_config
from .tuning_loop import _baseline_defaults, run_single_trial
from .artifact_bundle import save_orchestrator_bundle


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]


# backwards-compat aliases (used internally and in CLI)
_utc_now_iso = utc_now_iso
_new_run_id = new_run_id


# ---------------------------------------------------------------------------
# 1. Data ingestion
# ---------------------------------------------------------------------------

def download_ohlcv(
    symbol: str,
    timeframe: str = "1h",
    *,
    start_date: str = "2022-01-01 00:00:00",
    end_date: Optional[str] = None,
    rate_limit: bool = True,
    verbose: bool = True,
    config_path: Optional[str | Path] = None,
    merge_mtf: bool = True,
) -> Path:
    """Скачивает OHLCV в ``data/ohlcv/<slug>.parquet`` (с MTF при ``config_path`` + ``merge_mtf``)."""
    if config_path is not None:
        from orchestration.canonical_pipeline import download_canonical_ohlcv

        return download_canonical_ohlcv(
            symbol,
            timeframe,
            config_path=config_path,
            rate_limit=rate_limit,
            verbose=verbose,
            merge_mtf=merge_mtf,
        )

    from data_layer.loaders.okx_ohlcv_loader import OKXDataLoader
    from data_layer.storage import save_ohlcv

    sp = paths_for(symbol, timeframe)
    end_date = end_date or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    sp.parquet.parent.mkdir(parents=True, exist_ok=True)

    loader = OKXDataLoader(rate_limit=rate_limit)
    df = loader.fetch_all_ohlcv(
        sp.symbol.replace("-", "/"),
        timeframe,
        start_date,
        end_date,
        verbose=verbose,
    )
    if df.empty:
        raise RuntimeError(f"No OHLCV downloaded for {sp.symbol} {timeframe}")
    save_ohlcv(df, sp.parquet, fmt="parquet")
    return sp.parquet


# ---------------------------------------------------------------------------
# 2. Feature build
# ---------------------------------------------------------------------------

def build_features(
    parquet_path: Path,
    *,
    config_path: Optional[str | Path] = None,
    save_to: Optional[Path] = None,
) -> pd.DataFrame:
    if config_path is not None:
        from orchestration.canonical_pipeline import build_canonical_features

        return build_canonical_features(
            parquet_path,
            config_path=config_path,
            output_path=save_to,
        )
    raw = pd.read_parquet(parquet_path)
    fe = FeatureEngine(raw, FeatureEngineeringConfig())
    fe.add_indicators()
    return fe.get_processed_data()


def split_train_holdout(
    features: pd.DataFrame,
    holdout_fraction: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Хронологический split: первые (1-h)% — train, последние h% — holdout."""
    if not 0.05 <= holdout_fraction <= 0.5:
        raise ValueError("holdout_fraction must be in [0.05, 0.5]")
    n = len(features)
    if n < 200:
        raise ValueError(f"Not enough rows for split: {n}")
    cut = int(n * (1.0 - holdout_fraction))
    return features.iloc[:cut].copy(), features.iloc[cut:].copy()


def _merge_symbol_tuning_overrides(
    params: Dict[str, Any],
    sp: SymbolPaths,
    *,
    allow_two_model_override: bool = False,
) -> Dict[str, Any]:
    """Эталон + overrides (``tuning_best_for``); не понижать ``model_keys`` до 2 без флага."""
    from orchestration.symbols import tuning_best_for

    over = dict(tuning_best_for(sp.symbol, sp.timeframe) or {})
    if not over:
        return params
    if not allow_two_model_override:
        mk = over.get("model_keys")
        if isinstance(mk, list) and len(mk) < 4 and set(mk) <= {"lgb", "xgb"}:
            over.pop("model_keys", None)
    return {**params, **over}


def _span_of(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {"start": None, "end": None, "rows": 0}
    return {
        "start": str(df.index[0]),
        "end": str(df.index[-1]),
        "rows": int(len(df)),
    }


# ---------------------------------------------------------------------------
# 3. Tuning per symbol
# ---------------------------------------------------------------------------

def tune_for_symbol(
    sp: SymbolPaths,
    train_features: pd.DataFrame,
    *,
    max_trials: int = 30,
    journal: Optional[BacktestResultsJournal] = None,
    on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
    config_path: str | Path = "config.yaml",
    allow_two_model_override: bool = False,
) -> Dict[str, Any]:
    """
    Локальный refine WFO на ``train_features``.

    Возвращает dict: ``{best_params, best_summary, run_ids, trials_run}``.
    """
    from .tuning_loop import iter_refine_baseline_candidates, _score_report

    from orchestration.ml_deps import require_tensorflow

    journal = journal or BacktestResultsJournal()
    baseline = _baseline_defaults(config_path, canonical=True)
    require_tensorflow(baseline.get("model_keys"))
    candidates = iter_refine_baseline_candidates(baseline, canonical=True)

    best_score: Tuple = (-1,)
    best_params: Optional[Dict[str, Any]] = None
    best_report: Optional[Dict[str, Any]] = None
    run_ids: List[str] = []

    parquet_label = str(sp.parquet.resolve())

    for i, params in enumerate(candidates, 1):
        if i > max_trials:
            break
        params = dict(params)
        params["_label"] = f"tune_{i}"
        try:
            report = run_single_trial(
                train_features,
                params,
                parquet_label=parquet_label,
            )
        except Exception as exc:
            if on_progress:
                on_progress({"step": "tune", "trial": i, "error": str(exc)})
            continue
        report["max_rows"] = len(train_features)
        report["symbol"] = sp.symbol
        report["timeframe"] = sp.timeframe
        report["stage"] = "tune"
        report["train_span"] = _span_of(train_features)
        rid = journal.append_run(
            report,
            label=f"{sp.slug}/tune_{i}",
            params={k: v for k, v in params.items() if not k.startswith("_")},
        )
        run_ids.append(rid)

        sc = _score_report(report)
        if sc > best_score:
            best_score = sc
            best_params = {k: v for k, v in params.items() if not k.startswith("_")}
            best_report = report
        if on_progress:
            s = report["summary"]
            on_progress({
                "step": "tune",
                "trial": i,
                "max_trials": max_trials,
                "sharpe": s.get("mean_sharpe"),
                "pf": s.get("mean_profit_factor"),
                "wfe": s.get("mean_wfe"),
                "acceptance": report["criteria"]["acceptance"]["passed"],
                "run_id": rid,
            })
        if report["criteria"]["acceptance"]["passed"] and report["criteria"]["target"]["passed"]:
            break

    if best_params:
        write_symbol_config(
            sp.symbol,
            sp.timeframe,
            tuning_best=best_params,
            allow_two_model_override=allow_two_model_override,
        )
    return {
        "best_params": best_params or {},
        "best_summary": (best_report or {}).get("summary", {}),
        "best_run_id": run_ids[-1] if run_ids else None,
        "run_ids": run_ids,
        "trials_run": len(run_ids),
    }


# ---------------------------------------------------------------------------
# 4. Holdout (one-shot, no parameter search)
# ---------------------------------------------------------------------------

def holdout_for_symbol(
    sp: SymbolPaths,
    train_features: pd.DataFrame,
    holdout_features: pd.DataFrame,
    *,
    journal: Optional[BacktestResultsJournal] = None,
    config_path: str | Path = "config/profiles/canonical_4model.yaml",
) -> Dict[str, Any]:
    """
    Один WFO-прогон по holdout-сегменту с **зафиксированными** параметрами
    из ``config/symbols/<slug>.yaml``. Подбора нет — это оценочный запуск.
    """
    from orchestration import TrainingOrchestrator
    from risk_management import OrchestratorRiskBridge

    from .tuning_loop import _baseline_defaults
    from .benchmark_runner import build_report

    from orchestration.ml_deps import require_tensorflow

    params = _merge_symbol_tuning_overrides(
        _baseline_defaults(config_path, canonical=True),
        sp,
        allow_two_model_override=False,
    )
    require_tensorflow(params.get("model_keys"))

    from .tuning_loop import _fit_wfo_windows_to_rows

    cfg = _fit_wfo_windows_to_rows(
        orchestrator_config_from_params(params),
        len(train_features) + len(holdout_features),
    )
    # На holdout-сегменте достаточно одного длинного фолда: train ⊂ holdout-history.
    # Используем все ``train_features`` как IS, ``holdout_features`` — OOS.
    cfg.train_window_size = max(cfg.train_window_size, max(200, int(len(train_features) * 0.6)))
    cfg.test_window_size = min(cfg.test_window_size, max(50, int(len(holdout_features) * 0.5)))

    full = pd.concat([train_features, holdout_features])
    y = default_horizon_labels(
        full["close"],
        cfg.prediction_horizon,
        min_return=float(getattr(cfg, "label_min_return", 0.0) or 0.0),
    )
    models = build_orchestration_models(cfg, full, dl_epochs=5)
    orch = TrainingOrchestrator(cfg)
    cap = float(params.get("max_position_fraction", 1.0) or 1.0)
    risk = (
        OrchestratorRiskBridge(max_position_fraction=cap)
        if params.get("use_risk_bridge")
        else None
    )
    orch.initialize(
        models=models,
        regime_detector=regime_detector_for(params.get("regime", "momentum")),
        meta_weighting=meta_weighting_from_config(cfg),
        risk_manager=risk,
    )
    folds = orch.walk_forward_backtest(full, y)
    # Оставляем только фолды, целиком лежащие в holdout-сегменте.
    if not folds.empty and len(holdout_features) > 0:
        boundary = full.index[len(train_features)]
        # ``Fold`` — порядковый номер; точечной информации о датах в DataFrame нет,
        # поэтому фильтруем по числу фолдов: в WFO они идут хронологически.
        # Для honest-holdout берём последние ⌈len(holdout)/test_window⌉ фолдов.
        n_keep = max(1, len(holdout_features) // max(cfg.test_window_size, 1))
        folds = folds.tail(n_keep).reset_index(drop=True)

    report = build_report(
        folds,
        parquet=str(sp.parquet.resolve()),
        feature_rows=len(full),
        max_rows=None,
        config_path=config_path,
        label=f"{sp.slug}/holdout",
        params={**params, "profile": params.get("profile")},
        symbol=sp.symbol,
        timeframe=sp.timeframe,
        stage="holdout",
        train_span=_span_of(train_features),
        holdout_span=_span_of(holdout_features),
    )
    journal = journal or BacktestResultsJournal()
    rid = journal.append_run(
        report,
        label=f"{sp.slug}/holdout",
        params=params,
    )
    return {
        "run_id": rid,
        "summary": report["summary"],
        "acceptance_passed": report["criteria"]["acceptance"]["passed"],
        "target_passed": report["criteria"]["target"]["passed"],
        "criteria": report["criteria"],
    }


# ---------------------------------------------------------------------------
# 5. Final fit on all data + bundle
# ---------------------------------------------------------------------------

def train_final_for_symbol(
    sp: SymbolPaths,
    features: pd.DataFrame,
    *,
    run_id: Optional[str] = None,
    config_path: str | Path = "config/profiles/canonical_4model.yaml",
) -> Path:
    """
    Финальное обучение на ВСЕХ доступных барах символа и сохранение bundle.

    ВАЖНО: вызывается **только после** прохождения holdout, иначе будет подгонка.
    Возвращает путь к bundle-директории.
    """
    from orchestration.symbols import paths_for as _paths_for

    rid = run_id or _new_run_id()
    bundle_dir = sp.artifacts_root / rid
    bundle_dir.mkdir(parents=True, exist_ok=True)

    from orchestration.ml_deps import require_tensorflow

    params = _merge_symbol_tuning_overrides(
        _baseline_defaults(config_path, canonical=True),
        sp,
        allow_two_model_override=False,
    )
    require_tensorflow(params.get("model_keys"))

    cfg = orchestrator_config_from_params(params)
    cols = infer_training_feature_columns(features)
    if "close" not in features.columns:
        raise ValueError("features must include 'close'")

    y = default_horizon_labels(
        features["close"],
        cfg.prediction_horizon,
        min_return=float(getattr(cfg, "label_min_return", 0.0) or 0.0),
    )
    train_mask = ~y.isna()
    X_train = features.loc[train_mask]
    y_train = y.loc[train_mask]

    models = build_orchestration_models(cfg, features, feature_columns=cols, dl_epochs=5)
    for k, m in models.items():
        if hasattr(m, "fit"):
            m.fit(X_train, y_train)

    # Train-only meta threshold для inference
    from orchestration import TrainingOrchestrator

    orch = TrainingOrchestrator(cfg)
    orch.initialize(
        models=models,
        regime_detector=regime_detector_for(params.get("regime", "momentum")),
        meta_weighting=meta_weighting_from_config(cfg),
    )
    train_result = orch.run_pipeline(X_train)
    thr = orch._calibrate_test_meta_threshold(train_result.meta_probabilities)

    save_orchestrator_bundle(
        bundle_dir,
        config=cfg,
        models=models,
        regime_detector=regime_detector_for(params.get("regime", "momentum")),
        feature_columns=cols,
        train_meta_threshold=float(thr) if thr is not None else None,
    )
    sp.write_latest_pointer(rid)
    return bundle_dir


# ---------------------------------------------------------------------------
# 6. Manifest
# ---------------------------------------------------------------------------

@dataclass
class SymbolManifest:
    symbol: str
    timeframe: str
    slug: str
    created_at: str
    parquet: str
    train_span: Dict[str, Any]
    holdout_span: Dict[str, Any]
    full_span: Dict[str, Any]
    tuning_best_params: Dict[str, Any]
    tune_run_ids: List[str] = field(default_factory=list)
    holdout_run_id: Optional[str] = None
    holdout_summary: Dict[str, Any] = field(default_factory=dict)
    holdout_acceptance_passed: bool = False
    holdout_target_passed: bool = False
    artifact_bundle: Optional[str] = None
    bundle_run_id: Optional[str] = None
    ready_for_paper: bool = False
    ready_for_live: bool = False
    notes: str = ""


def write_symbol_manifest(sp: SymbolPaths, manifest: SymbolManifest) -> Path:
    sp.artifacts_root.mkdir(parents=True, exist_ok=True)
    out = sp.artifacts_root / "manifest.json"
    out.write_text(json.dumps(asdict(manifest), indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def read_symbol_manifest(sp: SymbolPaths) -> Optional[Dict[str, Any]]:
    p = sp.artifacts_root / "manifest.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 7. End-to-end orchestration
# ---------------------------------------------------------------------------

def _progress_writer(path: Path) -> Callable[[Dict[str, Any]], None]:
    path.parent.mkdir(parents=True, exist_ok=True)

    def _write(payload: Dict[str, Any]) -> None:
        payload = {**payload, "ts": _utc_now_iso()}
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    return _write


def prepare_symbol(
    symbol: str,
    timeframe: str = "1h",
    *,
    download: bool = False,
    start_date: str = "2022-01-01 00:00:00",
    end_date: Optional[str] = None,
    holdout_fraction: float = 0.2,
    max_trials: int = 30,
    do_train_final: bool = True,
    journal_root: str | Path = "docs/backtest_journal",
    config_path: str | Path = "config/profiles/canonical_4model.yaml",
    allow_two_model_override: bool = False,
    on_step: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """
    Высокоуровневый сценарий: данные → признаки → tune → holdout → final fit.

    Идемпотентен в части артефактов (каждый запуск пишет новый run_id), но
    переписывает указатель ``LATEST.txt``. Для блокировки поверх — внешняя очередь.
    """
    sp = paths_for(symbol, timeframe)
    task_id = _new_run_id()
    progress_path = sp.artifacts_root / "active" / f"{task_id}.jsonl"
    log = _progress_writer(progress_path)
    journal = BacktestResultsJournal(journal_root)

    def _emit(step: str, payload: Dict[str, Any]) -> None:
        log({"step": step, **payload})
        if on_step:
            on_step(step, payload)

    _emit("start", {"symbol": sp.symbol, "timeframe": sp.timeframe, "slug": sp.slug})

    from orchestration.canonical_pipeline import (
        features_parquet_for,
        journal_profile_tag,
        prepare_canonical_dataset,
    )

    profile_tag = journal_profile_tag(config_path)

    from orchestration.benchmark_runner import orchestrator_config_from_yaml
    from orchestration.ml_deps import require_tensorflow

    canonical_cfg = orchestrator_config_from_yaml(config_path)
    require_tensorflow(canonical_cfg.model_keys)

    # 1. Data (+ MTF when using canonical profile)
    if download or not sp.parquet.is_file():
        _emit("download", {"status": "begin", "profile": profile_tag})
        if Path(config_path).is_file():
            prepare_canonical_dataset(
                symbol,
                timeframe,
                config_path=config_path,
                download=True,
                verbose=False,
            )
        else:
            download_ohlcv(
                symbol,
                timeframe,
                start_date=start_date,
                end_date=end_date,
                verbose=False,
            )
        _emit("download", {"status": "done", "parquet": str(sp.parquet)})
    else:
        _emit("download", {"status": "skipped", "parquet": str(sp.parquet)})

    if not sp.parquet.is_file():
        raise FileNotFoundError(f"parquet not found: {sp.parquet}")

    # 2. Features + split
    _emit("features", {"status": "begin", "profile": profile_tag})
    feat_path = features_parquet_for(symbol, timeframe)
    features = build_features(
        sp.parquet,
        config_path=config_path if Path(config_path).is_file() else None,
        save_to=feat_path,
    )
    train_part, holdout_part = split_train_holdout(features, holdout_fraction)
    full_span = _span_of(features)
    train_span = _span_of(train_part)
    holdout_span = _span_of(holdout_part)
    from orchestration.canonical_pipeline import data_collection_dates, expected_bar_count

    start_d, end_d = data_collection_dates(config_path)
    need_1h = expected_bar_count(start_d, end_d, timeframe)
    if len(features) < need_1h:
        raise RuntimeError(
            f"Only {len(features)} feature rows (expected ~{need_1h}+ for {timeframe}). "
            f"OHLCV download is incomplete — re-run with `--download` "
            f"(check data/ohlcv/{sp.slug}.parquet and OKX rate limits)."
        )

    _emit("features", {
        "status": "done",
        "rows": len(features),
        "train_rows": len(train_part),
        "holdout_rows": len(holdout_part),
        "train_span": train_span,
        "holdout_span": holdout_span,
    })

    # 3. Tune
    _emit("tune", {"status": "begin", "max_trials": max_trials})

    def _on_progress(payload: Dict[str, Any]) -> None:
        _emit("tune", payload)

    tune_out = tune_for_symbol(
        sp,
        train_part,
        max_trials=max_trials,
        journal=journal,
        on_progress=_on_progress,
        config_path=config_path,
        allow_two_model_override=allow_two_model_override,
    )
    _emit("tune", {
        "status": "done",
        "trials_run": tune_out["trials_run"],
        "best_summary": tune_out["best_summary"],
    })

    # 4. Holdout
    _emit("holdout", {"status": "begin"})
    holdout_out = holdout_for_symbol(
        sp,
        train_part,
        holdout_part,
        journal=journal,
        config_path=config_path,
    )
    _emit("holdout", {
        "status": "done",
        "acceptance_passed": holdout_out["acceptance_passed"],
        "target_passed": holdout_out["target_passed"],
        "summary": holdout_out["summary"],
        "run_id": holdout_out["run_id"],
    })

    # 5. Final fit (только если holdout PASS)
    bundle_run_id: Optional[str] = None
    bundle_path: Optional[Path] = None
    if do_train_final and holdout_out["acceptance_passed"]:
        _emit("train_final", {"status": "begin"})
        bundle_run_id = _new_run_id()
        bundle_path = train_final_for_symbol(
            sp,
            features,
            run_id=bundle_run_id,
            config_path=config_path,
        )
        _emit("train_final", {"status": "done", "bundle": str(bundle_path), "run_id": bundle_run_id})
    else:
        _emit("train_final", {"status": "skipped", "reason": "holdout not passed" if not holdout_out["acceptance_passed"] else "do_train_final=False"})

    # 6. Manifest
    manifest = SymbolManifest(
        symbol=sp.symbol,
        timeframe=sp.timeframe,
        slug=sp.slug,
        created_at=_utc_now_iso(),
        parquet=str(sp.parquet),
        train_span=train_span,
        holdout_span=holdout_span,
        full_span=full_span,
        tuning_best_params=tune_out["best_params"],
        tune_run_ids=tune_out["run_ids"],
        holdout_run_id=holdout_out["run_id"],
        holdout_summary=holdout_out["summary"],
        holdout_acceptance_passed=bool(holdout_out["acceptance_passed"]),
        holdout_target_passed=bool(holdout_out["target_passed"]),
        artifact_bundle=str(bundle_path) if bundle_path else None,
        bundle_run_id=bundle_run_id,
        ready_for_paper=bool(holdout_out["acceptance_passed"] and bundle_path is not None),
        ready_for_live=False,
    )
    write_symbol_manifest(sp, manifest)
    _emit("done", {"manifest": str(sp.artifacts_root / "manifest.json")})

    return {
        "symbol": sp.symbol,
        "timeframe": sp.timeframe,
        "manifest": asdict(manifest),
        "task_id": task_id,
        "progress_log": str(progress_path),
    }
