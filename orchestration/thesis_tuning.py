"""
Multilevel thesis hyperparameter search: fast → refine → confirm.

Replaces ad-hoc grid in thesis_push_4model.py with structured levels + pruning.
"""

from __future__ import annotations

from orchestration.dl_training import configure_tf_runtime

configure_tf_runtime()

import json
import random
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import yaml

from backtesting.results_journal import BacktestResultsJournal
from orchestration.dl_training import clear_tf_session
from orchestration.feature_store import load_features
from orchestration.symbols import REPO_ROOT, paths_for
from orchestration.tuning_config import THESIS_4MODEL_SEED, merge_thesis_params
from orchestration.tuning_loop import _score_report, run_single_trial

SHORTLIST_PATH = REPO_ROOT / "docs" / "thesis_tune_shortlist.json"
DEFAULT_TUNING_YAML = REPO_ROOT / "config" / "profiles" / "thesis_tuning.yaml"
TURBO_TUNING_YAML = REPO_ROOT / "config" / "profiles" / "thesis_tuning_turbo.yaml"


def resolve_tuning_yaml(path: Optional[str | Path] = None, profile: str = "default") -> Path:
    if path:
        return Path(path)
    if profile == "turbo":
        return TURBO_TUNING_YAML
    return DEFAULT_TUNING_YAML

# Never copied from fast/refine shortlist — always taken from params_for_level(level).
_LEVEL_STRUCTURAL: Dict[str, frozenset] = {
    "fast": frozenset(
        {
            "max_rows",
            "max_wfo_folds",
            "dl_epochs",
            "dl_batch_size",
            "tune_level",
            "profile",
            "config_path",
        }
    ),
    "refine": frozenset(
        {
            "max_rows",
            "max_wfo_folds",
            "dl_epochs",
            "dl_batch_size",
            "tune_level",
            "profile",
            "config_path",
        }
    ),
    "confirm": frozenset(
        {
            "max_rows",
            "max_wfo_folds",
            "walk_forward_step",
            "dl_epochs",
            "dl_batch_size",
            "tune_level",
            "profile",
            "config_path",
        }
    ),
}

# Local patches around quality preset (plan M4)
_REFINE_PATCHES: List[Dict[str, Any]] = [
    {},
    {"ensemble_mode": "fixed_range"},
    {"min_signal_margin": 0.08},
    {"min_signal_margin": 0.10},
    {"direction_threshold": 0.58},
    {"direction_threshold": 0.62},
    {"trade_mode": "long_only"},
    {"prediction_horizon": 12},
    {"max_position_fraction": 0.5},
    {"volatility_filter_percentile": 90.0},
    {"label_min_return": 0.004},
    {"signal_strategy": "momentum_confirm"},
]


def load_tuning_yaml(path: Optional[Path] = None) -> Dict[str, Any]:
    p = path or DEFAULT_TUNING_YAML
    if not p.is_file():
        return {"levels": {}, "top_k": 5}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def level_config(level: str, yaml_cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    raw = yaml_cfg or load_tuning_yaml()
    return dict((raw.get("levels") or {}).get(level) or {})


def params_for_level(
    level: str,
    *,
    base: Optional[Dict[str, Any]] = None,
    yaml_cfg: Optional[Dict[str, Any]] = None,
    tabular_first: bool = False,
) -> Dict[str, Any]:
    """Merge THESIS seed + level overrides from thesis_tuning.yaml."""
    lc = level_config(level, yaml_cfg)
    mode = "fast" if level == "fast" else ("quality" if level == "confirm" else "fast+quality")
    out = merge_thesis_params(base or THESIS_4MODEL_SEED, mode=mode)
    for key in (
        "max_rows",
        "max_wfo_folds",
        "walk_forward_step",
        "dl_epochs",
        "dl_batch_size",
    ):
        if key in lc:
            out[key] = lc[key]
    out["tune_level"] = level
    out["_mixed_precision"] = bool(lc.get("mixed_precision", level != "confirm"))
    out["_dl_batch_size"] = int(lc.get("dl_batch_size", 64))
    out["_tabular_profile"] = lc.get("tabular_profile", "fast" if level != "confirm" else "confirm")
    if tabular_first and level in ("fast", "refine"):
        out["model_keys"] = ["lgb", "xgb"]
    return out


def merge_candidate_params(
    level: str,
    base: Dict[str, Any],
    patch: Dict[str, Any],
) -> Dict[str, Any]:
    """Hyperparams from patch/shortlist; structural knobs always from level base."""
    locked = _LEVEL_STRUCTURAL.get(level, frozenset())
    private = {k: v for k, v in base.items() if str(k).startswith("_")}
    structural = {k: base[k] for k in locked if k in base}
    tunable = {
        k: v
        for k, v in patch.items()
        if k not in locked and not str(k).startswith("_")
    }
    base_defaults = {
        k: v
        for k, v in base.items()
        if k not in locked and not str(k).startswith("_") and k not in tunable
    }
    return {**base_defaults, **tunable, **structural, **private}


def _random_fast_params(rng: random.Random) -> Dict[str, Any]:
    return {
        "min_signal_margin": rng.choice([0.06, 0.08, 0.10, 0.12]),
        "direction_threshold": rng.choice([0.56, 0.58, 0.60, 0.62]),
        "ensemble_mode": rng.choice(["fixed_range", "regime_adaptive"]),
        "prediction_horizon": rng.choice([12, 24]),
        "walk_forward_step": rng.choice([400, 500]),
        "trade_mode": rng.choice(["both", "long_only"]),
        "max_position_fraction": rng.choice([1.0, 0.5, 0.35]),
    }


def iter_level_candidates(
    level: str,
    *,
    shortlist: Optional[List[Dict[str, Any]]] = None,
    n_trials: Optional[int] = None,
    yaml_cfg: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    raw = yaml_cfg or load_tuning_yaml()
    lc = level_config(level, raw)
    n = n_trials or int(lc.get("n_trials", 12))
    base = params_for_level(
        level,
        yaml_cfg=raw,
        tabular_first=bool(raw.get("tabular_first_fast_refine")),
    )
    out: List[Dict[str, Any]] = []

    if level == "fast":
        rng = random.Random(42)
        for _ in range(n):
            out.append(merge_candidate_params(level, base, _random_fast_params(rng)))
        return out

    if level == "refine" and shortlist:
        for item in shortlist[:n]:
            p = deepcopy(item.get("params") or item)
            out.append(merge_candidate_params(level, base, p))
        for patch in _REFINE_PATCHES:
            if len(out) >= n:
                break
            seed = shortlist[0].get("params", shortlist[0]) if shortlist else {}
            out.append(merge_candidate_params(level, base, {**seed, **patch}))
        return out[:n]

    if level == "confirm":
        # Top shortlist entries first (fast/refine winners), then local patches
        if shortlist:
            ranked = sorted(
                shortlist,
                key=lambda x: float((x.get("summary") or {}).get("mean_sharpe") or -1e9),
                reverse=True,
            )
            for item in ranked[: min(n, len(ranked))]:
                p = deepcopy(item.get("params") or item)
                out.append(merge_candidate_params(level, base, p))
        for patch in _REFINE_PATCHES:
            if len(out) >= n:
                break
            out.append(merge_candidate_params(level, base, patch))
        return out[:n]

    return out


def _fold_sharpe(fold: Dict[str, Any]) -> float:
    for key in ("Sharpe Ratio", "sharpe_ratio", "sharpe", "mean_sharpe"):
        if key in fold and fold[key] is not None:
            try:
                return float(fold[key])
            except (TypeError, ValueError):
                pass
    return 0.0


class FoldPruner:
    """Median-style early stop after >= min_folds completed folds."""

    def __init__(self, min_folds: int = 2, min_completed_trials: int = 3):
        self.min_folds = min_folds
        self.min_completed_trials = min_completed_trials
        self._completed_sharpes: List[float] = []

    def record_completed_trials(self, sharpes: List[float]) -> None:
        self._completed_sharpes = list(sharpes)

    def should_prune(self, fold_metrics: List[Dict[str, Any]]) -> bool:
        if len(fold_metrics) < self.min_folds:
            return False
        if len(self._completed_sharpes) < self.min_completed_trials:
            return False
        import statistics

        trial_sharpes = [_fold_sharpe(f) for f in fold_metrics]
        mean_s = sum(trial_sharpes) / len(trial_sharpes)
        med = statistics.median(self._completed_sharpes)
        return mean_s < med


def should_prune_trial(
    fold_metrics: List[Dict[str, Any]],
    *,
    completed_trial_sharpes: List[float],
    min_folds: int = 2,
    enabled: bool = True,
) -> bool:
    if not enabled:
        return False
    pruner = FoldPruner(min_folds=min_folds)
    pruner.record_completed_trials(completed_trial_sharpes)
    return pruner.should_prune(fold_metrics)


def _load_features_for_symbol(
    symbol: str,
    timeframe: str,
    *,
    max_rows: Optional[int],
    config_path: str | Path,
    use_feature_cache: bool,
) -> Any:
    sp = paths_for(symbol, timeframe)
    if use_feature_cache:
        return load_features(
            symbol,
            timeframe,
            max_rows=max_rows,
            config_path=config_path,
            use_cache=True,
        )
    from orchestration.real_data_benchmark import features_from_ohlcv_parquet

    return features_from_ohlcv_parquet(sp.parquet, max_rows=max_rows)


def _run_trial_with_meta(
    features,
    params: Dict[str, Any],
    *,
    config_path: str | Path,
    parquet_label: str,
    journal: BacktestResultsJournal,
    label: str,
    on_fold_done: Optional[Callable] = None,
    pruner: Optional[FoldPruner] = None,
) -> Tuple[Optional[Dict[str, Any]], float, bool, Optional[str]]:
    from orchestration.tuning_loop import TrialPrunedError, run_single_trial

    t0 = time.perf_counter()
    pruned = False
    report = None
    err_msg: Optional[str] = None
    p = dict(params)

    try:
        report = run_single_trial(
            features,
            p,
            config_path=config_path,
            parquet_label=parquet_label,
            on_fold_done=on_fold_done,
            pruner=pruner,
            dl_batch_size=int(p.get("_dl_batch_size", 64)),
            mixed_precision=bool(p.get("_mixed_precision", False)),
            tune_level=str(p.get("tune_level", "")),
        )
    except TrialPrunedError:
        pruned = True
    except Exception as exc:
        err_msg = f"{type(exc).__name__}: {exc}"
        journal.append_run(
            {
                "parquet": parquet_label,
                "feature_rows": len(features),
                "summary": {},
                "criteria": {
                    "acceptance": {"passed": False, "checks": []},
                    "target": {"passed": False, "checks": []},
                },
                "fold_metrics": [],
            },
            label=f"{label}_ERROR",
            params={k: v for k, v in p.items() if not k.startswith("_")},
            notes=err_msg,
        )
        return None, time.perf_counter() - t0, False, err_msg

    elapsed = time.perf_counter() - t0
    if report is None and pruned:
        return None, elapsed, True, None
    return report, elapsed, pruned, None


def save_shortlist(
    entries: List[Dict[str, Any]],
    *,
    path: Path = SHORTLIST_PATH,
    top_k: int = 5,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    def _rank_key(e: Dict[str, Any]) -> tuple:
        rep = e.get("report")
        if not rep:
            return (-1,)
        try:
            return _score_report(rep)
        except (KeyError, TypeError):
            ms = float((rep.get("summary") or {}).get("mean_sharpe") or -1e9)
            return (0, 0, 0, 0, ms, 0, 0, 0)

    ranked = sorted(entries, key=_rank_key, reverse=True)[:top_k]
    payload = {
        "top_k": top_k,
        "entries": [
            {
                "params": {k: v for k, v in (e.get("params") or {}).items() if not str(k).startswith("_")},
                "summary": (e.get("report") or {}).get("summary"),
                "mean_sharpe": float(((e.get("report") or {}).get("summary") or {}).get("mean_sharpe") or 0),
                "acceptance_passed": bool(
                    ((e.get("report") or {}).get("criteria") or {})
                    .get("acceptance", {})
                    .get("passed")
                ),
                "tune_level": e.get("params", {}).get("tune_level"),
            }
            for e in ranked
        ],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_shortlist(path: Path = SHORTLIST_PATH) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data.get("entries") or [])


def run_level(
    level: str,
    symbol: str,
    timeframe: str = "1h",
    *,
    config_path: str | Path = "config/profiles/canonical_4model.yaml",
    journal_root: str | Path = "docs/backtest_journal",
    use_feature_cache: bool = True,
    shortlist: Optional[List[Dict[str, Any]]] = None,
    n_trials: Optional[int] = None,
    tuning_yaml: Optional[str | Path] = None,
    tuning_profile: str = "default",
) -> Dict[str, Any]:
    yaml_path = resolve_tuning_yaml(tuning_yaml, tuning_profile)
    raw = load_tuning_yaml(yaml_path)
    lc = level_config(level, raw)
    base = params_for_level(
        level,
        yaml_cfg=raw,
        tabular_first=bool(raw.get("tabular_first_fast_refine")),
    )
    max_rows = lc.get("max_rows", base.get("max_rows"))
    candidates = iter_level_candidates(level, shortlist=shortlist, n_trials=n_trials, yaml_cfg=raw)
    sp = paths_for(symbol, timeframe)
    features = _load_features_for_symbol(
        symbol,
        timeframe,
        max_rows=max_rows,
        config_path=config_path,
        use_feature_cache=use_feature_cache,
    )
    journal = BacktestResultsJournal(journal_root)
    pruner_type = str(lc.get("pruner", "none"))
    fold_pruner = (
        FoldPruner(
            min_folds=2,
            min_completed_trials=int(lc.get("pruner_min_completed_trials", 5)),
        )
        if pruner_type not in ("none", "")
        else None
    )
    completed_sharpes: List[float] = []
    trial_results: List[Dict[str, Any]] = []
    best_report = None
    best_params = None
    best_score = (-1,)
    run_ids: List[str] = []
    trials_attempted = 0
    trials_failed = 0
    trials_pruned = 0
    last_error: Optional[str] = None

    for idx, patch in enumerate(candidates):
        params = merge_candidate_params(level, base, patch)
        params["_label"] = f"{level}_{idx + 1}"
        if fold_pruner:
            fold_pruner.record_completed_trials(completed_sharpes)

        trials_attempted += 1
        clear_tf_session()
        report, elapsed, pruned, err = _run_trial_with_meta(
            features,
            params,
            config_path=config_path,
            parquet_label=str(sp.parquet.resolve()),
            journal=journal,
            label=params["_label"],
            pruner=fold_pruner,
        )
        if err:
            trials_failed += 1
            last_error = err
        if pruned:
            trials_pruned += 1
        meta = {
            "tune_level": level,
            "symbol": symbol,
            "stage": level,
            "elapsed_sec": round(elapsed, 2),
            "pruned": pruned,
        }
        if report is None:
            continue
        report.setdefault("params", {})
        report["params"].update(meta)
        ms = float((report.get("summary") or {}).get("mean_sharpe") or -1e9)
        completed_sharpes.append(ms)
        rid = journal.append_run(
            report,
            label=params["_label"],
            params={**{k: v for k, v in params.items() if not k.startswith("_")}, **meta},
        )
        run_ids.append(rid)
        entry = {"params": params, "report": report}
        trial_results.append(entry)
        sc = _score_report(report)
        if sc > best_score:
            best_score = sc
            best_report = report
            best_params = params

    top_k = int(raw.get("top_k", 5))
    if level in ("fast", "refine"):
        save_shortlist(trial_results, top_k=top_k)

    return {
        "level": level,
        "symbol": symbol,
        "timeframe": timeframe,
        "trials_attempted": trials_attempted,
        "trials_failed": trials_failed,
        "trials_pruned": trials_pruned,
        "trials_run": len(run_ids),
        "last_error": last_error,
        "pruned_count": trials_pruned,
        "best_params": {k: v for k, v in (best_params or {}).items() if not k.startswith("_")},
        "best_report": best_report,
        "run_ids": run_ids,
        "shortlist_path": str(SHORTLIST_PATH) if level in ("fast", "refine") else None,
        "acceptance_passed": bool(
            best_report
            and best_report.get("criteria", {}).get("acceptance", {}).get("passed")
        ),
    }


def run_fast_search(
    symbol: str,
    timeframe: str = "1h",
    **kwargs: Any,
) -> Dict[str, Any]:
    return run_level("fast", symbol, timeframe, **kwargs)


def run_refine(
    symbol: str,
    timeframe: str = "1h",
    *,
    shortlist: Optional[List[Dict[str, Any]]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    sl = shortlist if shortlist is not None else load_shortlist()
    return run_level("refine", symbol, timeframe, shortlist=sl, **kwargs)


def run_confirm(
    symbol: str,
    timeframe: str = "1h",
    *,
    shortlist: Optional[List[Dict[str, Any]]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    sl = shortlist if shortlist is not None else load_shortlist()
    return run_level("confirm", symbol, timeframe, shortlist=sl, **kwargs)


def run_multilevel(
    symbol: str,
    timeframe: str = "1h",
    *,
    phase: str = "all",
    **kwargs: Any,
) -> Dict[str, Any]:
    """``phase``: fast | refine | confirm | all."""
    results: Dict[str, Any] = {}
    if phase in ("fast", "all"):
        results["fast"] = run_fast_search(symbol, timeframe, **kwargs)
    shortlist = load_shortlist()
    if phase in ("refine", "all"):
        results["refine"] = run_refine(symbol, timeframe, shortlist=shortlist, **kwargs)
        shortlist = load_shortlist()
    if phase in ("confirm", "all"):
        results["confirm"] = run_confirm(symbol, timeframe, shortlist=shortlist, **kwargs)
    return results
