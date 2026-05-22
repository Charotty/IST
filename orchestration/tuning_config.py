"""
Сборка ``OrchestratorConfig`` из параметров тюнинга / per-symbol YAML.

Для 4 моделей с ``regime_adaptive`` берётся canonical-профиль (веса trend/range),
а не равные 0.25 на каждую модель.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from orchestration.orchestrator_config import (
    OrchestratorConfig,
    _DEFAULT_BREAKOUT_WEIGHTS,
    _DEFAULT_RANGE_WEIGHTS,
    _DEFAULT_TREND_WEIGHTS,
    _renormalize_weight_subset,
)

# Семя: 2-model acceptance (журнал 20260516T130409Z) + полный canonical ансамбль
THESIS_4MODEL_SEED: Dict[str, Any] = {
    "profile": "canonical_4model",
    "config_path": "config/profiles/canonical_4model.yaml",
    "model_keys": ["lgb", "gru", "xgb", "cnn"],
    "ensemble_mode": "regime_adaptive",
    "apply_decision_pipeline": True,
    "regime": "momentum",
    "prediction_horizon": 24,
    "direction_threshold": 0.6,
    "min_signal_margin": 0.08,
    "meta_threshold_mode": "percentile",
    "trade_mode": "both",
    "use_risk_bridge": False,
    "train_window_size": 1500,
    "test_window_size": 170,
    "walk_forward_step": 360,
    "signal_strategy": "ensemble",
    "label_min_return": 0.0,
    "volatility_filter_percentile": 0.0,
    "max_position_fraction": 1.0,
    "dl_epochs": 3,
    "max_wfo_folds": 0,
}

# Task 1 — fast hyperparameter search (~3–8× fewer fold-trains vs full WFO)
THESIS_FAST_PRESET: Dict[str, Any] = {
    "dl_epochs": 2,
    "max_wfo_folds": 8,
    "walk_forward_step": 500,
    "max_rows": 6000,
}

# Task 2 — quality / acceptance-oriented (journal 2-model + 4 model_keys)
THESIS_QUALITY_PRESET: Dict[str, Any] = {
    "ensemble_mode": "fixed_range",
    "direction_threshold": 0.6,
    "min_signal_margin": 0.08,
    "meta_threshold_mode": "percentile",
    "prediction_horizon": 24,
    "train_window_size": 1500,
    "test_window_size": 170,
    "walk_forward_step": 360,
    "dl_epochs": 3,
    "max_wfo_folds": 0,
    "max_rows": 8000,
}


def merge_thesis_params(
    base: Optional[Dict[str, Any]] = None,
    *,
    mode: str = "default",
) -> Dict[str, Any]:
    """``mode``: ``default`` | ``fast`` | ``quality`` | ``fast+quality``."""
    out = {**THESIS_4MODEL_SEED, **(base or {})}
    if mode in ("fast", "fast+quality"):
        out.update({k: v for k, v in THESIS_FAST_PRESET.items() if k != "max_rows"})
    if mode in ("quality", "fast+quality"):
        out.update({k: v for k, v in THESIS_QUALITY_PRESET.items() if k != "max_rows"})
    return out


_TUNABLE_ORCH_FIELDS = (
    "model_keys",
    "ensemble_mode",
    "apply_decision_pipeline",
    "direction_threshold",
    "meta_threshold_mode",
    "meta_threshold",
    "signal_threshold",
    "trade_mode",
    "min_signal_margin",
    "signal_strategy",
    "momentum_sma_period",
    "label_min_return",
    "volatility_filter_percentile",
    "max_position_fraction",
    "train_window_size",
    "test_window_size",
    "walk_forward_step",
    "prediction_horizon",
    "embargo_period",
    "feature_window_size",
    "enable_purge",
    "allow_meta_label_on_test",
    "safe_label_generation",
    "max_wfo_folds",
)


def _is_four_model_setup(p: Dict[str, Any]) -> bool:
    mk = p.get("model_keys") or THESIS_4MODEL_SEED["model_keys"]
    return bool(set(mk) >= {"gru", "cnn"})


def _apply_regime_weights(cfg: OrchestratorConfig) -> None:
    mk = list(cfg.model_keys)
    cfg.trend_weights = _renormalize_weight_subset(_DEFAULT_TREND_WEIGHTS, mk)
    cfg.range_weights = _renormalize_weight_subset(_DEFAULT_RANGE_WEIGHTS, mk)
    cfg.breakout_weights = _renormalize_weight_subset(_DEFAULT_BREAKOUT_WEIGHTS, mk)
    cfg._validate_weights()


def _apply_param_overrides(cfg: OrchestratorConfig, p: Dict[str, Any]) -> None:
    for key in _TUNABLE_ORCH_FIELDS:
        if key in p and p[key] is not None:
            setattr(cfg, key, p[key])
    if "model_keys" in p and p["model_keys"]:
        cfg.model_keys = list(p["model_keys"])
        if cfg.ensemble_mode == "regime_adaptive" or _is_four_model_setup(p):
            _apply_regime_weights(cfg)
        else:
            w = {k: 1.0 / len(cfg.model_keys) for k in cfg.model_keys}
            cfg.trend_weights = w.copy()
            cfg.range_weights = w.copy()
            cfg.breakout_weights = w.copy()


def orchestrator_config_from_tuning_params(
    p: Dict[str, Any],
    *,
    config_path: str | Path = "config.yaml",
) -> OrchestratorConfig:
    """
    Построить конфиг: canonical 4-model + regime weights, затем поля из ``p``.
    """
    from orchestration.benchmark_runner import (
        load_canonical_config_path,
        lgb_xgb_config,
        orchestrator_config_from_yaml,
    )

    merged = {**THESIS_4MODEL_SEED, **{k: v for k, v in p.items() if not str(k).startswith("_")}}
    cp = merged.get("config_path") or config_path

    if _is_four_model_setup(merged):
        cfg = orchestrator_config_from_yaml(load_canonical_config_path(cp))
    else:
        cfg = lgb_xgb_config(
            train_window_size=int(merged["train_window_size"]),
            test_window_size=int(merged["test_window_size"]),
            walk_forward_step=int(merged["walk_forward_step"]),
            prediction_horizon=int(merged["prediction_horizon"]),
            direction_threshold=float(merged["direction_threshold"]),
            apply_decision_pipeline=bool(merged.get("apply_decision_pipeline", True)),
            meta_threshold_mode=str(merged.get("meta_threshold_mode", "median")),
            ensemble_mode=str(merged.get("ensemble_mode", "regime_adaptive")),
            min_signal_margin=float(merged.get("min_signal_margin", 0.0)),
            trade_mode=str(merged.get("trade_mode", "both")),
        )
    _apply_param_overrides(cfg, merged)
    return cfg


def dl_epochs_from_params(p: Dict[str, Any], default: int = 3) -> int:
    return int(p.get("dl_epochs", default))
