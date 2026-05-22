"""
Tabular model acceleration profiles (threads, GPU, n_estimators by tune level).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from orchestration.symbols import REPO_ROOT

_DEFAULT_YAML = REPO_ROOT / "config" / "profiles" / "thesis_tuning.yaml"


@dataclass
class TabularTrainingProfile:
    profile: str = "confirm"
    n_estimators: int = 200
    lgb_extra: Optional[Dict[str, Any]] = None
    xgb_extra: Optional[Dict[str, Any]] = None
    tabular_device: str = "cpu"


def _load_tabular_accel(path: Optional[Path] = None) -> Dict[str, Any]:
    p = path or _DEFAULT_YAML
    if not p.is_file():
        return {}
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return raw.get("tabular_accel") or {}


def _resolve_num_threads(cfg_val: Any) -> int:
    if cfg_val is None or int(cfg_val) < 0:
        return min(os.cpu_count() or 4, 16)
    return min(int(cfg_val), 16)


def _lgb_device_type(requested: str) -> str:
    if requested != "gpu":
        return "cpu"
    try:
        import lightgbm as lgb

        m = lgb.LGBMClassifier(device_type="gpu", n_estimators=1, verbosity=-1)
        # probe fit on tiny data would be heavy; trust build flag
        return "gpu"
    except Exception:
        return "cpu"


def _xgb_device(requested: str) -> str:
    if requested not in ("cuda", "gpu"):
        return "cpu"
    try:
        import xgboost as xgb
        import numpy as np

        X = np.zeros((8, 2), dtype=np.float32)
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        m = xgb.XGBClassifier(
            n_estimators=1,
            tree_method="hist",
            device="cuda",
            verbosity=0,
        )
        m.fit(X, y)
        return "cuda"
    except Exception:
        return "cpu"


def tabular_profile_for_level(
    tune_level: str,
    *,
    yaml_path: Optional[Path] = None,
) -> TabularTrainingProfile:
    """``tune_level``: fast | refine | confirm → tabular_profile fast or confirm."""
    accel = _load_tabular_accel(yaml_path)
    lgb_cfg = accel.get("lightgbm") or {}
    xgb_cfg = accel.get("xgboost") or {}
    level = tune_level if tune_level in ("fast", "refine", "confirm") else "confirm"
    is_fast = level in ("fast", "refine")
    n_lgb = int(lgb_cfg.get("n_estimators_fast" if is_fast else "n_estimators_confirm", 200))
    n_xgb = int(xgb_cfg.get("n_estimators_fast" if is_fast else "n_estimators_confirm", 200))
    lgb_device = _lgb_device_type(str(lgb_cfg.get("device_type", "cpu")))
    xgb_device = _xgb_device(str(xgb_cfg.get("device", "cpu")))
    tabular_device = xgb_device if xgb_device == "cuda" else lgb_device
    lgb_extra: Dict[str, Any] = {
        "num_threads": _resolve_num_threads(lgb_cfg.get("num_threads", -1)),
        "max_bin": int(lgb_cfg.get("max_bin", 255 if lgb_device == "cpu" else 63)),
    }
    if lgb_device == "gpu":
        lgb_extra["device_type"] = "gpu"
    xgb_extra: Dict[str, Any] = {
        "tree_method": str(xgb_cfg.get("tree_method", "hist")),
    }
    if xgb_device == "cuda":
        xgb_extra["device"] = "cuda"
    return TabularTrainingProfile(
        profile="fast" if is_fast else "confirm",
        n_estimators=max(n_lgb, n_xgb),
        lgb_extra={**lgb_extra, "n_estimators": n_lgb},
        xgb_extra={**xgb_extra, "n_estimators": n_xgb},
        tabular_device=tabular_device,
    )
