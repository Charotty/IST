"""
Symbol-aware пути и per-symbol YAML конфиги.

Позволяет работать с несколькими парами одновременно:

* parquet:    ``data/ohlcv/<SYMBOL>_<TF>.parquet`` (`/` в символе → `-`)
* config:     ``config/symbols/<SYMBOL>_<TF>.yaml`` (наследует ``backtesting`` из base)
* artifacts:  ``artifacts/<SYMBOL>_<TF>/<run_id>/`` + symlink/маркер ``LATEST``
* runs:       поле ``symbol`` в записях ``BacktestResultsJournal``

Этот модуль — единственный источник правды по умолчанию, чтобы CLI и GUI
не разбредались по магическим путям.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "ohlcv"
SYMBOLS_DIR = REPO_ROOT / "config" / "symbols"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
BASE_CONFIG = REPO_ROOT / "config.yaml"

_SLUG_RE = re.compile(r"[^A-Z0-9]+")


def normalize_symbol(symbol: str) -> str:
    """``BTC/USDT`` / ``btc-usdt`` → ``BTC-USDT``."""
    s = symbol.strip().upper().replace("/", "-").replace("_", "-")
    s = _SLUG_RE.sub("-", s).strip("-")
    return s


def slug(symbol: str, timeframe: str) -> str:
    return f"{normalize_symbol(symbol)}_{timeframe.lower()}"


@dataclass
class SymbolPaths:
    symbol: str
    timeframe: str
    slug: str
    parquet: Path
    config_yaml: Path
    artifacts_root: Path

    def latest_bundle(self) -> Optional[Path]:
        ptr = self.artifacts_root / "LATEST.txt"
        if ptr.is_file():
            run_id = ptr.read_text(encoding="utf-8").strip()
            cand = self.artifacts_root / run_id
            if cand.is_dir():
                return cand
        if not self.artifacts_root.is_dir():
            return None
        runs = sorted(
            (p for p in self.artifacts_root.iterdir() if p.is_dir() and p.name != "active"),
            key=lambda p: p.name,
            reverse=True,
        )
        return runs[0] if runs else None

    def write_latest_pointer(self, run_id: str) -> None:
        self.artifacts_root.mkdir(parents=True, exist_ok=True)
        (self.artifacts_root / "LATEST.txt").write_text(run_id, encoding="utf-8")


def paths_for(symbol: str, timeframe: str = "1h") -> SymbolPaths:
    s = slug(symbol, timeframe)
    return SymbolPaths(
        symbol=normalize_symbol(symbol),
        timeframe=timeframe.lower(),
        slug=s,
        parquet=DATA_DIR / f"{s}.parquet",
        config_yaml=SYMBOLS_DIR / f"{s}.yaml",
        artifacts_root=ARTIFACTS_DIR / s,
    )


def list_known_symbols() -> List[Dict[str, Any]]:
    """Список пар: parquet есть / есть конфиг / есть бандл LATEST."""
    seen: Dict[str, Dict[str, Any]] = {}
    if DATA_DIR.is_dir():
        for f in DATA_DIR.glob("*.parquet"):
            name = f.stem
            seen.setdefault(name, {"slug": name, "parquet": str(f), "config": None, "latest": None})
    if SYMBOLS_DIR.is_dir():
        for f in SYMBOLS_DIR.glob("*.yaml"):
            name = f.stem
            entry = seen.setdefault(name, {"slug": name, "parquet": None, "config": None, "latest": None})
            entry["config"] = str(f)
    if ARTIFACTS_DIR.is_dir():
        for d in ARTIFACTS_DIR.iterdir():
            if not d.is_dir():
                continue
            entry = seen.setdefault(d.name, {"slug": d.name, "parquet": None, "config": None, "latest": None})
            ptr = d / "LATEST.txt"
            if ptr.is_file():
                entry["latest"] = ptr.read_text(encoding="utf-8").strip()
    return [seen[k] for k in sorted(seen)]


def load_base_config(base_yaml: Optional[Path] = None) -> Dict[str, Any]:
    """Читает корневой ``config.yaml`` (или путь, переданный аргументом).

    Лукап выполняется через текущее значение ``BASE_CONFIG`` модуля, чтобы
    тесты могли подменять путь через monkeypatch без перезапуска интерпретатора.
    """
    p = base_yaml if base_yaml is not None else BASE_CONFIG
    if not p.is_file():
        return {}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def merged_config(symbol: str, timeframe: str = "1h") -> Dict[str, Any]:
    """
    Объединяет ``config.yaml`` (base) с ``config/symbols/<slug>.yaml`` (override).
    Per-symbol перекрывает base только для тех ключей, которые в нём заданы.
    """
    base = load_base_config()
    p = paths_for(symbol, timeframe).config_yaml
    if not p.is_file():
        return base
    override = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return _deep_merge(base, override)


def write_symbol_config(
    symbol: str,
    timeframe: str,
    *,
    tuning_best: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
    allow_two_model_override: bool = False,
) -> Path:
    sp = paths_for(symbol, timeframe)
    sp.config_yaml.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {"symbol": sp.symbol, "timeframe": sp.timeframe}
    if tuning_best:
        tb = dict(tuning_best)
        mk = tb.get("model_keys")
        if (
            not allow_two_model_override
            and isinstance(mk, list)
            and len(mk) < 4
            and set(mk) <= {"lgb", "xgb"}
        ):
            tb.pop("model_keys", None)
        payload["orchestration_tuning_best"] = tb
    if extra:
        payload.update(extra)
    sp.config_yaml.write_text(
        yaml.dump(payload, default_flow_style=False, allow_unicode=True),
        encoding="utf-8",
    )
    return sp.config_yaml


def tuning_best_for(symbol: str, timeframe: str = "1h") -> Dict[str, Any]:
    """Per-symbol tuning_best (если задан), иначе из base ``config.yaml``."""
    sp = paths_for(symbol, timeframe)
    if sp.config_yaml.is_file():
        raw = yaml.safe_load(sp.config_yaml.read_text(encoding="utf-8")) or {}
        if isinstance(raw, dict) and raw.get("orchestration_tuning_best"):
            return dict(raw["orchestration_tuning_best"])
    base = load_base_config()
    return dict(base.get("orchestration_tuning_best") or {})


def _deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out
