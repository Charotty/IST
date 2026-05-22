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
THESIS_REFERENCE_BASELINE = REPO_ROOT / "config" / "reference" / "thesis_4model_reference.yaml"
DEFAULT_BASELINE_REF = "config/reference/thesis_4model_reference.yaml"

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


def resolve_baseline_path(ref: str) -> Path:
    p = Path(ref)
    if not p.is_absolute():
        p = REPO_ROOT / ref
    return p.resolve()


def load_reference_baseline(ref: Optional[str] = None) -> Dict[str, Any]:
    """Эталонный YAML (лучший confirm); меняется одним файлом."""
    path = resolve_baseline_path(ref or DEFAULT_BASELINE_REF)
    if not path.is_file():
        path = THESIS_REFERENCE_BASELINE
    if not path.is_file():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_symbol_yaml_raw(symbol: str, timeframe: str = "1h") -> Dict[str, Any]:
    p = paths_for(symbol, timeframe).config_yaml
    if not p.is_file():
        return {}
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return dict(raw) if isinstance(raw, dict) else {}


def tuning_overrides_from_full(
    full: Dict[str, Any],
    reference: Dict[str, Any],
) -> Dict[str, Any]:
    """Только ключи, отличающиеся от эталона (для per-symbol YAML)."""
    out: Dict[str, Any] = {}
    for k, v in full.items():
        rv = reference.get(k)
        if rv != v:
            out[k] = v
    return out


def resolve_tuning_best_block(raw_symbol: Dict[str, Any]) -> Dict[str, Any]:
    """
    ``baseline_ref`` + ``orchestration_overrides`` → полный ``orchestration_tuning_best``.
    Legacy: только ``orchestration_tuning_best`` в symbol YAML (без baseline_ref).
    """
    if not raw_symbol:
        return {}
    legacy = raw_symbol.get("orchestration_tuning_best")
    ref_key = raw_symbol.get("baseline_ref")
    overrides = raw_symbol.get("orchestration_overrides")

    if ref_key:
        ref_doc = load_reference_baseline(str(ref_key))
        base = dict(ref_doc.get("orchestration_tuning_best") or {})
        if isinstance(overrides, dict) and overrides:
            return _deep_merge(base, overrides)
        return base

    if isinstance(legacy, dict) and legacy:
        return dict(legacy)
    return {}


def merged_config(symbol: str, timeframe: str = "1h") -> Dict[str, Any]:
    """
    Объединяет ``config.yaml`` (base) с ``config/symbols/<slug>.yaml``.
    Эффективный ``orchestration_tuning_best`` подставляется из эталона + overrides.
    """
    base = load_base_config()
    raw = load_symbol_yaml_raw(symbol, timeframe)
    if not raw:
        return base
    override = dict(raw)
    tb = resolve_tuning_best_block(raw)
    if tb:
        override["orchestration_tuning_best"] = tb
    return _deep_merge(base, override)


def write_symbol_config(
    symbol: str,
    timeframe: str,
    *,
    tuning_best: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
    allow_two_model_override: bool = False,
    baseline_ref: Optional[str] = None,
    store_as_overrides: bool = True,
) -> Path:
    sp = paths_for(symbol, timeframe)
    sp.config_yaml.parent.mkdir(parents=True, exist_ok=True)
    ref_path = baseline_ref or DEFAULT_BASELINE_REF
    payload: Dict[str, Any] = {
        "symbol": sp.symbol,
        "timeframe": sp.timeframe,
        "baseline_ref": ref_path,
    }
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
        ref_tb = dict(
            load_reference_baseline(ref_path).get("orchestration_tuning_best") or {}
        )
        if store_as_overrides and ref_tb:
            overrides = tuning_overrides_from_full(tb, ref_tb)
            if overrides:
                payload["orchestration_overrides"] = overrides
        else:
            payload["orchestration_tuning_best"] = tb
    if extra:
        payload.update(extra)
    sp.config_yaml.write_text(
        yaml.dump(payload, default_flow_style=False, allow_unicode=True),
        encoding="utf-8",
    )
    return sp.config_yaml


def tuning_best_for(symbol: str, timeframe: str = "1h") -> Dict[str, Any]:
    """Эталон + per-symbol overrides; иначе legacy symbol YAML; иначе base config."""
    raw = load_symbol_yaml_raw(symbol, timeframe)
    resolved = resolve_tuning_best_block(raw)
    if resolved:
        return resolved
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
