"""
Disk-backed feature cache per symbol (canonical FeatureManager path).

Avoids rebuilding MTF + indicators on every tune/report trial.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import yaml

from orchestration.canonical_pipeline import (
    FEATURES_DIR,
    build_canonical_features,
    features_parquet_for,
    resolve_profile_path,
)
from orchestration.symbols import REPO_ROOT, paths_for

DEFAULT_PROFILE = "config/profiles/canonical_4model.yaml"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def manifest_path(symbol: str, timeframe: str = "1h") -> Path:
    return features_parquet_for(symbol, timeframe).with_suffix(".manifest.json")


def _hash_sections(profile_path: Path) -> str:
    raw = yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}
    payload = {
        "feature_engineering": raw.get("feature_engineering") or {},
        "synchronization": raw.get("synchronization") or {},
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _columns_hash(df: pd.DataFrame) -> str:
    cols = ",".join(sorted(df.columns.astype(str)))
    return hashlib.sha256(cols.encode("utf-8")).hexdigest()[:16]


def read_manifest(symbol: str, timeframe: str = "1h") -> Optional[Dict[str, Any]]:
    path = manifest_path(symbol, timeframe)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_manifest(
    symbol: str,
    timeframe: str,
    df: pd.DataFrame,
    *,
    config_path: str | Path = DEFAULT_PROFILE,
    ohlcv_path: Optional[Path] = None,
) -> Dict[str, Any]:
    profile = resolve_profile_path(config_path)
    feat_path = features_parquet_for(symbol, timeframe)
    meta = {
        "symbol": paths_for(symbol, timeframe).symbol,
        "timeframe": timeframe,
        "slug": paths_for(symbol, timeframe).slug,
        "built_at": _utc_now(),
        "config_path": str(profile),
        "config_hash": _hash_sections(profile),
        "row_count": len(df),
        "columns_hash": _columns_hash(df),
        "features_parquet": str(feat_path),
        "ohlcv_parquet": str(ohlcv_path) if ohlcv_path else None,
    }
    manifest_path(symbol, timeframe).write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return meta


def is_stale(
    symbol: str,
    timeframe: str = "1h",
    *,
    config_path: str | Path = DEFAULT_PROFILE,
) -> bool:
    feat_path = features_parquet_for(symbol, timeframe)
    if not feat_path.is_file():
        return True
    man = read_manifest(symbol, timeframe)
    if man is None:
        return True
    profile = resolve_profile_path(config_path)
    return man.get("config_hash") != _hash_sections(profile)


def build_features(
    symbol: str,
    timeframe: str = "1h",
    *,
    config_path: str | Path = DEFAULT_PROFILE,
    force: bool = False,
    ohlcv_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Build canonical features parquet + manifest (full rebuild)."""
    sp = paths_for(symbol, timeframe)
    ohlcv = Path(ohlcv_path) if ohlcv_path else sp.parquet
    if not ohlcv.is_file():
        raise FileNotFoundError(f"OHLCV parquet missing: {ohlcv}")
    feat_path = features_parquet_for(symbol, timeframe)
    if not force and feat_path.is_file() and not is_stale(symbol, timeframe, config_path=config_path):
        return pd.read_parquet(feat_path)
    df = build_canonical_features(ohlcv, config_path=config_path, output_path=feat_path)
    write_manifest(symbol, timeframe, df, config_path=config_path, ohlcv_path=ohlcv)
    return df


def build_if_needed(
    symbol: str,
    timeframe: str = "1h",
    *,
    config_path: str | Path = DEFAULT_PROFILE,
    force: bool = False,
    ohlcv_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    if force or is_stale(symbol, timeframe, config_path=config_path):
        if not force and features_parquet_for(symbol, timeframe).is_file():
            import warnings

            warnings.warn(
                f"Feature cache stale for {symbol} {timeframe}; rebuilding.",
                stacklevel=2,
            )
        return build_features(
            symbol,
            timeframe,
            config_path=config_path,
            force=True,
            ohlcv_path=ohlcv_path,
        )
    return pd.read_parquet(features_parquet_for(symbol, timeframe))


def load_features(
    symbol: str,
    timeframe: str = "1h",
    *,
    max_rows: Optional[int] = None,
    config_path: str | Path = DEFAULT_PROFILE,
    use_cache: bool = True,
    force_rebuild: bool = False,
    ohlcv_path: Optional[str | Path] = None,
    columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Load features from cache or build from OHLCV.

    ``max_rows`` slices tail after load (no recompute).
    """
    if use_cache:
        df = build_if_needed(
            symbol,
            timeframe,
            config_path=config_path,
            force=force_rebuild,
            ohlcv_path=ohlcv_path,
        )
    else:
        sp = paths_for(symbol, timeframe)
        ohlcv = Path(ohlcv_path) if ohlcv_path else sp.parquet
        df = build_canonical_features(ohlcv, config_path=config_path)

    if columns:
        keep = [c for c in columns if c in df.columns]
        df = df[keep]
    if max_rows is not None and len(df) > max_rows:
        df = df.iloc[-max_rows:].copy()
    return df


def load_features_from_parquet(
    parquet_path: str | Path,
    *,
    symbol: Optional[str] = None,
    timeframe: str = "1h",
    max_rows: Optional[int] = None,
    use_feature_cache: bool = False,
    config_path: str | Path = DEFAULT_PROFILE,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """
    Entry for report/tune when caller passes OHLCV path.

    With ``use_feature_cache`` and ``symbol``, reads ``data/features/<slug>.parquet``.
    Otherwise falls back to inline FeatureEngine (legacy).
    """
    if use_feature_cache and symbol:
        sp = paths_for(symbol, timeframe)
        return load_features(
            symbol,
            timeframe,
            max_rows=max_rows,
            config_path=config_path,
            use_cache=True,
            force_rebuild=force_rebuild,
            ohlcv_path=parquet_path if Path(parquet_path).resolve() == sp.parquet.resolve() else parquet_path,
        )
    from orchestration.real_data_benchmark import features_from_ohlcv_parquet

    return features_from_ohlcv_parquet(parquet_path, max_rows=max_rows)
