"""
Canonical data path: OKX OHLCV → MTF merge → FeatureManager (profile YAML).

Used by ``prepare-symbol`` and ablation scripts with ``config/profiles/canonical_4model.yaml``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import yaml

from feature_engineering.config import FeatureEngineeringConfig
from feature_engineering.feature_manager import FeatureManager
from feature_engineering.storage import save_features
from orchestration.benchmark_runner import load_canonical_config_path
from orchestration.symbols import DATA_DIR, REPO_ROOT, paths_for
from synchronization.config import SynchronizationConfig
from synchronization.multi_timeframe_engine import MultiTimeframeEngine

FEATURES_DIR = REPO_ROOT / "data" / "features"
PROFILE_NAME = "canonical_4model"

# ~90% of hourly bars from start→end (OKX gaps tolerated)
_MIN_COVERAGE_RATIO = 0.85
_TF_HOURS = {"1h": 1.0, "4h": 4.0, "15m": 0.25}


def resolve_profile_path(config_path: str | Path = "config.yaml") -> Path:
    return load_canonical_config_path(config_path)


def load_profile_yaml(config_path: str | Path = "config.yaml") -> Dict[str, Any]:
    path = resolve_profile_path(config_path)
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def expected_bar_count(start_date: str, end_date: str, timeframe: str) -> int:
    """Rough expected bars for OKX history sanity checks."""
    start = pd.Timestamp(start_date, tz="UTC")
    end = pd.Timestamp(end_date, tz="UTC")
    hours = max(0.0, (end - start).total_seconds() / 3600.0)
    bar_h = _TF_HOURS.get(timeframe.lower(), 1.0)
    return max(1, int(hours / bar_h * _MIN_COVERAGE_RATIO))


def _validate_ohlcv_coverage(
    df: pd.DataFrame,
    *,
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
) -> None:
    need = expected_bar_count(start_date, end_date, timeframe)
    got = len(df)
    if got >= need:
        return
    last = df.index[-1] if got else None
    raise RuntimeError(
        f"Incomplete OKX download for {symbol} {timeframe}: got {got} bars, "
        f"expected at least ~{need} ({start_date} → {end_date}). "
        f"Last bar: {last}. "
        "Often caused by rate limits when fetching 1h+15m+4h in sequence — retry "
        "`prepare-symbol --download` or run downloads separately."
    )


def data_collection_dates(config_path: str | Path = "config.yaml") -> Tuple[str, str]:
    """``(start_date, end_date)`` from profile ``data_collection`` (end defaults to UTC now)."""
    raw = load_profile_yaml(config_path)
    dc = raw.get("data_collection") or {}
    start = str(dc.get("start_date", "2022-01-01 00:00:00"))
    end = dc.get("end_date")
    if end is None or str(end).strip() in ("", "null", "None"):
        end = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    return start, str(end)


def features_parquet_for(symbol: str, timeframe: str = "1h") -> Path:
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    return FEATURES_DIR / f"{paths_for(symbol, timeframe).slug}.parquet"


def download_ohlcv_frame(
    symbol: str,
    timeframe: str,
    *,
    start_date: str,
    end_date: str,
    rate_limit: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    from data_layer.loaders.okx_ohlcv_loader import OKXDataLoader

    loader = OKXDataLoader(rate_limit=rate_limit)
    sym = paths_for(symbol, timeframe).symbol.replace("-", "/")
    df = loader.fetch_all_ohlcv(sym, timeframe, start_date, end_date, verbose=verbose)
    if df.empty:
        raise RuntimeError(f"No OHLCV for {sym} {timeframe}")
    return df


def download_canonical_ohlcv(
    symbol: str,
    timeframe: str = "1h",
    *,
    config_path: str | Path = "config.yaml",
    rate_limit: bool = True,
    verbose: bool = False,
    merge_mtf: bool = True,
) -> Path:
    """
  Download base OHLCV (+ auxiliary TFs), optionally merge MTF columns, save to ``data/ohlcv/<slug>.parquet``.
    """
    from data_layer.storage import save_ohlcv

    profile = resolve_profile_path(config_path)
    start_date, end_date = data_collection_dates(profile)
    sp = paths_for(symbol, timeframe)
    sp.parquet.parent.mkdir(parents=True, exist_ok=True)

    sym = paths_for(symbol, timeframe).symbol.replace("-", "/")

    def _fetch(tf: str) -> pd.DataFrame:
        df = download_ohlcv_frame(
            symbol,
            tf,
            start_date=start_date,
            end_date=end_date,
            rate_limit=rate_limit,
            verbose=verbose,
        )
        _validate_ohlcv_coverage(
            df,
            symbol=sym,
            timeframe=tf,
            start_date=start_date,
            end_date=end_date,
        )
        return df

    if not merge_mtf:
        base_df = _fetch(timeframe)
        save_ohlcv(base_df, sp.parquet, fmt="parquet")
        return sp.parquet

    sync_cfg = SynchronizationConfig.from_yaml(profile)
    sync_cfg = sync_cfg.model_copy(update={"drop_na_after_merge": False})
    aux_order = sorted(
        sync_cfg.auxiliary_timeframes,
        key=lambda tf: expected_bar_count(start_date, end_date, tf),
    )
    aux_frames: Dict[str, pd.DataFrame] = {}
    for tf in aux_order:
        aux_sp = paths_for(symbol, tf)
        aux_df = _fetch(tf)
        save_ohlcv(aux_df, aux_sp.parquet, fmt="parquet")
        aux_frames[tf] = aux_df

    # Base 1h last (most requests) after aux TFs succeeded.
    base_df = _fetch(timeframe)
    raw_path = sp.parquet.with_name(f"{sp.slug}_ohlcv_raw.parquet")
    save_ohlcv(base_df, raw_path, fmt="parquet")

    engine = MultiTimeframeEngine(config=sync_cfg, frames=aux_frames)
    merged = engine.compute_and_merge(base_df)
    mtf_cols = list(FeatureEngineeringConfig.from_yaml(profile).mtf_columns)
    if mtf_cols:
        present = [c for c in mtf_cols if c in merged.columns]
        if present:
            merged[present] = merged[present].ffill()
    save_ohlcv(merged, sp.parquet, fmt="parquet")
    return sp.parquet


def merge_mtf_from_disk(
    symbol: str,
    timeframe: str = "1h",
    *,
    config_path: str | Path = "config.yaml",
) -> Path:
    """Merge existing auxiliary parquets onto base OHLCV (no download)."""
    from data_layer.storage import save_ohlcv

    profile = resolve_profile_path(config_path)
    sp = paths_for(symbol, timeframe)
    if not sp.parquet.is_file():
        raise FileNotFoundError(sp.parquet)
    sync_cfg = SynchronizationConfig.from_yaml(profile)
    aux_frames: Dict[str, pd.DataFrame] = {}
    for tf in sync_cfg.auxiliary_timeframes:
        aux_path = paths_for(symbol, tf).parquet
        if not aux_path.is_file():
            raise FileNotFoundError(f"Missing auxiliary parquet: {aux_path}")
        aux_frames[tf] = pd.read_parquet(aux_path)
    base_df = pd.read_parquet(sp.parquet)
    engine = MultiTimeframeEngine(config=sync_cfg, frames=aux_frames)
    merged = engine.compute_and_merge(base_df)
    save_ohlcv(merged, sp.parquet, fmt="parquet")
    return sp.parquet


def build_canonical_features(
    ohlcv_path: str | Path,
    *,
    config_path: str | Path = "config.yaml",
    output_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """FeatureManager transform using ``feature_engineering`` section from profile."""
    profile = resolve_profile_path(config_path)
    fe_cfg = FeatureEngineeringConfig.from_yaml(profile)
    df = pd.read_parquet(ohlcv_path)
    result = FeatureManager(fe_cfg).transform(df)
    if output_path is not None:
        save_features(result, output_path)
    return result


def prepare_canonical_dataset(
    symbol: str,
    timeframe: str = "1h",
    *,
    config_path: str | Path = "config.yaml",
    download: bool = False,
    rate_limit: bool = True,
    verbose: bool = False,
) -> Tuple[Path, Path]:
    """
    Full phase-1 data path: OHLCV (+MTF) → features parquet.

    Returns ``(ohlcv_parquet, features_parquet)``.
    """
    sp = paths_for(symbol, timeframe)
    if download or not sp.parquet.is_file():
        download_canonical_ohlcv(
            symbol,
            timeframe,
            config_path=config_path,
            rate_limit=rate_limit,
            verbose=verbose,
        )
    elif _needs_mtf_columns(sp.parquet, config_path):
        merge_mtf_from_disk(symbol, timeframe, config_path=config_path)

    feat_path = features_parquet_for(symbol, timeframe)
    build_canonical_features(sp.parquet, config_path=config_path, output_path=feat_path)
    return sp.parquet, feat_path


def _needs_mtf_columns(ohlcv_path: Path, config_path: str | Path) -> bool:
    fe_cfg = FeatureEngineeringConfig.from_yaml(resolve_profile_path(config_path))
    if not fe_cfg.mtf_columns:
        return False
    cols = set(pd.read_parquet(ohlcv_path).columns)
    return not all(c in cols for c in fe_cfg.mtf_columns)


def journal_profile_tag(config_path: str | Path = "config.yaml") -> str:
    path = resolve_profile_path(config_path)
    if path.name == "canonical_4model.yaml":
        return PROFILE_NAME
    return path.stem
