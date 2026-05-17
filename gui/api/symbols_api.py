"""Symbol list, paths, merged config — wraps ``orchestration.symbols``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from orchestration.canonical_pipeline import features_parquet_for
from orchestration.symbols import (
    list_known_symbols,
    merged_config,
    normalize_symbol,
    paths_for,
    slug as make_slug,
)

from gui.api.types import DataHealth, SymbolEntry


class SymbolsApi:
    def list_symbols(self) -> List[SymbolEntry]:
        known = list_known_symbols()
        out: List[SymbolEntry] = []
        for row in known:
            slug_name = row["slug"]
            parts = slug_name.rsplit("_", 1)
            if len(parts) == 2:
                sym, tf = parts[0], parts[1]
            else:
                sym, tf = slug_name, "1h"
            sp = paths_for(sym.replace("-", "/") if "/" not in sym else sym, tf)
            feat = features_parquet_for(sp.symbol, sp.timeframe)
            out.append(
                SymbolEntry(
                    slug=slug_name,
                    symbol=sp.symbol,
                    timeframe=sp.timeframe,
                    parquet_ohlcv=Path(row["parquet"]) if row.get("parquet") else sp.parquet,
                    parquet_features=feat if feat.is_file() else None,
                    config_yaml=Path(row["config"]) if row.get("config") else sp.config_yaml,
                    latest_bundle_run_id=row.get("latest"),
                    has_bundle=bool(row.get("latest")),
                )
            )
        return out

    def resolve(self, symbol: str, timeframe: str = "1h") -> SymbolEntry:
        sp = paths_for(symbol, timeframe)
        feat = features_parquet_for(sp.symbol, sp.timeframe)
        latest = None
        bundle = sp.latest_bundle()
        if bundle is not None:
            latest = bundle.name
        return SymbolEntry(
            slug=sp.slug,
            symbol=sp.symbol,
            timeframe=sp.timeframe,
            parquet_ohlcv=sp.parquet,
            parquet_features=feat if feat.is_file() else None,
            config_yaml=sp.config_yaml if sp.config_yaml.is_file() else None,
            latest_bundle_run_id=latest,
            has_bundle=bundle is not None,
        )

    def merged_config(self, symbol: str, timeframe: str = "1h") -> Dict[str, Any]:
        return merged_config(symbol, timeframe)

    def data_health(self, symbol: str, timeframe: str = "1h") -> DataHealth:
        sp = paths_for(symbol, timeframe)
        feat_path = features_parquet_for(sp.symbol, sp.timeframe)
        health = DataHealth(
            slug=sp.slug,
            ohlcv_path=sp.parquet,
            features_path=feat_path,
            ohlcv_exists=sp.parquet.is_file(),
            features_exists=feat_path.is_file(),
        )
        if health.ohlcv_exists:
            import pandas as pd

            df = pd.read_parquet(sp.parquet, columns=["close"])
            health.ohlcv_rows = len(df)
            if len(df):
                health.ohlcv_last_ts = str(df.index[-1])
        if health.features_exists:
            import pandas as pd

            df = pd.read_parquet(feat_path, columns=["close"])
            health.features_rows = len(df)
            if len(df):
                health.features_last_ts = str(df.index[-1])
        return health

    @staticmethod
    def normalize_symbol(symbol: str) -> str:
        return normalize_symbol(symbol)

    @staticmethod
    def slug(symbol: str, timeframe: str) -> str:
        return make_slug(symbol, timeframe)
