"""Save OHLCV DataFrames to disk."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def default_output_path(
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
    output_dir: str | Path = "data/ohlcv",
    fmt: str = "parquet",
) -> Path:
    """Build ``{output_dir}/{SYMBOL}_{tf}_{start}_{end}.parquet``."""
    safe_symbol = symbol.replace("/", "-")
    start = pd.Timestamp(start_date).strftime("%Y%m%d")
    end = pd.Timestamp(end_date).strftime("%Y%m%d")
    ext = "parquet" if fmt == "parquet" else "csv"
    return Path(output_dir) / f"{safe_symbol}_{timeframe}_{start}_{end}.{ext}"


def save_ohlcv(df: pd.DataFrame, path: str | Path, fmt: str | None = None) -> Path:
    """Write OHLCV to ``path``; format from extension if ``fmt`` is omitted."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    file_fmt = (fmt or out.suffix.lstrip(".")).lower()
    if file_fmt == "parquet":
        df.to_parquet(out)
    elif file_fmt == "csv":
        df.to_csv(out)
    else:
        raise ValueError(f"Unsupported format: {file_fmt!r} (use parquet or csv)")
    return out.resolve()
