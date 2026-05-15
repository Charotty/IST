"""Persist synchronized datasets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_synced(df: pd.DataFrame, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    return out.resolve()
