"""Persist feature datasets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_features(df: pd.DataFrame, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    suffix = out.suffix.lower()
    if suffix == ".csv":
        df.to_csv(out)
    else:
        df.to_parquet(out)
    return out.resolve()
