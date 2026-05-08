from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


class DataCleaner:
    """Small deterministic cleaning helpers for tabular market data."""

    def remove_duplicates(self, data: pd.DataFrame) -> pd.DataFrame:
        if "timestamp" in data.columns:
            return data.drop_duplicates(subset=["timestamp"], keep="first").reset_index(drop=True)
        return data.drop_duplicates(keep="first").reset_index(drop=True)

    def handle_missing_values(self, data: pd.DataFrame, method: str = "forward_fill") -> pd.DataFrame:
        cleaned = data.copy()
        if method in {"forward_fill", "ffill", "forward"}:
            cleaned = cleaned.ffill().bfill()
        elif method in {"backward_fill", "bfill", "backward"}:
            cleaned = cleaned.bfill().ffill()
        elif method == "drop":
            cleaned = cleaned.dropna()
        elif method == "zero":
            cleaned = cleaned.fillna(0)
        else:
            raise ValueError(f"Unknown missing-value method: {method}")
        return cleaned

    def remove_outliers(
        self,
        data: pd.DataFrame,
        columns: Iterable[str] | None = None,
        std_threshold: float = 5.0,
    ) -> pd.DataFrame:
        cleaned = data.copy()
        columns = list(columns or cleaned.select_dtypes(include=[np.number]).columns)
        mask = pd.Series(True, index=cleaned.index)
        for column in columns:
            if column not in cleaned.columns:
                continue
            series = cleaned[column]
            median = series.median()
            mad = (series - median).abs().median()
            if mad == 0 or pd.isna(mad):
                std = series.std()
                if std == 0 or pd.isna(std):
                    continue
                score = (series - series.mean()).abs() / std
            else:
                score = 0.6745 * (series - median).abs() / mad
            mask &= score <= std_threshold
        return cleaned.loc[mask].reset_index(drop=True)

    def replace_outliers(
        self,
        data: pd.DataFrame,
        columns: Iterable[str] | None = None,
        std_threshold: float = 5.0,
    ) -> pd.DataFrame:
        cleaned = data.copy()
        columns = list(columns or cleaned.select_dtypes(include=[np.number]).columns)
        for column in columns:
            if column not in cleaned.columns:
                continue
            series = cleaned[column]
            median = series.median()
            mad = (series - median).abs().median()
            if mad == 0 or pd.isna(mad):
                std = series.std()
                if std == 0 or pd.isna(std):
                    continue
                score = (series - series.mean()).abs() / std
            else:
                score = 0.6745 * (series - median).abs() / mad
            cleaned.loc[score > std_threshold, column] = median
        return cleaned.reset_index(drop=True)

    def validate_types(self, data: pd.DataFrame) -> bool:
        numeric_columns = [c for c in data.columns if c != "timestamp"]
        return all(pd.api.types.is_numeric_dtype(data[column]) for column in numeric_columns)

    def validate_schema(self, data: pd.DataFrame, required_columns: Iterable[str]) -> bool:
        return set(required_columns).issubset(data.columns)

    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        if "timestamp" in data.columns:
            duplicate_fraction = data.duplicated(subset=["timestamp"]).mean()
        else:
            duplicate_fraction = data.duplicated().mean()
        cleaned = data.copy() if duplicate_fraction > 0.5 else self.remove_duplicates(data)
        cleaned = self.handle_missing_values(cleaned, method="forward_fill")
        numeric_columns = [c for c in ["open", "high", "low", "close", "volume"] if c in cleaned.columns]
        if numeric_columns:
            cleaned[numeric_columns] = cleaned[numeric_columns].astype(float)
            cleaned = self.replace_outliers(cleaned, numeric_columns, std_threshold=5.0)
            cleaned = self.handle_missing_values(cleaned, method="forward_fill")
        return cleaned.reset_index(drop=True)
