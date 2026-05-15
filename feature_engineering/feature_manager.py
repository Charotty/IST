"""End-to-end feature pipeline: indicators + optional microstructure."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from feature_engineering.config import (
    DIRECTION_FEATURE_COLUMNS,
    FeatureEngineeringConfig,
)
from feature_engineering.feature_engine import FeatureEngine
from feature_engineering.microstructure.simulator import simulate_l2_features


class FeatureManager:
    """Orchestrates ``FeatureEngine`` and microstructure modes."""

    def __init__(self, config: FeatureEngineeringConfig | None = None):
        self.config = config or FeatureEngineeringConfig()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        engine = FeatureEngine(df, self.config)
        result = engine.add_indicators().get_processed_data()
        return self._apply_microstructure(result)

    def _apply_microstructure(self, df: pd.DataFrame) -> pd.DataFrame:
        mode = self.config.microstructure.mode.lower()
        if mode == "off":
            return df
        if mode == "simulated":
            return simulate_l2_features(
                df, seed=self.config.microstructure.simulation_seed
            )
        if mode == "live":
            raise NotImplementedError(
                "live microstructure requires L2 snapshots via data_layer; "
                "use l2_adapter.align_l2_series() and join manually for now"
            )
        raise ValueError(f"Unknown microstructure.mode: {mode!r}")

    def direction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return only the standard Direction/DL feature matrix (must exist on ``df``)."""
        missing = [c for c in DIRECTION_FEATURE_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing direction features: {missing}")
        return df[DIRECTION_FEATURE_COLUMNS]

    @classmethod
    def from_parquet(
        cls,
        input_path: str | Path,
        config: FeatureEngineeringConfig | None = None,
    ) -> pd.DataFrame:
        df = pd.read_parquet(input_path)
        return cls(config).transform(df)
