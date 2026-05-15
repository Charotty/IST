"""Feature engineering configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class BaseIndicatorsConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    ema_fast: int = 20
    ema_slow: int = 50
    rsi_length: int = 14
    atr_length: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    adx_length: int = 14
    vol_window: int = 20
    vol_annualize_factor: float = 24.0**0.5  # √24 for hourly bars


class MicrostructureConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    mode: str = "simulated"  # simulated | live | off
    depth: int = 20
    simulation_seed: int = 42


class FeatureEngineeringConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    base_indicators: BaseIndicatorsConfig = Field(default_factory=BaseIndicatorsConfig)
    microstructure: MicrostructureConfig = Field(default_factory=MicrostructureConfig)
    mtf_columns: list[str] = Field(
        default_factory=lambda: ["rsi_15m", "ema_slope_15m", "rsi_4h", "adx_4h"]
    )
    drop_na: bool = True

    @classmethod
    def from_yaml(cls, path: str | Path) -> "FeatureEngineeringConfig":
        with open(path, encoding="utf-8") as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}
        section = raw.get("feature_engineering", raw)
        return cls.model_validate(section)


# Etalon training set (Direction / DL) per README
DIRECTION_FEATURE_COLUMNS = [
    "rsi",
    "macd_hist",
    "ema_slope",
    "adx",
    "rsi_15m",
    "ema_slope_15m",
    "rsi_4h",
    "adx_4h",
]
