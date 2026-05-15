"""Configuration for the data layer (OHLCV loading)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class MultiTimeframeConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")
    enabled: bool = True
    timeframes: list[str] = Field(default_factory=lambda: ["15m", "4h"])


class DataLayerConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    exchange: str = "okx"
    symbol: str = "BTC/USDT"
    timeframe: str = "1h"
    start_date: str = "2020-01-01 00:00:00"
    end_date: str = "2026-01-01 00:00:00"
    multi_timeframe: MultiTimeframeConfig = Field(default_factory=MultiTimeframeConfig)
    rate_limit: bool = True

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DataLayerConfig":
        with open(path, encoding="utf-8") as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}
        section = raw.get("data_layer", raw)
        return cls.model_validate(section)

    def all_timeframes(self) -> list[str]:
        """Base timeframe first, then additional MTF intervals (no duplicates)."""
        tfs = [self.timeframe]
        if self.multi_timeframe.enabled:
            for tf in self.multi_timeframe.timeframes:
                if tf not in tfs:
                    tfs.append(tf)
        return tfs
