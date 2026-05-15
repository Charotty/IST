"""Synchronization layer configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class SynchronizationConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    base_timeframe: str = "1h"
    auxiliary_timeframes: list[str] = Field(default_factory=lambda: ["15m", "4h"])
    resample_rule: str | None = None
    fill_method: str = "ffill"
    drop_na_after_merge: bool = True

    @property
    def rule(self) -> str:
        """Pandas resample rule (defaults to ``base_timeframe``)."""
        return self.resample_rule or self.base_timeframe

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SynchronizationConfig":
        with open(path, encoding="utf-8") as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}
        section = raw.get("synchronization", raw)
        return cls.model_validate(section)
