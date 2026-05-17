"""Чтение/запись per-symbol конфига для GUI."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

from orchestration.symbols import merged_config, paths_for, tuning_best_for, write_symbol_config


class ConfigApi:
    """Редактирование ``config/symbols/<slug>.yaml`` (orchestration_tuning_best)."""

    def orchestration_section(self, symbol: str, timeframe: str = "1h") -> Dict[str, Any]:
        tb = tuning_best_for(symbol, timeframe)
        if tb:
            return dict(tb)
        merged = merged_config(symbol, timeframe)
        orch = merged.get("orchestration") or {}
        return dict(orch) if isinstance(orch, dict) else {}

    def merged(self, symbol: str, timeframe: str = "1h") -> Dict[str, Any]:
        return merged_config(symbol, timeframe)

    def symbol_config_path(self, symbol: str, timeframe: str = "1h") -> Path:
        return paths_for(symbol, timeframe).config_yaml

    def save_orchestration(
        self,
        symbol: str,
        timeframe: str,
        values: Dict[str, Any],
    ) -> Path:
        clean = {k: v for k, v in values.items() if v is not None}
        return write_symbol_config(symbol, timeframe, tuning_best=clean)

    def save_raw_yaml(self, symbol: str, timeframe: str, text: str) -> Path:
        path = self.symbol_config_path(symbol, timeframe)
        path.parent.mkdir(parents=True, exist_ok=True)
        # проверка синтаксиса
        parsed = yaml.safe_load(text) or {}
        if not isinstance(parsed, dict):
            raise ValueError("YAML должен быть объектом (mapping)")
        path.write_text(text, encoding="utf-8")
        return path

    def load_raw_yaml(self, symbol: str, timeframe: str) -> Tuple[str, bool]:
        path = self.symbol_config_path(symbol, timeframe)
        if not path.is_file():
            base = {
                "symbol": paths_for(symbol, timeframe).symbol,
                "timeframe": timeframe,
                "orchestration_tuning_best": self.orchestration_section(symbol, timeframe),
            }
            return yaml.dump(base, default_flow_style=False, allow_unicode=True), False
        return path.read_text(encoding="utf-8"), True

    def test_pipeline_ready(self, symbol: str, timeframe: str = "1h") -> Dict[str, Any]:
        """Быстрая проверка: parquet, features, bundle."""
        sp = paths_for(symbol, timeframe)
        from orchestration.canonical_pipeline import features_parquet_for

        feat = features_parquet_for(sp.symbol, sp.timeframe)
        bundle = sp.latest_bundle()
        checks = {
            "ohlcv": sp.parquet.is_file(),
            "features": feat.is_file(),
            "symbol_config": sp.config_yaml.is_file(),
            "artifact_bundle": bundle is not None,
        }
        return {
            "ok": all(checks.values()),
            "checks": checks,
            "bundle_run_id": bundle.name if bundle else None,
        }
