from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency fallback
    yaml = None


DEFAULT_CONFIG: dict[str, Any] = {
    "data": {
        "exchange": "okx",
        "symbols": ["BTC/USDT"],
        "timeframe": "1h",
    },
    "target": {
        "horizon": 5,
        "threshold": 0.002,
        "target_type": "direction",
    },
    "features": {
        "return_periods": [1, 5, 15],
        "volatility_windows": [5, 15],
    },
    "model": {
        "type": "economic_boosting",
        "n_estimators": 100,
        "learning_rate": 0.05,
        "max_depth": 3,
    },
    "backtesting": {
        "initial_capital": 10000.0,
        "commission_rate": 0.001,
        "slippage_rate": 0.0005,
    },
}


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load project configuration from JSON/YAML or return defaults."""
    if path is None:
        for candidate in (Path("its_project/config.yaml"), Path("config.json")):
            if candidate.exists():
                path = candidate
                break

    if path is None:
        return DEFAULT_CONFIG.copy()

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(config_path)

    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() == ".json":
        loaded = json.loads(text)
    elif config_path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required to load YAML configuration")
        loaded = yaml.safe_load(text) or {}
    else:
        raise ValueError(f"Unsupported config format: {config_path.suffix}")

    return _deep_merge(DEFAULT_CONFIG, loaded)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = {key: value.copy() if isinstance(value, dict) else value for key, value in base.items()}
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
