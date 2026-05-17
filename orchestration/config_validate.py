"""
Валидация сквозного ``config.yaml`` при старте пайплайна.

Проверяет секции ``orchestration`` и ``feature_engineering`` (pydantic).
Неизвестные ключи в ``orchestration`` отбрасываются в ``OrchestratorConfig.from_yaml``
с предупреждением — см. ``validate_pipeline_config`` для отчёта.
"""

from __future__ import annotations

import warnings
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from orchestration.orchestrator_config import OrchestratorConfig
from feature_engineering.config import FeatureEngineeringConfig


def _orch_field_names() -> set:
    return {f.name for f in fields(OrchestratorConfig)}


def validate_pipeline_config(yaml_path: Path) -> Tuple[List[str], List[str]]:
    """
    Возвращает ``(errors, warnings)`` после разбора конфига.

    ``errors`` — блокирующие проблемы; ``warnings`` — в т.ч. неизвестные ключи YAML.
    """
    errors: List[str] = []
    warns: List[str] = []
    path = Path(yaml_path)
    if not path.is_file():
        return ([f"Config file not found: {path}"], [])

    try:
        raw: Dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as e:
        return ([f"YAML parse error: {e}"], [])

    orch = raw.get("orchestration")
    if not isinstance(orch, dict):
        errors.append("Missing or invalid 'orchestration' mapping")
    else:
        unknown_o = set(orch) - _orch_field_names()
        if unknown_o:
            warns.append(
                f"orchestration: keys not in OrchestratorConfig (will be ignored): {sorted(unknown_o)}"
            )
        try:
            OrchestratorConfig.from_yaml(path)
        except Exception as e:
            errors.append(f"OrchestratorConfig: {e}")

    try:
        FeatureEngineeringConfig.from_yaml(path)
    except Exception as e:
        errors.append(f"FeatureEngineeringConfig: {e}")

    for section in ("data_layer", "logging"):
        if section not in raw:
            warns.append(f"Optional section '{section}' absent (OK if unused)")

    return (errors, warns)


def raise_if_invalid(yaml_path: Path) -> None:
    """Бросает ``ValueError``, если есть ошибки валидации."""
    errors, warns = validate_pipeline_config(yaml_path)
    for w in warns:
        warnings.warn(w, UserWarning, stacklevel=2)
    if errors:
        raise ValueError("; ".join(errors))