"""
Конфигурация расчёта метрик бэктеста (частота баров, risk-free, издержки).
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# Часовые бары: 24 * 365
DEFAULT_BARS_PER_YEAR = 8760


@dataclass
class MetricsConfig:
    bars_per_year: int = DEFAULT_BARS_PER_YEAR
    risk_free_rate: float = 0.02
    commission: float = 0.0006
    slippage: float = 0.0002


@dataclass
class CriteriaThresholds:
    """Пороги уровня (а) acceptable или (б) target."""

    min_oos_profit_factor: float = 1.2
    min_oos_sharpe: float = 0.5
    min_wfe: float = 0.5
    max_drawdown_pct: float = -35.0
    min_recovery_factor: float = 1.0
    min_folds: int = 5
    min_trade_events: int = 30
    min_total_return_pct: float = 0.0
    min_calmar: Optional[float] = None
    min_sortino: Optional[float] = None


@dataclass
class BacktestingConfig:
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    acceptance: CriteriaThresholds = field(
        default_factory=lambda: CriteriaThresholds(
            min_oos_profit_factor=1.2,
            min_oos_sharpe=0.5,
            min_wfe=0.5,
            max_drawdown_pct=-35.0,
            min_recovery_factor=1.0,
            min_folds=5,
            min_trade_events=30,
            min_total_return_pct=0.0,
        )
    )
    target: CriteriaThresholds = field(
        default_factory=lambda: CriteriaThresholds(
            min_oos_profit_factor=1.6,
            min_oos_sharpe=1.0,
            min_wfe=0.7,
            max_drawdown_pct=-20.0,
            min_recovery_factor=2.0,
            min_folds=8,
            min_trade_events=100,
            min_total_return_pct=0.0,
            min_calmar=1.0,
            min_sortino=0.5,
        )
    )


def _criteria_from_dict(d: Optional[Dict[str, Any]]) -> CriteriaThresholds:
    if not d:
        return CriteriaThresholds()
    known = {f.name for f in fields(CriteriaThresholds)}
    return CriteriaThresholds(**{k: v for k, v in d.items() if k in known})


def load_backtesting_config(yaml_path: str | Path = "config.yaml") -> BacktestingConfig:
    path = Path(yaml_path)
    if not path.is_file():
        return BacktestingConfig()

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    bt = raw.get("backtesting") or {}
    sim = bt.get("simulation") or {}

    metrics = MetricsConfig(
        bars_per_year=int(bt.get("bars_per_year", DEFAULT_BARS_PER_YEAR)),
        risk_free_rate=float((bt.get("analysis") or {}).get("risk_free_rate", 0.02)),
        commission=float(sim.get("commission", 0.0006)),
        slippage=float(sim.get("slippage", 0.0002)),
    )
    return BacktestingConfig(
        metrics=metrics,
        acceptance=_criteria_from_dict(bt.get("acceptance")),
        target=_criteria_from_dict(bt.get("target")),
    )
