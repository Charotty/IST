"""
Walk-forward validation — совместимо с ``TrainingOrchestrator`` и ``Backtester``.

Удалены импорты несуществующих ``models.direction_model`` / ``models.dl_model_builder``.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Optional

import pandas as pd

from .time_series_splitter import TimeSeriesSplitter
from .backtester import Backtester
from .performance_metrics import PerformanceMetrics

if TYPE_CHECKING:
    from orchestration.training_orchestrator import TrainingOrchestrator


def run_orchestrator_walk_forward_backtest(
    orchestrator: "TrainingOrchestrator",
    features: pd.DataFrame,
    targets: pd.Series,
    *,
    commission: float = 0.0006,
    slippage: float = 0.0002,
) -> pd.DataFrame:
    """
    Обертка над ``TrainingOrchestrator.walk_forward_backtest`` (единый WFO + метрики).
    """
    return orchestrator.walk_forward_backtest(
        features, targets, commission=commission, slippage=slippage
    )


def run_walk_forward_validation(df, n_splits=5):
    """
    .. deprecated::
        Использовал несуществующие модули. Вызывайте
        ``run_orchestrator_walk_forward_backtest`` с инициализированным
        ``TrainingOrchestrator`` или ``python -m orchestration`` для smoke-теста.
    """
    warnings.warn(
        "run_walk_forward_validation is removed: use TrainingOrchestrator.walk_forward_backtest "
        "or run_orchestrator_walk_forward_backtest. See orchestration.integration_smoke.",
        DeprecationWarning,
        stacklevel=2,
    )
    raise NotImplementedError(
        "Старый путь (DirectionModel) удалён. Используйте TrainingOrchestrator.walk_forward_backtest("
        "features, targets) при тех же model_keys, что в OrchestratorConfig / config.yaml."
    )


def run_integrated_wfo(df, n_splits=3, window_size=24, features=None):
    """
    .. deprecated::
        См. ``run_walk_forward_validation``.
    """
    warnings.warn(
        "run_integrated_wfo is removed: use TrainingOrchestrator.walk_forward_backtest.",
        DeprecationWarning,
        stacklevel=2,
    )
    raise NotImplementedError(
        "Старый integrated WFO (DLModelBuilder) удалён. Используйте TrainingOrchestrator "
        "с полным набором моделей из config и walk_forward_backtest."
    )


def run_simple_signal_wfo(
    df: pd.DataFrame,
    signal_column: str,
    n_splits: int = 5,
    train_size: float = 0.7,
    commission: float = 0.0006,
    slippage: float = 0.0002,
    position_size_column: Optional[str] = None,
) -> pd.DataFrame:
    """
    Простой WFO без ML: колонка сигнала уже в ``df`` (например после batch inference).

    Полезно для проверки связки «сигнал → Backtester» на готовых данных.
    """
    if "close" not in df.columns:
        raise ValueError("df must contain 'close'")
    if signal_column not in df.columns:
        raise ValueError(f"Missing signal column: {signal_column}")

    splitter = TimeSeriesSplitter(n_splits=n_splits, train_size=train_size)
    stats = []
    bt = Backtester(commission=commission, slippage=slippage)

    for i, (_, test_chunk) in enumerate(splitter.split(df)):
        sig = test_chunk[signal_column]
        if position_size_column and position_size_column in test_chunk.columns:
            pos = test_chunk[position_size_column].values
        else:
            pos = None
        perf = bt.run(test_chunk[["close"]], sig, position_size=pos)
        m = PerformanceMetrics(perf).calculate_metrics()
        m["Fold"] = i + 1
        stats.append(m)

    return pd.DataFrame(stats)
