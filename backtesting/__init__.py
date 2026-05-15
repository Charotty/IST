"""
Backtesting Layer

Модуль для симуляции стратегии на истории, walk-forward валидации, метрик и логов экспериментов.
"""

from .backtester import Backtester
from .performance_metrics import PerformanceMetrics
from .time_series_splitter import TimeSeriesSplitter
from .walk_forward import run_walk_forward_validation, run_integrated_wfo
from .experiment_logger import ExperimentLogger

__all__ = [
    'Backtester',
    'PerformanceMetrics',
    'TimeSeriesSplitter',
    'run_walk_forward_validation',
    'run_integrated_wfo',
    'ExperimentLogger',
]
