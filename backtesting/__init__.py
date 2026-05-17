"""
Backtesting Layer

Модуль для симуляции стратегии на истории, walk-forward валидации, метрик и логов экспериментов.
"""

from .backtester import Backtester
from .performance_metrics import PerformanceMetrics
from .metrics_config import (
    BacktestingConfig,
    CriteriaThresholds,
    MetricsConfig,
    load_backtesting_config,
)
from .criteria_evaluator import (
    evaluate_backtest_levels,
    evaluate_criteria,
    summarize_wfo_folds,
)
from .results_journal import BacktestResultsJournal
from .time_series_splitter import TimeSeriesSplitter
from .walk_forward import (
    run_walk_forward_validation,
    run_integrated_wfo,
    run_orchestrator_walk_forward_backtest,
    run_simple_signal_wfo,
)
from .experiment_logger import ExperimentLogger

__all__ = [
    'Backtester',
    'PerformanceMetrics',
    'MetricsConfig',
    'BacktestingConfig',
    'CriteriaThresholds',
    'load_backtesting_config',
    'evaluate_criteria',
    'evaluate_backtest_levels',
    'summarize_wfo_folds',
    'BacktestResultsJournal',
    'TimeSeriesSplitter',
    'run_walk_forward_validation',
    'run_integrated_wfo',
    'run_orchestrator_walk_forward_backtest',
    'run_simple_signal_wfo',
    'ExperimentLogger',
]
