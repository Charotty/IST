"""Unit tests for extended PerformanceMetrics."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtesting.backtester import Backtester
from backtesting.performance_metrics import PerformanceMetrics


def _synthetic_backtest(n: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    close = 100 + rng.standard_normal(n).cumsum() * 0.2
    df = pd.DataFrame({"close": close})
    signals = np.where(rng.random(n) > 0.55, 1, np.where(rng.random(n) < 0.45, -1, 0))
    return Backtester().run(df, pd.Series(signals))


def test_extended_metrics_present():
    perf = _synthetic_backtest()
    m = PerformanceMetrics(perf).calculate_metrics()
    for key in (
        "Sharpe Ratio",
        "Sortino Ratio",
        "Calmar Ratio",
        "CAGR (%)",
        "Profit Factor",
        "Max Drawdown (%)",
        "Recovery Factor",
        "Ulcer Index",
        "Time Underwater (bars)",
        "Max Consecutive Losses",
        "Trade Events",
        "Alpha vs Benchmark (%)",
    ):
        assert key in m.index
    assert np.isfinite(m["Sharpe Ratio"])


def test_annualized_return_positive():
    ar = PerformanceMetrics.annualized_return(0.1, 8760, 8760)
    assert abs(ar - 0.1) < 1e-6
