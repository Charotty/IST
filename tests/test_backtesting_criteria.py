"""Criteria evaluator on synthetic fold table."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtesting.criteria_evaluator import evaluate_backtest_levels, summarize_wfo_folds
from backtesting.metrics_config import BacktestingConfig, CriteriaThresholds


def test_summarize_and_acceptance_pass():
    folds = pd.DataFrame(
        [
            {
                "Fold": 1,
                "Sharpe Ratio": 0.8,
                "Profit Factor": 1.5,
                "Total Return (%)": 2.0,
                "Walk-Forward Efficiency": 0.6,
                "Max Drawdown (%)": -10.0,
                "Recovery Factor": 1.5,
                "Trade Events": 40,
                "Calmar Ratio": 1.2,
                "Sortino Ratio": 0.9,
            },
            {
                "Fold": 2,
                "Sharpe Ratio": 0.7,
                "Profit Factor": 1.4,
                "Total Return (%)": 1.0,
                "Walk-Forward Efficiency": 0.55,
                "Max Drawdown (%)": -12.0,
                "Recovery Factor": 1.3,
                "Trade Events": 35,
                "Calmar Ratio": 1.0,
                "Sortino Ratio": 0.8,
            },
        ]
    )
    summary = summarize_wfo_folds(folds)
    assert summary["n_folds"] == 2
    assert summary["mean_sharpe"] > 0.5

    cfg = BacktestingConfig(
        acceptance=CriteriaThresholds(
            min_folds=2,
            min_trade_events=50,
            min_oos_profit_factor=1.2,
            min_oos_sharpe=0.5,
            min_wfe=0.5,
            max_drawdown_pct=-35.0,
            min_recovery_factor=1.0,
            min_total_return_pct=0.0,
        )
    )
    ev = evaluate_backtest_levels(folds, cfg)
    assert ev["acceptance"]["passed"] is True
