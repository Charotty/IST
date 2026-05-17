"""
PerformanceMetrics — расчёт метрик эффективности стратегии (бэктест).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .metrics_config import DEFAULT_BARS_PER_YEAR, MetricsConfig


class PerformanceMetrics:
    def __init__(
        self,
        backtest_results: pd.DataFrame,
        config: Optional[MetricsConfig] = None,
    ):
        """
        :param backtest_results: DataFrame с результатами от Backtester.run()
        """
        self.results = backtest_results
        self.config = config or MetricsConfig()
        self.returns = self.results["net_returns"].fillna(0)
        self._bars_per_year = self.config.bars_per_year

    @staticmethod
    def annualized_return(
        total_return: float,
        n_bars: int,
        bars_per_year: int = DEFAULT_BARS_PER_YEAR,
    ) -> float:
        if n_bars <= 0:
            return 0.0
        base = 1.0 + float(total_return)
        if base <= 0:
            return -1.0
        return base ** (bars_per_year / n_bars) - 1.0

    @staticmethod
    def max_consecutive_losses(returns: pd.Series) -> int:
        losing = (returns < 0).astype(int)
        max_streak = 0
        current = 0
        for v in losing:
            if v == 1:
                current += 1
                max_streak = max(max_streak, current)
            else:
                current = 0
        return max_streak

    @staticmethod
    def time_underwater_bars(drawdown: pd.Series) -> int:
        """Число баров с drawdown < 0 (под водой)."""
        return int((drawdown < 0).sum())

    @staticmethod
    def ulcer_index(drawdown: pd.Series) -> float:
        """RMS просадок (отрицательные значения drawdown)."""
        dd = drawdown[drawdown < 0]
        if len(dd) == 0:
            return 0.0
        return float(np.sqrt(np.mean(dd.values ** 2)))

    def calculate_metrics(self) -> pd.Series:
        """Расчёт метрик эффективности."""
        metrics: dict = {}
        n = len(self.returns)
        bpy = self._bars_per_year
        rf_annual = self.config.risk_free_rate
        rf_per_bar = rf_annual / bpy if bpy else 0.0

        total_return = float(self.results["cum_strategy_returns"].iloc[-1] - 1)
        metrics["Total Return (%)"] = total_return * 100
        metrics["CAGR (%)"] = self.annualized_return(total_return, n, bpy) * 100

        std = float(self.returns.std())
        mean_ret = float(self.returns.mean())
        excess_mean = mean_ret - rf_per_bar
        metrics["Sharpe Ratio"] = (
            (excess_mean / std * np.sqrt(bpy)) if std != 0 else 0.0
        )

        downside = self.returns[self.returns < rf_per_bar]
        down_std = float(downside.std()) if len(downside) > 1 else 0.0
        metrics["Sortino Ratio"] = (
            (excess_mean / down_std * np.sqrt(bpy)) if down_std != 0 else 0.0
        )

        gains = self.returns[self.returns > 0].sum()
        losses = abs(self.returns[self.returns < 0].sum())
        metrics["Profit Factor"] = (gains / losses) if losses != 0 else np.inf

        wins = len(self.returns[self.returns > 0])
        active = len(self.returns[self.returns != 0])
        metrics["Win Rate (%)"] = (wins / active * 100) if active > 0 else 0.0

        max_dd = float(self.results["drawdown"].min())
        metrics["Max Drawdown (%)"] = max_dd * 100
        metrics["Recovery Factor"] = (
            (total_return / abs(max_dd)) if max_dd != 0 else 0.0
        )

        ann_return = self.annualized_return(total_return, n, bpy)
        metrics["Calmar Ratio"] = (
            (ann_return / abs(max_dd)) if max_dd != 0 else 0.0
        )

        metrics["Annualized Volatility (%)"] = (
            std * np.sqrt(bpy) * 100 if std != 0 else 0.0
        )
        metrics["Time Underwater (bars)"] = self.time_underwater_bars(
            self.results["drawdown"]
        )
        metrics["Ulcer Index"] = self.ulcer_index(self.results["drawdown"])
        metrics["Max Consecutive Losses"] = self.max_consecutive_losses(self.returns)

        if "trades" in self.results.columns:
            metrics["Trade Events"] = int((self.results["trades"] > 0).sum())
        else:
            metrics["Trade Events"] = int(active)

        if "cum_market_returns" in self.results.columns:
            mkt_total = float(self.results["cum_market_returns"].iloc[-1] - 1)
            metrics["Benchmark Return (%)"] = mkt_total * 100
            metrics["Alpha vs Benchmark (%)"] = (total_return - mkt_total) * 100
            mkt_std = float(self.results["market_returns"].fillna(0).std())
            metrics["Benchmark Sharpe Ratio"] = (
                (self.results["market_returns"].mean() - rf_per_bar)
                / mkt_std
                * np.sqrt(bpy)
                if mkt_std != 0
                else 0.0
            )

        return pd.Series(metrics)

    def annualized_return_from_results(self) -> float:
        total_return = float(self.results["cum_strategy_returns"].iloc[-1] - 1)
        return self.annualized_return(
            total_return, len(self.results), self._bars_per_year
        )
