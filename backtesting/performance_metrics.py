"""
PerformanceMetrics - Расчет метрик эффективности стратегии.
"""

import pandas as pd
import numpy as np


class PerformanceMetrics:
    def __init__(self, backtest_results):
        """
        Инициализация с результатами бэктеста.
        
        :param backtest_results: DataFrame с результатами от Backtester.run()
        """
        self.results = backtest_results
        self.returns = self.results['net_returns'].fillna(0)

    def calculate_metrics(self):
        """
        Расчет основных метрик эффективности.
        
        :return: Series с метриками
        """
        metrics = {}

        # 1. Общая доходность
        total_return = (self.results['cum_strategy_returns'].iloc[-1] - 1)

        # 2. Коэффициент Шарпа (годовой, предполагая часовые данные)
        # 24 * 365 = 8760 часов в году
        std = self.returns.std()
        metrics['Sharpe Ratio'] = (self.returns.mean() / std * np.sqrt(8760)) if std != 0 else 0

        # 3. Profit Factor
        gains = self.returns[self.returns > 0].sum()
        losses = abs(self.returns[self.returns < 0].sum())
        metrics['Profit Factor'] = (gains / losses) if losses != 0 else np.inf

        # 4. Win Rate
        # Доля положительных баров с ненулевым return
        wins = len(self.returns[self.returns > 0])
        total_trades = len(self.returns[self.returns != 0])
        metrics['Win Rate (%)'] = (wins / total_trades * 100) if total_trades > 0 else 0

        # 5. Максимальная просадка (MDD)
        metrics['Max Drawdown (%)'] = self.results['drawdown'].min() * 100

        # 6. Recovery Factor
        metrics['Recovery Factor'] = (total_return / abs(self.results['drawdown'].min())) if self.results['drawdown'].min() != 0 else 0

        # 7. Total Return
        metrics['Total Return (%)'] = total_return * 100

        return pd.Series(metrics)
