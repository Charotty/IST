"""
Backtester - Движок для симуляции торговли на исторических данных.
"""

import pandas as pd
import numpy as np


class Backtester:
    def __init__(self, commission=0.0005, slippage=0.0001):
        """
        Движок для симуляции торговли.
        
        :param commission: Комиссия за сделку (0.0005 = 0.05%)
        :param slippage: Проскальзывание (0.0001 = 0.01%)
        """
        self.commission = commission
        self.slippage = slippage

    def run(self, df, signals):
        """
        Запуск симуляции на всех данных.
        
        :param df: DataFrame с ценами (close)
        :param signals: Серия сигналов (-1, 0, 1)
        :return: DataFrame с результатами бэктеста
        """
        results = df[['close']].copy()
        results['signal'] = signals

        # 1. Расчет доходности актива (Buy & Hold)
        results['market_returns'] = results['close'].pct_change()

        # 2. Расчет доходности стратегии (до издержек)
        # Исполнение на close следующего бара (shift(1))
        results['strategy_returns'] = results['signal'].shift(1) * results['market_returns']

        # 3. Моделирование издержек (комиссия + проскальзывание)
        # Издержки только при смене сигнала
        results['trades'] = results['signal'].diff().fillna(0).abs()
        results['costs'] = results['trades'] * (self.commission + self.slippage)

        # 4. Чистая доходность
        results['net_returns'] = results['strategy_returns'] - results['costs']

        # 5. Кумулятивные показатели
        results['cum_market_returns'] = (1 + results['market_returns'].fillna(0)).cumprod()
        results['cum_strategy_returns'] = (1 + results['net_returns'].fillna(0)).cumprod()

        # 6. Расчет просадки (Drawdown)
        peak = results['cum_strategy_returns'].expanding(min_periods=1).max()
        results['drawdown'] = (results['cum_strategy_returns'] - peak) / peak

        return results
