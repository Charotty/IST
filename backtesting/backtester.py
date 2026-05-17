"""
Backtester - Движок для симуляции торговли на исторических данных.
"""

import pandas as pd
import numpy as np


class Backtester:
    def __init__(self, commission=0.0006, slippage=0.0002):
        """
        Движок для симуляции торговли.
        
        NOTE: Aligned with ExecutionConfig defaults (0.0006 + 0.0002 = 0.0008 total)
        
        :param commission: Комиссия за сделку (0.0006 = 0.06%)
        :param slippage: Проскальзывание (0.0002 = 0.02%)
        """
        self.commission = commission
        self.slippage = slippage

    def run(self, df, signals, position_size=None):
        """
        Запуск симуляции на всех данных.
        
        :param df: DataFrame с ценами (close)
        :param signals: Серия сигналов (-1, 0, 1)
        :param position_size: Опционально, доля капитала / множитель риска по барам (как ``final_pos_size``
            из ``RiskPipeline``). Если задано, доходность стратегии масштабируется как ``signal * pos_size * ret``.
        :return: DataFrame с результатами бэктеста
        """
        results = df[['close']].copy()
        results['signal'] = signals
        if position_size is not None:
            pos = pd.Series(position_size)
            if len(pos) != len(results):
                raise ValueError("position_size length must match df/signals length")
            results['position_size'] = pos.values
        else:
            results['position_size'] = 1.0

        # 1. Расчет доходности актива (Buy & Hold)
        results['market_returns'] = results['close'].pct_change()

        # 2. Расчет доходности стратегии (до издержек)
        # Исполнение на close следующего бара (shift(1)); масштаб позиции как в risk-слое
        results['strategy_returns'] = (
            results['signal'].shift(1) * results['position_size'].shift(1).fillna(0)
            * results['market_returns']
        )

        # 3. Моделирование издержек (комиссия + проскальзывание)
        # Издержки только при смене *эффективной* экспозиции (сигнал × размер)
        eff_exposure = results['signal'] * results['position_size']
        results['trades'] = eff_exposure.diff().fillna(0).abs()
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
