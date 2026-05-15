"""
Position Sizer

ATR-based position sizing (статический).

Эталонная реализация из ist.py.

Параметры по умолчанию:
- risk_per_trade: 1% капитала
- account_size: 10000
- ATR stop mult: 2.0
"""

import numpy as np
import pandas as pd
from typing import Optional, Union


class PositionSizer:
    """
    ATR-based position sizer.
    
    Рассчитывает размер позиции на основе ATR и риска на сделку.
    """
    
    def __init__(
        self,
        risk_per_trade: float = 0.01,
        account_size: float = 10000,
        atr_stop_multiplier: float = 2.0,
    ):
        """
        Initialize PositionSizer.
        
        Args:
            risk_per_trade: Риск на сделку (доля капитала, по умолчанию 1%)
            account_size: Размер капитала (по умолчанию 10000)
            atr_stop_multiplier: Множитель ATR для стоп-лосса (по умолчанию 2.0)
        """
        self.risk_per_trade = risk_per_trade
        self.account_size = account_size
        self.atr_stop_multiplier = atr_stop_multiplier
    
    def calculate_sizes(
        self,
        df: pd.DataFrame,
        signal_col: str = 'final_signal',
        atr_col: str = 'atr',
        close_col: str = 'close',
    ) -> pd.DataFrame:
        """
        Рассчитать размер позиции для каждого временного шага.
        
        Args:
            df: DataFrame с данными OHLCV, ATR и сигналами
            signal_col: Название колонки с сигналами
            atr_col: Название колонки с ATR
            close_col: Название колонки с ценой закрытия
            
        Returns:
            DataFrame с добавленными колонками pos_size и final_pos_size
        """
        df = df.copy()
        
        # Рассчитать риск на сделку
        risk_amount = self.account_size * self.risk_per_trade
        
        # Рассчитать размер позиции на основе ATR
        # pos_size = risk_amount / (atr * stop_multiplier)
        df['pos_size'] = risk_amount / (df[atr_col] * self.atr_stop_multiplier)
        
        # Применить сигнал для определения направления
        # final_pos_size = pos_size * abs(final_signal)
        df['final_pos_size'] = df['pos_size'] * np.abs(df[signal_col])
        
        return df
    
    def calculate_position_size(
        self,
        atr: float,
        signal: float,
        close_price: Optional[float] = None,
    ) -> float:
        """
        Рассчитать размер позиции для одного временного шага.
        
        Args:
            atr: Значение ATR
            signal: Торговый сигнал (-1, 0, 1)
            close_price: Цена закрытия (опционально, для валидации)
            
        Returns:
            Размер позиции (в единицах актива)
        """
        risk_amount = self.account_size * self.risk_per_trade
        pos_size = risk_amount / (atr * self.atr_stop_multiplier)
        final_pos_size = pos_size * abs(signal)
        
        return final_pos_size
    
    def set_risk_per_trade(self, risk_per_trade: float):
        """
        Установить риск на сделку.
        
        Args:
            risk_per_trade: Новый риск на сделку (доля капитала)
        """
        if 0 < risk_per_trade < 1:
            self.risk_per_trade = risk_per_trade
        else:
            raise ValueError("risk_per_trade должен быть между 0 и 1")
    
    def set_account_size(self, account_size: float):
        """
        Установить размер капитала.
        
        Args:
            account_size: Новый размер капитала
        """
        if account_size > 0:
            self.account_size = account_size
        else:
            raise ValueError("account_size должен быть положительным")
    
    def set_atr_stop_multiplier(self, multiplier: float):
        """
        Установить множитель ATR для стоп-лосса.
        
        Args:
            multiplier: Новый множитель ATR
        """
        if multiplier > 0:
            self.atr_stop_multiplier = multiplier
        else:
            raise ValueError("atr_stop_multiplier должен быть положительным")
    
    def get_config(self) -> dict:
        """
        Получить текущую конфигурацию.
        
        Returns:
            Словарь с параметрами конфигурации
        """
        return {
            'risk_per_trade': self.risk_per_trade,
            'account_size': self.account_size,
            'atr_stop_multiplier': self.atr_stop_multiplier,
        }
