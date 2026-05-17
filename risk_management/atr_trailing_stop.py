"""
ATR Trailing Stop

ATR trailing stop logic.

Эталонная реализация из ist.py.

Параметры:
- atr_mult: 3.0 (по умолчанию)

Логика:
- long: trailing_stop = max(prev_stop, close - atr * mult)
- exit когда close < trailing_stop
- short: симметрично
"""

import numpy as np
import pandas as pd
from typing import Optional


def apply_atr_trailing_stop(
    df: pd.DataFrame,
    signal_col: str = 'final_signal',
    atr_col: str = 'atr',
    close_col: str = 'close',
    atr_mult: float = 3.0,
) -> pd.DataFrame:
    """
    Применить ATR trailing stop к сигналам.
    
    Args:
        df: DataFrame с данными OHLCV, ATR и сигналами
        signal_col: Название колонки с сигналами
        atr_col: Название колонки с ATR
        close_col: Название колонки с ценой закрытия
        atr_mult: Множитель ATR для trailing stop (по умолчанию 3.0)
        
    Returns:
        DataFrame с добавленными колонками trailing_stop и exit_signal
    """
    df = df.copy()

    # Инициализировать trailing stop
    df['trailing_stop'] = np.nan
    ts_col = df.columns.get_loc('trailing_stop')
    close_ix = df.columns.get_loc(close_col)
    atr_ix = df.columns.get_loc(atr_col)
    
    # Инициализировать exit signal
    df['exit_signal'] = 0
    
    # Рассчитать trailing stop для long позиций
    long_mask = df[signal_col] == 1
    
    # Для long: trailing_stop = max(prev_stop, close - atr * mult)
    for i in range(1, len(df)):
        if long_mask.iloc[i]:
            if pd.notna(df.iat[i - 1, ts_col]):
                new_stop = df.iat[i, close_ix] - df.iat[i, atr_ix] * atr_mult
                df.iat[i, ts_col] = max(df.iat[i - 1, ts_col], new_stop)
            else:
                df.iat[i, ts_col] = df.iat[i, close_ix] - df.iat[i, atr_ix] * atr_mult
    
    # Рассчитать trailing stop для short позиций
    short_mask = df[signal_col] == -1
    
    # Для short: trailing_stop = min(prev_stop, close + atr * mult)
    for i in range(1, len(df)):
        if short_mask.iloc[i]:
            if pd.notna(df.iat[i - 1, ts_col]):
                new_stop = df.iat[i, close_ix] + df.iat[i, atr_ix] * atr_mult
                df.iat[i, ts_col] = min(df.iat[i - 1, ts_col], new_stop)
            else:
                df.iat[i, ts_col] = df.iat[i, close_ix] + df.iat[i, atr_ix] * atr_mult
    
    # Определить exit signals
    # Long exit: когда close < trailing_stop
    long_exit = (df[signal_col] == 1) & (df[close_col] < df['trailing_stop'])
    
    # Short exit: когда close > trailing_stop
    short_exit = (df[signal_col] == -1) & (df[close_col] > df['trailing_stop'])
    
    # Exit signal = 1 когда нужно выйти
    df['exit_signal'] = np.where(long_exit | short_exit, 1, 0)
    
    return df


def combine_signals_with_exit(
    df: pd.DataFrame,
    signal_col: str = 'final_signal',
    exit_col: str = 'exit_signal',
) -> np.ndarray:
    """
    Комбинировать торговые сигналы с exit signals.
    
    Args:
        df: DataFrame с сигналами и exit signals
        signal_col: Название колонки с торговыми сигналами
        exit_col: Название колонки с exit signals
        
    Returns:
        Комбинированный сигнал (exit_signal = 1 -> signal = 0)
    """
    combined = np.where(df[exit_col] == 1, 0, df[signal_col])
    return combined


class ATRTrailingStop:
    """
    Класс для управления ATR trailing stop.
    """
    
    def __init__(
        self,
        atr_mult: float = 3.0,
        signal_col: str = 'final_signal',
        atr_col: str = 'atr',
        close_col: str = 'close',
    ):
        """
        Initialize ATRTrailingStop.
        
        Args:
            atr_mult: Множитель ATR для trailing stop
            signal_col: Название колонки с сигналами
            atr_col: Название колонки с ATR
            close_col: Название колонки с ценой закрытия
        """
        self.atr_mult = atr_mult
        self.signal_col = signal_col
        self.atr_col = atr_col
        self.close_col = close_col
    
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Применить trailing stop к DataFrame.
        
        Args:
            df: DataFrame с данными
            
        Returns:
            DataFrame с добавленными колонками trailing_stop и exit_signal
        """
        return apply_atr_trailing_stop(
            df=df,
            signal_col=self.signal_col,
            atr_col=self.atr_col,
            close_col=self.close_col,
            atr_mult=self.atr_mult,
        )
    
    def get_combined_signals(self, df: pd.DataFrame) -> np.ndarray:
        """
        Получить комбинированные сигналы с учетом exit signals.
        
        Args:
            df: DataFrame с данными (должен содержать exit_signal)
            
        Returns:
            Комбинированный сигнал
        """
        return combine_signals_with_exit(
            df=df,
            signal_col=self.signal_col,
            exit_col='exit_signal',
        )
    
    def set_atr_multiplier(self, multiplier: float):
        """
        Установить множитель ATR.
        
        Args:
            multiplier: Новый множитель ATR
        """
        if multiplier > 0:
            self.atr_mult = multiplier
        else:
            raise ValueError("atr_mult должен быть положительным")
