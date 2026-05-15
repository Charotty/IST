"""
Risk Pipeline

Объединяет stop + sizer + RL hook.

Рекомендуемый порядок применения:
Meta-Learning → final_signal / integrated_signal
    → [опционально] ATR trailing → combined_signal
    → PositionSizer → final_pos_size
    → [опционально] RL → risk_multiplier на доходность
    → Backtesting / Execution
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Union

from .position_sizer import PositionSizer
from .atr_trailing_stop import ATRTrailingStop, apply_atr_trailing_stop


class RiskPipeline:
    """
    Pipeline для управления рисками.
    
    Объединяет ATR trailing stop, position sizing и RL risk multiplier.
    """
    
    def __init__(
        self,
        position_sizer: Optional[PositionSizer] = None,
        atr_trailing_stop: Optional[ATRTrailingStop] = None,
        rl_overlay_enabled: bool = False,
        config: Optional[Dict] = None,
    ):
        """
        Initialize RiskPipeline.
        
        Args:
            position_sizer: PositionSizer instance (создастся по умолчанию)
            atr_trailing_stop: ATRTrailingStop instance (создастся по умолчанию)
            rl_overlay_enabled: Включить RL overlay
            config: Конфигурация pipeline
        """
        if config is None:
            config = {}
        
        # Position Sizer
        position_sizer_config = config.get('position_sizer', {})
        self.position_sizer = position_sizer or PositionSizer(
            risk_per_trade=position_sizer_config.get('risk_per_trade', 0.01),
            account_size=position_sizer_config.get('account_size', 10000),
            atr_stop_multiplier=position_sizer_config.get('atr_stop_multiplier', 2.0),
        )
        
        # ATR Trailing Stop
        trailing_stop_config = config.get('trailing_stop', {})
        self.trailing_stop_enabled = trailing_stop_config.get('enabled', True)
        self.atr_trailing_stop = atr_trailing_stop or ATRTrailingStop(
            atr_mult=trailing_stop_config.get('atr_mult', 3.0),
            signal_col=trailing_stop_config.get('signal_column', 'final_signal'),
        )
        
        # RL Overlay
        rl_config = config.get('rl_overlay', {})
        self.rl_overlay_enabled = rl_overlay_enabled or rl_config.get('enabled', False)
        self.rl_source = rl_config.get('source', 'rl_layer')
    
    def apply_pipeline(
        self,
        df: pd.DataFrame,
        signal_col: str = 'final_signal',
        atr_col: str = 'atr',
        close_col: str = 'close',
        rl_risk_multiplier: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Применить полный pipeline управления рисками.
        
        Порядок применения:
        1. ATR trailing stop (если включен)
        2. Position sizing
        3. RL risk multiplier (если включен)
        
        Args:
            df: DataFrame с данными OHLCV, ATR и сигналами
            signal_col: Название колонки с сигналами
            atr_col: Название колонки с ATR
            close_col: Название колонки с ценой закрытия
            rl_risk_multiplier: RL risk multiplier array (опционально)
            
        Returns:
            DataFrame с добавленными колонками:
            - trailing_stop, exit_signal (если trailing stop включен)
            - combined_signal (если trailing stop включен)
            - pos_size, final_pos_size
            - rl_adjusted_return (если RL включен)
        """
        df = df.copy()
        
        # Шаг 1: ATR Trailing Stop
        if self.trailing_stop_enabled:
            df = self.atr_trailing_stop.apply(df)
            # Комбинировать сигналы с exit signals
            df['combined_signal'] = self.atr_trailing_stop.get_combined_signals(df)
            working_signal_col = 'combined_signal'
        else:
            working_signal_col = signal_col
        
        # Шаг 2: Position Sizing
        df = self.position_sizer.calculate_sizes(
            df=df,
            signal_col=working_signal_col,
            atr_col=atr_col,
            close_col=close_col,
        )
        
        # Шаг 3: RL Risk Multiplier
        if self.rl_overlay_enabled and rl_risk_multiplier is not None:
            # Применить RL risk multiplier к доходности
            # strategy_return = final_signal * pct_change.shift(-1) * risk_multiplier
            if 'pct_change' in df.columns:
                df['rl_adjusted_return'] = df[working_signal_col] * df['pct_change'].shift(-1) * rl_risk_multiplier
            else:
                # Если pct_change нет, используем close
                df['pct_change'] = df[close_col].pct_change()
                df['rl_adjusted_return'] = df[working_signal_col] * df['pct_change'].shift(-1) * rl_risk_multiplier
        
        return df
    
    def apply_trailing_stop_only(
        self,
        df: pd.DataFrame,
        signal_col: str = 'final_signal',
        atr_col: str = 'atr',
        close_col: str = 'close',
    ) -> pd.DataFrame:
        """
        Применить только ATR trailing stop.
        
        Args:
            df: DataFrame с данными
            signal_col: Название колонки с сигналами
            atr_col: Название колонки с ATR
            close_col: Название колонки с ценой закрытия
            
        Returns:
            DataFrame с добавленными колонками trailing_stop и exit_signal
        """
        return self.atr_trailing_stop.apply(df)
    
    def apply_position_sizing_only(
        self,
        df: pd.DataFrame,
        signal_col: str = 'final_signal',
        atr_col: str = 'atr',
        close_col: str = 'close',
    ) -> pd.DataFrame:
        """
        Применить только position sizing.
        
        Args:
            df: DataFrame с данными
            signal_col: Название колонки с сигналами
            atr_col: Название колонки с ATR
            close_col: Название колонки с ценой закрытия
            
        Returns:
            DataFrame с добавленными колонками pos_size и final_pos_size
        """
        return self.position_sizer.calculate_sizes(
            df=df,
            signal_col=signal_col,
            atr_col=atr_col,
            close_col=close_col,
        )
    
    def get_config(self) -> Dict:
        """
        Получить текущую конфигурацию pipeline.
        
        Returns:
            Словарь с конфигурацией
        """
        return {
            'position_sizer': self.position_sizer.get_config(),
            'trailing_stop': {
                'enabled': self.trailing_stop_enabled,
                'atr_mult': self.atr_trailing_stop.atr_mult,
                'signal_column': self.atr_trailing_stop.signal_col,
            },
            'rl_overlay': {
                'enabled': self.rl_overlay_enabled,
                'source': self.rl_source,
            },
        }
    
    def update_config(self, config: Dict):
        """
        Обновить конфигурацию pipeline.
        
        Args:
            config: Словарь с новой конфигурацией
        """
        if 'position_sizer' in config:
            ps_config = config['position_sizer']
            if 'risk_per_trade' in ps_config:
                self.position_sizer.set_risk_per_trade(ps_config['risk_per_trade'])
            if 'account_size' in ps_config:
                self.position_sizer.set_account_size(ps_config['account_size'])
            if 'atr_stop_multiplier' in ps_config:
                self.position_sizer.set_atr_stop_multiplier(ps_config['atr_stop_multiplier'])
        
        if 'trailing_stop' in config:
            ts_config = config['trailing_stop']
            if 'enabled' in ts_config:
                self.trailing_stop_enabled = ts_config['enabled']
            if 'atr_mult' in ts_config:
                self.atr_trailing_stop.set_atr_multiplier(ts_config['atr_mult'])
        
        if 'rl_overlay' in config:
            rl_config = config['rl_overlay']
            if 'enabled' in rl_config:
                self.rl_overlay_enabled = rl_config['enabled']
            if 'source' in rl_config:
                self.rl_source = rl_config['source']
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'RiskPipeline':
        """
        Создать RiskPipeline из YAML конфигурации.
        
        Args:
            yaml_path: Путь к YAML файлу конфигурации
            
        Returns:
            RiskPipeline instance
        """
        import yaml
        
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Извлечь секцию risk_management если есть
        if 'risk_management' in config:
            config = config['risk_management']
        
        return cls(config=config)
    
    def to_yaml(self, yaml_path: str):
        """
        Сохранить конфигурацию в YAML файл.
        
        Args:
            yaml_path: Путь для сохранения YAML файла
        """
        import yaml
        
        config = self.get_config()
        
        with open(yaml_path, 'w') as f:
            yaml.dump({'risk_management': config}, f, default_flow_style=False)
