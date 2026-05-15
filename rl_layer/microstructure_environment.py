"""
Microstructure Trading Environment for RL-based risk management.

This environment extends the ensemble environment with order book imbalance (OBI).
Designed for use when live L2 order book data is available.
"""

import numpy as np
from typing import Tuple, Dict, Any
import pandas as pd


class MicrostructureRLenv:
    """
    RL environment with order book imbalance (OBI) for microstructure-aware risk management.
    
    Action space: 3 discrete risk levels
    - 0: 0.005 (0.5%) - Low risk
    - 1: 0.01 (1%) - Medium risk
    - 2: 0.02 (2%) - High risk
    
    Observation space: 11-dimensional state
    - rsi, volatility, regime_pred, vol_spike_prob
    - ensemble_prob, meta_prob
    - ema_slope, adx, rsi_15m, rsi_4h
    - order_book_imbalance (OBI)
    """
    
    def __init__(self, df: pd.DataFrame, initial_balance: float = 10000.0):
        """
        Initialize the microstructure trading environment.
        
        Args:
            df: DataFrame with OHLCV data and features including:
                - final_signal: Signal from meta-learning (-1 to 1)
                - ensemble_prob: Ensemble probability
                - order_book_imbalance: Order book imbalance metric
                - rsi, volatility, regime_pred
                - vol_spike_prob, meta_prob
                - ema_slope, adx, rsi_15m, rsi_4h
            initial_balance: Starting balance for backtesting
        """
        self.df = df.reset_index(drop=True).copy()
        self.initial_balance = initial_balance
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0.0
        self.total_profit = 0.0
        
        # Risk level mapping
        self.risk_map = {
            0: 0.005,  # 0.5% - Low risk
            1: 0.01,   # 1% - Medium risk
            2: 0.02    # 2% - High risk
        }
        
        # Action and observation space
        self.action_size = 3
        self.state_size = 11
        
        # Required columns
        self.required_columns = [
            'final_signal', 'rsi', 'volatility', 'regime_pred',
            'vol_spike_prob', 'ensemble_prob', 'meta_prob',
            'ema_slope', 'adx', 'rsi_15m', 'rsi_4h',
            'order_book_imbalance', 'close'
        ]
        
        self._validate_data()
        
    def _validate_data(self):
        """Validate that required columns exist in DataFrame."""
        missing_cols = [col for col in self.required_columns if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
    
    def reset(self) -> np.ndarray:
        """
        Reset the environment to initial state.
        
        Returns:
            Initial state observation
        """
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0.0
        self.total_profit = 0.0
        return self._get_state(self.current_step)
    
    def _get_state(self, step: int) -> np.ndarray:
        """
        Get state observation at given step.
        
        Args:
            step: Current step index
            
        Returns:
            11-dimensional state vector with order_book_imbalance
        """
        row = self.df.iloc[step]
        state = np.array([
            row['rsi'],
            row['volatility'],
            row['regime_pred'],
            row['vol_spike_prob'],
            row['ensemble_prob'],
            row['meta_prob'],
            row['ema_slope'],
            row['adx'],
            row['rsi_15m'],
            row['rsi_4h'],
            row['order_book_imbalance']  # OBI feature
        ], dtype=np.float32)
        
        # Handle NaN values
        state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return state
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Risk level action (0, 1, or 2)
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        if self.current_step >= len(self.df) - 2:
            return self._get_state(self.current_step), 0.0, True, {}
        
        # Get risk percentage from action
        risk_pct = self.risk_map[action]
        
        # Get signal and price change
        current_row = self.df.iloc[self.current_step]
        next_row = self.df.iloc[self.current_step + 1]
        
        signal = current_row['final_signal']
        current_close = current_row['close']
        future_close = next_row['close']
        
        # Calculate price change
        price_change = (future_close - current_close) / current_close
        
        # Calculate reward: signal * price_change * risk_pct * 100
        reward = signal * price_change * risk_pct * 100
        
        # Update step
        self.current_step += 1
        
        # Check if done
        done = self.current_step >= len(self.df) - 2
        
        # Get next state
        next_state = self._get_state(self.current_step)
        
        # Info dictionary
        info = {
            'risk_pct': risk_pct,
            'signal': signal,
            'price_change': price_change,
            'balance': self.balance,
            'step': self.current_step,
            'obi': current_row['order_book_imbalance']
        }
        
        return next_state, reward, done, info
    
    def render(self, mode='human'):
        """Render the environment state (optional)."""
        if mode == 'human':
            current_row = self.df.iloc[self.current_step]
            print(f"Step: {self.current_step}, Balance: {self.balance:.2f}, OBI: {current_row['order_book_imbalance']:.4f}")
