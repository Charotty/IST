from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np


@dataclass
class Trade:
    """Trade information."""
    timestamp: int
    symbol: str
    side: str  # 'buy' or 'sell'
    price: float
    size: float
    commission: float
    slippage: float
    pnl: Optional[float] = None  # Filled when position closed


@dataclass
class BacktestResult:
    """Backtest results."""
    trades: List[Trade]
    equity_curve: np.ndarray
    returns: np.ndarray
    metrics: Dict[str, float]
    positions: pd.DataFrame
    
    def summary(self) -> str:
        """Brief summary of results."""
        return f"""
Backtest Summary:
-----------------
Total Return: {self.metrics['total_return']:.2%}
Sharpe Ratio: {self.metrics['sharpe_ratio']:.2f}
Max Drawdown: {self.metrics['max_drawdown']:.2%}
Win Rate: {self.metrics['win_rate']:.2%}
Profit Factor: {self.metrics['profit_factor']:.2f}
Total Trades: {self.metrics['num_trades']}
        """


class BaseBacktester(ABC):
    """Base class for backtesting."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        
        # Trading costs
        self.commission_rate = config.get('commission_rate', 0.001)  # 0.1%
        self.slippage_rate = config.get('slippage_rate', 0.0005)  # 0.05%
        
        # Initial capital
        self.initial_capital = config.get('initial_capital', 10000)
        
        # History
        self.trades: List[Trade] = []
        self.equity_history: List[float] = [self.initial_capital]
        self.current_capital = self.initial_capital
        self.current_position: Optional[Dict[str, Any]] = None
    
    @abstractmethod
    def run(
        self,
        data: pd.DataFrame,
        strategy: Any,  # Model or Decision Engine
    ) -> BacktestResult:
        """
        Run backtest.
        
        Args:
            data: Historical data (MUST be sorted by time!)
            strategy: Trading strategy (model + decision maker)
        
        Returns:
            BacktestResult with full statistics
        """
        pass
    
    def calculate_commission(self, price: float, size: float) -> float:
        """Calculate commission."""
        return price * size * self.commission_rate
    
    def calculate_slippage(self, price: float, size: float, side: str) -> float:
        """Calculate slippage."""
        slippage_amount = price * self.slippage_rate
        
        # Price increases for buy, decreases for sell
        if side == 'buy':
            return slippage_amount
        else:
            return -slippage_amount
    
    def execute_trade(
        self,
        timestamp: int,
        symbol: str,
        side: str,
        price: float,
        size: float
    ) -> Trade:
        """Simulate trade execution."""
        
        # Account for slippage
        slippage = self.calculate_slippage(price, size, side)
        execution_price = price + slippage
        
        # Account for commission
        commission = self.calculate_commission(execution_price, size)
        
        # Update capital
        if side == 'buy':
            cost = execution_price * size + commission
            self.current_capital -= cost
        else:  # sell
            proceeds = execution_price * size - commission
            self.current_capital += proceeds
        
        # Create trade record
        trade = Trade(
            timestamp=timestamp,
            symbol=symbol,
            side=side,
            price=execution_price,
            size=size,
            commission=commission,
            slippage=abs(slippage)
        )
        
        self.trades.append(trade)
        self.equity_history.append(self.current_capital)
        
        return trade
