from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from its_project.backtesting.base import BaseBacktester, BacktestResult, Trade


@dataclass
class BaselineResult:
    """Results from baseline strategy."""
    name: str
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_holding_period: float
    volatility: float
    calmar_ratio: float


class BaselineStrategies:
    """
    Collection of baseline trading strategies for comparison.
    
    Includes:
    - Buy & Hold
    - Random Trading
    - Simple Moving Average Crossover
    - Mean Reversion
    - Momentum
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        
        # Common parameters
        self.initial_capital = config.get("initial_capital", 10000.0)
        self.commission_rate = config.get("commission_rate", 0.001)
        self.slippage_rate = config.get("slippage_rate", 0.0005)
        
    def buy_and_hold(self, data: pd.DataFrame) -> BaselineResult:
        """
        Buy & Hold strategy.
        
        Buy at the beginning and hold until the end.
        """
        prices = data['close']
        
        # Calculate returns
        returns = prices.pct_change().dropna()
        
        # Total return
        total_return = (prices.iloc[-1] / prices.iloc[0]) - 1
        
        # Annualized return
        days = (prices.index[-1] - prices.index[0]).days
        annualized_return = (1 + total_return) ** (365.25 / days) - 1
        
        # Sharpe ratio
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        
        # Maximum drawdown
        max_drawdown = self._calculate_max_drawdown(prices)
        
        # Volatility
        volatility = returns.std() * np.sqrt(252)
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return BaselineResult(
            name="Buy & Hold",
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=1.0,  # Always wins in the end (assuming positive return)
            profit_factor=float('inf') if total_return > 0 else 0,
            total_trades=1,
            avg_holding_period=days,
            volatility=volatility,
            calmar_ratio=calmar_ratio
        )
    
    def random_trading(self, data: pd.DataFrame, n_simulations: int = 100) -> BaselineResult:
        """
        Random trading strategy.
        
        Generate random buy/sell signals.
        """
        np.random.seed(42)
        
        results = []
        
        for sim in range(n_simulations):
            # Generate random signals
            signals = np.random.choice([0, 1, 2], size=len(data), p=[0.3, 0.4, 0.3])  # SELL, HOLD, BUY
            
            # Calculate returns from random trading
            returns = self._calculate_strategy_returns(data, signals)
            
            # Calculate metrics
            total_return = returns.sum()
            annualized_return = self._annualize_return(total_return, len(data))
            sharpe_ratio = self._calculate_sharpe_ratio(returns)
            max_drawdown = self._calculate_max_drawdown_from_returns(returns)
            volatility = returns.std() * np.sqrt(252)
            
            # Trade statistics
            trades = self._count_trades(signals)
            win_rate = self._calculate_win_rate(returns)
            profit_factor = self._calculate_profit_factor(returns)
            
            results.append(BaselineResult(
                name=f"Random Trading {sim+1}",
                total_return=total_return,
                annualized_return=annualized_return,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                profit_factor=profit_factor,
                total_trades=trades,
                avg_holding_period=1.0,  # Simplified
                volatility=volatility,
                calmar_ratio=annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
            ))
        
        # Return average results
        avg_result = self._average_baseline_results(results, "Random Trading")
        return avg_result
    
    def moving_average_crossover(self, data: pd.DataFrame, fast_window: int = 10, 
                               slow_window: int = 50) -> BaselineResult:
        """
        Moving Average Crossover strategy.
        
        Buy when fast MA crosses above slow MA, sell when opposite.
        """
        prices = data['close']
        
        # Calculate moving averages
        fast_ma = prices.rolling(window=fast_window).mean()
        slow_ma = prices.rolling(window=slow_window).mean()
        
        # Generate signals
        signals = np.zeros(len(data))
        
        # Buy signal (fast MA > slow MA)
        signals[fast_ma > slow_ma] = 2  # BUY
        
        # Sell signal (fast MA < slow MA)
        signals[fast_ma < slow_ma] = 0  # SELL
        
        # Hold signal (equal)
        signals[fast_ma == slow_ma] = 1  # HOLD
        
        # Calculate returns
        returns = self._calculate_strategy_returns(data, signals)
        
        # Calculate metrics
        total_return = returns.sum()
        annualized_return = self._annualize_return(total_return, len(data))
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        max_drawdown = self._calculate_max_drawdown_from_returns(returns)
        volatility = returns.std() * np.sqrt(252)
        
        # Trade statistics
        trades = self._count_trades(signals)
        win_rate = self._calculate_win_rate(returns)
        profit_factor = self._calculate_profit_factor(returns)
        
        return BaselineResult(
            name=f"MA Crossover ({fast_window}/{slow_window})",
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=trades,
            avg_holding_period=self._calculate_avg_holding_period(signals),
            volatility=volatility,
            calmar_ratio=annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        )
    
    def mean_reversion(self, data: pd.DataFrame, lookback_window: int = 20, 
                      entry_threshold: float = 2.0, exit_threshold: float = 0.5) -> BaselineResult:
        """
        Mean reversion strategy.
        
        Buy when price is below mean by entry_threshold standard deviations,
        sell when above, exit when within exit_threshold.
        """
        prices = data['close']
        
        # Calculate rolling mean and standard deviation
        rolling_mean = prices.rolling(window=lookback_window).mean()
        rolling_std = prices.rolling(window=lookback_window).std()
        
        # Calculate z-score
        z_score = (prices - rolling_mean) / rolling_std
        
        # Generate signals
        signals = np.ones(len(data))  # Default HOLD
        
        # Buy signal (oversold)
        signals[z_score < -entry_threshold] = 2  # BUY
        
        # Sell signal (overbought)
        signals[z_score > entry_threshold] = 0  # SELL
        
        # Exit signals (return to mean)
        signals[(abs(z_score) < exit_threshold)] = 1  # HOLD
        
        # Calculate returns
        returns = self._calculate_strategy_returns(data, signals)
        
        # Calculate metrics
        total_return = returns.sum()
        annualized_return = self._annualize_return(total_return, len(data))
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        max_drawdown = self._calculate_max_drawdown_from_returns(returns)
        volatility = returns.std() * np.sqrt(252)
        
        # Trade statistics
        trades = self._count_trades(signals)
        win_rate = self._calculate_win_rate(returns)
        profit_factor = self._calculate_profit_factor(returns)
        
        return BaselineResult(
            name=f"Mean Reversion ({lookback_window})",
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=trades,
            avg_holding_period=self._calculate_avg_holding_period(signals),
            volatility=volatility,
            calmar_ratio=annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        )
    
    def momentum_strategy(self, data: pd.DataFrame, lookback_window: int = 20,
                         momentum_threshold: float = 0.02) -> BaselineResult:
        """
        Momentum strategy.
        
        Buy when momentum is positive and above threshold, sell when negative.
        """
        prices = data['close']
        
        # Calculate momentum (price change over lookback window)
        momentum = prices.pct_change(lookback_window)
        
        # Generate signals
        signals = np.ones(len(data))  # Default HOLD
        
        # Buy signal (positive momentum above threshold)
        signals[momentum > momentum_threshold] = 2  # BUY
        
        # Sell signal (negative momentum below -threshold)
        signals[momentum < -momentum_threshold] = 0  # SELL
        
        # Calculate returns
        returns = self._calculate_strategy_returns(data, signals)
        
        # Calculate metrics
        total_return = returns.sum()
        annualized_return = self._annualize_return(total_return, len(data))
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        max_drawdown = self._calculate_max_drawdown_from_returns(returns)
        volatility = returns.std() * np.sqrt(252)
        
        # Trade statistics
        trades = self._count_trades(signals)
        win_rate = self._calculate_win_rate(returns)
        profit_factor = self._calculate_profit_factor(returns)
        
        return BaselineResult(
            name=f"Momentum ({lookback_window})",
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=trades,
            avg_holding_period=self._calculate_avg_holding_period(signals),
            volatility=volatility,
            calmar_ratio=annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        )
    
    def _calculate_strategy_returns(self, data: pd.DataFrame, signals: np.ndarray) -> np.ndarray:
        """Calculate returns from trading signals."""
        prices = data['close']
        returns = np.zeros(len(prices))
        
        current_position = 0  # 0: no position, 1: long, -1: short
        entry_price = 0
        
        for i in range(1, len(prices)):
            signal = signals[i]
            
            # Handle position changes
            if signal == 2 and current_position <= 0:  # BUY signal
                if current_position == -1:  # Close short position
                    returns[i] = (entry_price - prices[i]) / entry_price
                current_position = 1
                entry_price = prices[i]
                
            elif signal == 0 and current_position >= 0:  # SELL signal
                if current_position == 1:  # Close long position
                    returns[i] = (prices[i] - entry_price) / entry_price
                current_position = -1
                entry_price = prices[i]
                
            elif signal == 1 and current_position != 0:  # HOLD signal, close position
                if current_position == 1:
                    returns[i] = (prices[i] - entry_price) / entry_price
                else:  # current_position == -1
                    returns[i] = (entry_price - prices[i]) / entry_price
                current_position = 0
                entry_price = 0
        
        # Apply transaction costs
        returns -= self.commission_rate + self.slippage_rate
        
        return returns
    
    def _count_trades(self, signals: np.ndarray) -> int:
        """Count number of trades."""
        trades = 0
        prev_signal = 1  # Start with HOLD
        
        for signal in signals:
            if signal != prev_signal and signal != 1:  # Changed to BUY or SELL
                trades += 1
            prev_signal = signal
        
        return trades
    
    def _calculate_win_rate(self, returns: np.ndarray) -> float:
        """Calculate win rate."""
        winning_trades = np.sum(returns > 0)
        total_trades = np.sum(returns != 0)
        
        return winning_trades / total_trades if total_trades > 0 else 0
    
    def _calculate_profit_factor(self, returns: np.ndarray) -> float:
        """Calculate profit factor."""
        gross_profit = np.sum(returns[returns > 0])
        gross_loss = abs(np.sum(returns[returns < 0]))
        
        return gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    def _calculate_avg_holding_period(self, signals: np.ndarray) -> float:
        """Calculate average holding period in bars."""
        periods = []
        current_period = 0
        in_position = False
        
        for signal in signals:
            if signal != 1:  # BUY or SELL
                if not in_position:
                    in_position = True
                    current_period = 1
                else:
                    current_period += 1
            else:  # HOLD
                if in_position:
                    periods.append(current_period)
                    in_position = False
                    current_period = 0
        
        return np.mean(periods) if periods else 1.0
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        
        daily_rf = risk_free_rate / 252
        excess_returns = returns - daily_rf
        
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown."""
        peak = np.maximum.accumulate(prices)
        drawdown = (prices - peak) / peak
        return abs(drawdown.min())
    
    def _calculate_max_drawdown_from_returns(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown from returns."""
        cumulative = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - peak) / peak
        return abs(drawdown.min())
    
    def _annualize_return(self, total_return: float, num_periods: int) -> float:
        """Annualize return."""
        if num_periods == 0:
            return 0.0
        
        # Assuming daily data
        years = num_periods / 252.0
        return (1 + total_return) ** (1 / years) - 1
    
    def _average_baseline_results(self, results: List[BaselineResult], name: str) -> BaselineResult:
        """Average multiple baseline results."""
        if not results:
            return BaselineResult(name, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        return BaselineResult(
            name=name,
            total_return=np.mean([r.total_return for r in results]),
            annualized_return=np.mean([r.annualized_return for r in results]),
            sharpe_ratio=np.mean([r.sharpe_ratio for r in results]),
            max_drawdown=np.mean([r.max_drawdown for r in results]),
            win_rate=np.mean([r.win_rate for r in results]),
            profit_factor=np.mean([r.profit_factor for r in results if r.profit_factor != float('inf')]),
            total_trades=int(np.mean([r.total_trades for r in results])),
            avg_holding_period=np.mean([r.avg_holding_period for r in results]),
            volatility=np.mean([r.volatility for r in results]),
            calmar_ratio=np.mean([r.calmar_ratio for r in results])
        )
    
    def run_all_baselines(self, data: pd.DataFrame) -> Dict[str, BaselineResult]:
        """Run all baseline strategies."""
        results = {}
        
        # Buy & Hold
        results['buy_hold'] = self.buy_and_hold(data)
        
        # Random Trading
        results['random'] = self.random_trading(data)
        
        # Moving Average Crossover
        results['ma_crossover_10_50'] = self.moving_average_crossover(data, 10, 50)
        results['ma_crossover_20_100'] = self.moving_average_crossover(data, 20, 100)
        
        # Mean Reversion
        results['mean_reversion_20'] = self.mean_reversion(data, 20)
        results['mean_reversion_50'] = self.mean_reversion(data, 50)
        
        # Momentum
        results['momentum_20'] = self.momentum_strategy(data, 20)
        results['momentum_50'] = self.momentum_strategy(data, 50)
        
        return results
    
    def compare_with_model(self, model_result: BacktestResult, 
                          baseline_results: Dict[str, BaselineResult]) -> Dict[str, Any]:
        """Compare model performance with baselines."""
        comparison = {}
        
        # Extract model metrics
        model_metrics = model_result.metrics
        
        # Compare with each baseline
        for name, baseline in baseline_results.items():
            comparison[name] = {
                'model_return': model_metrics.get('total_return', 0),
                'baseline_return': baseline.total_return,
                'return_alpha': model_metrics.get('total_return', 0) - baseline.total_return,
                'model_sharpe': model_metrics.get('sharpe_ratio', 0),
                'baseline_sharpe': baseline.sharpe_ratio,
                'sharpe_alpha': model_metrics.get('sharpe_ratio', 0) - baseline.sharpe_ratio,
                'model_max_dd': abs(model_metrics.get('max_drawdown', 0)),
                'baseline_max_dd': abs(baseline.max_drawdown),
                'drawdown_improvement': abs(baseline.max_drawdown) - abs(model_metrics.get('max_drawdown', 0))
            }
        
        # Find best baseline
        best_baseline = max(baseline_results.values(), key=lambda x: x.sharpe_ratio)
        
        comparison['best_baseline'] = {
            'name': best_baseline.name,
            'sharpe_ratio': best_baseline.sharpe_ratio,
            'total_return': best_baseline.total_return,
            'model_vs_best_sharpe': model_metrics.get('sharpe_ratio', 0) - best_baseline.sharpe_ratio,
            'model_vs_best_return': model_metrics.get('total_return', 0) - best_baseline.total_return
        }
        
        # Overall assessment
        model_sharpe = model_metrics.get('sharpe_ratio', 0)
        if model_sharpe > best_baseline.sharpe_ratio * 1.1:
            assessment = "EXCELLENT"
        elif model_sharpe > best_baseline.sharpe_ratio:
            assessment = "GOOD"
        elif model_sharpe > best_baseline.sharpe_ratio * 0.9:
            assessment = "ACCEPTABLE"
        else:
            assessment = "POOR"
        
        comparison['overall_assessment'] = assessment
        
        return comparison
