from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from its_project.backtesting.base import BacktestResult, Trade

logger = logging.getLogger(__name__)


class BacktestMetrics:
    """Enhanced backtesting metrics with realistic cost calculations."""
    
    @staticmethod
    def calculate_commission(
        trade_value: float,
        commission_rate: float,
        exchange_fee: float = 0.0001,
        slippage_cost: float = 0.0
    ) -> float:
        """
        Calculate total transaction cost including commission and slippage.
        
        Args:
            trade_value: Value of the trade
            commission_rate: Broker commission rate (e.g., 0.001 for 0.1%)
            exchange_fee: Exchange fee (default 0.01%)
            slippage_cost: Slippage cost calculated separately
            
        Returns:
            Total transaction cost
        """
        broker_commission = trade_value * commission_rate
        exchange_cost = trade_value * exchange_fee
        total_cost = broker_commission + exchange_cost + slippage_cost
        
        return total_cost
    
    @staticmethod
    def calculate_slippage(
        order_size: float,
        market_price: float,
        order_book_depth: Optional[Dict[str, Any]] = None,
        slippage_model: str = "linear",
        slippage_params: Dict[str, Any] = None
    ) -> float:
        """
        Calculate realistic slippage based on order size and market conditions.
        
        Args:
            order_size: Order size in base currency
            market_price: Current market price
            order_book_depth: Order book depth information
            slippage_model: Model type ("linear", "percentage", "volume_impact")
            slippage_params: Model-specific parameters
            
        Returns:
            Slippage cost in price units
        """
        params = slippage_params or {}
        
        if slippage_model == "linear":
            # Linear slippage: slippage = base_rate * order_size
            base_rate = params.get("base_rate", 0.0001)  # 0.01% base slippage
            size_factor = params.get("size_factor", 0.000001)  # Additional slippage per unit
            
            slippage_rate = base_rate + (order_size * size_factor)
            slippage_cost = market_price * slippage_rate
            
        elif slippage_model == "percentage":
            # Percentage-based slippage
            percentage_rate = params.get("percentage_rate", 0.0005)  # 0.05%
            slippage_cost = market_price * order_size * percentage_rate
            
        elif slippage_model == "volume_impact":
            # Volume impact model based on order book
            if order_book_depth:
                slippage_cost = BacktestMetrics._calculate_volume_impact_slippage(
                    order_size, market_price, order_book_depth, params
                )
            else:
                # Fallback to linear model
                slippage_cost = BacktestMetrics.calculate_slippage(
                    order_size, market_price, None, "linear", params
                )
        else:
            raise ValueError(f"Unknown slippage model: {slippage_model}")
        
        return slippage_cost
    
    @staticmethod
    def _calculate_volume_impact_slippage(
        order_size: float,
        market_price: float,
        order_book: Dict[str, Any],
        params: Dict[str, Any]
    ) -> float:
        """Calculate slippage based on order book volume impact."""
        bids = order_book.get("bids", [])
        asks = order_book.get("asks", [])
        
        if not bids or not asks:
            # Fallback to linear model
            base_rate = params.get("base_rate", 0.0001)
            return market_price * base_rate * order_size
        
        # Calculate volume impact
        total_volume = sum(float(level[1]) for level in bids + asks)
        volume_ratio = order_size / (total_volume + 1e-10)
        
        # Impact function: impact = a * volume_ratio^b
        impact_a = params.get("impact_a", 0.001)
        impact_b = params.get("impact_b", 0.5)
        
        impact_rate = impact_a * (volume_ratio ** impact_b)
        
        return market_price * impact_rate * order_size
    
    @staticmethod
    def calculate_realistic_pnl(
        trades: List[Trade],
        commission_rate: float = 0.001,
        slippage_model: str = "linear",
        slippage_params: Dict[str, Any] = None
    ) -> List[float]:
        """
        Calculate realistic P&L including all transaction costs.
        
        Args:
            trades: List of trades
            commission_rate: Commission rate
            slippage_model: Slippage model
            slippage_params: Slippage parameters
            
        Returns:
            List of net returns after costs
        """
        net_returns = []
        slippage_params = slippage_params or {}
        
        for trade in trades:
            # Calculate entry costs
            entry_value = trade.entry_price * trade.size
            entry_slippage = BacktestMetrics.calculate_slippage(
                trade.size, trade.entry_price, None, slippage_model, slippage_params
            )
            entry_commission = BacktestMetrics.calculate_commission(
                entry_value, commission_rate
            )
            
            # Calculate exit costs
            exit_value = trade.exit_price * trade.size if trade.exit_price else 0
            exit_slippage = 0
            exit_commission = 0
            
            if trade.exit_price:
                exit_slippage = BacktestMetrics.calculate_slippage(
                    trade.size, trade.exit_price, None, slippage_model, slippage_params
                )
                exit_commission = BacktestMetrics.calculate_commission(
                    exit_value, commission_rate
                )
            
            # Calculate gross P&L
            gross_pnl = trade.pnl
            
            # Subtract all costs
            total_costs = entry_slippage + entry_commission + exit_slippage + exit_commission
            net_pnl = gross_pnl - total_costs
            
            # Calculate percentage return
            if entry_value > 0:
                net_return = net_pnl / entry_value
            else:
                net_return = 0.0
            
            net_returns.append(net_return)
        
        return net_returns
    
    @staticmethod
    def calculate_enhanced_metrics(
        result: BacktestResult,
        commission_rate: float = 0.001,
        slippage_model: str = "linear",
        slippage_params: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Calculate enhanced backtesting metrics with realistic costs.
        
        Args:
            result: Backtest result
            commission_rate: Commission rate
            slippage_model: Slippage model
            slippage_params: Slippage parameters
            
        Returns:
            Enhanced metrics dictionary
        """
        # Calculate realistic returns with costs
        net_returns = BacktestMetrics.calculate_realistic_pnl(
            result.trades, commission_rate, slippage_model, slippage_params
        )
        
        if not net_returns:
            return {"error": "No trades to calculate metrics"}
        
        returns_array = np.array(net_returns)
        
        # Basic metrics
        total_return = np.sum(returns_array)
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array)
        
        # Risk-adjusted metrics
        sharpe_ratio = BacktestMetrics._calculate_sharpe_ratio(returns_array)
        sortino_ratio = BacktestMetrics._calculate_sortino_ratio(returns_array)
        calmar_ratio = BacktestMetrics._calculate_calmar_ratio(returns_array)
        
        # Drawdown metrics
        max_drawdown = BacktestMetrics._calculate_max_drawdown(returns_array)
        drawdown_duration = BacktestMetrics._calculate_drawdown_duration(returns_array)
        
        # Trade metrics
        win_rate = np.mean(returns_array > 0)
        loss_rate = np.mean(returns_array < 0)
        profit_factor = BacktestMetrics._calculate_profit_factor(returns_array)
        
        # Cost analysis
        gross_returns = [trade.pnl for trade in result.trades]
        gross_total = np.sum(gross_returns)
        total_costs = gross_total - total_return
        cost_ratio = total_costs / abs(gross_total) if gross_total != 0 else 0
        
        # Trade statistics
        trade_stats = BacktestMetrics._analyze_trade_statistics(result.trades)
        
        return {
            # Performance metrics
            "total_return": total_return,
            "annualized_return": BacktestMetrics._annualize_return(total_return, len(returns_array)),
            "mean_return": mean_return,
            "volatility": std_return * np.sqrt(252),
            
            # Risk-adjusted metrics
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "calmar_ratio": calmar_ratio,
            
            # Drawdown metrics
            "max_drawdown": max_drawdown,
            "max_drawdown_duration": drawdown_duration,
            "recovery_factor": total_return / abs(max_drawdown) if max_drawdown != 0 else 0,
            
            # Trade metrics
            "total_trades": len(result.trades),
            "win_rate": win_rate,
            "loss_rate": loss_rate,
            "profit_factor": profit_factor,
            "avg_win": np.mean(returns_array[returns_array > 0]) if np.any(returns_array > 0) else 0,
            "avg_loss": np.mean(returns_array[returns_array < 0]) if np.any(returns_array < 0) else 0,
            "largest_win": np.max(returns_array),
            "largest_loss": np.min(returns_array),
            
            # Cost analysis
            "gross_return": gross_total,
            "total_costs": total_costs,
            "cost_ratio": cost_ratio,
            "commission_costs": BacktestMetrics._estimate_commission_costs(result.trades, commission_rate),
            "slippage_costs": BacktestMetrics._estimate_slippage_costs(result.trades, slippage_model, slippage_params),
            
            # Trade statistics
            **trade_stats
        }
    
    @staticmethod
    def _calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio with annualized risk-free rate."""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        
        # Annualize returns and risk-free rate
        annual_risk_free = risk_free_rate / 252  # Daily risk-free rate
        excess_returns = returns - annual_risk_free
        
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    
    @staticmethod
    def _calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sortino ratio (downside deviation).
        
        Sortino Ratio measures risk-adjusted return considering only downside volatility.
        Formula: (Mean Return - Risk Free Rate) / Downside Deviation
        """
        if len(returns) == 0:
            return 0.0
        
        annual_risk_free = risk_free_rate / 252
        excess_returns = returns - annual_risk_free
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            # No downside returns - return very high Sortino or handle edge case
            return float('inf') if np.mean(excess_returns) > 0 else 0.0
        
        downside_deviation = np.std(downside_returns)
        if downside_deviation == 0:
            return 0.0
        
        return np.mean(excess_returns) / downside_deviation * np.sqrt(252)
    
    @staticmethod
    def _calculate_calmar_ratio(returns: np.ndarray) -> float:
        """Calculate Calmar ratio (annual return / max drawdown)."""
        if len(returns) == 0:
            return 0.0
        
        annual_return = BacktestMetrics._annualize_return(np.sum(returns), len(returns))
        max_dd = BacktestMetrics._calculate_max_drawdown(returns)
        
        if max_dd == 0:
            return 0.0
        
        return annual_return / abs(max_dd)
    
    @staticmethod
    def _calculate_max_drawdown(returns: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        if len(returns) == 0:
            return 0.0
        
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown)
    
    @staticmethod
    def _calculate_drawdown_duration(returns: np.ndarray) -> float:
        """Calculate maximum drawdown duration in periods."""
        if len(returns) == 0:
            return 0.0
        
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative < running_max
        
        if not np.any(drawdown):
            return 0.0
        
        # Find longest consecutive drawdown period
        durations = []
        current_duration = 0
        
        for is_dd in drawdown:
            if is_dd:
                current_duration += 1
            else:
                if current_duration > 0:
                    durations.append(current_duration)
                current_duration = 0
        
        if current_duration > 0:
            durations.append(current_duration)
        
        return max(durations) if durations else 0.0
    
    @staticmethod
    def _calculate_profit_factor(returns: np.ndarray) -> float:
        """Calculate profit factor."""
        if len(returns) == 0:
            return 0.0
        
        gross_profit = np.sum(returns[returns > 0])
        gross_loss = abs(np.sum(returns[returns < 0]))
        
        return gross_profit / (gross_loss + 1e-10)
    
    @staticmethod
    def _annualize_return(total_return: float, num_periods: int) -> float:
        """Annualize total return."""
        if num_periods == 0:
            return 0.0
        
        # Assuming daily returns
        years = num_periods / 252.0
        return (1 + total_return) ** (1 / years) - 1
    
    @staticmethod
    def _estimate_commission_costs(trades: List[Trade], commission_rate: float) -> float:
        """Estimate total commission costs."""
        total_commission = 0.0
        
        for trade in trades:
            if trade.entry_price:
                entry_value = trade.entry_price * trade.size
                total_commission += entry_value * commission_rate
            
            if trade.exit_price:
                exit_value = trade.exit_price * trade.size
                total_commission += exit_value * commission_rate
        
        return total_commission
    
    @staticmethod
    def _estimate_slippage_costs(
        trades: List[Trade],
        slippage_model: str,
        slippage_params: Dict[str, Any]
    ) -> float:
        """Estimate total slippage costs."""
        total_slippage = 0.0
        
        for trade in trades:
            if trade.entry_price:
                entry_slippage = BacktestMetrics.calculate_slippage(
                    trade.size, trade.entry_price, None, slippage_model, slippage_params
                )
                total_slippage += entry_slippage
            
            if trade.exit_price:
                exit_slippage = BacktestMetrics.calculate_slippage(
                    trade.size, trade.exit_price, None, slippage_model, slippage_params
                )
                total_slippage += exit_slippage
        
        return total_slippage
    
    @staticmethod
    def _analyze_trade_statistics(trades: List[Trade]) -> Dict[str, Any]:
        """Analyze detailed trade statistics."""
        if not trades:
            return {}
        
        # Trade durations
        durations = []
        for trade in trades:
            if trade.entry_time and trade.exit_time:
                duration = (trade.exit_time - trade.entry_time).total_seconds()
                durations.append(duration)
        
        # Trade sizes
        sizes = [abs(trade.size) for trade in trades]
        
        # Trade values
        entry_values = [trade.entry_price * trade.size for trade in trades if trade.entry_price]
        
        return {
            "avg_trade_duration": np.mean(durations) if durations else 0,
            "median_trade_duration": np.median(durations) if durations else 0,
            "avg_trade_size": np.mean(sizes),
            "median_trade_size": np.median(sizes),
            "avg_trade_value": np.mean(entry_values) if entry_values else 0,
            "total_volume": sum(sizes),
        }
