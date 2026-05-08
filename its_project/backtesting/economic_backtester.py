from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from its_project.backtesting.base import BaseBacktester, BacktestResult, Trade
from its_project.decision.decision import Signal, Action
from its_project.models.base import BaseModel


@dataclass
class EconomicTrade:
    """Enhanced trade with economic metrics."""
    id: str
    symbol: str
    action: str  # 'buy', 'sell', 'hold'
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    size: float
    entry_confidence: float
    exit_confidence: Optional[float]
    commission: float = 0.0
    slippage: float = 0.0
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    return_pct: float = 0.0
    holding_period: Optional[int] = None  # in periods
    market_return: float = 0.0  # Market return during holding period
    alpha: float = 0.0  # Excess return over market
    sharpe_contribution: float = 0.0  # Contribution to portfolio Sharpe


@dataclass
class EconomicMetrics:
    """Economic performance metrics."""
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    trading_frequency: float  # Trades per period
    avg_holding_period: float
    market_alpha: float  # Alpha over market
    information_ratio: float  # Information ratio
    hit_ratio: float  # Percentage of profitable trades
    payoff_ratio: float  # Average win / average loss


class EconomicBacktester(BaseBacktester):
    """
    Economic backtester with realistic PnL calculation.
    
    Features:
    - Proper future return calculation
    - Transaction costs and slippage
    - Risk-adjusted performance metrics
    - Market comparison
    - Economic significance testing
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        
        # Trading parameters
        self.initial_capital = config.get("initial_capital", 10000.0)
        self.commission_rate = config.get("commission_rate", 0.001)  # 0.1%
        self.slippage_rate = config.get("slippage_rate", 0.0005)  # 0.05%
        self.position_size = config.get("position_size", 0.1)  # 10% of capital per trade
        
        # Risk management
        self.max_position_size = config.get("max_position_size", 0.3)  # 30% max
        self.stop_loss = config.get("stop_loss", 0.02)  # 2% stop loss
        self.take_profit = config.get("take_profit", 0.04)  # 4% take profit
        
        # Market benchmark
        self.use_market_benchmark = config.get("use_market_benchmark", True)
        self.market_symbol = config.get("market_symbol", "BTC/USDT")
        
        # Performance tracking
        self.trades: List[EconomicTrade] = []
        self.equity_curve: List[float] = []
        self.market_equity: List[float] = []
        self.current_capital: float = self.initial_capital
        self.current_position: Optional[EconomicTrade] = None
        
        # Statistics
        self.total_commission: float = 0.0
        self.total_slippage: float = 0.0
        self.total_trades: int = 0
        self.winning_trades: int = 0
        self.losing_trades: int = 0
    
    def run(
        self,
        data: pd.DataFrame,
        model: BaseModel,
        decision_maker: Any,
        target_calculator: Any = None
    ) -> BacktestResult:
        """Run economic backtest with proper PnL calculation."""
        
        # Validate data
        self._validate_data(data)
        
        print(f"Starting economic backtest on {len(data)} bars...")
        
        # Initialize
        self._initialize_backtest(data)
        
        # Process each time step
        for i in range(self.config.get('min_window', 100), len(data)):
            current_time = data.index[i]
            current_bar = data.iloc[i]
            
            # Use only historical data (no look-ahead)
            historical_data = data.iloc[:i]
            
            # Extract features
            features = self._extract_features(historical_data)
            
            # Get model prediction
            if len(features) > 0:
                prediction = model.predict(features[-1:])
                confidence = model.get_confidence(features[-1:])
                
                # Create signal
                signal = Signal(
                    action=self._prediction_to_action(prediction[0]),
                    confidence=confidence[0],
                    timestamp=int(current_time.timestamp()),
                    symbol=current_bar.get('symbol', 'BTC/USDT'),
                    metadata={
                        'price': current_bar['close'],
                        'prediction': prediction[0],
                        'features': features[-1:].tolist()
                    }
                )
                
                # Make decision
                market_state = self._create_market_state(current_bar, historical_data)
                decision = decision_maker.decide(signal, market_state)
                
                # Execute decision
                if decision:
                    self._execute_decision(decision, current_bar, current_time)
        
        # Close final position
        if self.current_position:
            self._close_position(data.iloc[-1], data.index[-1])
        
        # Calculate results
        return self._calculate_economic_results(data)
    
    def _validate_data(self, data: pd.DataFrame) -> None:
        """Validate input data."""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        if not data.index.is_monotonic_increasing:
            raise ValueError("Data must be sorted chronologically")
    
    def _initialize_backtest(self, data: pd.DataFrame) -> None:
        """Initialize backtest state."""
        self.trades = []
        self.equity_curve = [self.initial_capital]
        self.market_equity = [self.initial_capital]
        self.current_capital = self.initial_capital
        self.current_position = None
        self.total_commission = 0.0
        self.total_slippage = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
    
    def _extract_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract features without look-ahead bias."""
        # Try to use improved features if available
        try:
            from its_project.features.economic_features import EconomicFeatures
            from its_project.features.microstructure_features import MicrostructureFeatures
            
            # Economic features
            econ_config = {
                "return_periods": [1, 5, 15],
                "volatility_windows": [5, 15],
                "use_risk_features": True
            }
            econ_features = EconomicFeatures(econ_config)
            econ_feature_array = econ_features.calculate(data)
            
            # Microstructure features
            micro_config = {
                "use_order_book": False,
                "impact_window": 20,
                "efficiency_window": 50
            }
            micro_features = MicrostructureFeatures(micro_config)
            micro_feature_array = micro_features.calculate(data)
            
            # Combine features
            min_samples = min(econ_feature_array.shape[0], micro_feature_array.shape[0])
            combined_features = np.hstack([
                econ_feature_array[:min_samples], 
                micro_feature_array[:min_samples]
            ])
            
            return combined_features
            
        except Exception:
            # Fallback to basic OHLCV features
            features = data[['open', 'high', 'low', 'close', 'volume']].values
            return features
    
    def _prediction_to_action(self, prediction: int) -> Action:
        """Convert model prediction to action."""
        action_map = {0: 'sell', 1: 'hold', 2: 'buy'}
        return Action(action_map.get(prediction, 'hold'))
    
    def _create_market_state(self, current_bar: pd.Series, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """Create market state with additional context."""
        # Calculate recent volatility
        returns = historical_data['close'].pct_change().dropna()
        recent_volatility = returns.tail(20).std() if len(returns) > 20 else 0.0
        
        # Calculate recent trend
        recent_trend = (current_bar['close'] / historical_data['close'].iloc[-20] - 1) if len(historical_data) > 20 else 0.0
        
        # Market microstructure
        volume_ma = historical_data['volume'].tail(20).mean() if len(historical_data) > 20 else current_bar['volume']
        volume_ratio = current_bar['volume'] / volume_ma if volume_ma > 0 else 1.0
        
        return {
            'price': current_bar['close'],
            'volatility': recent_volatility,
            'trend': recent_trend,
            'volume_ratio': volume_ratio,
            'capital': self.current_capital,
            'in_position': self.current_position is not None
        }
    
    def _execute_decision(self, decision: Any, current_bar: pd.Series, current_time: datetime) -> None:
        """Execute trading decision with economic considerations."""
        if decision.action == Action.HOLD:
            return
        
        # Close existing position if needed
        if self.current_position:
            should_close = False
            
            if self.current_position.action == 'buy' and decision.action == Action.SELL:
                should_close = True
            elif self.current_position.action == 'sell' and decision.action == Action.BUY:
                should_close = True
            elif decision.action in [Action.BUY, Action.SELL]:
                # Close and reverse position
                should_close = True
            
            if should_close:
                self._close_position(current_bar, current_time)
        
        # Open new position
        if decision.action in [Action.BUY, Action.SELL]:
            action = 'buy' if decision.action == Action.BUY else 'sell'
            self._open_position(action, decision, current_bar, current_time)
    
    def _open_position(self, action: str, decision: Any, current_bar: pd.Series, current_time: datetime) -> None:
        """Open new position with economic calculations."""
        # Calculate position size
        available_capital = self.current_capital * self.max_position_size
        position_value = available_capital * self.position_size
        size = position_value / current_bar['close']
        
        # Calculate transaction costs
        commission = position_value * self.commission_rate
        slippage = position_value * self.slippage_rate
        
        # Total cost
        total_cost = commission + slippage
        
        # Check if enough capital
        if action == 'buy':
            required_capital = position_value + total_cost
            if required_capital > self.current_capital:
                return  # Not enough capital
        
        # Create trade
        trade_id = f"trade_{self.total_trades + 1}"
        trade = EconomicTrade(
            id=trade_id,
            symbol=current_bar.get('symbol', 'BTC/USDT'),
            action=action,
            entry_time=current_time,
            exit_time=None,
            entry_price=current_bar['close'],
            exit_price=None,
            size=size,
            entry_confidence=decision.confidence if hasattr(decision, 'confidence') else 0.5,
            exit_confidence=None,
            commission=commission,
            slippage=slippage
        )
        
        # Update capital
        if action == 'buy':
            self.current_capital -= required_capital
        else:  # sell
            # For short selling, we need margin
            margin_requirement = position_value * 0.5  # 50% margin
            if margin_requirement > self.current_capital:
                return  # Not enough margin
            self.current_capital -= margin_requirement
        
        # Update state
        self.current_position = trade
        self.total_trades += 1
        self.total_commission += commission
        self.total_slippage += slippage
    
    def _close_position(self, current_bar: pd.Series, current_time: datetime) -> None:
        """Close position and calculate PnL."""
        if not self.current_position:
            return
        
        trade = self.current_position
        
        # Calculate exit price with slippage
        exit_price = current_bar['close']
        if trade.action == 'buy':
            exit_price *= (1 - self.slippage_rate)  # Selling gets worse price
        else:
            exit_price *= (1 + self.slippage_rate)  # Buying to cover gets worse price
        
        # Calculate commission
        position_value = trade.size * exit_price
        exit_commission = position_value * self.commission_rate
        
        # Calculate PnL
        if trade.action == 'buy':
            # Long position
            gross_pnl = (exit_price - trade.entry_price) * trade.size
        else:
            # Short position
            gross_pnl = (trade.entry_price - exit_price) * trade.size
        
        # Net PnL after costs
        total_costs = trade.commission + trade.slippage + exit_commission
        net_pnl = gross_pnl - total_costs
        
        # Update trade
        trade.exit_time = current_time
        trade.exit_price = exit_price
        trade.gross_pnl = gross_pnl
        trade.net_pnl = net_pnl
        trade.return_pct = net_pnl / (trade.entry_price * trade.size) if trade.size > 0 else 0
        trade.holding_period = (current_time - trade.entry_time).total_seconds() / 60  # minutes
        
        # Calculate market return and alpha
        market_return = (exit_price - trade.entry_price) / trade.entry_price
        trade.market_return = market_return
        trade.alpha = trade.return_pct - market_return
        
        # Update capital
        if trade.action == 'buy':
            self.current_capital += position_value - exit_commission
        else:  # short
            # Return margin + PnL
            margin_return = trade.entry_price * trade.size * 0.5
            self.current_capital += margin_return + net_pnl
        
        # Update statistics
        if net_pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        # Store trade
        self.trades.append(trade)
        self.current_position = None
        self.total_commission += exit_commission
    
    def _calculate_economic_results(self, data: pd.DataFrame) -> BacktestResult:
        """Calculate comprehensive economic results."""
        # Calculate equity curve
        self._calculate_equity_curve(data)
        
        # Calculate market benchmark
        if self.use_market_benchmark:
            self._calculate_market_benchmark(data)
        
        # Calculate metrics
        metrics = self._calculate_economic_metrics(data)
        
        # Create positions DataFrame
        positions_df = self._create_positions_df()
        
        # Create trade objects for compatibility
        trades = self._create_trade_objects()
        
        return BacktestResult(
            trades=trades,
            equity_curve=np.array(self.equity_curve),
            returns=np.diff(self.equity_curve) / self.equity_curve[:-1],
            metrics=metrics,
            positions=positions_df
        )
    
    def _calculate_equity_curve(self, data: pd.DataFrame) -> None:
        """Calculate equity curve including unrealized PnL."""
        self.equity_curve = [self.initial_capital]
        
        for i, (timestamp, bar) in enumerate(data.iterrows()):
            if i == 0:
                continue
            
            current_equity = self.current_capital
            
            # Add unrealized PnL if in position
            if self.current_position:
                unrealized_pnl = self._calculate_unrealized_pnl(bar)
                current_equity += unrealized_pnl
            
            self.equity_curve.append(current_equity)
    
    def _calculate_market_benchmark(self, data: pd.DataFrame) -> None:
        """Calculate market benchmark (buy & hold)."""
        self.market_equity = [self.initial_capital]
        
        for i, bar in enumerate(data.iterrows()):
            if i == 0:
                continue
            
            # Buy & hold return
            if i > 0:
                prev_price = data.iloc[i-1]['close']
                current_price = bar[1]['close']
                market_return = (current_price - prev_price) / prev_price
                
                current_equity = self.market_equity[-1] * (1 + market_return)
                self.market_equity.append(current_equity)
    
    def _calculate_unrealized_pnl(self, current_bar: pd.Series) -> float:
        """Calculate unrealized PnL for current position."""
        if not self.current_position:
            return 0.0
        
        trade = self.current_position
        current_price = current_bar['close']
        
        if trade.action == 'buy':
            unrealized_pnl = (current_price - trade.entry_price) * trade.size
        else:
            unrealized_pnl = (trade.entry_price - current_price) * trade.size
        
        return unrealized_pnl
    
    def _calculate_economic_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive economic metrics."""
        if not self.trades:
            return {'error': 'No trades executed'}
        
        # Basic returns
        equity_array = np.array(self.equity_curve)
        returns = np.diff(equity_array) / equity_array[:-1]
        
        total_return = (equity_array[-1] / equity_array[0]) - 1
        
        # Risk metrics
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        sortino_ratio = self._calculate_sortino_ratio(returns)
        max_drawdown = self._calculate_max_drawdown(equity_array)
        calmar_ratio = (total_return / abs(max_drawdown)) if max_drawdown != 0 else 0
        
        # Trade metrics
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
        gross_profit = sum(t.gross_pnl for t in self.trades if t.gross_pnl > 0)
        gross_loss = abs(sum(t.gross_pnl for t in self.trades if t.gross_pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        wins = [t.net_pnl for t in self.trades if t.net_pnl > 0]
        losses = [t.net_pnl for t in self.trades if t.net_pnl < 0]
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        
        # Market comparison
        if self.use_market_benchmark and len(self.market_equity) > 1:
            market_array = np.array(self.market_equity)
            market_return = (market_array[-1] / market_array[0]) - 1
            alpha = total_return - market_return
            
            # Information ratio
            excess_returns = returns - np.diff(market_array) / market_array[:-1]
            information_ratio = np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) > 0 else 0
        else:
            market_return = 0
            alpha = total_return
            information_ratio = 0
        
        # Trading statistics
        avg_holding_period = np.mean([t.holding_period for t in self.trades if t.holding_period]) if self.trades else 0
        trading_frequency = self.total_trades / len(data)
        
        return {
            'total_return': total_return,
            'annualized_return': self._annualize_return(total_return, len(data)),
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'largest_win': max(t.net_pnl for t in self.trades),
            'largest_loss': min(t.net_pnl for t in self.trades),
            'trading_frequency': trading_frequency,
            'avg_holding_period': avg_holding_period,
            'market_return': market_return,
            'alpha': alpha,
            'information_ratio': information_ratio,
            'hit_ratio': win_rate,
            'payoff_ratio': avg_win / abs(avg_loss) if avg_loss != 0 else float('inf'),
            'total_trades': self.total_trades,
            'total_commission': self.total_commission,
            'total_slippage': self.total_slippage,
            'net_profit': sum(t.net_pnl for t in self.trades)
        }
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        
        daily_rf = risk_free_rate / 252
        excess_returns = returns - daily_rf
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    
    def _calculate_sortino_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio."""
        if len(returns) == 0:
            return 0.0
        
        daily_rf = risk_free_rate / 252
        excess_returns = returns - daily_rf
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf') if np.mean(excess_returns) > 0 else 0.0
        
        downside_std = np.std(downside_returns)
        if downside_std == 0:
            return 0.0
        
        return np.mean(excess_returns) / downside_std * np.sqrt(252)
    
    def _calculate_max_drawdown(self, equity_curve: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        if len(equity_curve) == 0:
            return 0.0
        
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak
        return abs(drawdown.min())
    
    def _annualize_return(self, total_return: float, num_periods: int) -> float:
        """Annualize return."""
        if num_periods == 0:
            return 0.0
        
        # Assuming daily data
        years = num_periods / 252.0
        return (1 + total_return) ** (1 / years) - 1
    
    def _create_positions_df(self) -> pd.DataFrame:
        """Create positions DataFrame."""
        positions_data = []
        
        for trade in self.trades:
            positions_data.append({
                'timestamp': trade.entry_time,
                'symbol': trade.symbol,
                'action': trade.action,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'size': trade.size,
                'commission': trade.commission,
                'slippage': trade.slippage,
                'gross_pnl': trade.gross_pnl,
                'net_pnl': trade.net_pnl,
                'return_pct': trade.return_pct,
                'holding_period': trade.holding_period,
                'alpha': trade.alpha
            })
        
        return pd.DataFrame(positions_data)
    
    def _create_trade_objects(self) -> List[Trade]:
        """Create Trade objects for compatibility."""
        trades = []
        
        for econ_trade in self.trades:
            trade = Trade(
                id=econ_trade.id,
                symbol=econ_trade.symbol,
                side=econ_trade.action,
                amount=econ_trade.size,
                price=econ_trade.entry_price,
                commission=econ_trade.commission,
                slippage=econ_trade.slippage,
                timestamp=econ_trade.entry_time,
                order_id=econ_trade.id,
                pnl=econ_trade.net_pnl
            )
            trades.append(trade)
        
        return trades
