from __future__ import annotations

from typing import Dict, Any
import pandas as pd
import numpy as np

from its_project.backtesting.base import BaseBacktester, BacktestResult, Trade
from its_project.decision.decision import Signal, Action
from its_project.models.base import BaseModel


class SimpleBacktester(BaseBacktester):
    """Simple backtester for long/short strategies with no look-ahead bias."""
    
    def run(
        self,
        data: pd.DataFrame,
        model: BaseModel,
        decision_maker: Any,  # DecisionMaker instance
    ) -> BacktestResult:
        """Run backtest with strict chronological order."""
        
        # CRITICAL: Check time sorting
        if not data['timestamp'].is_monotonic_increasing:
            raise ValueError("Data MUST be sorted by time!")
        
        print(f"Starting backtest on {len(data)} bars...")
        
        # Minimum window for feature calculation
        min_window = self.config.get('min_window', 100)
        
        # Reset state
        self.trades = []
        self.equity_history = [self.initial_capital]
        self.current_capital = self.initial_capital
        self.current_position = None
        
        for i in range(min_window, len(data)):
            # Use ONLY data BEFORE current timestamp
            historical_data = data.iloc[:i]
            current_bar = data.iloc[i]
            
            # Extract features using only past data!
            features = self._extract_features(historical_data)
            
            # Get signal from model (no future data!)
            prediction = model.predict(features[-1:])
            confidence = model.get_confidence(features[-1:])
            
            # Create signal
            signal = Signal(
                action=self._prediction_to_action(prediction[0]),
                confidence=confidence[0],
                timestamp=int(current_bar['timestamp']),
                symbol=current_bar.get('symbol', 'BTCUSDT'),
                metadata={'price': current_bar['close']}
            )
            
            # Make decision
            market_state = {'price': current_bar['close']}
            decision = decision_maker.decide(signal, market_state)
            
            # Execute decision
            if decision and decision.action != Action.HOLD:
                self._execute_decision(decision, current_bar)
        
        # Calculate results
        return self._calculate_results()
    
    def _extract_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract features using only historical data."""
        # Simple implementation - use OHLCV as features
        # In real implementation, this would use the feature pipeline
        features = data[['open', 'high', 'low', 'close', 'volume']].values
        return features
    
    def _prediction_to_action(self, prediction: int) -> Action:
        """Convert model prediction to action."""
        action_map = {0: 'sell', 1: 'hold', 2: 'buy'}
        return Action(action_map.get(prediction, 'hold'))
    
    def _execute_decision(self, decision: Any, current_bar: pd.Series) -> None:
        """Execute trading decision."""
        # Close existing position if needed
        if self.current_position:
            if (self.current_position['side'] == 'long' and decision.action == Action.SELL) or \
               (self.current_position['side'] == 'short' and decision.action == Action.BUY):
                self._close_position(current_bar)
        
        # Open new position
        if decision.action in [Action.BUY, Action.SELL]:
            side = 'buy' if decision.action == Action.BUY else 'sell'
            self._open_position(side, decision.size, current_bar)
    
    def _open_position(self, side: str, size: float, current_bar: pd.Series) -> None:
        """Open new position."""
        trade = self.execute_trade(
            timestamp=int(current_bar['timestamp']),
            symbol=current_bar.get('symbol', 'BTCUSDT'),
            side=side,
            price=current_bar['close'],
            size=size
        )
        
        self.current_position = {
            'side': side,
            'size': size,
            'entry_price': trade.price,
            'entry_time': trade.timestamp,
            'entry_trade': trade
        }
    
    def _close_position(self, current_bar: pd.Series) -> None:
        """Close existing position."""
        if not self.current_position:
            return
        
        side = 'sell' if self.current_position['side'] == 'long' else 'buy'
        
        trade = self.execute_trade(
            timestamp=int(current_bar['timestamp']),
            symbol=current_bar.get('symbol', 'BTCUSDT'),
            side=side,
            price=current_bar['close'],
            size=self.current_position['size']
        )
        
        # Calculate PnL
        if self.current_position['side'] == 'long':
            pnl = (trade.price - self.current_position['entry_price']) * self.current_position['size']
        else:
            pnl = (self.current_position['entry_price'] - trade.price) * self.current_position['size']
        
        # Update PnL for both trades
        self.current_position['entry_trade'].pnl = -pnl  # Opening trade PnL
        trade.pnl = pnl  # Closing trade PnL
        
        self.current_position = None
    
    def _calculate_results(self) -> BacktestResult:
        """Calculate final backtest results."""
        # Close any remaining position
        if self.current_position:
            # Use last price to close
            last_price = self.trades[-1].price if self.trades else self.initial_capital
            dummy_bar = pd.Series({'timestamp': self.trades[-1].timestamp, 'close': last_price})
            self._close_position(dummy_bar)
        
        # Calculate equity curve and returns
        equity_curve = np.array(self.equity_history)
        returns = np.diff(equity_curve) / equity_curve[:-1]
        
        # Calculate metrics
        metrics = self._calculate_metrics(equity_curve, returns)
        
        # Create positions DataFrame
        positions = self._create_positions_df()
        
        return BacktestResult(
            trades=self.trades,
            equity_curve=equity_curve,
            returns=returns,
            metrics=metrics,
            positions=positions
        )
    
    def _calculate_metrics(self, equity_curve: np.ndarray, returns: np.ndarray) -> Dict[str, float]:
        """Calculate performance metrics."""
        total_return = (equity_curve[-1] / equity_curve[0]) - 1
        
        # Sharpe ratio
        if len(returns) > 0 and returns.std() > 0:
            sharpe_ratio = np.sqrt(252) * (returns.mean() / returns.std())
        else:
            sharpe_ratio = 0.0
        
        # Maximum drawdown
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak
        max_drawdown = abs(drawdown.min())
        
        # Win rate
        winning_trades = [t for t in self.trades if t.pnl and t.pnl > 0]
        win_rate = len(winning_trades) / len(self.trades) if self.trades else 0.0
        
        # Profit factor
        gross_profit = sum(t.pnl for t in self.trades if t.pnl and t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl and t.pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'num_trades': len(self.trades),
            'avg_return': returns.mean() if len(returns) > 0 else 0.0,
            'volatility': returns.std() if len(returns) > 0 else 0.0,
        }
    
    def _create_positions_df(self) -> pd.DataFrame:
        """Create positions DataFrame."""
        positions_data = []
        
        for trade in self.trades:
            positions_data.append({
                'timestamp': trade.timestamp,
                'symbol': trade.symbol,
                'side': trade.side,
                'price': trade.price,
                'size': trade.size,
                'commission': trade.commission,
                'slippage': trade.slippage,
                'pnl': trade.pnl,
            })
        
        return pd.DataFrame(positions_data)
