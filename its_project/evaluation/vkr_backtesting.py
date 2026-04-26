"""
Backtesting for VKR Models
Calculates real trading metrics (Sharpe, Sortino, Win Rate) from model predictions
"""

import numpy as np
from typing import Dict, Any, List, Tuple
import logging

logger = logging.getLogger(__name__)


class VKRBacktester:
    """
    Backtester for VKR trading models.
    
    Simulates trading based on model predictions and calculates
    real trading metrics: Sharpe, Sortino, Win Rate, Total Trades.
    """
    
    def __init__(self, initial_balance: float = 10000.0, commission: float = 0.001):
        """
        Initialize backtester.
        
        Args:
            initial_balance: Starting balance in USDT
            commission: Trading commission (default 0.1%)
        """
        self.initial_balance = initial_balance
        self.commission = commission
        
    def backtest(self, predictions: np.ndarray, actual_prices: np.ndarray,
                 threshold: float = 0.6) -> Dict[str, float]:
        """
        Run backtest on model predictions.
        
        Args:
            predictions: Model predictions (0=SELL, 1=HOLD, 2=BUY)
            actual_prices: Actual price changes (returns)
            threshold: Confidence threshold for trade execution
            
        Returns:
            Dictionary with trading metrics
        """
        balance = self.initial_balance
        position = 0.0  # Amount of asset held
        entry_price = 0.0
        trades = []
        pnl_history = []
        
        for i in range(len(predictions) - 1):
            pred = predictions[i]
            price_change = actual_prices[i]
            
            # Skip if confidence too low (for probability-based predictions)
            if isinstance(pred, (list, np.ndarray)) and len(pred) > 1:
                max_prob = np.max(pred)
                if max_prob < threshold:
                    continue
                pred = np.argmax(pred)
            
            # Execute trades based on prediction
            if pred == 2 and position == 0:  # BUY signal
                # Buy
                entry_price = 1.0  # Normalized price
                position = balance / entry_price
                balance = 0
                trades.append(('BUY', i, entry_price))
                
            elif pred == 0 and position > 0:  # SELL signal
                # Sell
                exit_price = 1.0 + price_change
                balance = position * exit_price * (1 - self.commission)
                pnl = (exit_price - entry_price) / entry_price
                pnl_history.append(pnl)
                trades.append(('SELL', i, exit_price))
                position = 0
        
        # Close any remaining position
        if position > 0:
            balance = position * 1.0 * (1 - self.commission)
            pnl = (1.0 - entry_price) / entry_price
            pnl_history.append(pnl)
        
        # Calculate metrics
        total_return = (balance - self.initial_balance) / self.initial_balance
        
        if len(pnl_history) > 0:
            sharpe = self._calculate_sharpe(pnl_history)
            sortino = self._calculate_sortino(pnl_history)
            win_rate = sum(1 for p in pnl_history if p > 0) / len(pnl_history)
        else:
            sharpe = 0.0
            sortino = 0.0
            win_rate = 0.0
        
        return {
            'sharpe': sharpe,
            'sortino': sortino,
            'win_rate': win_rate,
            'total_trades': len(trades),
            'total_return': total_return,
            'final_balance': balance,
            'pnl_history': pnl_history
        }
    
    def _calculate_sharpe(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) < 2:
            return 0.0
        
        returns_array = np.array(returns)
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array)
        
        if std_return == 0:
            return 0.0
        
        # Annualized Sharpe (assuming daily returns)
        sharpe = (mean_return / std_return) * np.sqrt(252)
        return sharpe
    
    def _calculate_sortino(self, returns: List[float]) -> float:
        """Calculate Sortino ratio (downside deviation)."""
        if len(returns) < 2:
            return 0.0
        
        returns_array = np.array(returns)
        mean_return = np.mean(returns_array)
        
        # Downside deviation (only negative returns)
        downside_returns = returns_array[returns_array < 0]
        if len(downside_returns) == 0:
            return float('inf') if mean_return > 0 else 0.0
        
        downside_deviation = np.std(downside_returns)
        if downside_deviation == 0:
            return 0.0
        
        # Annualized Sortino
        sortino = (mean_return / downside_deviation) * np.sqrt(252)
        return sortino
    
    def backtest_with_probabilities(self, probabilities: np.ndarray, 
                                   actual_prices: np.ndarray,
                                   threshold: float = 0.65) -> Dict[str, float]:
        """
        Backtest with probability-based predictions.
        
        Args:
            probabilities: Model probability predictions [n_samples, n_classes]
            actual_prices: Actual price changes
            threshold: Confidence threshold for trade execution
            
        Returns:
            Dictionary with trading metrics
        """
        # Convert probabilities to predictions
        predictions = np.argmax(probabilities, axis=1)
        
        # Filter by confidence
        max_probs = np.max(probabilities, axis=1)
        mask = max_probs >= threshold
        
        # Only backtest high-confidence predictions
        filtered_predictions = predictions[mask]
        filtered_prices = actual_prices[mask]
        
        return self.backtest(filtered_predictions, filtered_prices, threshold)


def calculate_vkr_metrics(model, X_test: np.ndarray, y_test: np.ndarray,
                          backtester: VKRBacktester = None) -> Dict[str, float]:
    """
    Calculate VKR metrics for a trained model.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels (actual returns)
        backtester: Backtester instance (creates new if None)
        
    Returns:
        Dictionary with VKR metrics
    """
    if backtester is None:
        backtester = VKRBacktester()
    
    try:
        # Get predictions
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X_test)
            metrics = backtester.backtest_with_probabilities(probabilities, y_test)
        else:
            predictions = model.predict(X_test)
            metrics = backtester.backtest(predictions, y_test)
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculating VKR metrics: {e}")
        return {
            'sharpe': 0.0,
            'sortino': 0.0,
            'win_rate': 0.0,
            'total_trades': 0,
            'total_return': 0.0,
            'final_balance': backtester.initial_balance,
            'pnl_history': []
        }
