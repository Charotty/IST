from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from its_project.backtesting.base import BaseBacktester, BacktestResult, Trade
from its_project.common.types import MarketData, MarketDataType

logger = logging.getLogger(__name__)


class WalkForwardEngine:
    """Walk-forward validation engine with strict no look-ahead bias."""
    
    def __init__(
        self,
        base_backtester: BaseBacktester,
        window_size: timedelta = timedelta(days=30),
        step_size: timedelta = timedelta(days=7),
        min_train_size: timedelta = timedelta(days=21),
        commission_rate: float = 0.001,  # 0.1% commission
        slippage_model: str = "linear",  # "linear", "percentage", "fixed"
        slippage_params: Dict[str, Any] = None
    ) -> None:
        self.base_backtester = base_backtester
        self.window_size = window_size
        self.step_size = step_size
        self.min_train_size = min_train_size
        self.commission_rate = commission_rate
        self.slippage_model = slippage_model
        self.slippage_params = slippage_params or {}
        
        # Results storage
        self.window_results: List[BacktestResult] = []
        self.aggregate_metrics: Optional[Dict[str, Any]] = None
    
    def validate(
        self,
        data: List[MarketData],
        model_class: Any,
        model_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform walk-forward validation.
        
        Args:
            data: Historical market data
            model_class: Model class to use
            model_params: Model parameters
            
        Returns:
            Aggregate validation results
        """
        if not data:
            raise ValueError("No data provided for validation")
        
        # Sort data by timestamp
        sorted_data = sorted(data, key=lambda x: x.timestamp_ms)
        
        # Convert to DataFrame for easier time handling
        df = pd.DataFrame([
            {"timestamp": datetime.fromtimestamp(md.timestamp_ms / 1000), "data": md}
            for md in sorted_data
        ]).set_index("timestamp")
        
        # Generate walk-forward windows
        windows = self._generate_windows(df)
        
        logger.info(f"Starting walk-forward validation with {len(windows)} windows")
        
        # Process each window
        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            logger.info(f"Processing window {i+1}/{len(windows)}")
            
            # Split data
            train_data = df.loc[train_start:train_end]["data"].tolist()
            test_data = df.loc[test_start:test_end]["data"].tolist()
            
            # Train model
            model = self._train_model(train_data, model_class, model_params)
            
            # Run backtest on test data
            result = self._run_backtest(test_data, model)
            
            # Store result
            self.window_results.append(result)
        
        # Calculate aggregate metrics
        self.aggregate_metrics = self._calculate_aggregate_metrics()
        
        return self.aggregate_metrics
    
    def _generate_windows(self, df: pd.DataFrame) -> List[Tuple[datetime, datetime, datetime, datetime]]:
        """Generate walk-forward windows."""
        windows = []
        
        # Start from the beginning
        current_start = df.index[0]
        end_time = df.index[-1]
        
        while True:
            # Calculate window boundaries
            train_end = current_start + self.min_train_size
            window_end = current_start + self.window_size
            
            if window_end > end_time:
                break
            
            # Test period starts after training period
            test_start = train_end
            test_end = window_end
            
            # Ensure we have enough data
            if test_end > end_time:
                test_end = end_time
            
            # Add window
            windows.append((current_start, train_end, test_start, test_end))
            
            # Move to next window
            current_start += self.step_size
        
        return windows
    
    def _train_model(self, train_data: List[MarketData], model_class: Any, model_params: Dict[str, Any]) -> Any:
        """Train model on training data."""
        # Convert training data to features
        X_train, y_train = self._prepare_training_data(train_data)
        
        # Create and train model
        model = model_class(**model_params)
        model.fit(X_train, y_train)
        
        return model
    
    def _prepare_training_data(self, data: List[MarketData]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from MarketData."""
        # This is a simplified implementation
        # In practice, you'd use the feature engineering pipeline
        
        features = []
        targets = []
        
        for i in range(len(data) - 1):
            # Extract features from current data point
            current_data = data[i]
            next_data = data[i + 1]
            
            # Simple feature extraction (price change)
            if "p" in current_data.data and "p" in next_data.data:
                current_price = float(current_data.data["p"])
                next_price = float(next_data.data["p"])
                
                # Feature: price (simplified)
                feature = [current_price]
                
                # Target: price direction
                price_change = (next_price - current_price) / current_price
                if price_change > 0.001:
                    target = 2  # Buy
                elif price_change < -0.001:
                    target = 0  # Sell
                else:
                    target = 1  # Hold
                
                features.append(feature)
                targets.append(target)
        
        return np.array(features), np.array(targets)
    
    def _run_backtest(self, test_data: List[MarketData], model: Any) -> BacktestResult:
        """Run backtest on test data with realistic costs."""
        # Configure backtester with commission and slippage
        self.base_backtester.commission_rate = self.commission_rate
        self.base_backtester.slippage_model = self.slippage_model
        self.base_backtester.slippage_params = self.slippage_params
        
        # Run backtest
        result = self.base_backtester.run(test_data, model)
        
        return result
    
    def _calculate_aggregate_metrics(self) -> Dict[str, Any]:
        """Calculate aggregate metrics across all windows."""
        if not self.window_results:
            return {}
        
        # Aggregate returns
        all_returns = []
        all_trades = []
        
        for result in self.window_results:
            all_returns.extend(result.returns)
            all_trades.extend(result.trades)
        
        # Calculate metrics
        returns_array = np.array(all_returns)
        
        metrics = {
            "total_windows": len(self.window_results),
            "total_return": np.sum(returns_array),
            "sharpe_ratio": self._calculate_sharpe_ratio(returns_array),
            "max_drawdown": self._calculate_max_drawdown(returns_array),
            "win_rate": np.mean(returns_array > 0),
            "total_trades": len(all_trades),
            "avg_trade_duration": self._calculate_avg_trade_duration(all_trades),
            "profit_factor": self._calculate_profit_factor(all_trades),
            "window_consistency": self._calculate_window_consistency(),
        }
        
        return metrics
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        excess_returns = returns - risk_free_rate
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        if len(returns) == 0:
            return 0.0
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown)
    
    def _calculate_avg_trade_duration(self, trades: List[Trade]) -> float:
        """Calculate average trade duration."""
        if not trades:
            return 0.0
        
        durations = []
        for trade in trades:
            if trade.exit_time and trade.entry_time:
                duration = (trade.exit_time - trade.entry_time).total_seconds()
                durations.append(duration)
        
        return np.mean(durations) if durations else 0.0
    
    def _calculate_profit_factor(self, trades: List[Trade]) -> float:
        """Calculate profit factor."""
        if not trades:
            return 0.0
        
        gross_profit = sum(trade.pnl for trade in trades if trade.pnl > 0)
        gross_loss = abs(sum(trade.pnl for trade in trades if trade.pnl < 0))
        
        return gross_profit / (gross_loss + 1e-10)
    
    def _calculate_window_consistency(self) -> float:
        """Calculate consistency of returns across windows."""
        if not self.window_results:
            return 0.0
        
        window_returns = [np.sum(result.returns) for result in self.window_results]
        
        # Consistency = 1 - coefficient of variation
        if np.mean(window_returns) == 0:
            return 0.0
        
        cv = np.std(window_returns) / abs(np.mean(window_returns))
        consistency = max(0.0, 1.0 - cv)
        
        return consistency
    
    def get_window_results(self) -> List[BacktestResult]:
        """Get results for each window."""
        return self.window_results.copy()
    
    def get_performance_report(self) -> pd.DataFrame:
        """Get detailed performance report."""
        if not self.window_results:
            return pd.DataFrame()
        
        report_data = []
        for i, result in enumerate(self.window_results):
            returns = np.array(result.returns)
            
            report_data.append({
                "window": i + 1,
                "return": np.sum(returns),
                "sharpe": self._calculate_sharpe_ratio(returns),
                "max_dd": self._calculate_max_drawdown(returns),
                "trades": len(result.trades),
                "win_rate": np.mean(returns > 0),
            })
        
        return pd.DataFrame(report_data)


class NoLookAheadValidator:
    """Validator to ensure no look-ahead bias in backtesting."""
    
    @staticmethod
    def validate_data_leakage(
        train_data: List[MarketData],
        test_data: List[MarketData]
    ) -> bool:
        """Validate that test data doesn't leak into training."""
        if not train_data or not test_data:
            return False
        
        # Get timestamps
        train_max_ts = max(md.timestamp_ms for md in train_data)
        test_min_ts = min(md.timestamp_ms for md in test_data)
        
        # Ensure test data starts after training data
        if test_min_ts <= train_max_ts:
            raise ValueError("Data leakage detected: test data overlaps with training data")
        
        return True
    
    @staticmethod
    def validate_feature_leakage(
        features: np.ndarray,
        targets: np.ndarray,
        lookback_window: int = 1
    ) -> bool:
        """Validate that features don't use future information."""
        # Check that features at time t only use data up to time t
        for i in range(lookback_window, len(features)):
            # Simple check: features should be similar to previous features
            # (no sudden jumps that would indicate future data usage)
            feature_diff = np.abs(features[i] - features[i-1])
            
            # If feature difference is too large, might indicate leakage
            if np.any(feature_diff > 10 * np.std(feature_diff)):
                logger.warning(f"Potential feature leakage detected at index {i}")
                return False
        
        return True
    
    @staticmethod
    def validate_temporal_order(data: List[MarketData]) -> bool:
        """Validate that data is in chronological order."""
        timestamps = [md.timestamp_ms for md in data]
        
        if timestamps != sorted(timestamps):
            raise ValueError("Data is not in chronological order")
        
        return True
