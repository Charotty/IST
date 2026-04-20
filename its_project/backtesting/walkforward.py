from __future__ import annotations

from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np

from its_project.backtesting.base import BaseBacktester, BacktestResult
from its_project.backtesting.simple import SimpleBacktester


class WalkForwardValidator:
    """
    Walk-forward validation for robust strategy testing.
    
    Implements rolling window validation to prevent overfitting
    and test strategy stability over time.
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.train_size = config.get('train_size', 252)  # 1 year
        self.test_size = config.get('test_size', 63)     # 3 months
        self.step_size = config.get('step_size', 21)     # 1 month
        self.min_window = config.get('min_window', 100)
        
    def validate(
        self,
        data: pd.DataFrame,
        model_factory: Any,
        decision_maker_factory: Any,
    ) -> List[BacktestResult]:
        """
        Perform walk-forward validation.
        
        Args:
            data: Historical data (MUST be sorted by time)
            model_factory: Function to create new model instances
            decision_maker_factory: Function to create decision makers
        
        Returns:
            List of BacktestResult for each window
        """
        # CRITICAL: Check time sorting
        if not data['timestamp'].is_monotonic_increasing:
            raise ValueError("Data MUST be sorted by time!")
        
        results = []
        total_bars = len(data)
        
        # Calculate windows
        windows = self._calculate_windows(total_bars)
        
        print(f"Starting walk-forward validation with {len(windows)} windows...")
        
        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            print(f"\nWindow {i+1}/{len(windows)}:")
            print(f"  Train: {train_start} to {train_end} ({train_end-train_start} bars)")
            print(f"  Test:  {test_start} to {test_end} ({test_end-test_start} bars)")
            
            # Split data
            train_data = data.iloc[train_start:train_end]
            test_data = data.iloc[test_start:test_end]
            
            # Create fresh model and decision maker
            model = model_factory()
            decision_maker = decision_maker_factory()
            
            # Train model on training data
            self._train_model(model, train_data)
            
            # Run backtest on test data
            backtester = SimpleBacktester(self.config)
            result = backtester.run(test_data, model, decision_maker)
            
            # Store results
            results.append(result)
            
            # Print window summary
            print(f"  Return: {result.metrics['total_return']:.2%}")
            print(f"  Sharpe: {result.metrics['sharpe_ratio']:.2f}")
            print(f"  Trades: {result.metrics['num_trades']}")
        
        # Calculate aggregate statistics
        self._print_aggregate_results(results)
        
        return results
    
    def _calculate_windows(self, total_bars: int) -> List[Tuple[int, int, int, int]]:
        """Calculate walk-forward windows."""
        windows = []
        
        start = 0
        while start + self.train_size + self.test_size <= total_bars:
            train_start = start
            train_end = start + self.train_size
            test_start = train_end
            test_end = test_start + self.test_size
            
            windows.append((train_start, train_end, test_start, test_end))
            
            start += self.step_size
        
        return windows
    
    def _train_model(self, model: Any, train_data: pd.DataFrame) -> None:
        """Train model on training data."""
        # Extract features from training data
        features = self._extract_features(train_data)
        
        # Create simple labels (next return direction)
        returns = train_data['close'].pct_change().shift(-1)
        labels = np.where(returns > 0.001, 2, np.where(returns < -0.001, 0, 1))
        labels = labels[:-1]  # Remove last NaN
        
        # Use only complete samples
        valid_idx = ~np.isnan(labels)
        X_train = features[valid_idx]
        y_train = labels[valid_idx]
        
        if len(X_train) < self.min_window:
            print("  Warning: Insufficient training data, skipping...")
            return
        
        # Train model
        model.fit(X_train, y_train)
    
    def _extract_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract features from data."""
        # Simple implementation - use OHLCV
        features = data[['open', 'high', 'low', 'close', 'volume']].values
        return features
    
    def _print_aggregate_results(self, results: List[BacktestResult]) -> None:
        """Print aggregate statistics across all windows."""
        if not results:
            return
        
        # Calculate aggregate metrics
        returns = [r.metrics['total_return'] for r in results]
        sharpes = [r.metrics['sharpe_ratio'] for r in results]
        drawdowns = [r.metrics['max_drawdown'] for r in results]
        win_rates = [r.metrics['win_rate'] for r in results]
        
        print(f"\n{'='*60}")
        print("WALK-FORWARD VALIDATION SUMMARY")
        print(f"{'='*60}")
        print(f"Windows tested: {len(results)}")
        print(f"Average return: {np.mean(returns):.2%} (±{np.std(returns):.2%})")
        print(f"Average Sharpe: {np.mean(sharpes):.2f} (±{np.std(sharpes):.2f})")
        print(f"Average max drawdown: {np.mean(drawdowns):.2%}")
        print(f"Average win rate: {np.mean(win_rates):.2%}")
        
        # Consistency checks
        positive_returns = sum(1 for r in returns if r > 0)
        print(f"Windows with positive returns: {positive_returns}/{len(results)} ({positive_returns/len(results):.1%})")
        
        positive_sharpes = sum(1 for s in sharpes if s > 0)
        print(f"Windows with positive Sharpe: {positive_sharpes}/{len(results)} ({positive_sharpes/len(results):.1%})")
        
        # Stability assessment
        return_std = np.std(returns)
        if return_std < 0.1:  # 10% standard deviation
            print("Strategy appears STABLE (low return variance)")
        elif return_std < 0.2:
            print("Strategy appears MODERATELY STABLE")
        else:
            print("Strategy appears UNSTABLE (high return variance)")
        
        # Performance consistency
        if positive_returns / len(results) > 0.7:
            print("Strategy shows CONSISTENT performance")
        elif positive_returns / len(results) > 0.5:
            print("Strategy shows MODERATE consistency")
        else:
            print("Strategy shows POOR consistency")


class MultiAssetWalkForward(WalkForwardValidator):
    """
    Walk-forward validation for multiple assets simultaneously.
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.assets = config.get('assets', ['BTCUSDT', 'ETHUSDT'])
    
    def validate_multi_asset(
        self,
        data_dict: Dict[str, pd.DataFrame],
        model_factory: Any,
        decision_maker_factory: Any,
    ) -> Dict[str, List[BacktestResult]]:
        """
        Perform walk-forward validation on multiple assets.
        
        Args:
            data_dict: Dictionary of asset -> DataFrame
            model_factory: Function to create model instances
            decision_maker_factory: Function to create decision makers
        
        Returns:
            Dictionary of asset -> list of BacktestResult
        """
        results = {}
        
        for asset in self.assets:
            if asset not in data_dict:
                print(f"Warning: No data for {asset}, skipping...")
                continue
            
            print(f"\n{'='*40}")
            print(f"VALIDATING ASSET: {asset}")
            print(f"{'='*40}")
            
            asset_data = data_dict[asset]
            asset_results = self.validate(asset_data, model_factory, decision_maker_factory)
            results[asset] = asset_results
        
        # Print cross-asset comparison
        self._print_cross_asset_comparison(results)
        
        return results
    
    def _print_cross_asset_comparison(self, results_dict: Dict[str, List[BacktestResult]]) -> None:
        """Print comparison across assets."""
        print(f"\n{'='*60}")
        print("CROSS-ASSET COMPARISON")
        print(f"{'='*60}")
        
        for asset, results in results_dict.items():
            if not results:
                continue
            
            avg_return = np.mean([r.metrics['total_return'] for r in results])
            avg_sharpe = np.mean([r.metrics['sharpe_ratio'] for r in results])
            win_rate = np.mean([r.metrics['win_rate'] for r in results])
            
            print(f"{asset}:")
            print(f"  Avg Return: {avg_return:.2%}")
            print(f"  Avg Sharpe: {avg_sharpe:.2f}")
            print(f"  Win Rate: {win_rate:.2%}")
        
        # Find best performing asset
        best_asset = None
        best_sharpe = -float('inf')
        
        for asset, results in results_dict.items():
            if results:
                avg_sharpe = np.mean([r.metrics['sharpe_ratio'] for r in results])
                if avg_sharpe > best_sharpe:
                    best_sharpe = avg_sharpe
                    best_asset = asset
        
        if best_asset:
            print(f"\nBest performing asset: {best_asset} (Sharpe: {best_sharpe:.2f})")
