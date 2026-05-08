from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from enum import Enum


class TargetType(Enum):
    """Types of economic targets."""
    RETURNS = "returns"
    DIRECTION = "direction"
    VOLATILITY = "volatility"
    SHARPE = "sharpe"


class EconomicTargetCalculator:
    """
    Economic target calculator with proper future return formulation.
    
    Target definition based on future returns:
    - r_{t+h} = (p_{t+h} - p_t) / p_t
    - Target classes: SELL (r_{t+h} < -threshold), HOLD (|r_{t+h}| ≤ threshold), BUY (r_{t+h} > threshold)
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        
        # Target parameters
        self.horizon = config.get("horizon", 5)  # h in periods (e.g., 5 minutes)
        self.threshold = config.get("threshold", 0.002)  # 0.2% threshold
        self.target_type = TargetType(config.get("target_type", "direction"))
        
        # Risk-adjusted parameters
        self.risk_free_rate = config.get("risk_free_rate", 0.02)
        self.volatility_window = config.get("volatility_window", 20)
        
        # Validation
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if self.horizon <= 0:
            raise ValueError("Horizon must be positive")
        if self.threshold <= 0:
            raise ValueError("Threshold must be positive")
        if self.volatility_window <= 0:
            raise ValueError("Volatility window must be positive")
    
    def calculate_returns(self, prices: pd.Series) -> pd.Series:
        """
        Calculate future returns r_{t+h}.
        
        Args:
            prices: Price series indexed by time
            
        Returns:
            Series of future returns
        """
        # Calculate log returns
        log_returns = np.log(prices / prices.shift(1))
        
        # Calculate cumulative returns over horizon
        future_returns = log_returns.rolling(window=self.horizon).sum().shift(-self.horizon)
        
        return future_returns
    
    def calculate_volatility(self, returns: pd.Series) -> pd.Series:
        """
        Calculate rolling volatility for risk adjustment.
        
        Args:
            returns: Return series
            
        Returns:
            Rolling volatility series
        """
        return returns.rolling(window=self.volatility_window).std()
    
    def calculate_sharpe_adjusted_returns(self, returns: pd.Series, volatility: pd.Series) -> pd.Series:
        """
        Calculate Sharpe ratio adjusted returns.
        
        Args:
            returns: Return series
            volatility: Volatility series
            
        Returns:
            Sharpe adjusted returns
        """
        # Annualize (assuming daily data)
        annual_risk_free = self.risk_free_rate / 252
        excess_returns = returns - annual_risk_free
        
        # Sharpe ratio
        sharpe = excess_returns / (volatility + 1e-10)
        
        return sharpe
    
    def create_direction_target(self, future_returns: pd.Series) -> pd.Series:
        """
        Create direction-based target classes.
        
        Args:
            future_returns: Future returns r_{t+h}
            
        Returns:
            Target classes: 0 (SELL), 1 (HOLD), 2 (BUY)
        """
        # Define thresholds
        sell_threshold = -self.threshold
        buy_threshold = self.threshold
        
        # Create target classes
        target = pd.Series(index=future_returns.index, dtype=int)
        
        # SELL: future return < -threshold
        target[future_returns < sell_threshold] = 0
        
        # HOLD: |future return| ≤ threshold
        target[(future_returns >= sell_threshold) & (future_returns <= buy_threshold)] = 1
        
        # BUY: future return > threshold
        target[future_returns > buy_threshold] = 2
        
        return target
    
    def create_returns_target(self, future_returns: pd.Series) -> pd.Series:
        """
        Create continuous returns target for regression.
        
        Args:
            future_returns: Future returns
            
        Returns:
            Continuous target values
        """
        return future_returns
    
    def create_volatility_target(self, future_volatility: pd.Series) -> pd.Series:
        """
        Create volatility-based target.
        
        Args:
            future_volatility: Future volatility
            
        Returns:
            Volatility target classes
        """
        # Quantile-based classification
        q33 = future_volatility.quantile(0.33)
        q67 = future_volatility.quantile(0.67)
        
        target = pd.Series(index=future_volatility.index, dtype=int)
        target[future_volatility < q33] = 0  # Low volatility
        target[(future_volatility >= q33) & (future_volatility < q67)] = 1  # Medium
        target[future_volatility >= q67] = 2  # High volatility
        
        return target
    
    def calculate_target(self, data: pd.DataFrame) -> Tuple[pd.Series, Dict[str, Any]]:
        """
        Calculate economic target with proper future return formulation.
        
        Args:
            data: DataFrame with OHLCV data indexed by time
            
        Returns:
            Tuple of (target_series, metadata)
        """
        # Validate input
        self._validate_data(data)
        
        # Calculate prices
        prices = data['close']
        
        # Calculate future returns r_{t+h}
        future_returns = self.calculate_returns(prices)
        
        # Calculate volatility for risk adjustment
        returns = np.log(prices / prices.shift(1))
        volatility = self.calculate_volatility(returns)
        
        # Create target based on type
        if self.target_type == TargetType.DIRECTION:
            target = self.create_direction_target(future_returns)
        elif self.target_type == TargetType.RETURNS:
            target = self.create_returns_target(future_returns)
        elif self.target_type == TargetType.VOLATILITY:
            future_volatility = volatility.shift(-self.horizon)
            target = self.create_volatility_target(future_volatility)
        elif self.target_type == TargetType.SHARPE:
            sharpe_adjusted = self.calculate_sharpe_adjusted_returns(future_returns, volatility)
            target = self.create_direction_target(sharpe_adjusted)
        else:
            raise ValueError(f"Unknown target type: {self.target_type}")
        
        # Calculate metadata
        metadata = self._calculate_metadata(target, future_returns, volatility)
        
        return target, metadata
    
    def _validate_data(self, data: pd.DataFrame) -> None:
        """Validate input data."""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must be indexed by datetime")
        
        if len(data) < self.horizon + self.volatility_window:
            raise ValueError(f"Insufficient data: need at least {self.horizon + self.volatility_window} periods")
    
    def _calculate_metadata(self, target: pd.Series, future_returns: pd.Series, 
                          volatility: pd.Series) -> Dict[str, Any]:
        """Calculate target metadata."""
        # Remove NaN values
        valid_target = target.dropna()
        valid_returns = future_returns.dropna()
        valid_volatility = volatility.dropna()
        
        # Class distribution
        class_counts = valid_target.value_counts().sort_index()
        class_distribution = class_counts.to_dict()
        
        # Return statistics
        return_stats = {
            'mean': valid_returns.mean(),
            'std': valid_returns.std(),
            'min': valid_returns.min(),
            'max': valid_returns.max(),
            'skew': valid_returns.skew(),
            'kurtosis': valid_returns.kurtosis()
        }
        
        # Volatility statistics
        vol_stats = {
            'mean': valid_volatility.mean(),
            'std': valid_volatility.std(),
            'min': valid_volatility.min(),
            'max': valid_volatility.max()
        }
        
        # Economic metrics
        expected_return = valid_returns.mean()
        expected_volatility = valid_volatility.mean()
        sharpe_ratio = (expected_return - self.risk_free_rate/252) / (expected_volatility + 1e-10)
        
        return {
            'target_type': self.target_type.value,
            'horizon': self.horizon,
            'threshold': self.threshold,
            'class_distribution': class_distribution,
            'total_samples': len(valid_target),
            'return_statistics': return_stats,
            'volatility_statistics': vol_stats,
            'economic_metrics': {
                'expected_return': expected_return,
                'expected_volatility': expected_volatility,
                'sharpe_ratio': sharpe_ratio,
                'risk_adjusted_return': expected_return / (expected_volatility + 1e-10)
            }
        }
    
    def get_target_labels(self) -> Dict[int, str]:
        """Get target label mapping."""
        if self.target_type in [TargetType.DIRECTION, TargetType.SHARPE]:
            return {0: "SELL", 1: "HOLD", 2: "BUY"}
        elif self.target_type == TargetType.VOLATILITY:
            return {0: "LOW_VOL", 1: "MED_VOL", 2: "HIGH_VOL"}
        else:
            return {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}
    
    def calculate_class_weights(self, target: pd.Series) -> Dict[int, float]:
        """
        Calculate class weights for imbalanced data.
        
        Args:
            target: Target series
            
        Returns:
            Dictionary of class weights
        """
        class_counts = target.value_counts().sort_index()
        total_samples = len(target)
        
        # Inverse frequency weighting
        weights = {}
        for class_id, count in class_counts.items():
            weight = total_samples / (len(class_counts) * count)
            weights[class_id] = weight
        
        return weights
