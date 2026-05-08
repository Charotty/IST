from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional


class ReturnsTargetCalculator:
    """
    Simplified returns-based target calculator.
    
    Focus on pure return prediction without complex transformations.
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.horizon = config.get("horizon", 5)
        self.use_log_returns = config.get("use_log_returns", True)
        self.normalize = config.get("normalize", False)
    
    def calculate_simple_returns(self, prices: pd.Series) -> pd.Series:
        """Calculate simple returns."""
        return prices.pct_change()
    
    def calculate_log_returns(self, prices: pd.Series) -> pd.Series:
        """Calculate log returns."""
        return np.log(prices / prices.shift(1))
    
    def calculate_future_returns(self, prices: pd.Series) -> pd.Series:
        """Calculate future returns over horizon."""
        if self.use_log_returns:
            returns = self.calculate_log_returns(prices)
            # Cumulative log returns
            future_returns = returns.rolling(window=self.horizon).sum().shift(-self.horizon)
        else:
            # Simple compound returns
            future_prices = prices.shift(-self.horizon)
            future_returns = (future_prices - prices) / prices
        
        return future_returns
    
    def normalize_returns(self, returns: pd.Series) -> pd.Series:
        """Normalize returns using rolling statistics."""
        mean = returns.rolling(window=50).mean()
        std = returns.rolling(window=50).std()
        
        normalized = (returns - mean) / (std + 1e-10)
        return normalized
    
    def calculate_target(self, data: pd.DataFrame) -> Tuple[pd.Series, Dict[str, Any]]:
        """Calculate returns target."""
        prices = data['close']
        
        # Calculate future returns
        future_returns = self.calculate_future_returns(prices)
        
        # Normalize if requested
        if self.normalize:
            target = self.normalize_returns(future_returns)
        else:
            target = future_returns
        
        # Metadata
        metadata = {
            'target_type': 'returns',
            'horizon': self.horizon,
            'use_log_returns': self.use_log_returns,
            'normalize': self.normalize,
            'statistics': {
                'mean': target.mean(),
                'std': target.std(),
                'min': target.min(),
                'max': target.max(),
                'valid_samples': target.notna().sum()
            }
        }
        
        return target, metadata
