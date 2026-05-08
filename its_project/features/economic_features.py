from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from its_project.features.base import BaseFeature


class EconomicFeatures(BaseFeature):
    """
    Economic features without look-ahead bias.
    
    All features use only historical data up to time t.
    No future information leakage.
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        
        # Feature parameters
        self.return_periods = config.get("return_periods", [1, 5, 15, 60])
        self.volatility_windows = config.get("volatility_windows", [5, 15, 60])
        self.momentum_windows = config.get("momentum_windows", [5, 15, 60])
        self.ma_windows = config.get("ma_windows", [5, 15, 60])
        
        # Microstructure features
        self.use_microstructure = config.get("use_microstructure", True)
        self.order_flow_window = config.get("order_flow_window", 10)
        
        # Risk features
        self.use_risk_features = config.get("use_risk_features", True)
        self.var_window = config.get("var_window", 20)
        self.sharpe_window = config.get("sharpe_window", 60)
    
    def calculate_returns(self, prices: pd.Series) -> Dict[str, pd.Series]:
        """Calculate returns for different periods."""
        returns = {}
        
        for period in self.return_periods:
            # Simple returns
            simple_ret = prices.pct_change(period)
            returns[f"return_{period}"] = simple_ret
            
            # Log returns
            log_ret = np.log(prices / prices.shift(period))
            returns[f"log_return_{period}"] = log_ret
            
            # Excess returns (assuming risk-free rate)
            risk_free_daily = 0.02 / 252  # 2% annual
            excess_ret = simple_ret - risk_free_daily * period
            returns[f"excess_return_{period}"] = excess_ret
        
        return returns
    
    def calculate_volatility(self, returns: pd.Series) -> Dict[str, pd.Series]:
        """Calculate volatility measures."""
        volatility = {}
        
        for window in self.volatility_windows:
            # Rolling standard deviation
            vol_std = returns.rolling(window=window).std()
            volatility[f"volatility_std_{window}"] = vol_std
            
            # Parkinson volatility (using high/low)
            if hasattr(returns, 'name'):  # If we have OHLC data
                # This would be calculated from high/low prices
                pass
            
            # Realized volatility (squared returns)
            realized_vol = (returns ** 2).rolling(window=window).sum()
            volatility[f"realized_vol_{window}"] = realized_vol
        
        return volatility
    
    def calculate_momentum(self, prices: pd.Series) -> Dict[str, pd.Series]:
        """Calculate momentum indicators."""
        momentum = {}
        
        for window in self.momentum_windows:
            # Price momentum
            price_momentum = prices / prices.shift(window) - 1
            momentum[f"price_momentum_{window}"] = price_momentum
            
            # Return momentum (average of recent returns)
            if f"return_1" in self.calculate_returns(prices):
                returns_1 = self.calculate_returns(prices)[f"return_1"]
                return_momentum = returns_1.rolling(window=window).mean()
                momentum[f"return_momentum_{window}"] = return_momentum
        
        return momentum
    
    def calculate_moving_averages(self, prices: pd.Series) -> Dict[str, pd.Series]:
        """Calculate moving averages and deviations."""
        ma_features = {}
        
        for window in self.ma_windows:
            # Simple moving average
            sma = prices.rolling(window=window).mean()
            ma_features[f"sma_{window}"] = sma
            
            # Exponential moving average
            ema = prices.ewm(span=window).mean()
            ma_features[f"ema_{window}"] = ema
            
            # Price deviations
            sma_deviation = (prices - sma) / sma
            ema_deviation = (prices - ema) / ema
            ma_features[f"sma_deviation_{window}"] = sma_deviation
            ma_features[f"ema_deviation_{window}"] = ema_deviation
            
            # Moving average crossover
            if window > 5:
                short_sma = prices.rolling(window=5).mean()
                crossover = (sma - short_sma) / short_sma
                ma_features[f"sma_crossover_{window}_5"] = crossover
        
        return ma_features
    
    def calculate_risk_features(self, returns: pd.Series, prices: pd.Series) -> Dict[str, pd.Series]:
        """Calculate risk management features."""
        risk_features = {}
        
        # Value at Risk (VaR)
        var_95 = returns.rolling(window=self.var_window).quantile(0.05)
        var_99 = returns.rolling(window=self.var_window).quantile(0.01)
        risk_features["var_95"] = var_95
        risk_features["var_99"] = var_99
        
        # Expected Shortfall (ES)
        es_95 = returns.rolling(window=self.var_window).apply(
            lambda x: x[x <= x.quantile(0.05)].mean()
        )
        risk_features["expected_shortfall_95"] = es_95
        
        # Maximum drawdown
        rolling_max = prices.rolling(window=self.var_window).max()
        drawdown = (prices - rolling_max) / rolling_max
        max_dd = drawdown.rolling(window=self.var_window).min()
        risk_features["max_drawdown"] = max_dd
        
        # Sharpe ratio
        mean_return = returns.rolling(window=self.sharpe_window).mean()
        vol_return = returns.rolling(window=self.sharpe_window).std()
        sharpe = mean_return / (vol_return + 1e-10)
        risk_features["sharpe_ratio"] = sharpe
        
        # Sortino ratio
        negative_returns = returns.copy()
        negative_returns[negative_returns > 0] = 0
        downside_vol = negative_returns.rolling(window=self.sharpe_window).std()
        sortino = mean_return / (downside_vol + 1e-10)
        risk_features["sortino_ratio"] = sortino
        
        return risk_features
    
    def calculate_microstructure_features(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate market microstructure features."""
        micro_features = {}
        
        if not self.use_microstructure:
            return micro_features
        
        # Price efficiency
        high_low_ratio = data['high'] / data['low']
        micro_features["high_low_ratio"] = high_low_ratio
        
        # Volume weighted average price (VWAP)
        if 'volume' in data.columns:
            typical_price = (data['high'] + data['low'] + data['close']) / 3
            vwap = (typical_price * data['volume']).rolling(window=self.order_flow_window).sum() / \
                   data['volume'].rolling(window=self.order_flow_window).sum()
            micro_features["vwap"] = vwap
            
            # VWAP deviation
            vwap_deviation = (data['close'] - vwap) / vwap
            micro_features["vwap_deviation"] = vwap_deviation
        
        # Order flow imbalance (proxy)
        if 'volume' in data.columns:
            volume_change = data['volume'].pct_change()
            price_change = data['close'].pct_change()
            
            # Kyle's lambda (price impact)
            lambda_kyle = price_change.rolling(window=self.order_flow_window).cov(volume_change) / \
                        volume_change.rolling(window=self.order_flow_window).var()
            micro_features["kyles_lambda"] = lambda_kyle
        
        # Amihud illiquidity
        if 'volume' in data.columns:
            daily_return = data['close'].pct_change()
            amihud = (abs(daily_return) / data['volume']).rolling(window=self.order_flow_window).mean()
            micro_features["amihud_illiquidity"] = amihud
        
        return micro_features
    
    def calculate(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate all features without look-ahead bias."""
        self.validate_input(data)
        
        prices = data['close']
        returns = prices.pct_change()
        
        all_features = {}
        
        # Returns
        return_features = self.calculate_returns(prices)
        all_features.update(return_features)
        
        # Volatility
        vol_features = self.calculate_volatility(returns)
        all_features.update(vol_features)
        
        # Momentum
        momentum_features = self.calculate_momentum(prices)
        all_features.update(momentum_features)
        
        # Moving averages
        ma_features = self.calculate_moving_averages(prices)
        all_features.update(ma_features)
        
        # Risk features
        if self.use_risk_features:
            risk_features = self.calculate_risk_features(returns, prices)
            all_features.update(risk_features)
        
        # Microstructure features
        micro_features = self.calculate_microstructure_features(data)
        all_features.update(micro_features)
        
        # Convert to array
        feature_df = pd.DataFrame(all_features)
        
        # Handle NaN values (only forward fill to avoid look-ahead)
        feature_df = feature_df.ffill().fillna(0)
        
        # Store feature names
        self._feature_names = list(feature_df.columns)
        
        return feature_df.values
    
    def get_feature_names(self) -> List[str]:
        """Get feature names."""
        return self._feature_names.copy() if hasattr(self, '_feature_names') else []
