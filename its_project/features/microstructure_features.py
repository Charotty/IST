from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from its_project.features.base import BaseFeature


class MicrostructureFeatures(BaseFeature):
    """
    Market microstructure features for high-frequency trading.
    
    Features based on order book dynamics, price impact, and market efficiency.
    No look-ahead bias - all calculations use only historical data.
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        
        # Order book features
        self.use_order_book = config.get("use_order_book", False)
        self.depth_levels = config.get("depth_levels", 5)
        self.imbalance_window = config.get("imbalance_window", 10)
        
        # Price impact features
        self.impact_window = config.get("impact_window", 20)
        self.impact_lags = config.get("impact_lags", [1, 5, 10])
        
        # Market efficiency features
        self.efficiency_window = config.get("efficiency_window", 100)
        self.hurst_window = config.get("hurst_window", 50)
        
        # Volume features
        self.volume_window = config.get("volume_window", 20)
        self.volume_lags = config.get("volume_lags", [1, 5, 10])
        
        # Volatility features
        self.volatility_windows = config.get("volatility_windows", [5, 10, 20, 50])
        self.realized_vol_window = config.get("realized_vol_window", 20)
        
        # Liquidity features
        self.liquidity_window = config.get("liquidity_window", 20)
        self.spread_window = config.get("spread_window", 10)
    
    def calculate_price_impact_features(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate price impact and Kyle's lambda."""
        impact_features = {}
        
        # Calculate returns and volume changes
        returns = data['close'].pct_change()
        volume_changes = data['volume'].pct_change()
        
        # Kyle's lambda (price impact coefficient)
        for lag in self.impact_lags:
            if lag >= len(returns):
                continue
                
            # Calculate covariance and variance
            cov_returns_volume = returns.rolling(window=self.impact_window).cov(volume_changes.shift(-lag))
            var_volume = volume_changes.rolling(window=self.impact_window).var()
            
            # Kyle's lambda
            kyles_lambda = cov_returns_volume / (var_volume + 1e-10)
            impact_features[f"kyles_lambda_{lag}"] = kyles_lambda
        
        # Amihud illiquidity ratio
        for lag in self.impact_lags:
            if lag >= len(returns):
                continue
                
            abs_returns = abs(returns)
            amihud = (abs_returns / data['volume']).rolling(window=self.impact_window).mean()
            impact_features[f"amihud_illiquidity_{lag}"] = amihud
        
        # Price impact function
        for window in [10, 20, 50]:
            if window >= len(returns):
                continue
                
            # Volume-weighted price impact
            volume_weighted_returns = returns * data['volume']
            volume_weighted_impact = volume_weighted_returns.rolling(window=window).sum() / \
                                   data['volume'].rolling(window=window).sum()
            impact_features[f"volume_weighted_impact_{window}"] = volume_weighted_impact
        
        return impact_features
    
    def calculate_order_book_features(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate order book imbalance features."""
        book_features = {}
        
        if not self.use_order_book:
            return book_features
        
        # Simulated order book imbalance using price and volume
        # In practice, this would use real order book data
        
        # Bid-ask spread estimation
        high_low_spread = (data['high'] - data['low']) / data['close']
        book_features["bid_ask_spread_estimate"] = high_low_spread.rolling(window=self.spread_window).mean()
        
        # Order flow imbalance (proxy)
        price_change = data['close'].pct_change()
        volume_change = data['volume'].pct_change()
        
        # Positive price change with high volume = buying pressure
        buying_pressure = (price_change > 0) * volume_change
        selling_pressure = (price_change < 0) * abs(volume_change)
        
        order_flow_imbalance = (buying_pressure - selling_pressure).rolling(window=self.imbalance_window).sum()
        book_features["order_flow_imbalance"] = order_flow_imbalance
        
        # Volume-weighted price
        vwap = ((data['high'] + data['low'] + data['close']) / 3 * data['volume']).rolling(window=self.imbalance_window).sum() / \
               data['volume'].rolling(window=self.imbalance_window).sum()
        
        vwap_deviation = (data['close'] - vwap) / vwap
        book_features["vwap_deviation"] = vwap_deviation
        
        # Depth imbalance (using volume as proxy)
        volume_ma = data['volume'].rolling(window=self.imbalance_window).mean()
        depth_imbalance = (data['volume'] - volume_ma) / volume_ma
        book_features["depth_imbalance"] = depth_imbalance
        
        return book_features
    
    def calculate_market_efficiency_features(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate market efficiency and predictability features."""
        efficiency_features = {}
        
        returns = data['close'].pct_change().dropna()
        
        # Autocorrelations at different lags
        for lag in [1, 5, 10, 20, 50]:
            if lag >= len(returns):
                continue
                
            autocorr = returns.rolling(window=self.efficiency_window).apply(
                lambda x: x.autocorr(lag=lag) if len(x) > lag else 0
            )
            efficiency_features[f"autocorr_{lag}"] = autocorr
        
        # Variance ratio test for random walk
        for window in [20, 50, 100]:
            if window >= len(returns):
                continue
                
            # Variance ratio: Var(r_t) / Var(r_t + r_{t+1} + ... + r_{t+k-1})
            single_period_var = returns.rolling(window=window).var()
            multi_period_var = returns.rolling(window=window).sum().var()
            
            variance_ratio = single_period_var / (multi_period_var + 1e-10)
            efficiency_features[f"variance_ratio_{window}"] = variance_ratio
        
        # Hurst exponent (simplified calculation)
        for window in [50, 100]:
            if window >= len(returns):
                continue
                
            hurst = returns.rolling(window=window).apply(self._calculate_hurst_exponent)
            efficiency_features[f"hurst_exponent_{window}"] = hurst
        
        # Predictability score (based on linear regression R²)
        for window in [20, 50]:
            if window >= len(returns):
                continue
                
            predictability = returns.rolling(window=window).apply(self._calculate_predictability)
            efficiency_features[f"predictability_{window}"] = predictability
        
        return efficiency_features
    
    def calculate_volatility_features(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate advanced volatility features."""
        vol_features = {}
        
        returns = data['close'].pct_change()
        
        # Realized volatility
        for window in self.volatility_windows:
            if window >= len(returns):
                continue
                
            realized_vol = np.sqrt((returns ** 2).rolling(window=window).sum())
            vol_features[f"realized_vol_{window}"] = realized_vol
        
        # Parkinson volatility (using high/low)
        for window in self.volatility_windows:
            if window >= len(data):
                continue
                
            hl_ratio = np.log(data['high'] / data['low'])
            parkinson_vol = np.sqrt((hl_ratio ** 2).rolling(window=window).sum() / (4 * np.log(2)))
            vol_features[f"parkinson_vol_{window}"] = parkinson_vol
        
        # Garman-Klass volatility
        for window in self.volatility_windows:
            if window >= len(data):
                continue
                
            log_hl = np.log(data['high'] / data['low'])
            log_co = np.log(data['close'] / data['open'])
            
            gk_vol = np.sqrt(
                0.5 * (log_hl ** 2).rolling(window=window).sum() - 
                (2 * np.log(2) - 1) * (log_co ** 2).rolling(window=window).sum()
            )
            vol_features[f"garman_klass_vol_{window}"] = gk_vol
        
        # Volatility of volatility
        for window in [20, 50]:
            if window >= len(returns):
                continue
                
            vol_of_vol = realized_vol.rolling(window=window).std()
            vol_features[f"vol_of_vol_{window}"] = vol_of_vol
        
        # Volatility regime detection
        vol_ma = realized_vol.rolling(window=50).mean()
        vol_regime = (realized_vol > vol_ma).astype(int)
        vol_features["volatility_regime"] = vol_regime
        
        return vol_features
    
    def calculate_volume_features(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate volume-based microstructure features."""
        volume_features = {}
        
        volume = data['volume']
        
        # Volume autocorrelations
        for lag in self.volume_lags:
            if lag >= len(volume):
                continue
                
            volume_autocorr = volume.rolling(window=self.volume_window).apply(
                lambda x: x.autocorr(lag=lag) if len(x) > lag else 0
            )
            volume_features[f"volume_autocorr_{lag}"] = volume_autocorr
        
        # Volume momentum
        for window in [5, 10, 20]:
            if window >= len(volume):
                continue
                
            volume_momentum = volume.pct_change().rolling(window=window).sum()
            volume_features[f"volume_momentum_{window}"] = volume_momentum
        
        # Relative volume
        volume_ma = volume.rolling(window=self.volume_window).mean()
        relative_volume = volume / volume_ma
        volume_features["relative_volume"] = relative_volume
        
        # Volume-weighted average price (VWAP) deviation
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        vwap = (typical_price * volume).rolling(window=self.volume_window).sum() / \
               volume.rolling(window=self.volume_window).sum()
        
        vwap_deviation = (data['close'] - vwap) / vwap
        volume_features["vwap_deviation"] = vwap_deviation
        
        # On-balance volume (OBV)
        price_change = data['close'].diff()
        obv = np.where(price_change > 0, volume, np.where(price_change < 0, -volume, 0))
        obv_cumsum = pd.Series(obv, index=data.index).cumsum()
        volume_features["obv"] = obv_cumsum
        
        # Volume rate of change
        volume_roc = volume.pct_change()
        volume_features["volume_roc"] = volume_roc
        
        return volume_features
    
    def calculate_liquidity_features(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate liquidity and market depth features."""
        liquidity_features = {}
        
        # Liquidity ratio (volume / price volatility)
        returns = data['close'].pct_change()
        vol = returns.rolling(window=self.liquidity_window).std()
        volume_ma = data['volume'].rolling(window=self.liquidity_window).mean()
        
        liquidity_ratio = volume_ma / (vol + 1e-10)
        liquidity_features["liquidity_ratio"] = liquidity_ratio
        
        # Market depth (proxy using volume)
        depth_score = volume_ma.rolling(window=self.liquidity_window).std()
        liquidity_features["market_depth"] = depth_score
        
        # Price continuity (gap analysis)
        price_gaps = data['close'].diff().abs()
        gap_ma = price_gaps.rolling(window=self.liquidity_window).mean()
        liquidity_features["price_continuity"] = gap_ma
        
        # Trading intensity
        trading_intensity = data['volume'].rolling(window=self.liquidity_window).sum()
        liquidity_features["trading_intensity"] = trading_intensity
        
        return liquidity_features
    
    def _calculate_hurst_exponent(self, returns: pd.Series) -> float:
        """Calculate Hurst exponent for a single window."""
        try:
            if len(returns) < 10:
                return 0.5  # Random walk
            
            # Calculate range of scales
            lags = range(2, min(20, len(returns) // 2))
            
            # Calculate R/S analysis
            rs_values = []
            
            for lag in lags:
                # Calculate cumulative sum
                cumulative = np.cumsum(returns - returns.mean())
                
                # Calculate range
                r_range = np.max(cumulative) - np.min(cumulative)
                
                # Calculate standard deviation
                s_std = np.std(returns)
                
                if s_std > 0:
                    rs_values.append(r_range / s_std)
            
            if len(rs_values) < 2:
                return 0.5
            
            # Linear regression in log-log space
            log_lags = np.log(lags[:len(rs_values)])
            log_rs = np.log(rs_values)
            
            # Calculate slope (Hurst exponent)
            slope = np.polyfit(log_lags, log_rs, 1)[0]
            
            return slope
            
        except:
            return 0.5
    
    def _calculate_predictability(self, returns: pd.Series) -> float:
        """Calculate predictability score using linear regression R²."""
        try:
            if len(returns) < 10:
                return 0.0
            
            # Create lagged features
            X = np.column_stack([returns.shift(i) for i in range(1, 6)])
            y = returns.values[5:]  # Remove first 5 NaN values
            
            # Remove NaN rows
            valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
            X = X[valid_mask]
            y = y[valid_mask]
            
            if len(X) < 5:
                return 0.0
            
            # Fit linear regression
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(X, y)
            
            # Calculate R²
            r_squared = model.score(X, y)
            
            return max(0.0, r_squared)  # Ensure non-negative
            
        except:
            return 0.0
    
    def calculate(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate all microstructure features."""
        self.validate_input(data)
        
        all_features = {}
        
        # Price impact features
        impact_features = self.calculate_price_impact_features(data)
        all_features.update(impact_features)
        
        # Order book features
        book_features = self.calculate_order_book_features(data)
        all_features.update(book_features)
        
        # Market efficiency features
        efficiency_features = self.calculate_market_efficiency_features(data)
        all_features.update(efficiency_features)
        
        # Volatility features
        vol_features = self.calculate_volatility_features(data)
        all_features.update(vol_features)
        
        # Volume features
        volume_features = self.calculate_volume_features(data)
        all_features.update(volume_features)
        
        # Liquidity features
        liquidity_features = self.calculate_liquidity_features(data)
        all_features.update(liquidity_features)
        
        # Convert to array
        feature_df = pd.DataFrame(all_features)
        
        # Handle NaN values (forward fill only, no look-ahead)
        feature_df = feature_df.ffill().fillna(0)
        
        # Store feature names
        self._feature_names = list(feature_df.columns)
        
        return feature_df.values
    
    def get_feature_names(self) -> List[str]:
        """Get feature names."""
        return self._feature_names.copy() if hasattr(self, '_feature_names') else []
