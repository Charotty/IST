"""
Regime Detection Module

Detects current market conditions: trend vs flat, bullish vs bearish, high volatility vs low volatility.
Primary model: LightGBM
"""

import lightgbm as lgb
import pandas as pd
import numpy as np


class RegimeDetector:
    """
    LightGBM-based regime detection model.
    
    Determines:
    - trend vs flat
    - bullish vs bearish
    - high volatility vs low volatility
    """
    
    def __init__(self, n_estimators=100, learning_rate=0.05):
        """
        Initialize regime detector.
        
        :param n_estimators: Number of trees
        :param learning_rate: Learning rate
        """
        self.model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            objective='binary',
            metric='binary_logloss',
            importance_type='gain',
            verbose=-1
        )
        self.feature_cols = None
        self.is_fitted = False
    
    def prepare_labels(self, df, window=24, threshold_mult=1.5):
        """
        Prepare labels for regime detection.
        
        Labels: 1 if strong movement, 0 otherwise
        
        :param df: DataFrame with OHLCV data
        :param window: Window for returns calculation
        :param threshold_mult: Threshold multiplier for strong movement
        :return: Series of labels
        """
        returns = df['close'].pct_change(window).abs()
        rolling_vol = returns.rolling(window=window * 5).mean()
        y = (returns > rolling_vol * threshold_mult).astype(int)
        return y
    
    def train(self, df, y):
        """
        Train the regime detector.
        
        :param df: DataFrame with features
        :param y: Labels
        """
        if self.feature_cols is None:
            # Default features
            self.feature_cols = ['adx', 'ema_slope', 'volatility', 'atr']
        
        X = df[self.feature_cols]
        self.model.fit(X, y)
        self.is_fitted = True
    
    def predict(self, df):
        """
        Predict regime.
        
        :param df: DataFrame with features
        :return: Series of regime predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")
        
        X = df[self.feature_cols]
        return self.model.predict(X)
    
    def predict_proba(self, df):
        """
        Predict regime probabilities.
        
        :param df: DataFrame with features
        :return: DataFrame with probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")
        
        X = df[self.feature_cols]
        return self.model.predict_proba(X)
    
    def get_regime_info(self, df):
        """
        Get full regime information.
        
        :param df: DataFrame with features
        :return: Dict with regime information
        """
        probs = self.predict_proba(df)
        regime_pred = self.predict(df)
        
        # Determine market regime based on predictions
        strong_movement_prob = probs[:, 1]
        
        # Simple regime classification
        if strong_movement_prob.mean() > 0.6:
            market_regime = "trend"
        else:
            market_regime = "range"
        
        # Volatility regime (simplified)
        if 'volatility' in df.columns:
            avg_vol = df['volatility'].mean()
            if avg_vol > df['volatility'].quantile(0.7):
                volatility_regime = "high_vol"
            else:
                volatility_regime = "low_vol"
        else:
            volatility_regime = "unknown"
        
        return {
            "market_regime": market_regime,
            "volatility_regime": volatility_regime,
            "trade_allowed": market_regime == "trend",
            "strong_movement_prob": strong_movement_prob.mean()
        }
