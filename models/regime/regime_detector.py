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
        
        Uses bar-level inference (last row only) to avoid data leakage.
        In backtesting, this ensures regime is determined using only historical data.
        
        :param df: DataFrame with features
        :return: Dict with regime information including regime_pred aligned to df rows
                  (1 = trend-like / strong movement, 0 = range-like), same length as df.
        """
        probs = np.asarray(self.predict_proba(df))
        strong_movement_prob = probs[:, 1].astype(float)
        n = len(df)
        regime_pred = np.asarray(self.predict(df), dtype=np.int64).reshape(-1)
        if regime_pred.shape[0] != n:
            regime_pred = np.resize(regime_pred, n)

        last_prob = float(strong_movement_prob[-1])
        
        # Simple regime classification based on last bar
        if last_prob > 0.6:
            market_regime = "trend"
        else:
            market_regime = "range"
        
        # Volatility regime based on last bar (bar-level inference)
        if 'volatility' in df.columns:
            last_vol = df['volatility'].iloc[-1]
            # Use rolling quantile for context (e.g., last 100 bars) instead of full df
            vol_rolling_quantile = df['volatility'].iloc[-100:].quantile(0.7) if len(df) >= 100 else df['volatility'].quantile(0.7)
            if last_vol > vol_rolling_quantile:
                volatility_regime = "high_vol"
            else:
                volatility_regime = "low_vol"
        else:
            volatility_regime = "unknown"
        
        # Trade allowed logic: allow trading in both regimes, but models should specialize
        # Trend-following models: trend regime
        # Mean-reversion models: range regime
        return {
            "market_regime": market_regime,
            "volatility_regime": volatility_regime,
            "trade_allowed": True,  # Allow trading in both regimes
            "strong_movement_prob": float(last_prob),
            "regime_pred": regime_pred,
        }
