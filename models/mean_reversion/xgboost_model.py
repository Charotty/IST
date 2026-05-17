"""
Mean Reversion Prediction Module

Mean reversion prediction in ranging markets using XGBoost/CatBoost.
Activated during flat regimes and low trend strength environments.
"""

import xgboost as xgb
import pandas as pd
import numpy as np


class XGBoostMeanReversionModel:
    """
    XGBoost-based mean reversion prediction model.
    
    Best suited for:
    - tabular market structure
    - threshold behavior
    - indicator-based reversals
    """
    
    def __init__(self, n_estimators=200, learning_rate=0.05, max_depth=6):
        """
        Initialize XGBoost mean reversion model.
        
        :param n_estimators: Number of trees
        :param learning_rate: Learning rate
        :param max_depth: Maximum tree depth
        """
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            objective='binary:logistic',
            eval_metric='logloss',
            use_label_encoder=False,
            verbosity=0
        )
        self.feature_cols = None
        self.is_fitted = False
    
    def prepare_labels(self, df, lookback=20, threshold=2.0, safe_mode=True):
        """
        Prepare labels for mean reversion.
        
        Labels: 1 if price reverts to mean, 0 otherwise
        
        :param df: DataFrame with OHLCV data
        :param lookback: Lookback period for mean calculation
        :param threshold: Z-score threshold for reversal
        :param safe_mode: If True, uses safe label generation without future leakage
        :return: Series of labels
        """
        if safe_mode:
            from utils.data_leakage_prevention import create_safe_mean_reversion_labels
            return create_safe_mean_reversion_labels(df, lookback, threshold)
        
        # UNSAFE MODE - Only use for backtesting, not for production training
        # Calculate z-score deviation
        mean_price = df['close'].rolling(window=lookback).mean()
        std_price = df['close'].rolling(window=lookback).std()
        z_score = (df['close'] - mean_price) / std_price
        
        # Future reversion: price moves back towards mean
        future_mean = df['close'].shift(-lookback).rolling(window=lookback).mean()
        future_z = (df['close'].shift(-lookback) - future_mean) / std_price.shift(-lookback)
        
        # Label: 1 if z-score was extreme and future z-score is less extreme
        y = ((z_score.abs() > threshold) & (future_z.abs() < z_score.abs())).astype(int)
        return y
    
    def train(self, df, y):
        """
        Train the mean reversion model.
        
        :param df: DataFrame with features
        :param y: Labels
        """
        if self.feature_cols is None:
            # Default features for mean reversion
            self.feature_cols = ['rsi', 'ema_slope', 'volatility', 'atr']
        
        X = df[self.feature_cols]
        self.model.fit(X, y)
        self.is_fitted = True
    
    def predict(self, df):
        """
        Predict reversal probability.
        
        :param df: DataFrame with features
        :return: Series of reversal probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")
        
        X = df[self.feature_cols]
        return self.model.predict_proba(X)[:, 1]
    
    def get_prediction_info(self, df):
        """
        Get detailed prediction information.
        
        :param df: DataFrame with features
        :return: Dict with prediction info
        """
        probs = self.predict(df)
        
        return {
            "reversal_probability": float(np.mean(probs))
        }
