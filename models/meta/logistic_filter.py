"""
Meta Filter Module

Final trade quality filtering using Logistic Regression.
Determines whether the signal should actually be executed.
"""

from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np


class LogisticMetaFilter:
    """
    Logistic regression-based meta filter for trade quality.
    
    Chosen for:
    - interpretability
    - stability
    - low overfitting risk
    """
    
    def __init__(self, C=1.0, penalty='l2'):
        """
        Initialize logistic meta filter.
        
        :param C: Regularization strength
        :param penalty: Regularization type
        """
        self.model = LogisticRegression(C=C, penalty=penalty, solver='lbfgs', max_iter=1000)
        self.feature_cols = None
        self.is_fitted = False
    
    def prepare_labels(self, df, horizon=12, fee=0.001, min_profit=0.005):
        """
        Prepare labels for trade quality filtering.
        
        Labels: 1 if trade is profitable after costs, 0 otherwise
        
        :param df: DataFrame with OHLCV data
        :param horizon: Prediction horizon
        :param fee: Trading fee
        :param min_profit: Minimum profit threshold
        :return: Series of labels
        """
        future_ret = df['close'].pct_change(horizon).shift(-horizon)
        
        # If direction_prob is available, use it to determine side
        if 'direction_prob' in df.columns:
            side = np.where(df['direction_prob'] > 0.5, 1, -1)
        else:
            # Default to long if no direction prob
            side = 1
        
        profit = side * future_ret
        y = (profit > fee + min_profit).astype(int)
        return y
    
    def train(self, df, y):
        """
        Train the meta filter.
        
        :param df: DataFrame with features
        :param y: Labels
        """
        if self.feature_cols is None:
            # Default features for meta filter
            self.feature_cols = ['model_confidence', 'volatility', 'regime_confidence']
            # Use available features if defaults not present
            available_cols = [col for col in self.feature_cols if col in df.columns]
            if len(available_cols) == 0:
                # Fallback to any numeric columns
                self.feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        X = df[self.feature_cols]
        self.model.fit(X, y)
        self.is_fitted = True
    
    def predict(self, df):
        """
        Predict trade quality.
        
        :param df: DataFrame with features
        :return: Series of trade quality predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")
        
        X = df[self.feature_cols]
        return self.model.predict_proba(X)[:, 1]
    
    def should_take_trade(self, df, threshold=0.5):
        """
        Determine whether to take the trade.
        
        :param df: DataFrame with features
        :param threshold: Probability threshold
        :return: Boolean indicating whether to take trade
        """
        probs = self.predict(df)
        return probs.mean() > threshold
    
    def get_prediction_info(self, df):
        """
        Get detailed prediction information.
        
        :param df: DataFrame with features
        :return: Dict with prediction info
        """
        probs = self.predict(df)
        
        return {
            "take_trade": self.should_take_trade(df),
            "trade_quality_prob": float(np.mean(probs))
        }
