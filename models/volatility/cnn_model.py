"""
Volatility Breakout Prediction Module

Volatility breakout prediction using CNN.
Predicts volatility expansion, breakout probability, explosive movement conditions.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping


class CNNVolatilityModel:
    """
    CNN-based volatility breakout prediction model.
    
    Best suited for:
    - local temporal structures
    - volatility clustering
    - compression-expansion detection
    """
    
    def __init__(self, window_size=24, n_features=8, lr=0.0005, dropout=0.4):
        """
        Initialize CNN volatility model.
        
        :param window_size: Size of input window
        :param n_features: Number of input features
        :param lr: Learning rate
        :param dropout: Dropout rate
        """
        self.window_size = window_size
        self.n_features = n_features
        self.lr = lr
        self.dropout = dropout
        self.model = None
        self.is_fitted = False
        self.feature_cols = None
    
    def build_model(self):
        """Build CNN model architecture."""
        self.model = Sequential([
            Input(shape=(self.window_size, self.n_features)),
            Conv1D(filters=64, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=32, kernel_size=3, activation='relu'),
            Flatten(),
            Dense(16, activation='relu'),
            Dropout(self.dropout),
            Dense(1, activation='sigmoid')
        ])
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    
    def prepare_sequences(self, df: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for CNN training.
        
        :param df: DataFrame with features
        :param y: Target series
        :return: X, y arrays
        """
        if self.feature_cols is None:
            # Exclude leaky features: meta_prob, signals, OBI, future fields
            leaky_patterns = ['meta_prob', 'signal', 'order_book_imbalance', 'obi', 'future_', 'target_']
            self.feature_cols = [
                col for col in df.columns 
                if col != 'close' and not any(pattern in col.lower() for pattern in leaky_patterns)
            ]
        
        Xs, ys = [], []
        for i in range(len(df) - self.window_size):
            Xs.append(df[self.feature_cols].iloc[i:(i + self.window_size)].values)
            ys.append(y.iloc[i + self.window_size])
        
        return np.array(Xs), np.array(ys)
    
    def prepare_labels(self, df, window=6, threshold_pct=0.5, safe_mode=True):
        """
        Prepare labels for volatility breakout.
        
        Labels: 1 if volatility breakout, 0 otherwise
        
        :param df: DataFrame with OHLCV data
        :param window: Window for future volatility calculation
        :param threshold_pct: Threshold percentage for breakout
        :param safe_mode: If True, uses safe label generation without future leakage
        :return: Series of labels
        """
        if safe_mode:
            from utils.data_leakage_prevention import create_safe_volatility_labels
            return create_safe_volatility_labels(df, window, threshold_pct)
        
        # UNSAFE MODE - Only use for backtesting, not for production training
        if 'volatility' not in df.columns:
            # Calculate volatility if not present
            df = df.copy()
            df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
            df['volatility'] = df['log_ret'].rolling(window=20).std()
        
        future_max_vol = df['volatility'].shift(-window).rolling(window).max()
        current_avg_vol = df['volatility'].rolling(48).mean()
        
        y = (future_max_vol > current_avg_vol * (1 + threshold_pct)).astype(int)
        return y
    
    def train(self, df: pd.DataFrame, y: pd.Series, epochs=10, batch_size=64, validation_split=0.1):
        """
        Train the CNN model.
        
        Uses time-based validation split to avoid temporal data leakage.
        Last validation_split portion of data is used for validation.
        
        :param df: DataFrame with features
        :param y: Target series
        :param epochs: Number of epochs
        :param batch_size: Batch size
        :param validation_split: Validation split ratio (time-based, not random)
        """
        if self.model is None:
            self.build_model()
        
        X, y_seq = self.prepare_sequences(df, y)
        
        # Time-based split: last validation_split portion for validation
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y_seq[:split_idx], y_seq[split_idx:]
        
        early_stopping = EarlyStopping(patience=5, restore_best_weights=True)
        
        self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
            verbose=0
        )
        
        self.is_fitted = True
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict breakout probability.
        
        :param df: DataFrame with features
        :return: Array of probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")
        
        if self.feature_cols is None:
            raise ValueError("Feature columns not set")
        
        Xs = []
        for i in range(len(df) - self.window_size):
            Xs.append(df[self.feature_cols].iloc[i:(i + self.window_size)].values)
        
        X = np.array(Xs)
        probs = self.model.predict(X, verbose=0).flatten()
        
        # Pad with zeros for the first window_size rows
        probs_padded = np.zeros(len(df))
        probs_padded[self.window_size:] = probs
        
        return probs_padded
    
    def get_prediction_info(self, df: pd.DataFrame) -> dict:
        """
        Get detailed prediction information.
        
        :param df: DataFrame with features
        :return: Dict with prediction info
        """
        probs = self.predict(df)
        
        return {
            "breakout_probability": float(np.mean(probs))
        }
