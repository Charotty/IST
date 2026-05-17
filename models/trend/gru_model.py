"""
Trend Prediction Module

Directional trend continuation prediction using GRU/LSTM.
Activated during trending markets and directional momentum regimes.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping


class GRUTrendModel:
    """
    GRU-based trend prediction model.
    
    Best suited for:
    - sequential dependencies
    - momentum persistence
    - temporal pattern learning
    """
    
    def __init__(self, window_size=24, n_features=8, lr=0.0005, dropout=0.2):
        """
        Initialize GRU trend model.
        
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
        """Build GRU model architecture."""
        self.model = Sequential([
            Input(shape=(self.window_size, self.n_features)),
            GRU(64, return_sequences=True),
            Dropout(self.dropout),
            GRU(32),
            Dropout(self.dropout),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
    
    def prepare_sequences(self, df: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for GRU training.
        
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
    
    def train(self, df: pd.DataFrame, y: pd.Series, epochs=10, batch_size=64, validation_split=0.1):
        """
        Train the GRU model.
        
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
        Predict trend probabilities.
        
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
            "long_probability": float(np.mean(probs)),
            "short_probability": float(np.mean(1 - probs))
        }
