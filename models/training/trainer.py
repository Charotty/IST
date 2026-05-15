"""
Training Utilities Module

Training utilities for model training, validation, and hyperparameter optimization.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from typing import Callable, Dict, Any, Tuple, Optional
import warnings


class ModelTrainer:
    """
    Utility class for training models with proper time-series splitting.
    """
    
    def __init__(self, test_size=0.2, random_state=42):
        """
        Initialize model trainer.
        
        :param test_size: Test set size
        :param random_state: Random state for reproducibility
        """
        self.test_size = test_size
        self.random_state = random_state
        self.training_history = []
    
    def time_series_split(self, df: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data for time series training (no shuffling).
        
        :param df: DataFrame with features
        :param y: Target series
        :return: X_train, X_test, y_train, y_test
        """
        split_idx = int(len(df) * (1 - self.test_size))
        
        X_train = df.iloc[:split_idx]
        X_test = df.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        return X_train, X_test, y_train, y_test
    
    def walk_forward_split(self, df: pd.DataFrame, y: pd.Series, n_splits: int = 5):
        """
        Walk-forward cross-validation split.
        
        :param df: DataFrame with features
        :param y: Target series
        :param n_splits: Number of splits
        :yield: (train_idx, test_idx) tuples
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        for train_idx, test_idx in tscv.split(df):
            yield df.iloc[train_idx], df.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]
    
    def train_model(self, model: Any, X_train: pd.DataFrame, y_train: pd.Series, 
                    X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None,
                    **training_kwargs) -> Dict[str, Any]:
        """
        Train a model with optional validation.
        
        :param model: Model instance with fit method
        :param X_train: Training features
        :param y_train: Training labels
        :param X_val: Validation features (optional)
        :param y_val: Validation labels (optional)
        :param training_kwargs: Additional training arguments
        :return: Training history dict
        """
        training_info = {
            'train_size': len(X_train),
            'val_size': len(X_val) if X_val is not None else 0,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        try:
            if X_val is not None and y_val is not None:
                # Train with validation if supported
                if hasattr(model, 'fit'):
                    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], **training_kwargs)
            else:
                # Simple training
                if hasattr(model, 'fit'):
                    model.fit(X_train, y_train, **training_kwargs)
            
            training_info['status'] = 'success'
        except Exception as e:
            training_info['status'] = 'failed'
            training_info['error'] = str(e)
        
        self.training_history.append(training_info)
        return training_info
    
    def evaluate_model(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate a trained model.
        
        :param model: Trained model
        :param X_test: Test features
        :param y_test: Test labels
        :return: Evaluation metrics
        """
        metrics = {}
        
        try:
            # Get predictions
            if hasattr(model, 'predict'):
                y_pred = model.predict(X_test)
                
                # Calculate accuracy if classification
                if len(np.unique(y_test)) <= 10:  # Assume classification
                    from sklearn.metrics import accuracy_score, precision_score, recall_score
                    metrics['accuracy'] = accuracy_score(y_test, y_pred)
                    metrics['precision'] = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                    metrics['recall'] = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # Get probabilities if available
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_test)
                metrics['avg_confidence'] = np.mean(np.max(y_prob, axis=1))
        
        except Exception as e:
            metrics['error'] = str(e)
        
        return metrics
    
    def hyperparameter_search(self, model_class: Callable, param_grid: Dict[str, Any],
                              X_train: pd.DataFrame, y_train: pd.Series,
                              X_val: pd.DataFrame, y_val: pd.Series,
                              scoring: Callable = None) -> Dict[str, Any]:
        """
        Simple grid search for hyperparameter optimization.
        
        :param model_class: Model class to instantiate
        :param param_grid: Dictionary of parameter grids
        :param X_train: Training features
        :param y_train: Training labels
        :param X_val: Validation features
        :param y_val: Validation labels
        :param scoring: Scoring function
        :return: Best parameters and score
        """
        from itertools import product
        
        # Generate all parameter combinations
        keys = param_grid.keys()
        values = param_grid.values()
        combinations = [dict(zip(keys, v)) for v in product(*values)]
        
        best_score = -np.inf
        best_params = None
        
        for params in combinations:
            try:
                model = model_class(**params)
                model.fit(X_train, y_train)
                
                if scoring:
                    score = scoring(model, X_val, y_val)
                else:
                    # Default to accuracy
                    y_pred = model.predict(X_val)
                    from sklearn.metrics import accuracy_score
                    score = accuracy_score(y_val, y_pred)
                
                if score > best_score:
                    best_score = score
                    best_params = params
            
            except Exception as e:
                warnings.warn(f"Failed to train with params {params}: {e}")
        
        return {
            'best_params': best_params,
            'best_score': best_score
        }
