"""
Data Leakage Prevention Module

Prevents future data leakage in target generation and train/test splits.
Implements purge and embargo mechanisms to ensure no information from
future periods contaminates training or evaluation.

Key concepts:
- Purge: Remove H bars from end of training data where H is the prediction horizon
- Embargo: Add buffer period between train and test to prevent overlap
- No meta-label on test: Never compute labels using future data on test sets
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Union
from dataclasses import dataclass


@dataclass
class LeakageConfig:
    """Configuration for data leakage prevention."""
    horizon: int = 12  # Prediction horizon in bars
    embargo: int = 5  # Embargo period between train and test in bars
    purge: bool = True  # Whether to apply purge (remove H bars from end of train)
    allow_meta_label_on_test: bool = False  # Never allow meta-label on test


class DataLeakagePreventer:
    """
    Prevents data leakage in target generation and train/test splits.
    
    Usage:
    1. Use safe_label_generation() instead of direct shift(-horizon)
    2. Use safe_train_test_split() instead of standard split
    3. Use safe_walk_forward_split() for WFO with proper embargo
    """
    
    def __init__(self, config: Optional[LeakageConfig] = None):
        """
        Initialize data leakage preventer.
        
        Args:
            config: LeakageConfig with horizon, embargo, and purge settings
        """
        self.config = config or LeakageConfig()
    
    def safe_label_generation(
        self,
        df: pd.DataFrame,
        horizon: int,
        label_func: callable,
        **kwargs
    ) -> pd.Series:
        """
        Generate labels safely without future leakage.
        
        Instead of: df['close'].shift(-horizon)
        Use: This method which ensures labels are only generated where
        future data is available and purges appropriately.
        
        Args:
            df: DataFrame with features
            horizon: Prediction horizon
            label_func: Function that generates labels (receives df, horizon, **kwargs)
            **kwargs: Additional arguments for label_func
            
        Returns:
            Series of labels with NaN where future data is not available
        """
        # Create a copy to avoid modifying original
        df_copy = df.copy()
        
        # Generate labels using the provided function
        labels = label_func(df_copy, horizon, **kwargs)
        
        # Set NaN for last 'horizon' bars (no future data available)
        labels.iloc[-horizon:] = np.nan
        
        return labels
    
    def safe_train_test_split(
        self,
        df: pd.DataFrame,
        labels: pd.Series,
        test_ratio: float = 0.2,
        horizon: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Safe train/test split with purge and embargo.
        
        Args:
            df: DataFrame with features
            labels: Series with labels
            test_ratio: Ratio of test data
            horizon: Prediction horizon (uses config.horizon if None)
            
        Returns:
            Tuple of (train_df, train_labels, test_df, test_labels)
        """
        horizon = horizon or self.config.horizon
        
        # Calculate split point
        split_idx = int(len(df) * (1 - test_ratio))
        
        # Apply purge: remove horizon bars from end of train
        if self.config.purge:
            train_end = split_idx - horizon
            train_end = max(train_end, horizon)  # Ensure we have enough data
        else:
            train_end = split_idx
        
        # Apply embargo: add buffer between train and test
        test_start = split_idx + self.config.embargo
        
        # Split data
        train_df = df.iloc[:train_end]
        train_labels = labels.iloc[:train_end]
        
        test_df = df.iloc[test_start:]
        test_labels = labels.iloc[test_start:]
        
        # Ensure test labels don't use future data (set NaN for last horizon bars)
        if not self.config.allow_meta_label_on_test:
            test_labels = test_labels.copy()
            test_labels.iloc[-horizon:] = np.nan
        
        return train_df, train_labels, test_df, test_labels
    
    def safe_walk_forward_split(
        self,
        df: pd.DataFrame,
        labels: pd.Series,
        train_size: int,
        test_size: int,
        step: int,
        horizon: Optional[int] = None
    ) -> list:
        """
        Safe walk-forward split with purge and embargo.
        
        Args:
            df: DataFrame with features
            labels: Series with labels
            train_size: Training window size
            test_size: Test window size
            step: Step size for walk-forward
            horizon: Prediction horizon (uses config.horizon if None)
            
        Returns:
            List of tuples (train_df, train_labels, test_df, test_labels)
        """
        horizon = horizon or self.config.horizon
        splits = []
        
        n_samples = len(df)
        
        for start_idx in range(0, n_samples - train_size - test_size - horizon + 1, step):
            # Train period
            train_start = start_idx
            train_end = start_idx + train_size
            
            # Apply purge to train
            if self.config.purge:
                train_end_purged = train_end - horizon
                train_end_purged = max(train_end_purged, train_start + horizon)
            else:
                train_end_purged = train_end
            
            # Apply embargo: buffer between train and test
            test_start = train_end + self.config.embargo
            test_end = test_start + test_size
            
            # Ensure we don't go beyond data
            if test_end > n_samples - horizon:
                break
            
            # Split data
            train_df = df.iloc[train_start:train_end_purged]
            train_labels = labels.iloc[train_start:train_end_purged]
            
            test_df = df.iloc[test_start:test_end]
            test_labels = labels.iloc[test_start:test_end]
            
            # Ensure test labels don't use future data
            if not self.config.allow_meta_label_on_test:
                test_labels = test_labels.copy()
                test_labels.iloc[-horizon:] = np.nan
            
            splits.append((train_df, train_labels, test_df, test_labels))
        
        return splits
    
    def validate_no_leakage(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        train_labels: pd.Series,
        test_labels: pd.Series
    ) -> dict:
        """
        Validate that there is no data leakage between train and test.
        
        Args:
            train_df: Training features
            test_df: Test features
            train_labels: Training labels
            test_labels: Test labels
            
        Returns:
            Dict with validation results
        """
        results = {
            'has_overlap': False,
            'train_end_idx': train_df.index[-1],
            'test_start_idx': test_df.index[0],
            'gap': None,
            'train_labels_nan_ratio': train_labels.isna().mean(),
            'test_labels_nan_ratio': test_labels.isna().mean(),
            'validation_passed': True
        }
        
        # Check for temporal overlap
        if train_df.index[-1] >= test_df.index[0]:
            results['has_overlap'] = True
            results['validation_passed'] = False
        
        # Calculate gap
        if len(train_df) > 0 and len(test_df) > 0:
            results['gap'] = (test_df.index[0] - train_df.index[-1]).total_seconds() / 3600  # in hours
        
        # Check if test labels have NaN at the end (proper purge)
        if not self.config.allow_meta_label_on_test:
            horizon = self.config.horizon
            if len(test_labels) >= horizon:
                last_horizon_nan = test_labels.iloc[-horizon:].isna().all()
                if not last_horizon_nan:
                    results['validation_passed'] = False
                    results['test_labels_purge_failed'] = True
        
        return results
    
    def get_safe_horizon_shift(self, df: pd.DataFrame, horizon: int) -> pd.Series:
        """
        Safe alternative to df.shift(-horizon).
        
        Instead of: df['close'].shift(-horizon)
        Use: preventer.get_safe_horizon_shift(df['close'], horizon)
        
        This returns the shifted values but with NaN for the last horizon bars
        where future data is not available.
        
        Args:
            df: Series to shift
            horizon: Horizon to shift
            
        Returns:
            Shifted series with NaN where future data is not available
        """
        shifted = df.shift(-horizon)
        shifted.iloc[-horizon:] = np.nan
        return shifted


def create_safe_mean_reversion_labels(
    df: pd.DataFrame,
    lookback: int = 20,
    threshold: float = 2.0,
    preventer: Optional[DataLeakagePreventer] = None
) -> pd.Series:
    """
    Create mean reversion labels safely without future leakage.
    
    Args:
        df: DataFrame with OHLCV data
        lookback: Lookback period for mean calculation
        threshold: Z-score threshold for reversal
        preventer: DataLeakagePreventer instance
        
    Returns:
        Series of labels with proper purge
    """
    if preventer is None:
        preventer = DataLeakagePreventer()
    
    # Calculate z-score deviation
    mean_price = df['close'].rolling(window=lookback).mean()
    std_price = df['close'].rolling(window=lookback).std()
    z_score = (df['close'] - mean_price) / std_price
    
    # Future reversion: price moves back towards mean
    future_mean = preventer.get_safe_horizon_shift(df['close'], lookback).rolling(window=lookback).mean()
    future_std = preventer.get_safe_horizon_shift(std_price, lookback)
    future_z = (preventer.get_safe_horizon_shift(df['close'], lookback) - future_mean) / future_std
    
    # Label: 1 if z-score was extreme and future z-score is less extreme
    y = ((z_score.abs() > threshold) & (future_z.abs() < z_score.abs())).astype(int)
    
    # Apply purge
    y.iloc[-lookback:] = np.nan
    
    return y


def create_safe_volatility_labels(
    df: pd.DataFrame,
    window: int = 6,
    threshold_pct: float = 0.5,
    preventer: Optional[DataLeakagePreventer] = None
) -> pd.Series:
    """
    Create volatility breakout labels safely without future leakage.
    
    Args:
        df: DataFrame with OHLCV data
        window: Window for future volatility calculation
        threshold_pct: Threshold percentage for breakout
        preventer: DataLeakagePreventer instance
        
    Returns:
        Series of labels with proper purge
    """
    if preventer is None:
        preventer = DataLeakagePreventer()
    
    if 'volatility' not in df.columns:
        df = df.copy()
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = df['log_ret'].rolling(window=20).std()
    
    future_max_vol = preventer.get_safe_horizon_shift(df['volatility'], window).rolling(window).max()
    current_avg_vol = df['volatility'].rolling(48).mean()
    
    y = (future_max_vol > current_avg_vol * (1 + threshold_pct)).astype(int)
    
    # Apply purge
    y.iloc[-window:] = np.nan
    
    return y


def create_safe_meta_labels(
    df: pd.DataFrame,
    horizon: int = 12,
    fee: float = 0.001,
    min_profit: float = 0.005,
    preventer: Optional[DataLeakagePreventer] = None,
    direction_prob_same_bar_allowed: bool = False,
) -> pd.Series:
    """
    Create meta labels safely without future leakage.
    
    Args:
        df: DataFrame with OHLCV data
        horizon: Prediction horizon
        fee: Trading fee
        min_profit: Minimum profit threshold
        preventer: DataLeakagePreventer instance
        direction_prob_same_bar_allowed: When False (default), any ``direction_prob`` on the row
            is lagged by one bar so the label does not reuse a same-bar direction estimate
            (reduces leakage / circular coupling). For strictly out-of-fold OOS preds, insert
            a column ``direction_prob_oos`` aligned to the timestamp of the predicting bar.

    Notes:
        Long-horizon return uses ``pct_change(horizon)`` on the horizon-shifted close (total
        return over ``horizon`` bars after aligning with ``get_safe_horizon_shift``), not per-bar drift.
        
    Returns:
        Series of labels with proper purge
    """
    if preventer is None:
        preventer = DataLeakagePreventer()
    
    shifted_close = preventer.get_safe_horizon_shift(df['close'], horizon)
    future_ret = shifted_close.pct_change(horizon, fill_method=None)
    
    # If direction_prob is available, use it to determine side
    col = None
    if 'direction_prob_oos' in df.columns:
        col = df['direction_prob_oos']
    elif 'direction_prob' in df.columns:
        col = df['direction_prob']
        if not direction_prob_same_bar_allowed:
            col = col.shift(1)
    if col is not None:
        side = np.where(col > 0.5, 1, -1)
    else:
        side = 1
    
    profit = side * future_ret
    y = (profit > fee + min_profit).astype(int)
    
    # Apply purge
    y.iloc[-horizon:] = np.nan
    
    return y


def compute_safe_threshold(
    values: Union[pd.Series, np.ndarray],
    mode: str = "median",
    window: Optional[int] = None,
    expanding: bool = False,
    train_threshold: Optional[float] = None,
    percentile_q: float = 0.5,
) -> Union[float, pd.Series]:
    """
    Compute threshold safely without look-ahead leakage.
    
    Instead of computing threshold over entire sample (including future),
    use one of:
    - Fixed threshold from training data
    - Rolling window using only past data
    - Expanding window using only past data
    
    Args:
        values: Values to compute threshold from
        mode: Threshold mode ('median', 'mean', 'percentile')
        window: Window size for rolling (required unless expanding or train_threshold)
        expanding: If True, use expanding window instead of rolling
        train_threshold: Fixed threshold from training (if provided, returns this scalar)
        percentile_q: Quantile in (0,1) when mode=='percentile' (default 0.5 = median)
        
    Returns:
        Threshold value (if train_threshold provided) or Series of thresholds
    """
    if train_threshold is not None:
        return train_threshold
    
    values = pd.Series(values)
    
    if not expanding and window is None:
        raise ValueError(
            "compute_safe_threshold requires train_threshold, rolling window (window=...), "
            "or expanding=True. Computing a global threshold on this series is forbidden here "
            "— use compute_train_threshold(train_values_only) off-line then pass train_threshold."
        )

    pct = percentile_q if 0 < percentile_q < 1 else 0.5

    if expanding:
        if mode == "median":
            threshold = values.expanding(min_periods=1).median()
        elif mode == "mean":
            threshold = values.expanding(min_periods=1).mean()
        elif mode == "percentile":
            threshold = values.expanding(min_periods=1).quantile(pct)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    else:
        if mode == "median":
            threshold = values.rolling(window, min_periods=1).median()
        elif mode == "mean":
            threshold = values.rolling(window, min_periods=1).mean()
        elif mode == "percentile":
            threshold = values.rolling(window, min_periods=1).quantile(pct)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    return threshold


def compute_train_threshold(
    train_values: Union[pd.Series, np.ndarray],
    mode: str = "median"
) -> float:
    """
    Compute threshold from training data only.
    
    This threshold can then be used for inference without look-ahead.
    
    Args:
        train_values: Training values
        mode: Threshold mode ('median', 'mean', 'percentile')
        
    Returns:
        Fixed threshold value from training
    """
    train_values = pd.Series(train_values)
    
    if mode == "median":
        return float(train_values.median())
    elif mode == "mean":
        return float(train_values.mean())
    elif mode == "percentile":
        return float(train_values.quantile(0.5))
    else:
        raise ValueError(f"Unknown mode: {mode}")
