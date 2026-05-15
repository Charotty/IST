"""
Signal rules for decision layer.

Provides functions to compute final_signal and integrated_signal based on
direction_soft_signal, meta_prob, and meta_mgmt_prob.
"""

import numpy as np
import pandas as pd
from typing import Union, Literal


def compute_final_signal(
    direction_soft_signal: Union[np.ndarray, pd.Series],
    meta_prob: Union[np.ndarray, pd.Series],
    meta_threshold_mode: Literal["median", "mean", "fixed"] = "median",
    meta_threshold_value: float = None,
    direction_threshold: float = 0.52
) -> np.ndarray:
    """
    Compute final_signal using MetaFilter + MTF approach (Variant A).
    
    Rule:
    - meta_threshold = meta_prob.median() (or other mode)
    - final_signal = direction_soft_signal if meta_prob > meta_threshold and direction_soft_signal != 0
    - final_signal = 0 otherwise
    
    Args:
        direction_soft_signal: Soft signal from DirectionModel (thresholded at 0.52)
        meta_prob: Meta probability from MetaFilter
        meta_threshold_mode: Mode for computing meta_threshold ("median", "mean", "fixed")
        meta_threshold_value: Fixed threshold value if mode is "fixed"
        direction_threshold: Threshold for direction signal (default 0.52)
        
    Returns:
        final_signal array with values: 1 (Long), -1 (Short), 0 (Flat)
    """
    # Convert to numpy arrays if needed
    direction_soft_signal = np.asarray(direction_soft_signal)
    meta_prob = np.asarray(meta_prob)
    
    # Apply direction threshold
    direction_signal = np.where(
        direction_soft_signal > direction_threshold, 1,
        np.where(direction_soft_signal < -direction_threshold, -1, 0)
    )
    
    # Compute meta threshold
    if meta_threshold_mode == "median":
        meta_threshold = np.median(meta_prob)
    elif meta_threshold_mode == "mean":
        meta_threshold = np.mean(meta_prob)
    elif meta_threshold_mode == "fixed":
        if meta_threshold_value is None:
            raise ValueError("meta_threshold_value must be provided when mode is 'fixed'")
        meta_threshold = meta_threshold_value
    else:
        raise ValueError(f"Unknown meta_threshold_mode: {meta_threshold_mode}")
    
    # Apply meta filter
    final_signal = np.where(
        (meta_prob > meta_threshold) & (direction_signal != 0),
        direction_signal,
        0
    )
    
    return final_signal


def compute_integrated_signal(
    direction_soft_signal: Union[np.ndarray, pd.Series],
    meta_mgmt_prob: Union[np.ndarray, pd.Series],
    meta_threshold_mode: Literal["median", "mean", "fixed"] = "median",
    meta_threshold_value: float = None,
    direction_threshold: float = 0.52
) -> np.ndarray:
    """
    Compute integrated_signal using Regime-Adaptive Ensemble approach (Variant B - recommended).
    
    Rule:
    - meta_threshold = meta_mgmt_prob.median() (or other mode)
    - integrated_signal = direction_soft_signal if meta_mgmt_prob > meta_threshold and direction_soft_signal != 0
    - integrated_signal = 0 otherwise
    
    Args:
        direction_soft_signal: Soft signal from DirectionModel (thresholded at 0.52)
        meta_mgmt_prob: Meta management probability from Dynamic Ensemble
        meta_threshold_mode: Mode for computing meta_threshold ("median", "mean", "fixed")
        meta_threshold_value: Fixed threshold value if mode is "fixed"
        direction_threshold: Threshold for direction signal (default 0.52)
        
    Returns:
        integrated_signal array with values: 1 (Long), -1 (Short), 0 (Flat)
    """
    # Convert to numpy arrays if needed
    direction_soft_signal = np.asarray(direction_soft_signal)
    meta_mgmt_prob = np.asarray(meta_mgmt_prob)
    
    # Apply direction threshold
    direction_signal = np.where(
        direction_soft_signal > direction_threshold, 1,
        np.where(direction_soft_signal < -direction_threshold, -1, 0)
    )
    
    # Compute meta threshold
    if meta_threshold_mode == "median":
        meta_threshold = np.median(meta_mgmt_prob)
    elif meta_threshold_mode == "mean":
        meta_threshold = np.mean(meta_mgmt_prob)
    elif meta_threshold_mode == "fixed":
        if meta_threshold_value is None:
            raise ValueError("meta_threshold_value must be provided when mode is 'fixed'")
        meta_threshold = meta_threshold_value
    else:
        raise ValueError(f"Unknown meta_threshold_mode: {meta_threshold_mode}")
    
    # Apply meta filter
    integrated_signal = np.where(
        (meta_mgmt_prob > meta_threshold) & (direction_signal != 0),
        direction_signal,
        0
    )
    
    return integrated_signal


def apply_asymmetric_thresholds(
    direction_soft_signal: Union[np.ndarray, pd.Series],
    meta_prob: Union[np.ndarray, pd.Series],
    long_threshold: float = 0.52,
    short_threshold: float = 0.52,
    meta_threshold_mode: Literal["median", "mean", "fixed"] = "median",
    meta_threshold_value: float = None
) -> np.ndarray:
    """
    Apply asymmetric thresholds for long and short signals (Roadmap feature).
    
    Args:
        direction_soft_signal: Soft signal from DirectionModel
        meta_prob: Meta probability
        long_threshold: Threshold for long signals
        short_threshold: Threshold for short signals
        meta_threshold_mode: Mode for computing meta_threshold
        meta_threshold_value: Fixed threshold value if mode is "fixed"
        
    Returns:
        Signal array with asymmetric thresholds applied
    """
    # Convert to numpy arrays if needed
    direction_soft_signal = np.asarray(direction_soft_signal)
    meta_prob = np.asarray(meta_prob)
    
    # Apply asymmetric direction thresholds
    direction_signal = np.where(
        direction_soft_signal > long_threshold, 1,
        np.where(direction_soft_signal < -short_threshold, -1, 0)
    )
    
    # Compute meta threshold
    if meta_threshold_mode == "median":
        meta_threshold = np.median(meta_prob)
    elif meta_threshold_mode == "mean":
        meta_threshold = np.mean(meta_prob)
    elif meta_threshold_mode == "fixed":
        if meta_threshold_value is None:
            raise ValueError("meta_threshold_value must be provided when mode is 'fixed'")
        meta_threshold = meta_threshold_value
    else:
        raise ValueError(f"Unknown meta_threshold_mode: {meta_threshold_mode}")
    
    # Apply meta filter
    final_signal = np.where(
        (meta_prob > meta_threshold) & (direction_signal != 0),
        direction_signal,
        0
    )
    
    return final_signal
