"""
Signal rules for decision layer.

Provides functions to compute final_signal and integrated_signal based on
direction_soft_signal, meta_prob, and meta_mgmt_prob.
"""

import numpy as np
import pandas as pd
from typing import Union, Literal, Optional

_DEFAULT_SAFE_META_WINDOW = 100


def _meta_prob_exceeds_threshold(
    meta_prob: np.ndarray,
    meta_threshold: Union[float, pd.Series, np.ndarray],
) -> np.ndarray:
    """Element-wise comparison; supports scalar threshold or causal Series aligned to meta_prob."""
    mp = np.asarray(meta_prob, dtype=float)
    if isinstance(meta_threshold, pd.Series):
        mt = meta_threshold.to_numpy(dtype=float)
    else:
        mt = np.asarray(meta_threshold, dtype=float)
    if mt.ndim == 0:
        return mp > float(mt)
    if mt.shape != mp.shape:
        raise ValueError(
            "meta_threshold has wrong shape after rolling/expanding computation; "
            f"expected {mp.shape}, got {mt.shape}."
        )
    return mp > mt


def compute_final_signal(
    direction_soft_signal: Union[np.ndarray, pd.Series],
    meta_prob: Union[np.ndarray, pd.Series],
    meta_threshold_mode: Literal["median", "mean", "fixed"] = "median",
    meta_threshold_value: float = None,
    direction_threshold: float = 0.52,
    safe_mode: bool = True,
    train_threshold: Optional[float] = None,
    window: Optional[int] = None,
    expanding: bool = False
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
        safe_mode: If True, prevents look-ahead leakage in threshold computation
        train_threshold: Fixed threshold from training (for safe inference)
        window: Window size for rolling/expanding threshold
        expanding: If True, use expanding window instead of rolling
        
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
    if meta_threshold_mode == "fixed":
        if meta_threshold_value is None:
            raise ValueError("meta_threshold_value must be provided when mode is 'fixed'")
        meta_threshold = meta_threshold_value
    elif safe_mode:
        # Use safe threshold computation without look-ahead
        from utils.data_leakage_prevention import compute_safe_threshold
        eff_window = None if expanding else (window if window is not None else _DEFAULT_SAFE_META_WINDOW)
        meta_threshold = compute_safe_threshold(
            meta_prob, meta_threshold_mode, eff_window, expanding, train_threshold
        )
    else:
        # UNSAFE MODE - Only use for training, not for production inference
        if meta_threshold_mode == "median":
            meta_threshold = np.median(meta_prob)
        elif meta_threshold_mode == "mean":
            meta_threshold = np.mean(meta_prob)
        else:
            raise ValueError(f"Unknown meta_threshold_mode: {meta_threshold_mode}")
    
    meta_pass = _meta_prob_exceeds_threshold(meta_prob, meta_threshold)
    
    # Apply meta filter
    final_signal = np.where(
        meta_pass & (direction_signal != 0),
        direction_signal,
        0
    )
    
    return final_signal


def compute_integrated_signal(
    direction_soft_signal: Union[np.ndarray, pd.Series],
    meta_mgmt_prob: Union[np.ndarray, pd.Series],
    meta_threshold_mode: Literal["median", "mean", "fixed"] = "median",
    meta_threshold_value: float = None,
    direction_threshold: float = 0.52,
    safe_mode: bool = True,
    train_threshold: Optional[float] = None,
    window: Optional[int] = None,
    expanding: bool = False
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
        safe_mode: If True, prevents look-ahead leakage in threshold computation
        train_threshold: Fixed threshold from training (for safe inference)
        window: Window size for rolling/expanding threshold
        expanding: If True, use expanding window instead of rolling
        
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
    if meta_threshold_mode == "fixed":
        if meta_threshold_value is None:
            raise ValueError("meta_threshold_value must be provided when mode is 'fixed'")
        meta_threshold = meta_threshold_value
    elif safe_mode:
        # Use safe threshold computation without look-ahead
        from utils.data_leakage_prevention import compute_safe_threshold
        eff_window = None if expanding else (window if window is not None else _DEFAULT_SAFE_META_WINDOW)
        meta_threshold = compute_safe_threshold(
            meta_mgmt_prob, meta_threshold_mode, eff_window, expanding, train_threshold
        )
    else:
        # UNSAFE MODE - Only use for training, not for production inference
        if meta_threshold_mode == "median":
            meta_threshold = np.median(meta_mgmt_prob)
        elif meta_threshold_mode == "mean":
            meta_threshold = np.mean(meta_mgmt_prob)
        else:
            raise ValueError(f"Unknown meta_threshold_mode: {meta_threshold_mode}")
    
    meta_pass = _meta_prob_exceeds_threshold(meta_mgmt_prob, meta_threshold)
    
    # Apply meta filter
    integrated_signal = np.where(
        meta_pass & (direction_signal != 0),
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
    meta_threshold_value: float = None,
    safe_mode: bool = True,
    train_threshold: Optional[float] = None,
    window: Optional[int] = None,
    expanding: bool = False
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
        safe_mode: If True, prevents look-ahead leakage in threshold computation
        train_threshold: Fixed threshold from training (for safe inference)
        window: Window size for rolling/expanding threshold
        expanding: If True, use expanding window instead of rolling
        
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
    if meta_threshold_mode == "fixed":
        if meta_threshold_value is None:
            raise ValueError("meta_threshold_value must be provided when mode is 'fixed'")
        meta_threshold = meta_threshold_value
    elif safe_mode:
        # Use safe threshold computation without look-ahead
        from utils.data_leakage_prevention import compute_safe_threshold
        eff_window = None if expanding else (window if window is not None else _DEFAULT_SAFE_META_WINDOW)
        meta_threshold = compute_safe_threshold(
            meta_prob, meta_threshold_mode, eff_window, expanding, train_threshold
        )
    else:
        # UNSAFE MODE - Only use for training, not for production inference
        if meta_threshold_mode == "median":
            meta_threshold = np.median(meta_prob)
        elif meta_threshold_mode == "mean":
            meta_threshold = np.mean(meta_prob)
        else:
            raise ValueError(f"Unknown meta_threshold_mode: {meta_threshold_mode}")
    
    meta_pass = _meta_prob_exceeds_threshold(meta_prob, meta_threshold)
    
    # Apply meta filter
    final_signal = np.where(
        meta_pass & (direction_signal != 0),
        direction_signal,
        0
    )
    
    return final_signal
