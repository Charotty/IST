"""
Threshold Manager

Centralized threshold management.

Current Thresholds:
- Direction threshold: 0.52
- Meta threshold: Median adaptive threshold
- Ensemble mode: Regime-adaptive
"""

import numpy as np
from typing import Dict, Optional, Union
from dataclasses import dataclass


@dataclass
class ThresholdConfig:
    """Configuration for meta-learning thresholds."""
    direction_threshold: float = 0.52
    meta_threshold_mode: str = "median"
    default_meta_threshold: float = 0.5
    ensemble_mode: str = "regime_adaptive"
    dl_window_size: int = 24


class ThresholdManager:
    """
    Centralized threshold management for meta-learning layer.
    
    Manages all thresholds used in signal generation and ensemble aggregation.
    """
    
    def __init__(
        self,
        config: Optional[ThresholdConfig] = None,
    ):
        """
        Initialize ThresholdManager.
        
        Args:
            config: ThresholdConfig object (uses defaults if None)
        """
        self.config = config or ThresholdConfig()
        
        # Cache for adaptive thresholds
        self._cached_thresholds: Dict[str, float] = {}
    
    def get_direction_threshold(self) -> float:
        """
        Get direction threshold.
        
        Returns:
            Direction threshold value
        """
        return self.config.direction_threshold
    
    def set_direction_threshold(self, threshold: float):
        """
        Set direction threshold.
        
        Args:
            threshold: New direction threshold
        """
        if 0 < threshold < 1:
            self.config.direction_threshold = threshold
        else:
            raise ValueError("Direction threshold must be between 0 and 1")
    
    def get_meta_threshold(
        self,
        meta_prob: Optional[np.ndarray] = None,
        mode: Optional[str] = None,
        safe_mode: bool = True,
        train_threshold: Optional[float] = None,
        window: Optional[int] = None,
        expanding: bool = False
    ) -> Union[float, np.ndarray]:
        """
        Get meta threshold.
        
        Args:
            meta_prob: Meta probability array (required for adaptive modes)
            mode: Threshold mode ('median', 'fixed', 'adaptive', 'percentile')
            safe_mode: If True, prevents look-ahead leakage
            train_threshold: Fixed threshold from training (for safe inference)
            window: Window size for rolling/expanding threshold
            expanding: If True, use expanding window instead of rolling
            
        Returns:
            Meta threshold value (or array if using rolling/expanding)
        """
        mode = mode or self.config.meta_threshold_mode
        
        if mode == "fixed":
            return self.config.default_meta_threshold
        
        if meta_prob is None:
            raise ValueError("meta_prob required for adaptive modes")
        
        if safe_mode:
            # Use safe threshold computation without look-ahead
            from utils.data_leakage_prevention import compute_safe_threshold
            
            if train_threshold is not None:
                return train_threshold
            
            w = self.config.dl_window_size if window is None else window
            if mode == "adaptive":
                return compute_safe_threshold(
                    meta_prob, "percentile", w, expanding, None, percentile_q=0.75
                )
            if mode == "percentile":
                return compute_safe_threshold(
                    meta_prob, "percentile", w, expanding, None, percentile_q=0.60
                )
            
            smode = "median"
            if mode == "mean":
                smode = "mean"
            
            return compute_safe_threshold(
                meta_prob, smode, w, expanding, None
            )
        
        # UNSAFE MODE - Only use for training, not for production inference
        if mode == "median":
            return float(np.median(meta_prob))
        elif mode == "adaptive":
            return float(np.percentile(meta_prob, 75))
        elif mode == "percentile":
            return float(np.percentile(meta_prob, 60))
        else:
            raise ValueError(f"Unknown meta_threshold_mode: {mode}")
    
    def set_meta_threshold_mode(self, mode: str):
        """
        Set meta threshold mode.
        
        Args:
            mode: New threshold mode
        """
        valid_modes = ["median", "fixed", "adaptive", "percentile"]
        if mode not in valid_modes:
            raise ValueError(f"meta_threshold_mode must be one of {valid_modes}")
        self.config.meta_threshold_mode = mode
    
    def get_ensemble_mode(self) -> str:
        """
        Get ensemble mode.
        
        Returns:
            Ensemble mode string
        """
        return self.config.ensemble_mode
    
    def set_ensemble_mode(self, mode: str):
        """
        Set ensemble mode.
        
        Args:
            mode: New ensemble mode ('regime_adaptive', 'fixed_trend', 'fixed_range')
        """
        valid_modes = ["regime_adaptive", "fixed_trend", "fixed_range"]
        if mode not in valid_modes:
            raise ValueError(f"ensemble_mode must be one of {valid_modes}")
        self.config.ensemble_mode = mode
    
    def get_dl_window_size(self) -> int:
        """
        Get deep learning window size.
        
        Returns:
            Window size for deep learning models
        """
        return self.config.dl_window_size
    
    def set_dl_window_size(self, window_size: int):
        """
        Set deep learning window size.
        
        Args:
            window_size: New window size
        """
        if window_size > 0:
            self.config.dl_window_size = window_size
        else:
            raise ValueError("Window size must be positive")
    
    def get_all_thresholds(
        self,
        meta_prob: Optional[np.ndarray] = None,
    ) -> Dict[str, Union[float, str]]:
        """
        Get all current thresholds.
        
        Args:
            meta_prob: Meta probability array (for adaptive threshold calculation)
            
        Returns:
            Dictionary of all thresholds
        """
        thresholds = {
            "direction_threshold": self.config.direction_threshold,
            "meta_threshold_mode": self.config.meta_threshold_mode,
            "ensemble_mode": self.config.ensemble_mode,
            "dl_window_size": self.config.dl_window_size,
        }
        
        # Calculate current meta threshold if meta_prob provided
        if meta_prob is not None:
            thresholds["current_meta_threshold"] = self.get_meta_threshold(meta_prob)
        else:
            thresholds["current_meta_threshold"] = self.config.default_meta_threshold
        
        return thresholds
    
    def update_from_dict(self, config_dict: Dict[str, Union[float, str, int]]):
        """
        Update configuration from dictionary.
        
        Args:
            config_dict: Dictionary with configuration values
        """
        if "direction_threshold" in config_dict:
            self.set_direction_threshold(config_dict["direction_threshold"])
        
        if "meta_threshold_mode" in config_dict:
            self.set_meta_threshold_mode(config_dict["meta_threshold_mode"])
        
        if "default_meta_threshold" in config_dict:
            self.config.default_meta_threshold = config_dict["default_meta_threshold"]
        
        if "ensemble_mode" in config_dict:
            self.set_ensemble_mode(config_dict["ensemble_mode"])
        
        if "dl_window_size" in config_dict:
            self.set_dl_window_size(config_dict["dl_window_size"])
    
    def to_dict(self) -> Dict[str, Union[float, str, int]]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of configuration
        """
        return {
            "direction_threshold": self.config.direction_threshold,
            "meta_threshold_mode": self.config.meta_threshold_mode,
            "default_meta_threshold": self.config.default_meta_threshold,
            "ensemble_mode": self.config.ensemble_mode,
            "dl_window_size": self.config.dl_window_size,
        }
    
    def reset_to_defaults(self):
        """Reset all thresholds to default values."""
        self.config = ThresholdConfig()
        self._cached_thresholds.clear()
