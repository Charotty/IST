"""
Signal Assembler

Final signal generation layer.

Combines:
- directional prediction
- meta probability
- confidence thresholds
- regime filtering

Responsibilities:
- approve/reject trades
- reduce low-quality entries
- enforce confidence filtering
"""

import numpy as np
from typing import Dict, Optional, Tuple


class SignalAssembler:
    """
    Final signal generation layer.
    
    Combines directional prediction, meta probability, confidence thresholds,
    and regime filtering to produce the final trading signal.
    """
    
    def __init__(
        self,
        direction_threshold: float = 0.52,
        meta_threshold_mode: str = "median",
        default_meta_threshold: float = 0.5,
    ):
        """
        Initialize SignalAssembler.
        
        Args:
            direction_threshold: Threshold for directional signal (default: 0.52)
            meta_threshold_mode: Mode for meta threshold calculation ('median', 'fixed', 'adaptive')
            default_meta_threshold: Default meta threshold when mode is 'fixed'
        """
        self.direction_threshold = direction_threshold
        self.meta_threshold_mode = meta_threshold_mode
        self.default_meta_threshold = default_meta_threshold
        
    def assemble_signal(
        self,
        direction_soft_signal: np.ndarray,
        meta_prob: np.ndarray,
        meta_threshold: Optional[float] = None,
        regime_pred: Optional[np.ndarray] = None,
        regime_filter: Optional[Dict[int, bool]] = None,
    ) -> np.ndarray:
        """
        Assemble final trading signal.
        
        Current logic:
        final_signal = np.where(
            (meta_prob > meta_threshold)
            & (direction_soft_signal != 0),
            direction_soft_signal,
            0
        )
        
        Args:
            direction_soft_signal: Soft directional signal (-1 to 1)
            meta_prob: Meta probability from meta-filter model
            meta_threshold: Threshold for meta probability (optional, calculated if None)
            regime_pred: Regime prediction (1=trend, 0=range)
            regime_filter: Optional regime filter dict {regime: allow_trades}
            
        Returns:
            Final trading signal (-1, 0, 1)
        """
        # Calculate meta threshold if not provided
        if meta_threshold is None:
            meta_threshold = self._calculate_meta_threshold(meta_prob)
        
        # Apply regime filtering if provided
        if regime_pred is not None and regime_filter is not None:
            regime_mask = np.array([regime_filter.get(int(r), True) for r in regime_pred])
        else:
            regime_mask = np.ones_like(meta_prob, dtype=bool)
        
        # Apply confidence filtering and directional signal
        final_signal = np.where(
            (meta_prob > meta_threshold)
            & (direction_soft_signal != 0)
            & regime_mask,
            direction_soft_signal,
            0
        )
        
        return final_signal
    
    def _calculate_meta_threshold(self, meta_prob: np.ndarray) -> float:
        """
        Calculate meta threshold based on mode.
        
        Args:
            meta_prob: Meta probability array
            
        Returns:
            Calculated threshold
        """
        if self.meta_threshold_mode == "median":
            return np.median(meta_prob)
        elif self.meta_threshold_mode == "fixed":
            return self.default_meta_threshold
        elif self.meta_threshold_mode == "adaptive":
            # Adaptive: use 75th percentile
            return np.percentile(meta_prob, 75)
        else:
            raise ValueError(f"Unknown meta_threshold_mode: {self.meta_threshold_mode}")
    
    def get_signal_statistics(
        self,
        final_signal: np.ndarray,
        meta_prob: np.ndarray,
        direction_soft_signal: np.ndarray,
    ) -> Dict[str, float]:
        """
        Get statistics about the assembled signal.
        
        Args:
            final_signal: Final trading signal
            meta_prob: Meta probability
            direction_soft_signal: Directional soft signal
            
        Returns:
            Dictionary of statistics
        """
        return {
            "signal_count": np.sum(final_signal != 0),
            "long_count": np.sum(final_signal > 0),
            "short_count": np.sum(final_signal < 0),
            "no_trade_count": np.sum(final_signal == 0),
            "avg_meta_prob": np.mean(meta_prob[final_signal != 0]) if np.any(final_signal != 0) else 0.0,
            "avg_direction_strength": np.mean(np.abs(direction_soft_signal[final_signal != 0])) if np.any(final_signal != 0) else 0.0,
        }
    
    def apply_direction_threshold(
        self,
        direction_prob: np.ndarray,
    ) -> np.ndarray:
        """
        Apply direction threshold to probability.
        
        Args:
            direction_prob: Directional probability (0 to 1)
            
        Returns:
            Soft directional signal (-1, 0, 1)
        """
        direction_soft_signal = np.zeros_like(direction_prob)
        
        # Long signals
        direction_soft_signal[direction_prob > self.direction_threshold] = 1
        
        # Short signals
        direction_soft_signal[direction_prob < (1 - self.direction_threshold)] = -1
        
        return direction_soft_signal
