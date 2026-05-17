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

**Интеграция:** для сквозного пайплайна с оркестратором каноничен
``decision.DecisionPipeline`` (integrated). ``SignalAssembler`` — автономный/legacy путь
с похожей логикой порогов; не дублируйте правила в двух местах без явной причины.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, Union


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
        meta_threshold_safe: bool = True,
        train_threshold: Optional[float] = None,
        threshold_window: Optional[int] = None,
        threshold_expanding: bool = False,
    ):
        """
        Initialize SignalAssembler.
        
        Args:
            direction_threshold: Threshold for directional signal (default: 0.52)
            meta_threshold_mode: Mode for meta threshold calculation ('median', 'fixed', 'adaptive')
            default_meta_threshold: Default meta threshold when mode is 'fixed'
            meta_threshold_safe: If True, use causal rolling/expanding via compute_safe_threshold
            train_threshold: Optional train-only scalar threshold (applied to full series)
            threshold_window: Rolling window for safe meta threshold (default 100 when safe)
            threshold_expanding: Use expanding window for meta threshold
        """
        self.direction_threshold = direction_threshold
        self.meta_threshold_mode = meta_threshold_mode
        self.default_meta_threshold = default_meta_threshold
        self.meta_threshold_safe = meta_threshold_safe
        self.train_threshold = train_threshold
        self.threshold_window = threshold_window
        self.threshold_expanding = threshold_expanding
        
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

        mp = np.asarray(meta_prob)
        if isinstance(meta_threshold, pd.Series):
            thr = meta_threshold.to_numpy(dtype=float)
        else:
            thr = np.asarray(meta_threshold, dtype=float)
        if thr.ndim == 0:
            meta_pass = mp > float(thr)
        else:
            if thr.shape != mp.shape:
                raise ValueError("meta_threshold must be scalar or same shape as meta_prob")
            meta_pass = mp > thr

        # Apply regime filtering if provided
        if regime_pred is not None and regime_filter is not None:
            regime_mask = np.array([regime_filter.get(int(r), True) for r in regime_pred])
        else:
            regime_mask = np.ones_like(meta_prob, dtype=bool)
        
        # Apply confidence filtering and directional signal
        final_signal = np.where(
            meta_pass
            & (direction_soft_signal != 0)
            & regime_mask,
            direction_soft_signal,
            0,
        )
        
        return final_signal
    
    def _calculate_meta_threshold(
        self, meta_prob: np.ndarray
    ) -> Union[float, pd.Series]:
        """
        Calculate meta threshold based on mode.
        
        When meta_threshold_safe is True, delegates to compute_safe_threshold (causal rolling/expanding
        or train-only scalar).
        """
        arr = np.asarray(meta_prob)

        if self.meta_threshold_mode == "fixed":
            return float(self.default_meta_threshold)

        if not self.meta_threshold_safe:
            if self.meta_threshold_mode == "median":
                return float(np.median(arr))
            if self.meta_threshold_mode == "adaptive":
                return float(np.percentile(arr, 75))
            raise ValueError(f"Unknown meta_threshold_mode: {self.meta_threshold_mode}")

        from utils.data_leakage_prevention import compute_safe_threshold

        eff_window = (
            None
            if self.threshold_expanding
            else (self.threshold_window if self.threshold_window is not None else 100)
        )

        mode = "median"
        pct_q = 0.5
        if self.meta_threshold_mode == "adaptive":
            mode = "percentile"
            pct_q = 0.75
        elif self.meta_threshold_mode != "median":
            raise ValueError(f"Unknown meta_threshold_mode: {self.meta_threshold_mode}")

        thr = compute_safe_threshold(
            arr,
            mode,
            eff_window,
            self.threshold_expanding,
            self.train_threshold,
            percentile_q=pct_q,
        )
        if isinstance(thr, pd.Series):
            return thr
        if getattr(thr, 'ndim', 0) == 0:
            return float(thr)
        return thr
    
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
