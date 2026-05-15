"""
Signal Utils

Utility functions for signal processing and manipulation.
"""

import numpy as np
from typing import Dict, Optional, Tuple


class SignalUtils:
    """
    Utility functions for signal processing.
    """
    
    @staticmethod
    def softmax_to_signal(
        softmax_probs: np.ndarray,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Convert softmax probabilities to trading signals.
        
        Args:
            softmax_probs: Softmax probabilities (n_samples, n_classes)
            threshold: Threshold for signal generation
            
        Returns:
            Trading signals (-1, 0, 1)
        """
        # Assume binary classification: [down_prob, up_prob]
        if softmax_probs.ndim == 2 and softmax_probs.shape[1] == 2:
            up_prob = softmax_probs[:, 1]
        else:
            up_prob = softmax_probs
        
        signals = np.zeros_like(up_prob)
        signals[up_prob > threshold] = 1
        signals[up_prob < (1 - threshold)] = -1
        
        return signals
    
    @staticmethod
    def smooth_signals(
        signals: np.ndarray,
        window_size: int = 3,
    ) -> np.ndarray:
        """
        Smooth signals using moving average.
        
        Args:
            signals: Trading signals
            window_size: Window size for smoothing
            
        Returns:
            Smoothed signals
        """
        if window_size <= 1:
            return signals.copy()
        
        smoothed = np.convolve(signals, np.ones(window_size) / window_size, mode='same')
        
        # Convert back to -1, 0, 1
        final_signals = np.zeros_like(smoothed)
        final_signals[smoothed > 0.3] = 1
        final_signals[smoothed < -0.3] = -1
        
        return final_signals
    
    @staticmethod
    def filter_low_confidence(
        signals: np.ndarray,
        confidence: np.ndarray,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Filter out low-confidence signals.
        
        Args:
            signals: Trading signals
            confidence: Confidence scores
            threshold: Confidence threshold
            
        Returns:
            Filtered signals
        """
        filtered = signals.copy()
        filtered[confidence < threshold] = 0
        return filtered
    
    @staticmethod
    def get_signal_changes(
        signals: np.ndarray,
    ) -> np.ndarray:
        """
        Get signal change points.
        
        Args:
            signals: Trading signals
            
        Returns:
            Boolean array indicating where signals change
        """
        changes = np.zeros_like(signals, dtype=bool)
        changes[1:] = signals[1:] != signals[:-1]
        return changes
    
    @staticmethod
    def calculate_signal_persistence(
        signals: np.ndarray,
    ) -> Dict[str, float]:
        """
        Calculate signal persistence metrics.
        
        Args:
            signals: Trading signals
            
        Returns:
            Dictionary of persistence metrics
        """
        changes = SignalUtils.get_signal_changes(signals)
        change_indices = np.where(changes)[0]
        
        if len(change_indices) > 0:
            durations = np.diff(np.concatenate([[0], change_indices, [len(signals)]]))
            avg_duration = np.mean(durations)
            max_duration = np.max(durations)
            min_duration = np.min(durations)
        else:
            avg_duration = len(signals)
            max_duration = len(signals)
            min_duration = len(signals)
        
        return {
            "avg_duration": float(avg_duration),
            "max_duration": float(max_duration),
            "min_duration": float(min_duration),
            "total_changes": int(np.sum(changes)),
        }
    
    @staticmethod
    def align_signals(
        signals1: np.ndarray,
        signals2: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Align two signal arrays to the same length.
        
        Args:
            signals1: First signal array
            signals2: Second signal array
            
        Returns:
            Tuple of aligned signal arrays
        """
        min_len = min(len(signals1), len(signals2))
        return signals1[:min_len], signals2[:min_len]
