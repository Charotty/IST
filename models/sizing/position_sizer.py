"""
Position Sizing Module

Dynamic position sizing for exposure, leverage, and capital allocation.
Phase 1: Rule-based sizing
Phase 2: Bayesian sizing
Phase 3: RL-based allocation
"""

import numpy as np
import pandas as pd
from typing import Dict, Any


class PositionSizer:
    """
    Position sizer for dynamic position sizing.
    
    Phase 1: Rule-based sizing
    """
    
    def __init__(self, base_size=0.1, max_size=1.0, min_size=0.01):
        """
        Initialize position sizer.
        
        :param base_size: Base position size as fraction of capital
        :param max_size: Maximum position size
        :param min_size: Minimum position size
        """
        self.base_size = base_size
        self.max_size = max_size
        self.min_size = min_size
    
    def calculate_size(self, confidence: float, volatility: float, regime: str = 'neutral') -> float:
        """
        Calculate position size based on confidence and volatility.
        
        :param confidence: Model confidence (0-1)
        :param volatility: Current volatility
        :param regime: Market regime
        :return: Position size as fraction of capital
        """
        # Base size adjusted by confidence
        size = self.base_size * confidence
        
        # Volatility adjustment: lower size in high volatility
        vol_adjustment = 1.0 / (1.0 + volatility)
        size *= vol_adjustment
        
        # Regime adjustment
        if regime == 'trend':
            size *= 1.2  # Increase size in trend
        elif regime == 'range':
            size *= 0.8  # Decrease size in range
        
        # Clamp to min/max
        size = max(self.min_size, min(self.max_size, size))
        
        return size
    
    def calculate_kelly_size(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Calculate Kelly criterion position size.
        
        :param win_rate: Win rate (0-1)
        :param avg_win: Average win
        :param avg_loss: Average loss (positive value)
        :return: Kelly fraction
        """
        if avg_loss == 0:
            return 0.0
        
        kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        return max(0.0, min(1.0, kelly * 0.5))  # Half-Kelly for safety
    
    def get_position_info(self, signal: int, confidence: float, volatility: float, regime: str = 'neutral') -> Dict[str, Any]:
        """
        Get detailed position information.
        
        :param signal: Trading signal (-1, 0, 1)
        :param confidence: Model confidence
        :param volatility: Current volatility
        :param regime: Market regime
        :return: Dict with position info
        """
        if signal == 0:
            return {
                "position_size": 0.0,
                "direction": "neutral",
                "leverage": 1.0
            }
        
        size = self.calculate_size(confidence, volatility, regime)
        direction = "long" if signal > 0 else "short"
        
        return {
            "position_size": size,
            "direction": direction,
            "leverage": 1.0  # Phase 1: no leverage
        }
