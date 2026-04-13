from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
from enum import Enum


class Action(Enum):
    """Possible actions."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE_LONG = "close_long"
    CLOSE_SHORT = "close_short"


@dataclass
class Signal:
    """Signal from model."""
    action: Action
    confidence: float  # 0.0 - 1.0
    timestamp: int
    symbol: str
    metadata: Dict[str, Any]


@dataclass
class Decision:
    """Trading decision."""
    action: Action
    symbol: str
    size: float  # Position size
    price: Optional[float]  # Price (None for market order)
    stop_loss: Optional[float]
    take_profit: Optional[float]
    timestamp: int
    reason: str  # Why this decision was made


class BaseDecisionMaker(ABC):
    """Base class for decision making."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.confidence_threshold = config.get("confidence_threshold", 0.7)

    @abstractmethod
    def decide(
        self,
        signal: Signal,
        market_state: Dict[str, Any],
    ) -> Optional[Decision]:
        """
        Make decision based on signal.

        Args:
            signal: signal from model
            market_state: current market state

        Returns:
            Decision or None (if no action required)
        """
        pass

    def validate_signal(self, signal: Signal) -> bool:
        """Validate signal."""
        if signal.confidence < 0 or signal.confidence > 1:
            return False
        if signal.confidence < self.confidence_threshold:
            return False
        return True
